"""Persistent Echo-WM worker (file-based IPC), mirrors _evoke_worker.py.

Loaded ONCE inside Echo's venv (JoyAI-Echo/echo_wm/.venv). The MIND-side driver
(``src/drive_echo.py``) writes a JSON-Lines manifest, then spawns this process
pointed at it. The worker builds TI2VidOneStagePipeline once and processes
requests serially.

Why this exists: ``inference_wm.py`` is a one-shot CLI that constructs the
pipeline, generates a single clip and exits. Driving MIND through it reloads a
~47.8 GB checkpoint plus the Gemma text encoder for every one of 250 samples.
At ~5 min of sampling per clip the reloads dominated the wall clock; a full pass
was 25-30 hours, most of it spent loading the same weights again.

The pipeline is safe to reuse because everything that varies per sample is a
CALL argument (prompt, action condition, num_frames, seed, output path). The one
constructor-level input is ``action_config(width, height)``, so a single worker
is valid for a fixed resolution -- the driver starts one worker per distinct
(width, height), which in practice is one.

Manifest line format (one per line)::

    {"id": <int>, "tag": <str>, "image": <path>, "prompt": <str>,
     "action_str": "w-24,s-24", "target_path": <path>,
     "num_frames": 49, "fps": 24.0, "steps": 30, "seed": 2,
     "no_audio": true, "action_overlay": false}

Result line format (one per line, appended)::

    {"id": <int>, "ok": true,  "elapsed_s": 51.5, "target_path": <path>}
    {"id": <int>, "ok": false, "error": "<type>: <message>"}

num_frames varies per request -- mirror clips are fitted to their trajectory
(49 frames) while the main sets use 97 -- and that is a call argument, so both
kinds run through one loaded pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

# Echo's own entrypoint does this before importing ltx_*; the packages are not
# installed, they are vendored subdirectories of the repo.
ECHO_WM = Path(__file__).resolve().parent.parent.parent / "JoyAI-Echo" / "echo_wm"
if "ECHO_WM_ROOT" in __import__("os").environ:
    ECHO_WM = Path(__import__("os").environ["ECHO_WM_ROOT"])
sys.path.insert(0, str(ECHO_WM))
for _package in ("ltx-core/src", "ltx-pipelines/src"):
    sys.path.insert(0, str(ECHO_WM / _package))

from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: E402
from ltx_core.model.video_vae.tiling import TilingConfig  # noqa: E402
from ltx_core.model.video_vae.video_vae import get_video_chunks_number  # noqa: E402
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline  # noqa: E402
from ltx_pipelines.utils.args import ImageConditioningInput  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402

from helpers.action_camera import (  # noqa: E402
    DEFAULT_PITCH_LIMIT_DEG,
    DEFAULT_ROTATION_SPEED_DEG,
    DEFAULT_TRANSLATION_SPEED,
)
from helpers.action_condition import action_config, build_action_condition  # noqa: E402

NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "game UI, video game interface, HUD, heads-up display, menu, status bar, "
    "health bar, score, minimap, crosshair, reticle, buttons, icons, subtitles, "
    "captions, watermark, logo, text overlay, user interface"
)


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="JSON-Lines file with one request per line.")
    ap.add_argument("--results-path", type=Path, required=True,
                    help="Worker appends one JSON line per processed item.")
    ap.add_argument("--status-path", type=Path, required=True,
                    help="Worker rewrites a small status JSON snapshot here.")
    ap.add_argument("--config", type=Path, default=ECHO_WM / "configs" / "inference_wm.yaml")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--gemma-path", type=Path, required=True)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    video_cfg = cfg.get("video", {})
    action_cfg = cfg.get("action", {})
    width = args.width or video_cfg.get("width", 1280)
    height = args.height or video_cfg.get("height", 704)
    negative_prompt = cfg.get("negative_prompt", NEGATIVE_PROMPT)
    stg_scale = video_cfg.get("stg_scale", 1.0)
    stg_blocks = video_cfg.get("stg_blocks", [29])
    video_cfg_scale = video_cfg.get("video_cfg", 4.0)
    audio_cfg_scale = video_cfg.get("audio_cfg", 2.0)
    fov = action_cfg.get("fov_deg", 70.0)
    translation_speed = action_cfg.get("translation_speed", DEFAULT_TRANSLATION_SPEED)
    rotation_speed_deg = action_cfg.get("rotation_speed_deg", DEFAULT_ROTATION_SPEED_DEG)
    pitch_limit_deg = action_cfg.get("pitch_limit_deg", DEFAULT_PITCH_LIMIT_DEG)

    requests: list[dict] = []
    with open(args.manifest, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                requests.append(json.loads(line))

    total = len(requests)
    _write_status(args.status_path, {"phase": "loading", "done": 0, "total": total})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_load = time.time()
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=str(args.checkpoint), gemma_root=str(args.gemma_path), loras=(),
        device=device, action_config=action_config(width, height),
    )
    load_s = time.time() - t_load
    print(f"[echo-worker] pipeline loaded in {load_s:.1f}s for {total} request(s) "
          f"@ {width}x{height}", flush=True)
    _write_status(args.status_path, {"phase": "ready", "done": 0, "total": total,
                                     "load_s": round(load_s, 1)})

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    with open(args.results_path, "a", encoding="utf-8") as results_fh:
        for i, req in enumerate(requests):
            req_id = req.get("id", i)
            tag = req.get("tag", str(req_id))
            target_path = Path(req["target_path"])
            num_frames = int(req["num_frames"])
            fps = float(req.get("fps", 24.0))
            steps = int(req.get("steps", video_cfg.get("steps", 30)))
            seed = int(req.get("seed", 42))
            t0 = time.time()
            try:
                action = build_action_condition(
                    req["action_str"], num_frames=num_frames, width=width, height=height,
                    translation_speed=translation_speed,
                    rotation_speed_deg=rotation_speed_deg,
                    pitch_limit_deg=pitch_limit_deg,
                    fov_deg=fov, device=device, fps=fps,
                )
                video, audio = pipeline(
                    prompt=req["prompt"], negative_prompt=negative_prompt, seed=seed,
                    height=height, width=width, num_frames=num_frames, frame_rate=fps,
                    num_inference_steps=steps,
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=video_cfg_scale, stg_scale=stg_scale, stg_blocks=stg_blocks),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=audio_cfg_scale, stg_scale=stg_scale, stg_blocks=stg_blocks),
                    images=[ImageConditioningInput(str(req["image"]), 0, 1.0)],
                    action_cond=action,
                    video_tiling_config=TilingConfig.default(),
                )
                target_path.parent.mkdir(parents=True, exist_ok=True)
                encode_video(
                    video=video, fps=int(fps),
                    audio=None if req.get("no_audio", True) else audio,
                    output_path=str(target_path),
                    video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
                )
                elapsed = time.time() - t0
                done += 1
                results_fh.write(json.dumps({
                    "id": req_id, "ok": True, "elapsed_s": round(elapsed, 1),
                    "target_path": str(target_path)}) + "\n")
                print(f"[echo-worker] {i + 1}/{total} {tag} ok in {elapsed:.1f}s "
                      f"({num_frames}f)", flush=True)
            except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the batch
                failed += 1
                results_fh.write(json.dumps({
                    "id": req_id, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                print(f"[echo-worker] {i + 1}/{total} {tag} FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
            results_fh.flush()
            _write_status(args.status_path, {"phase": "running", "done": done,
                                             "failed": failed, "total": total})
            torch.cuda.empty_cache()

    _write_status(args.status_path, {"phase": "finished", "done": done,
                                     "failed": failed, "total": total})
    print(f"[echo-worker] finished: {done} ok, {failed} failed, load {load_s:.1f}s", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
