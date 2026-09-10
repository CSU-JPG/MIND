"""Drive Evoke (i2v-style, camera-controlled) from a MIND-Data tree.

Parallel to drive_helios_i2v.py: walks MIND-Data, for each sample extracts the first frame +
converts its action.json into an Evoke-compatible cam_c2w pose track (see utils/evoke_pose.py --
uses MIND's real actor_pos/actor_rpy / camera_pos/camera_rpy ground truth, not a ws/ad/ud/lr
dead-reckoning approximation), then dispatches to a persistent _evoke_worker.py that loads
EvokePipeline ONCE and processes all samples in a single subprocess.

Unlike Helios (image-conditioned, no action input -> action_space_test is skipped there), Evoke
DOES take a real camera pose trajectory, so this driver runs BOTH action_space_test and mem_test
by default across BOTH 1st_data and 3rd_data -- this is also how we find out whether Evoke's
first-person-only training shows up as a real perspective gap in the scores (it was never
demonstrated on 3rd-person footage anywhere in its own repo/examples).

Output layout::

    <test_root>/evoke/<perspective>/<test_type>/<gt_name>/video.mp4

Cross-venv: the worker lives in Evoke's own venv (C:\\workspace\\world\\Evoke\\.venv). See
_evoke_worker.py's docstring for the fp16-VAE / stage2-scheduler / CFG-off pitfalls that produce
silent near-pure-noise output if any one of them is missed.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np

from utils.evoke_pose import default_intrinsic, mind_action_to_c2w
from utils.scene_prompts import scene_prompt
from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4

EVOKE_REPO = Path(__file__).resolve().parent.parent.parent / "Evoke"
DEFAULT_EVOKE_VENV_PY = EVOKE_REPO / ".venv" / "Scripts" / "python.exe"
EVOKE_VENV_PY = Path(os.environ.get("EVOKE_VENV_PY", str(DEFAULT_EVOKE_VENV_PY)))
EVOKE_MODEL_PATH = os.environ.get("EVOKE_MODEL_PATH")  # resolved lazily in main() if unset

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")
# MIND-Data video.mp4 source resolution -- verified empirically across every sample checked
# (2026-08-17): always 1920x1080 regardless of perspective/test_type.
SOURCE_RESOLUTION = (1080, 1920)  # (h, w), matches load_pose_for_v2v's convention


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_image()
            img.save(out_path, "PNG")
            return
    raise RuntimeError(f"No frames decoded from {video_path}")


def gather_samples(gt_root: Path) -> list[dict]:
    samples: list[dict] = []
    for perspective in PERSPECTIVES:
        for test_type in TEST_TYPES:
            type_dir = gt_root / perspective / "test" / test_type
            if not type_dir.is_dir():
                continue
            for sample_dir in sorted(type_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                video = sample_dir / "video.mp4"
                action = sample_dir / "action.json"
                if not (video.exists() and action.exists()):
                    continue
                samples.append({
                    "perspective": perspective,
                    "test_type": test_type,
                    "gt_name": sample_dir.name,
                    "video": video,
                    "action": action,
                })
    return samples


def output_path(test_root: Path, model_name: str, sample: dict) -> Path:
    return test_root / model_name / sample["perspective"] / sample["test_type"] / sample["gt_name"] / "video.mp4"


def build_prompt(sample: dict, enhance: bool) -> str:
    """MIND's action.json carries no caption field -- fall back to a perspective-flavored default
    (same convention as drive_helios_i2v.py's build_prompt). Both the baseline
    and the --enhance-prompt text live in utils/scene_prompts.py so the wording
    can be iterated without touching driver logic, and so the other drivers can
    share one text rather than growing their own variants."""
    return scene_prompt(sample["perspective"], enhance)


def _resolve_model_path() -> str:
    if EVOKE_MODEL_PATH:
        return EVOKE_MODEL_PATH
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = list((hf_cache / "models--SII-YuanyangYin--Evoke" / "snapshots").glob("*"))
    if not snapshots:
        raise RuntimeError("No Evoke model snapshots found; run download_models.bat in the Evoke repo first.")
    return str(snapshots[0] / "evoke-base")


def run_persistent(samples: list[dict], args: argparse.Namespace, work_dir: Path) -> int:
    """Spawn the persistent Evoke worker ONCE and process all samples.

    Mirrors drive_helios_i2v.py's run_persistent: pre-extract first frames + build per-sample
    pose.npz into work_dir, write one manifest, spawn the worker, then report success/failure
    counts from results.jsonl.
    """
    ipc_dir = work_dir / "ipc" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ipc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ipc_dir / "manifest.jsonl"
    results_path = ipc_dir / "results.jsonl"
    status_path = ipc_dir / "status.json"

    items: list[dict] = []
    skipped = 0
    for idx, s in enumerate(samples):
        out = output_path(args.test_root, args.model_name, s)
        if out.exists():
            print(f"[skip] {s['perspective']}/{s['test_type']}/{s['gt_name']} -> {out} (exists)")
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        frame_png = work_dir / s["perspective"] / s["test_type"] / f"{s['gt_name']}.png"
        frame_png.parent.mkdir(parents=True, exist_ok=True)
        if s.get("frame_png_src") is not None:
            shutil.copy2(s["frame_png_src"], frame_png)
        elif not frame_png.exists():
            extract_first_frame(s["video"], frame_png)

        action_json = json.loads(Path(s["action"]).read_text(encoding="utf-8"))
        c2ws = mind_action_to_c2w(action_json, s["perspective"])
        num_frames = min(len(c2ws), args.num_frames) if args.num_frames else len(c2ws)
        # Evoke's hard floor: (latent_window_size-1)*4+1 = 33.
        num_frames = max(num_frames, 33)
        c2ws = c2ws[:num_frames] if len(c2ws) >= num_frames else np.concatenate(
            [c2ws, np.repeat(c2ws[-1:], num_frames - len(c2ws), axis=0)]
        )

        pose_npz = work_dir / s["perspective"] / s["test_type"] / f"{s['gt_name']}_pose.npz"
        pose_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pose_npz, cam_c2w=c2ws, intrinsics=default_intrinsic())

        items.append({
            "id": idx,
            "tag": f"{s['perspective']}/{s['test_type']}/{s['gt_name']}",
            "image": str(frame_png),
            "prompt": build_prompt(s, args.enhance_prompt),
            "pose_npz": str(pose_npz),
            "pose_source_resolution": list(SOURCE_RESOLUTION),
            "pose_fps": 24,  # action.json ticks == video frames, both at 24fps (verified empirically)
            "target_path": str(out),
            "height": args.height,
            "width": args.width,
            "num_frames": num_frames,
            "seed": args.seed,
            "fps": args.fps,
            "image_noise_sigma_min": args.image_noise_sigma_min,
            "image_noise_sigma_max": args.image_noise_sigma_max,
        })

    print(f"{skipped} already produced; manifesting {len(items)} new sample(s).")
    if not items:
        print("Nothing to do.")
        return 0

    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"[ipc] manifest : {manifest_path}")
    print(f"[ipc] results  : {results_path}")
    print(f"[ipc] status   : {status_path}")

    worker_script = Path(__file__).parent / "_evoke_worker.py"
    if not worker_script.exists():
        print(f"FATAL: worker not found at {worker_script}", file=sys.stderr)
        return 2

    model_path = _resolve_model_path()
    cmd = [
        str(EVOKE_VENV_PY), "-X", "utf8", str(worker_script),
        "--manifest", str(manifest_path),
        "--results-path", str(results_path),
        "--status-path", str(status_path),
        "--model-path", model_path,
    ]

    # Strip cross-venv pollution before spawning Evoke's interpreter (same fix as
    # drive_helios_i2v.py / drive_matrix2.py -- avoids SRE MAGIC stdlib mismatch between
    # MIND's Python 3.10 and Evoke's 3.11 venv).
    env = os.environ.copy()
    for key in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EVOKE_REPO"] = str(EVOKE_REPO)

    print(f"\nSpawning persistent worker (cwd={EVOKE_REPO}):")
    print("  " + " ".join(map(str, cmd)))
    print()

    t0 = time.perf_counter()
    rc = subprocess.run(cmd, cwd=str(EVOKE_REPO), env=env).returncode
    elapsed = time.perf_counter() - t0

    successes = 0
    failures: list[str] = []
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if result.get("ok"):
                    successes += 1
                    tp = result.get("target_path")
                    if tp and Path(tp).exists() and not result.get("skipped"):
                        # Find the matching sample for stats logging.
                        for s in samples:
                            if str(output_path(args.test_root, args.model_name, s)) == tp:
                                log_mp4(args.model_name, s["perspective"], s["test_type"], s["gt_name"], Path(tp))
                                break
                else:
                    failures.append(result.get("error", "?"))

    print(f"\n--- worker exit={rc}  elapsed={elapsed:.1f}s ---")
    print(f"--- {successes} succeeded, {len(failures)} failed, {skipped} pre-existing skipped ---")
    if failures:
        print("Sample errors (first 5):")
        for err in failures[:5]:
            print(f"  {err[:200]}")
        return 1
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root", type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="evoke", help="Subfolder name under test-root")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Temp dir for extracted frames + pose npz (default: <test-root>/.frames-evoke)")
    parser.add_argument("--only", nargs="+",
                        help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type", choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit", type=int, help="Only run first N matched samples")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip first N matched samples after filters; used for mid-run resume.")
    # 384x640 is Evoke's own paper/training resolution (Sec 4.1). 256x448 was tried first for
    # speed, but the coarse-to-fine pyramid's first stage (12x20, scaled off this res) got too
    # small to preserve fine subject detail at 256x448 -- confirmed by a controlled test on
    # Evoke's own turtle example (examples/i2v/): identical run at 384x640 recovered a subject
    # that was missing entirely at 256x448, while environment/scene-level content was fine at
    # either resolution. The MIND-driven grid/mesh artifact sample is likely the same failure mode.
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=61,
                        help="Capped to each sample's actual action.json length; floored at 33 "
                             "(Evoke's hard minimum). Shorter GT trajectories are held on their "
                             "last pose rather than padded with motion that isn't there.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--image-noise-sigma-min", type=float, default=0.0,
                        help="Pipeline default is 0.111; lower keeps the model anchored closer "
                             "to the seed-image pixels instead of drifting toward the (often "
                             "generic, since MIND has no captions) text prompt. Dropped from "
                             "0.02 -> 0.0 after a dark/detail-heavy MIND sample (mannequin + "
                             "neon signs) drifted to unrelated content (moonlit water) by frame "
                             "30 despite the 384x640 resolution fix -- pushing anchoring further "
                             "since there's no real caption to reinforce the actual scene content.")
    parser.add_argument("--image-noise-sigma-max", type=float, default=0.015,
                        help="Pipeline default is 0.135; see --image-noise-sigma-min.")
    # Off by default on purpose: MIND scores only compare across models when
    # every run uses the same prompt text, so an enhanced run is not comparable
    # with the existing baseline numbers. Keep the two sets of results apart.
    parser.add_argument("--enhance-prompt", action="store_true",
                        help="use the longer scene-anchoring prompts instead of the one-line "
                             "defaults; results are NOT comparable with baseline runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mirror-test", action="store_true", help="Also generate mirror_test outputs (additive).")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if not EVOKE_VENV_PY.exists():
        print(f"FATAL: Evoke venv python not found at {EVOKE_VENV_PY}", file=sys.stderr)
        return 2

    work_dir = args.work_dir or (args.test_root / ".frames-evoke")
    work_dir.mkdir(parents=True, exist_ok=True)

    samples = [] if args.mirror_only else gather_samples(args.gt_root)
    if args.mirror_test:
        samples += gather_mirror_samples(args.gt_root, args.mirror_action)
    if args.perspective:
        samples = [s for s in samples if s["perspective"] == args.perspective]
    if args.test_type:
        samples = [s for s in samples if s["test_type"] == args.test_type]
    if args.only:
        samples = [s for s in samples if any(sub.lower() in s["gt_name"].lower() for sub in args.only)]
    if args.start_index:
        if args.start_index >= len(samples):
            print(f"--start-index {args.start_index} is past the end of {len(samples)} matched sample(s); "
                  "nothing to do.")
            return 0
        samples = samples[args.start_index:]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("No samples matched.")
        return 1

    print(f"Will process {len(samples)} sample(s) via {EVOKE_VENV_PY.name}:")
    for s in samples:
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}")
    print()

    if args.dry_run:
        for s in samples:
            out = output_path(args.test_root, args.model_name, s)
            print(f"  [dry-run] {s['perspective']}/{s['test_type']}/{s['gt_name']} -> {out}")
        return 0

    rc = run_persistent(samples, args, work_dir)
    if rc != 0:
        return rc
    print(f"Done. {len(samples)} sample(s) processed (excluding any already-on-disk skips).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
