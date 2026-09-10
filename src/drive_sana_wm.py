"""Drive SANA-WM_bidirectional from MIND-Data into the MIND test layout.

Per-sample CLI invocation of Sana's inference_sana_wm.py (the SANA-WM_bidirectional
entry point — NOT the older inference_sana_video.py which uses an incompatible
pyrallis-based parser). Mirrors the working pattern in Sana\\test_sana_wm.bat.

For each (perspective, test_type, gt_name) sample we:
  1. Extract the first frame of video.mp4 -> a unique PNG in <work_dir>.
  2. Write the caption to a per-sample prompt .txt.
  3. Map MIND action.json's per-frame ws/ad/ud/lr ticks to Sana's action DSL
     (keys from wasdijkl, segments grouped by consecutive same-key-set frames).
  4. Copy Sana's demo_0_intrinsics.npy as the per-sample intrinsics (MIND has
     no real intrinsics — this is an approximation; same intrinsics for every
     sample is fine for first-pass eval).
  5. Spawn inference_sana_wm.py with --image / --prompt / --action / --intrinsics
     and the SANA-WM_bidirectional model + refiner paths.
  6. Relocate the produced mp4 to test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4.

MIND action -> Sana DSL key map (one held-key set per frame, then grouped):
    ws=0 -> 'w' (forward)        ad=0 -> 'a' (left)
    ws=1 -> 's' (backward)       ad=1 -> 'd' (right)                ad=2 -> -
    ud=0 -> 'i' (pitch up)       lr=0 -> 'j' (yaw left)
    ud=1 -> 'k' (pitch down)     lr=1 -> 'l' (yaw right)
    ud=2 -> -                    lr=2 -> -

Skip-if-exists: samples whose output mp4 already exists are skipped.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import av

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4

TEST_TYPES = ("action_space_test", "mem_test")
# 3rd_data first so the default (no --perspective filter) drives 3rd-person samples
# before 1st-person — SANA-WM_bidirectional is more naturally evaluated 3rd-first.
PERSPECTIVES = ("3rd_data", "1st_data")

DEFAULT_NUM_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_STEPS = 20
DEFAULT_CFG_SCALE = 5.0
DEFAULT_FLOW_SHIFT = 8.0
DEFAULT_TRANSLATION_SPEED = 0.055
DEFAULT_ROTATION_SPEED_DEG = 1.2
DEFAULT_SEED = 42


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame.to_image().save(out_path, "PNG")
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


def unique_stem(sample: dict) -> str:
    return f"{sample['perspective']}__{sample['test_type']}__{sample['gt_name']}"


def caption_for(sample: dict) -> str:
    if sample["perspective"] == "1st_data":
        return "First-person view exploring a 3D virtual environment."
    return "Third-person view of a character exploring a 3D virtual environment."


def mind_action_to_sana_dsl(action_json_path: Path, num_frames: int) -> str:
    """Map MIND action.json's per-frame ticks to Sana's '<keys>-<duration>' DSL.

    Reads at most `num_frames` entries from action.json['data'], converts each
    to a held-key set under the Sana grammar, then groups runs of identical
    held-key sets into segments. Idle frames (no keys held) become 'none-<n>'.
    """
    with open(action_json_path, encoding="utf-8") as f:
        action_data = json.load(f)
    entries = action_data.get("data", [])[:num_frames]
    if not entries:
        return f"none-{num_frames}"

    per_frame_keys: list[frozenset] = []
    for e in entries:
        held: set[str] = set()
        ws = e.get("ws")
        if ws == 0:
            held.add("w")
        elif ws == 1:
            held.add("s")
        ad = e.get("ad")
        if ad == 0:
            held.add("a")
        elif ad == 1:
            held.add("d")
        ud = e.get("ud")
        if ud == 0:
            held.add("i")
        elif ud == 1:
            held.add("k")
        lr = e.get("lr")
        if lr == 0:
            held.add("j")
        elif lr == 1:
            held.add("l")
        per_frame_keys.append(frozenset(held))

    # Pad to num_frames with the last seen key set so the segment count matches.
    while len(per_frame_keys) < num_frames:
        per_frame_keys.append(per_frame_keys[-1])

    segments: list[str] = []
    run_keys = per_frame_keys[0]
    run_len = 1
    for keys in per_frame_keys[1:]:
        if keys == run_keys:
            run_len += 1
        else:
            token = "none" if not run_keys else "".join(sorted(run_keys))
            segments.append(f"{token}-{run_len}")
            run_keys = keys
            run_len = 1
    token = "none" if not run_keys else "".join(sorted(run_keys))
    segments.append(f"{token}-{run_len}")
    return ",".join(segments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-root", type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root", type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="sana-wm", help="Subfolder name under test-root")
    parser.add_argument("--sana-repo", type=Path, required=True, help="Path to Sana checkout")
    parser.add_argument("--sana-py", type=Path, required=True, help="Sana .venv-wm python.exe")
    parser.add_argument("--sana-model-dir", type=Path, default=None,
                        help="SANA-WM_bidirectional model dir (default: <sana-repo>/output/pretrained_models/SANA-WM_bidirectional)")
    parser.add_argument("--intrinsics", type=Path, default=None,
                        help="Per-sample intrinsics .npy (default: <sana-repo>/asset/sana_wm/demo_0_intrinsics.npy)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Temp dir for first frames + prompts (default: <test-root>/.sana_wm_work)")
    parser.add_argument("--only", nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type", choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit", type=int, help="Only run first N matched samples")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--flow-shift", type=float, default=DEFAULT_FLOW_SHIFT)
    parser.add_argument("--translation-speed", type=float, default=DEFAULT_TRANSLATION_SPEED)
    parser.add_argument("--rotation-speed-deg", type=float, default=DEFAULT_ROTATION_SPEED_DEG)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--with-refiner", action="store_true",
                        help="Run the refiner pass too (off by default — matches test_sana_wm.bat's --no_refiner).")
    parser.add_argument("--mirror-test", action="store_true",
                        help="Also generate mirror_test outputs (additive). One mp4 per first-frame PNG.")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only generate mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    # Two layouts supported:
    #   1. Legacy local-dir: <sana-repo>/output/pretrained_models/SANA-WM_bidirectional/...
    #   2. HF hub cache:     ~/.cache/huggingface/hub/models--Efficient-Large-Model--SANA-WM_bidirectional/snapshots/<rev>/...
    # download_sana_wm.py now defaults to layout #2 (no --dest -> cache). Resolve
    # which one is on disk, preferring an explicit --sana-model-dir if given.
    if args.sana_model_dir:
        sana_model_dir = args.sana_model_dir
    else:
        legacy_dir = args.sana_repo / "output" / "pretrained_models" / "SANA-WM_bidirectional"
        if (legacy_dir / "dit" / "sana_wm_1600m_720p.safetensors").exists():
            sana_model_dir = legacy_dir
        else:
            # Try HF cache. snapshot_download with local_files_only=True returns
            # the snapshot dir path without re-downloading. Fails fast if missing.
            try:
                from huggingface_hub import snapshot_download as _snap
                sana_model_dir = Path(_snap(
                    repo_id="Efficient-Large-Model/SANA-WM_bidirectional",
                    local_files_only=True,
                ))
            except Exception:
                sana_model_dir = legacy_dir  # fall back so the error message points at the canonical path
    sana_entry = args.sana_repo / "inference_video_scripts" / "inference_sana_wm.py"
    sana_config = sana_model_dir / "config.yaml"
    sana_dit = sana_model_dir / "dit" / "sana_wm_1600m_720p.safetensors"
    sana_refiner_ckpt = sana_model_dir / "refiner" / "refiner.safetensors"
    sana_refiner_text_encoder = sana_model_dir / "refiner" / "text_encoder"
    intrinsics_path = args.intrinsics or (args.sana_repo / "asset" / "sana_wm" / "demo_0_intrinsics.npy")

    # The refiner is opt-in (--with-refiner). Only require its weights when
    # the user asks for the refiner pass; otherwise inference_sana_wm.py runs
    # with --no_refiner and never reads them, so partial-download installs
    # (dit + vae only, no refiner) can still drive the model.
    required_paths = [
        (sana_entry, "inference_sana_wm.py"),
        (sana_config, "SANA-WM config.yaml"),
        (sana_dit, "SANA-WM DiT weights"),
        (args.sana_py, "Sana .venv-wm python.exe"),
        (intrinsics_path, "intrinsics .npy"),
    ]
    if args.with_refiner:
        required_paths += [
            (sana_refiner_ckpt, "SANA-WM refiner weights"),
            (sana_refiner_text_encoder, "SANA-WM refiner text_encoder dir"),
        ]
    for path, label in required_paths:
        if not path.exists() and not args.dry_run:
            print(f"FATAL: {label} not found: {path}", file=sys.stderr)
            return 2

    work_dir = args.work_dir or (args.test_root / ".sana_wm_work")
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
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("No samples matched.")
        return 1

    pending: list[dict] = []
    skipped = 0
    for s in samples:
        target = output_path(args.test_root, args.model_name, s)
        if target.exists():
            skipped += 1
            continue
        pending.append({"sample": s, "target": target})

    print(f"Will process {len(pending)} sample(s) (skipped {skipped} already-staged).")
    if not pending:
        print("Nothing to do.")
        return 0

    # Env mirrors Sana\test_sana_wm.bat: pin CUDA_PATH to v12.8 so triton picks
    # the cu128 toolkit (not v13 if also installed), disable dynamo (triton-windows
    # lacks triton_key so @torch.compile breaks), and stay offline once cached.
    env = os.environ.copy()
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    env["CUDA_PATH"] = cuda_root
    env["CUDA_HOME"] = cuda_root
    env["PATH"] = f"{cuda_root}\\bin;" + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TORCHDYNAMO_DISABLE"] = "1"
    # NOTE: deliberately NOT setting HF_HUB_OFFLINE=1 (test_sana_wm.bat does, but
    # transformers' _patch_mistral_regex calls hf_api.model_info() to detect
    # base-mistral variants for gemma-2-2b-it, and without that metadata cached
    # locally the offline mode raises OfflineModeIsEnabled. Letting it reach
    # HF is a tiny one-shot check.

    failures: list[str] = []
    for i, m in enumerate(pending, 1):
        s = m["sample"]
        stem = unique_stem(s)
        frame_png = work_dir / f"{stem}.png"
        prompt_txt = work_dir / f"{stem}.txt"

        if not frame_png.exists():
            if s.get("frame_png_src") is not None:
                frame_png.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s["frame_png_src"], frame_png)
            else:
                extract_first_frame(s["video"], frame_png)
        prompt_txt.write_text(caption_for(s), encoding="utf-8")
        action_dsl = mind_action_to_sana_dsl(s["action"], args.num_frames)

        target = m["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        # inference_sana_wm.py writes <output_dir>/<name>_generated.mp4 (appends
        # _generated regardless of --name), so we ask Sana for "video" and then
        # rename video_generated.mp4 -> video.mp4 after the subprocess returns.
        out_name = "video"
        sana_out_mp4 = target.parent / f"{out_name}_generated.mp4"

        cmd = [
            str(args.sana_py), str(sana_entry),
            "--image", str(frame_png),
            "--prompt", str(prompt_txt),
            "--action", action_dsl,
            "--translation_speed", str(args.translation_speed),
            "--rotation_speed_deg", str(args.rotation_speed_deg),
            "--intrinsics", str(intrinsics_path),
            "--output_dir", str(target.parent),
            "--name", out_name,
            "--num_frames", str(args.num_frames),
            "--fps", str(args.fps),
            "--step", str(args.steps),
            "--cfg_scale", str(args.cfg_scale),
            "--flow_shift", str(args.flow_shift),
            "--seed", str(args.seed),
            "--config", str(sana_config),
            "--model_path", str(sana_dit),
            "--refiner_checkpoint", str(sana_refiner_ckpt),
            "--refiner_gemma_root", str(sana_refiner_text_encoder),
        ]
        if not args.with_refiner:
            cmd.append("--no_refiner")

        action_preview = action_dsl[:80] + ("..." if len(action_dsl) > 80 else "")
        print(f"\n[{i}/{len(pending)}] {s['perspective']}/{s['test_type']}/{s['gt_name']}")
        print(f"  action: {action_preview}")
        print(f"  out:    {target}")

        if args.dry_run:
            print("  [dry-run] " + " ".join(cmd))
            continue

        t0 = time.perf_counter()
        rc = subprocess.call(cmd, cwd=str(args.sana_repo), env=env)
        elapsed = time.perf_counter() - t0
        print(f"  rc={rc}  elapsed={elapsed:.1f}s")

        if rc == 0 and sana_out_mp4.exists():
            sana_out_mp4.replace(target)
        if rc == 0 and target.exists():
            log_mp4(args.model_name, s["perspective"], s["test_type"], s["gt_name"], target)
        else:
            failures.append(f"{stem} (rc={rc})")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for n in failures:
            print(f"  {n}")
        return 1
    print(f"Done. {len(pending)} sample(s) produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
