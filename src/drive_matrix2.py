"""Drive Matrix-Game-2 from a MIND-Data tree into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and, for each sample:
  1. Extracts the first frame from video.mp4 -> a temp PNG
  2. Calls Matrix-Game-2's inference.py with that frame
  3. Renames the produced demo.mp4 to video.mp4 at the standard MIND-tests path

Skip-if-exists: samples whose output mp4 already exists are skipped.

Differences from drive_matrix3.py:
  - Matrix-Game-2 inference.py CLI is different (--config_path, --checkpoint_path,
    --img_path, --output_folder, --num_output_frames, --seed, --pretrained_model_path).
  - It writes `demo.mp4` + `demo_icon.mp4` to --output_folder; we keep demo.mp4
    and discard demo_icon.mp4.
  - matrix2 doesn't support per-frame keyboard conditioning at inference time;
    samples are conditioned only on the first frame + the chosen yaml config.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import av

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4

MATRIX2_REPO = Path(__file__).resolve().parent.parent.parent / "matrix3" / "Matrix-Game-2"
MATRIX2_INFERENCE = MATRIX2_REPO / "inference.py"
# pretrained_model_path holds Wan2.1_VAE.pth + the safetensors weights.
MATRIX2_PRETRAINED = Path(
    os.environ.get(
        "MATRIX2_PRETRAINED",
        str(Path(__file__).resolve().parent.parent.parent / "Matrix-Game" / "Matrix-Game-2" / "Matrix-Game-2.0"),
    )
)
# Override via MATRIX2_VENV_PY env var to use a dedicated matrix2 venv if
# its torch stack differs from MIND's (defaults to MIND's venv python).
MATRIX2_VENV_PY = Path(
    os.environ.get("MATRIX2_VENV_PY", str(Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"))
)
# matrix2's CLI exposes three task configs (universal / gta_drive / templerun).
# Default to universal — closest match to MIND's first-person exploration corpus.
MATRIX2_CONFIG = MATRIX2_REPO / "configs" / "inference_yaml" / "inference_universal.yaml"

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")


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


def run_one(sample: dict, test_root: Path, model_name: str, work_dir: Path, dry_run: bool,
            num_output_frames: int, config_path: Path, checkpoint_path: str, seed: int) -> int:
    out = output_path(test_root, model_name, sample)
    if out.exists():
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} -> {out} (exists)")
        return 0

    frame_png = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.png"
    if sample.get("frame_png_src") is not None:
        frame_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["frame_png_src"], frame_png)
    else:
        extract_first_frame(sample["video"], frame_png)

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(MATRIX2_VENV_PY),
        str(MATRIX2_INFERENCE),
        "--config_path", str(config_path),
        "--img_path", str(frame_png),
        "--output_folder", str(out.parent),
        "--num_output_frames", str(num_output_frames),
        "--seed", str(seed),
        "--pretrained_model_path", str(MATRIX2_PRETRAINED),
    ]
    if checkpoint_path:
        cmd += ["--checkpoint_path", str(checkpoint_path)]

    print(f"\n=== {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} ===")
    print(f"img:    {frame_png}")
    print(f"out:    {out}")

    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return 0

    env = os.environ.copy()
    # Strip cross-venv pollution before spawning the matrix2 venv python (same
    # _sre.MAGIC mismatch trap that drive_matrix3 dodges).
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(MATRIX2_REPO), env=env)
    elapsed = time.perf_counter() - t0
    print(f"  rc={rc}  elapsed={elapsed:.1f}s")

    # matrix2 writes demo.mp4 + demo_icon.mp4 to --output_folder. Rename the
    # clean one to video.mp4 (MIND's expected filename) and drop the overlay one.
    demo = out.parent / "demo.mp4"
    demo_icon = out.parent / "demo_icon.mp4"
    if rc == 0 and demo.exists():
        demo.replace(out)
        if demo_icon.exists():
            demo_icon.unlink()
        log_mp4(model_name, sample["perspective"], sample["test_type"], sample["gt_name"], out)
    elif rc == 0 and not out.exists():
        print(f"  WARN: rc=0 but no demo.mp4 at {demo}; not staged.")
        rc = 3
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root", type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="matrix-game-2", help="Subfolder name under test-root")
    parser.add_argument("--work-dir", type=Path, default=None, help="Temp dir for extracted first frames (default: <test-root>/.frames)")
    parser.add_argument("--only", nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type", choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit", type=int, help="Only run first N matched samples")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N matched samples (applied AFTER filters, BEFORE --limit).")
    parser.add_argument("--config-path", type=Path, default=MATRIX2_CONFIG,
                        help=f"matrix2 yaml config (default: {MATRIX2_CONFIG.name}).")
    parser.add_argument("--checkpoint-path", type=str, default="",
                        help="Optional safetensors checkpoint override (empty = use pretrained_model_path's bundled weights).")
    parser.add_argument("--num-output-frames", type=int, default=150,
                        help="matrix2 --num_output_frames (default 150).")
    parser.add_argument("--seed", type=int, default=0, help="matrix2 --seed (default 0).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fps", type=int, default=24,
                        help="Accepted for bat-script consistency; matrix2 inference.py uses its own default.")
    parser.add_argument("--mirror-test", action="store_true",
                        help="Also generate mirror_test outputs (additive). One mp4 per first-frame PNG.")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only generate mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if not MATRIX2_INFERENCE.exists():
        print(f"FATAL: inference.py not found at {MATRIX2_INFERENCE}", file=sys.stderr)
        return 2
    if not MATRIX2_VENV_PY.exists():
        print(f"FATAL: python.exe not found at {MATRIX2_VENV_PY}", file=sys.stderr)
        return 2
    if not MATRIX2_PRETRAINED.exists():
        print(f"FATAL: pretrained_model_path not found at {MATRIX2_PRETRAINED}", file=sys.stderr)
        return 2
    if not args.config_path.exists():
        print(f"FATAL: config not found at {args.config_path}", file=sys.stderr)
        return 2

    work_dir = args.work_dir or (args.test_root / ".frames")
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
            print(f"--start-index {args.start_index} is past the end of {len(samples)} matched sample(s); nothing to do.")
            return 0
        samples = samples[args.start_index:]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("No samples matched.")
        return 1

    print(f"Will process {len(samples)} sample(s):")
    for s in samples:
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}")
    print()

    failures: list[str] = []
    for s in samples:
        rc = run_one(s, args.test_root, args.model_name, work_dir, args.dry_run,
                     num_output_frames=args.num_output_frames,
                     config_path=args.config_path,
                     checkpoint_path=args.checkpoint_path,
                     seed=args.seed)
        if rc != 0:
            failures.append(f"{s['perspective']}/{s['test_type']}/{s['gt_name']}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for name in failures:
            print(f"  {name}")
        return 1
    print(f"Done. {len(samples)} sample(s) produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
