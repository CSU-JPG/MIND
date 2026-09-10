"""Drive Matrix-Game-3 from a MIND-Data tree into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and, for each sample:
  1. Extracts the first frame from video.mp4 -> a temp PNG
  2. Loads action.json (uses `caption` as prompt)
  3. Calls Matrix-Game-3's generate.py with that frame + caption
  4. Writes the resulting mp4 to test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4

Skip-if-exists: samples whose output mp4 already exists are skipped.

Limitations:
  - Uses only the caption + first frame. Per-frame MIND actions (ws/ad/ud/lr) are
    NOT passed through; generate.py only supports a single prompt. To use the
    action sequence, add an actions-file path to generate.py and wire it here.
  - `mirror_test` produces 10 paths per gt_name — not yet supported. TODO.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import av
import psutil
from PIL import Image

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4

MATRIX3_REPO = Path(__file__).resolve().parent.parent.parent / "matrix3" / "Matrix-Game-3"
MATRIX3_GENERATE = MATRIX3_REPO / "generate.py"
MATRIX3_CKPT_DIR = "Matrix-Game-3.0"
# Override via MATRIX3_VENV_PY env var when the DeepVerse-shared venv is missing
# (e.g. reuse the MIND scoring venv if its torch stack is compatible).
MATRIX3_VENV_PY = Path(os.environ.get("MATRIX3_VENV_PY", str(Path(__file__).resolve().parent.parent.parent / "DeepVerse" / ".venv" / "Scripts" / "python.exe")))

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


def load_action_json(action_path: Path) -> dict:
    with open(action_path, encoding="utf-8") as f:
        return json.load(f)


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
            num_iterations: int | None = None, num_inference_steps: int | None = None) -> int:
    out = output_path(test_root, model_name, sample)
    if out.exists():
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} -> {out} (exists)")
        return 0

    if sample["perspective"] == "1st_data":
        prompt = "First-person view exploring a 3D virtual environment."
    else:
        prompt = "Third-person view of a character exploring a 3D virtual environment."

    frame_png = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.png"
    if sample.get("frame_png_src") is not None:
        frame_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["frame_png_src"], frame_png)
    else:
        extract_first_frame(sample["video"], frame_png)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Speed knobs. Priority: CLI flag > env var > default. Halving either
    # roughly halves wall-clock per clip at a quality cost; tune per run.
    m3_iters = str(num_iterations) if num_iterations is not None else os.environ.get("MIND_M3_ITERATIONS", "8")
    m3_steps = str(num_inference_steps) if num_inference_steps is not None else os.environ.get("MIND_M3_STEPS", "5")
    cmd = [
        str(MATRIX3_VENV_PY),
        str(MATRIX3_GENERATE),
        "--size", "704*1280",
        "--ckpt_dir", MATRIX3_CKPT_DIR,
        "--fa_version", "2",
        "--use_int8",
        "--num_iterations", m3_iters,
        "--num_inference_steps", m3_steps,
        "--sample_guide_scale", "1.0",
        "--image", str(frame_png),
        "--prompt", prompt,
        "--save_name", out.stem,
        "--seed", "42",
        "--lightvae_pruning_rate", "0.75",
        "--vae_type", "mg_lightvae_v2",
        "--output_dir", str(out.parent),
    ]

    print(f"\n=== {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} ===")
    print(f"prompt: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
    print(f"out:    {out}")

    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return 0

    env = os.environ.copy()
    # Strip cross-venv pollution before spawning the Matrix-Game-3 venv python.
    # The MIND venv (uv-managed 3.10) leaks VIRTUAL_ENV / PYTHONHOME / PYTHONPATH
    # into the subprocess; the Matrix-Game-3 interpreter then loads the host's
    # 3.10 stdlib, _sre.MAGIC mismatches the in-process MAGIC, and `import re`
    # crashes with `AssertionError: SRE module mismatch` before any user code runs.
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(MATRIX3_REPO), env=env)
    elapsed = time.perf_counter() - t0
    print(f"  rc={rc}  elapsed={elapsed:.1f}s")
    if rc == 0 and out.exists():
        log_mp4(model_name, sample["perspective"], sample["test_type"], sample["gt_name"], out)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root", type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="matrix-game-3", help="Subfolder name under test-root")
    parser.add_argument("--work-dir", type=Path, default=None, help="Temp dir for extracted first frames (default: <test-root>/.frames)")
    parser.add_argument("--only", nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type", choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit", type=int, help="Only run first N matched samples")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N matched samples (applied AFTER --perspective / --test-type / --only filters, BEFORE --limit). Use to resume mid-run.")
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="matrix3 generate.py --num_iterations (outer rollout loop). Default 8. Env fallback: MIND_M3_ITERATIONS.")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="matrix3 generate.py --num_inference_steps (diffusion steps per iter). Default 5. Env fallback: MIND_M3_STEPS.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fps", type=int, default=24,
                        help="Accepted for bat-script consistency; matrix3 generate.py uses its own default (24).")
    parser.add_argument("--mirror-test", action="store_true",
                        help="Also generate mirror_test outputs (additive). One mp4 per first-frame PNG.")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only generate mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if not MATRIX3_GENERATE.exists():
        print(f"FATAL: generate.py not found at {MATRIX3_GENERATE}", file=sys.stderr)
        return 2
    if not MATRIX3_VENV_PY.exists():
        print(f"FATAL: python.exe not found at {MATRIX3_VENV_PY}", file=sys.stderr)
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
                     num_iterations=args.num_iterations,
                     num_inference_steps=args.num_inference_steps)
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
