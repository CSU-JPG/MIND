"""Pair existing matrix3 outputs into the MIND layout and score them.

Picks every mp4 under <src-dir>, sorts them alphabetically, and pairs each one
with a MIND GT sample folder name. Then invokes src/process.py for scoring.

Numbers will only be meaningful if the test mp4 actually depicts the same scene
as its paired MIND GT sample. For your existing matrix3 outputs (from
Matrix-Game-3/demo_images), the pairing is arbitrary — this is a pipeline-only
smoke. Use src/drive_matrix3.py for real per-sample MIND-driven generation.

Usage:
    python src/score_matrix3.py
    python src/score_matrix3.py --metrics lcm,visual,dino,action
    python src/score_matrix3.py --src-dir C:\\foo\\outputs --limit 5
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_SRC = REPO.parent / "matrix3" / "Matrix-Game-3" / "output"
DEFAULT_GT_ROOT = REPO.parent / "MIND-Data"
DEFAULT_TEST_ROOT = REPO.parent / "MIND-tests" / "matrix-game-3"
DEFAULT_DINO_PATH = REPO / "dinov3_vitb16"
DEFAULT_METRICS = "lcm,visual,dino"
DEFAULT_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"

GT_SAMPLE_RE = re.compile(r"^data-\d+")


def list_source_mp4s(src_dir: Path) -> list[Path]:
    candidates = sorted(p for p in src_dir.glob("*.mp4") if p.is_file())
    # exclude obvious re-runs / variants — keep only NNN.mp4 (zero-padded 3-digit)
    pat = re.compile(r"^\d{3}\.mp4$")
    return [p for p in candidates if pat.match(p.name)]


def list_gt_samples(gt_root: Path, perspective: str, test_type: str) -> list[Path]:
    base = gt_root / perspective / "test" / test_type
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and GT_SAMPLE_RE.match(p.name))


def pair_and_link(src_mp4s: list[Path], gt_samples: list[Path], test_root: Path,
                  perspective: str, test_type: str, force: bool) -> list[tuple[Path, Path]]:
    dst_base = test_root / perspective / test_type
    dst_base.mkdir(parents=True, exist_ok=True)
    pairings: list[tuple[Path, Path]] = []
    n = min(len(src_mp4s), len(gt_samples))
    for i in range(n):
        src = src_mp4s[i]
        gt = gt_samples[i]
        dst_dir = dst_base / gt.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "video.mp4"
        if dst.exists() and not force:
            print(f"[skip] {src.name} -> {gt.name}/video.mp4 (exists)")
        else:
            shutil.copy2(src, dst)
            print(f"[link] {src.name} -> {gt.name}/video.mp4")
        pairings.append((src, dst))
    if len(src_mp4s) > n:
        print(f"  ({len(src_mp4s) - n} source mp4s unused — not enough GT samples)")
    if len(gt_samples) > n:
        print(f"  ({len(gt_samples) - n} GT samples unpaired — not enough source mp4s)")
    return pairings


def write_pairing_manifest(pairings: list[tuple[Path, Path]], test_root: Path) -> Path:
    manifest = test_root / "pairing_manifest.json"
    data = [{"source_mp4": str(src), "linked_to": str(dst)} for src, dst in pairings]
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest}")
    return manifest


def run_scorer(venv_py: Path, gt_root: Path, test_root: Path, dino_path: Path,
               metrics: str, num_gpus: int, video_max_time: int | None,
               extra_args: list[str]) -> int:
    process_py = REPO / "src" / "process.py"
    cmd = [
        str(venv_py),
        str(process_py),
        "--gt_root", str(gt_root),
        "--test_root", str(test_root),
        "--dino_path", str(dino_path),
        "--num_gpus", str(num_gpus),
        "--metrics", metrics,
    ]
    if video_max_time:
        cmd += ["--video_max_time", str(video_max_time)]
    cmd += extra_args

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print("\n=== invoking scorer ===")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC, help="Directory of source mp4s to pair into MIND layout.")
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--dino-path", type=Path, default=DEFAULT_DINO_PATH)
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Comma-separated MIND metrics.")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--perspective", choices=("1st_data", "3rd_data"), default="1st_data")
    parser.add_argument("--test-type", choices=("action_space_test", "mem_test"), default="action_space_test")
    parser.add_argument("--limit", type=int, help="Limit to first N source mp4s (after sort).")
    parser.add_argument("--force", action="store_true", help="Re-copy even if pair already exists.")
    parser.add_argument("--video-max-time", type=int, help="Pass --video_max_time to process.py.")
    parser.add_argument("--venv-py", type=Path, default=DEFAULT_VENV_PY)
    parser.add_argument("--no-score", action="store_true", help="Only pair; skip the scoring invocation.")
    parser.add_argument("scorer_args", nargs="*", help="Extra args passed through to process.py.")
    args = parser.parse_args()

    if not args.venv_py.exists():
        print(f"FATAL: venv python not found at {args.venv_py}", file=sys.stderr)
        return 2

    src_mp4s = list_source_mp4s(args.src_dir)
    if args.limit:
        src_mp4s = src_mp4s[: args.limit]
    if not src_mp4s:
        print(f"FATAL: no NNN.mp4 files in {args.src_dir}", file=sys.stderr)
        return 1

    gt_samples = list_gt_samples(args.gt_root, args.perspective, args.test_type)
    if not gt_samples:
        print(f"FATAL: no GT samples under {args.gt_root}/{args.perspective}/test/{args.test_type}", file=sys.stderr)
        return 1

    print(f"Pairing {len(src_mp4s)} source mp4(s) with {len(gt_samples)} GT sample(s)")
    print(f"  source dir: {args.src_dir}")
    print(f"  test_root:  {args.test_root}")
    print()

    pairings = pair_and_link(src_mp4s, gt_samples, args.test_root, args.perspective, args.test_type, args.force)

    # process.py os.listdir-s every (perspective, test_type) dir under test_root and
    # crashes on missing ones. Stub the unused combinations as empty dirs.
    for p in ("1st_data", "3rd_data"):
        for t in ("mem_test", "action_space_test", "mirror_test"):
            (args.test_root / p / t).mkdir(parents=True, exist_ok=True)

    write_pairing_manifest(pairings, args.test_root)

    if args.no_score:
        print("\n--no-score set; stopping after pairing.")
        return 0

    rc = run_scorer(
        args.venv_py, args.gt_root, args.test_root, args.dino_path,
        args.metrics, args.num_gpus, args.video_max_time, args.scorer_args,
    )
    print(f"\nScorer rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
