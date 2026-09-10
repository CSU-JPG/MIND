"""Drive LingBot-World (Fast) from a MIND-Data tree into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and for each sample:
  1. Extracts the first frame from video.mp4 -> temp PNG.
  2. Calls lingbot-world's generate.py with that frame + stock caption + the
     MIND action.json (passed through as --action_path).
  3. Writes the produced mp4 to:
        test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4

Skip-if-exists: samples whose output mp4 already exists are skipped.

Notes:
  - LingBot-World's generate.py task is `i2v-A14B` (only one currently shipped);
    the "fast" model is just a different --ckpt_dir pointing at the
    robbyant/lingbot-world-fast snapshot. Run download_fast.bat in the lingbot
    repo first, or pass --ckpt-dir to a custom location.
  - LingBot-World accepts `--action_path` natively (unlike Matrix-Game-3),
    so MIND's per-frame WASD actions ARE wired through here. That makes the
    `action` metric in MIND scoring meaningful (vs Matrix-Game-3 / DreamX where
    it isn't).

After running, score with MIND:
    python C:\\workspace\\world\\MIND\\src\\process.py \\
        --gt_root  C:\\workspace\\world\\MIND-Data \\
        --test_root C:\\workspace\\world\\MIND-tests\\lingbot-fast \\
        --metrics lcm,visual,dino,action,gsc

Usage:
    python src/drive_lingbot.py
    python src/drive_lingbot.py --limit 3 --perspective 1st_data --test-type action_space_test
    python src/drive_lingbot.py --dry-run
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

LINGBOT_REPO = Path(__file__).resolve().parent.parent.parent / "lingbot-world"
# Path A: use generate_fast.py against the fast-mini-cam ckpt. The original
# generate.py + base-cam-nf4 path failed on every sample (OSError: Error no
# file named config.json) because lingbot's WanModel.from_pretrained looked
# for a top-level config.json that doesn't ship in the NF4 release. The fast
# variant ships a complete ckpt layout and works out of the box.
LINGBOT_GENERATE = LINGBOT_REPO / "generate_fast.py"
LINGBOT_VENV_PY = LINGBOT_REPO / ".venv" / "Scripts" / "python.exe"
DEFAULT_CKPT_DIR = LINGBOT_REPO / "fast-mini-cam"  # holds VAE + T5 + lingbot_world_fast

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame.to_image().save(out_path, "PNG")
            return
    raise RuntimeError(f"No frames decoded from {video_path}")


def derive_caption(perspective: str) -> str:
    if perspective == "1st_data":
        return "First-person view exploring a 3D virtual environment in photorealistic style."
    return "Third-person view of a character exploring a 3D virtual environment in photorealistic style."


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
    return (
        test_root / model_name / sample["perspective"] / sample["test_type"]
        / sample["gt_name"] / "video.mp4"
    )


def run_one(sample: dict, test_root: Path, model_name: str, work_dir: Path,
            args: argparse.Namespace) -> int:
    out = output_path(test_root, model_name, sample)
    if out.exists() and not args.force:
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} -> {out} (exists)")
        return 0

    frame_png = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.png"
    if sample.get("frame_png_src") is not None:
        frame_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["frame_png_src"], frame_png)
    else:
        extract_first_frame(sample["video"], frame_png)
    caption = derive_caption(sample["perspective"])

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.venv_py),
        str(LINGBOT_GENERATE),
        "--task", args.task,
        "--size", args.size,
        "--ckpt_dir", str(args.ckpt_dir),
        "--image", str(frame_png),
        "--prompt", caption,
        # Path A intentionally drops --action_path. MIND's action.json doesn't
        # match what generate_fast expects (it wants a directory of poses.npy
        # + intrinsics.npy + WASD npys, not a flat action.json). Without
        # --action_path the model runs in image+text mode and produces a
        # plausible video; the action-metric scores will be loose but the
        # other 4 metrics (lcm/visual/dino/gsc) work fine. Path B (in
        # follow-up) will wire a proper converter.
        "--save_file", str(out),
        "--base_seed", str(args.seed),
        # NOTE: --sample_solver / --sample_steps / --sample_guide_scale are NOT
        # accepted by generate_fast.py (only generate.py has them). generate_fast
        # uses a fixed solver/scheduler bundled with the distilled model.
        "--t5_cpu",                # keeps VRAM headroom on 32 GB cards
        "--convert_model_dtype",   # always-on: avoids the OOM in test_fast.log
    ]
    if args.frame_num is not None:
        cmd += ["--frame_num", str(args.frame_num)]
    if args.sample_shift is not None:
        cmd += ["--sample_shift", str(args.sample_shift)]
    # --sample_guide_scale and --overlay_actions are not accepted by
    # generate_fast.py. The distilled fast model bakes the guidance scale in
    # and there is no overlay-actions visualizer in the fast pipeline.

    print(f"\n=== {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} ===")
    print(f"caption: {caption[:90]}{'...' if len(caption) > 90 else ''}")
    print(f"action_path: {sample['action']}")
    print(f"out:         {out}")

    if args.dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return 0

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Strip cross-venv pollution: scope's 3.12 venv vars would confuse lingbot's 3.10.
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT",
              "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(LINGBOT_REPO), env=env)
    elapsed = time.perf_counter() - t0
    print(f"  rc={rc}  elapsed={elapsed:.1f}s")
    if rc == 0 and out.exists():
        log_mp4(model_name, sample["perspective"], sample["test_type"], sample["gt_name"], out)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-root", type=Path, default=Path(__file__).resolve().parent.parent.parent / "MIND-Data")
    p.add_argument("--test-root", type=Path, default=Path(__file__).resolve().parent.parent.parent / "MIND-tests",
                   help="Parent dir; outputs land at <test-root>/<model-name>/<perspective>/...")
    p.add_argument("--model-name", default="lingbot-fast", help="Subfolder name under test-root")
    p.add_argument("--work-dir", type=Path, default=None, help="Temp dir (default: <test-root>/.frames)")
    p.add_argument("--only", nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    p.add_argument("--perspective", choices=PERSPECTIVES)
    p.add_argument("--test-type", choices=TEST_TYPES)
    p.add_argument("--limit", type=int, help="Only run first N matched samples")
    p.add_argument("--force", action="store_true", help="Re-run even if output exists")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--venv-py", type=Path, default=LINGBOT_VENV_PY)
    p.add_argument("--fps", type=int, default=24,
                   help="Accepted for bat-script consistency; lingbot sample_fps comes from wan_shared_cfg.sample_fps "
                        "(patched to 24 in wan/configs/shared_config.py).")

    # LingBot-World inference knobs
    p.add_argument("--task", default="i2v-A14B", help="Only i2v-A14B ships today")
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR,
                   help="LingBot-World checkpoint dir. Default: <lingbot>/fast (run download_fast.bat)")
    p.add_argument("--size", default="832*480",
                   help="One of SIZE_CONFIGS keys: 480*832, 832*480, 704*1280, 1280*704, 720*1280, 720*960")
    p.add_argument("--frame_num", type=int, default=None, help="Frame count (4n+1). Default: model's preset.")
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sample_shift", type=float, default=None)
    p.add_argument("--sample_guide_scale", type=float, default=None)
    p.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--convert_model_dtype", action="store_true",
                   help="Cast model params to lower dtype on load (less VRAM).")
    p.add_argument("--overlay_actions", action="store_true",
                   help="Draw WASD key state overlay on output frames (debug).")
    p.add_argument("--mirror-test", action="store_true",
                   help="Also generate mirror_test outputs (additive). One mp4 per first-frame PNG.")
    p.add_argument("--mirror-only", action="store_true",
                   help="Skip action_space_test + mem_test; only generate mirror_test. Implies --mirror-test.")
    p.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                   help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = p.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if not LINGBOT_GENERATE.exists():
        print(f"FATAL: generate.py not found at {LINGBOT_GENERATE}", file=sys.stderr)
        return 2
    if not args.venv_py.exists():
        print(f"FATAL: python.exe not found at {args.venv_py}", file=sys.stderr)
        print("Set up the venv first: cd C:\\workspace\\world\\lingbot-world && uv venv --python 3.10 .venv", file=sys.stderr)
        return 2
    if not args.ckpt_dir.exists() and not args.dry_run:
        print(f"FATAL: ckpt_dir not found at {args.ckpt_dir}", file=sys.stderr)
        print("Download the model first: cd C:\\workspace\\world\\lingbot-world && download_fast.bat", file=sys.stderr)
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
        rc = run_one(s, args.test_root, args.model_name, work_dir, args)
        if rc != 0:
            failures.append(f"{s['perspective']}/{s['test_type']}/{s['gt_name']}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for name in failures:
            print(f"  {name}")
        return 1
    print(f"Done. {len(samples)} sample(s) produced.")
    print(f"Next: score with run_mind.bat {args.model_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
