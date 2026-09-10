"""Drive DeepVerse from a MIND-Data tree into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and, for each sample whose target video.mp4 isn't already staged:
  1. Extracts the first frame from video.mp4 -> a unique PNG in <work_dir>.
  2. Reads action.json and maps per-frame MIND ticks (ws/ad/lr) into DeepVerse's
     action DSL: 7 segments of "(<trans><rot>)" codes for the 57-frame output
     (frame_per_chunk=8, 7 action prompts + 1 initial "empty").
  3. Spawns DeepVerse run.py with prompt_type=action.

DeepVerse hard-codes its output to "./output/generated_video.mp4", so we cwd
into the DeepVerse repo per sample, then rename the file into the MIND layout
at test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4.

Skip-if-exists: samples whose target video.mp4 already exists are excluded.
"""

import argparse
import json
import math
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
# 3rd_data first so default walk drives 3rd-person samples before 1st-person.
PERSPECTIVES = ("3rd_data", "1st_data")

# DeepVerse run.py constants (see DeepVerse/run.py main()).
DV_VIDEO_LENGTH = 57
DV_FRAMES_PER_CHUNK = 8       # save_video uses (i-1)//8+1 to index motion_prompts
DV_NUM_ACTION_SEGMENTS = 7    # 57 frames / 8 = 7 action chunks (+ 1 "empty" first)
DV_OUTPUT_REL = Path("output") / "generated_video.mp4"

# MIND action keys → DeepVerse translation codes.
# DeepVerse uses 8-way + stay (S/L/rL/B/rR/R/fR/F/fL) + rotation (N/L/R).
TRANS_CODES = {
    ( 0,  0): "S",
    ( 0, -1): "L",   # ad < 0  ⇒  strafe left
    ( 0,  1): "R",   # ad > 0  ⇒  strafe right
    ( 1,  0): "F",   # ws > 0  ⇒  forward
    (-1,  0): "B",   # ws < 0  ⇒  back
    ( 1, -1): "fL",
    ( 1,  1): "fR",
    (-1, -1): "rL",
    (-1,  1): "rR",
}
ROT_CODES = {
    0: "N",   # no yaw
   -1: "L",   # lr < 0  ⇒  yaw counterclockwise
    1: "R",   # lr > 0  ⇒  yaw clockwise
}


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
                if video.exists() and action.exists():
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


def _sign(v) -> int:
    if v > 0: return 1
    if v < 0: return -1
    return 0


def mind_actions_to_dv_dsl(action_data: list[dict]) -> str:
    """Map MIND per-frame ticks into a DeepVerse 7-segment action DSL string.

    Each segment is the majority-vote (sign of summed ws/ad/lr) over its chunk.
    Threshold: a direction needs >=30% of frames in the bucket to "win", else
    it's zero (stay/no-yaw).
    """
    if not action_data:
        return "".join(["(SN)"] * DV_NUM_ACTION_SEGMENTS)

    n = len(action_data)
    bucket_size = max(1, n // DV_NUM_ACTION_SEGMENTS)
    segments: list[str] = []
    for i in range(DV_NUM_ACTION_SEGMENTS):
        start = i * bucket_size
        end = n if i == DV_NUM_ACTION_SEGMENTS - 1 else (i + 1) * bucket_size
        chunk = action_data[start:end] or [{}]
        thresh = max(1, math.ceil(0.3 * len(chunk)))

        ws_p = sum(1 for d in chunk if d.get("ws", 0) > 0)
        ws_n = sum(1 for d in chunk if d.get("ws", 0) < 0)
        ad_p = sum(1 for d in chunk if d.get("ad", 0) > 0)
        ad_n = sum(1 for d in chunk if d.get("ad", 0) < 0)
        lr_p = sum(1 for d in chunk if d.get("lr", 0) > 0)
        lr_n = sum(1 for d in chunk if d.get("lr", 0) < 0)

        ws = 1 if ws_p >= thresh else -1 if ws_n >= thresh else 0
        ad = 1 if ad_p >= thresh else -1 if ad_n >= thresh else 0
        lr = 1 if lr_p >= thresh else -1 if lr_n >= thresh else 0

        trans = TRANS_CODES.get((ws, ad), "S")
        rot = ROT_CODES.get(lr, "N")
        segments.append(f"({trans}{rot})")

    return "".join(segments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-root",        type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root",      type=Path, required=True, help="Where to write generated test videos")
    parser.add_argument("--model-name",     default="deepverse", help="Subfolder name under test-root")
    parser.add_argument("--deepverse-repo", type=Path, required=True, help="Path to DeepVerse checkout")
    parser.add_argument("--deepverse-py",   type=Path, default=None,
                        help="DeepVerse python.exe (default: <deepverse-repo>/.venv/Scripts/python.exe)")
    parser.add_argument("--checkpoint",     type=Path, default=None,
                        help="Path to DeepVerse model checkpoint dir (default: <deepverse-repo>/checkpoint)")
    parser.add_argument("--work-dir",       type=Path, default=None,
                        help="Temp dir for first-frame PNGs (default: <test-root>/.deepverse_frames)")
    parser.add_argument("--only",           nargs="+", help="Only run samples whose gt_name contains any substring")
    parser.add_argument("--perspective",    choices=PERSPECTIVES)
    parser.add_argument("--test-type",      choices=TEST_TYPES)
    parser.add_argument("--limit",          type=int, help="Only run first N matched samples")
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--fps",            type=int, default=24, help="Traceability only; DeepVerse writes at 20 fps internally")
    parser.add_argument("--seed",           type=int, default=666)
    parser.add_argument("--add-depth",      action="store_true", help="Pass --add_depth to DeepVerse")
    parser.add_argument("--add-ply",        action="store_true", help="Pass --add_ply to DeepVerse")
    parser.add_argument("--mirror-test", action="store_true",
                        help="Also generate mirror_test outputs (additive). One mp4 per first-frame PNG.")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only generate mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    # Resolution order for the DeepVerse python: --deepverse-py CLI > DEEPVERSE_VENV_PY env > <repo>/.venv/Scripts/python.exe.
    dv_py = args.deepverse_py
    if dv_py is None:
        env_py = os.environ.get("DEEPVERSE_VENV_PY")
        dv_py = Path(env_py) if env_py else (args.deepverse_repo / ".venv" / "Scripts" / "python.exe")
    dv_entry = args.deepverse_repo / "run.py"
    ckpt = args.checkpoint or (args.deepverse_repo / "checkpoint")

    for path, label in [(dv_entry, "DeepVerse run.py"), (dv_py, "DeepVerse python"), (ckpt, "Checkpoint dir")]:
        if not path.exists() and not args.dry_run:
            print(f"FATAL: {label} not found: {path}", file=sys.stderr)
            return 2

    work_dir = args.work_dir or (args.test_root / ".deepverse_frames")
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
        stem = unique_stem(s)
        frame_png = work_dir / f"{stem}.png"
        if not frame_png.exists():
            if s.get("frame_png_src") is not None:
                frame_png.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s["frame_png_src"], frame_png)
            else:
                extract_first_frame(s["video"], frame_png)

        with open(s["action"], encoding="utf-8") as f:
            action_json = json.load(f)
        dsl = mind_actions_to_dv_dsl(action_json.get("data", []))

        pending.append({"sample": s, "stem": stem, "frame_png": frame_png, "target": target, "dsl": dsl})

    print(f"Will process {len(pending)} sample(s) (skipped {skipped} already-staged):")
    for m in pending:
        s = m["sample"]
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}  dsl={m['dsl']}")
    print()
    if not pending:
        print("Nothing to do.")
        return 0

    dv_out_abs = args.deepverse_repo / DV_OUTPUT_REL  # DeepVerse writes here every time

    env = os.environ.copy()
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env["VIRTUAL_ENV"] = str(args.deepverse_repo / ".venv")

    t0 = time.perf_counter()
    failed: list[str] = []
    for i, m in enumerate(pending, 1):
        s = m["sample"]
        print(f"\n=== [{i}/{len(pending)}] {s['perspective']}/{s['test_type']}/{s['gt_name']}  dsl={m['dsl']} ===")
        # DeepVerse run.py uses fire — argument style is --kwarg=value.
        cmd = [
            str(dv_py), str(dv_entry),
            "--input_image",  str(m["frame_png"]),
            "--model_path",   str(ckpt),
            "--prompt_type",  "action",
            "--prompt",       m["dsl"],
            "--seed",         str(args.seed),
        ]
        if args.add_depth: cmd.append("--add_depth")
        if args.add_ply:   cmd.append("--add_ply")
        print("cmd:\n  " + " ".join(cmd))

        if args.dry_run:
            continue

        # Clear stale output so a silent failure doesn't relocate the previous mp4.
        if dv_out_abs.exists():
            dv_out_abs.unlink()

        sample_t0 = time.perf_counter()
        rc = subprocess.call(cmd, cwd=str(args.deepverse_repo), env=env)
        sample_elapsed = time.perf_counter() - sample_t0
        print(f"  rc={rc}  elapsed={sample_elapsed:.1f}s")

        if rc != 0 or not dv_out_abs.exists():
            failed.append(f"{m['stem']}  rc={rc}  exists={dv_out_abs.exists()}")
            continue

        m["target"].parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dv_out_abs), str(m["target"]))
        print(f"  -> {m['target']}")
        log_mp4(args.model_name, s["perspective"], s["test_type"], s["gt_name"], m["target"])

    print(f"\nTotal elapsed: {time.perf_counter()-t0:.1f}s  "
          f"({(time.perf_counter()-t0)/max(1,len(pending)):.1f}s/sample)")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for line in failed:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
