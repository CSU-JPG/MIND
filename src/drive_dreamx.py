"""Drive DreamX-World from a MIND-Data tree into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and, for each sample that isn't already staged:
  1. Extracts the first frame from video.mp4 -> a unique PNG in <work_dir>.
  2. Maps MIND action.json (ws/ad/ud/lr per-frame ticks) to DreamX-World's
     discrete action_seq letters: w/s/a/d (translate), j/l (yaw), i/k (pitch).
  3. Adds a row to a single combined eval.json.

Then calls inference_dreamx5b.py ONCE with that combined eval.json. DreamX iterates
internally, so the 5B transformer + VAE + T5 load only once (~60-120s) regardless
of sample count — vs. once per sample if we spawned per-sample.

After inference, relocates each produced mp4 from <work_dir>/_outputs/<stem>_<action>.mp4
to test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4.

Skip-if-exists: samples whose target video.mp4 already exists are excluded from the run.
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

DREAMX_REPO = Path(__file__).resolve().parent.parent.parent / "DreamX-World"
# Override via DREAMX_VENV_PY env var when DreamX-World's own .venv is missing
# (e.g. share the MIND scoring venv if its torch stack is compatible).
DREAMX_VENV_PY = Path(os.environ.get("DREAMX_VENV_PY", str(DREAMX_REPO / ".venv" / "Scripts" / "python.exe")))
DREAMX_INFER = DREAMX_REPO / "inference_dreamx5b.py"
DREAMX_CONFIG = DREAMX_REPO / "configs" / "wan2.2" / "wan_ti2v_5b.yaml"
DREAMX_WAN = DREAMX_REPO / "Wan2.2-TI2V-5B"
DREAMX_TRANSFORMER = DREAMX_REPO / "DreamX-World-5B-Cam"

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")

# Full-resolution DreamX-World-5B-Cam defaults (matches inference_README.md).
# All six knobs below are CLI-overridable so a fast / smaller preset can be
# layered on top from the wrapper bat (drive_dreamx_small.bat passes the fp8
# + half-res + 30-step bundle as flags, leaving these untouched).
VIDEO_LENGTH = 121
HEIGHT = 704
WIDTH = 1280
FPS = 24
STEPS = 50
GUIDANCE = 3.0
SEED = 42
GPU_MEMORY_MODE = "none"  # "none", "model_full_load_and_qfloat8", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_image()
            img.save(out_path, "PNG")
            return
    raise RuntimeError(f"No frames decoded from {video_path}")


def mind_actions_to_dreamx(action_data: list[dict]) -> tuple[list[str], list[int]]:
    """Collapse MIND per-frame (ws/ad/ud/lr) ticks into DreamX (key, speed) segments.

    DreamX takes a list of letter-combo strings and a parallel speed list; each
    segment lasts ceil(video_length / len(action_seq)) frames. We bucket the MIND
    timeline into 4 segments and emit a direction only if ≥30% of frames in the
    bucket share it (else MIND's per-tick noise pollutes the sequence).
    """
    num_segments = 4
    if not action_data:
        return ["w"], [4]

    bucket_size = max(1, len(action_data) // num_segments)
    seq: list[str] = []
    speeds: list[int] = []
    for i in range(num_segments):
        start = i * bucket_size
        end = len(action_data) if i == num_segments - 1 else (i + 1) * bucket_size
        chunk = action_data[start:end]
        if not chunk:
            seq.append("w")
            speeds.append(4)
            continue
        thresh = max(1, int(0.3 * len(chunk)))
        # MIND axes are TRI-STATE (0=none, 1=dir A, 2=dir B), NOT signed: every
        # value is 0/1/2, so the old `>0`/`<0` test never saw direction B and
        # collapsed each axis to its first key. Decode value==1 vs value==2 and
        # map both directions to DreamX keys (keyboard convention; verified vs the
        # GT actor_pos/actor_rpy for ws/ud/lr):
        #   ws 1->w forward   2->s back
        #   ad 1->a left      2->d right
        #   ud 1->i pitch-up  2->k pitch-down
        #   lr 1->j yaw-left  2->l yaw-right
        axes = (("ws", "w", "s"), ("ad", "a", "d"), ("ud", "i", "k"), ("lr", "j", "l"))
        keys: list[str] = []
        per_axis_active = []
        for axis, key1, key2 in axes:
            c1 = sum(1 for d in chunk if d.get(axis, 0) == 1)
            c2 = sum(1 for d in chunk if d.get(axis, 0) == 2)
            per_axis_active.append(c1 + c2)
            if c1 >= thresh and c1 >= c2:
                keys.append(key1)
            elif c2 >= thresh:
                keys.append(key2)

        seq.append("".join(keys) if keys else "w")
        active = max(per_axis_active) if per_axis_active else 0
        speeds.append(min(8, max(2, int(active * 6 / max(1, len(chunk))))))
    return seq, speeds


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-root",    type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root",  type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="dreamx-world", help="Subfolder name under test-root")
    parser.add_argument("--work-dir",   type=Path, default=None, help="Temp dir for extracted first frames + eval.json (default: <test-root>/.frames)")
    parser.add_argument("--only",       nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type",   choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit",       type=int, help="Only run first N matched samples")
    parser.add_argument("--dry-run",     action="store_true")
    # Quality / speed overrides. Defaults match full-resolution DreamX-World-5B-Cam.
    parser.add_argument("--height",         type=int, default=HEIGHT, help=f"Output height (default {HEIGHT})")
    parser.add_argument("--width",          type=int, default=WIDTH,  help=f"Output width (default {WIDTH})")
    parser.add_argument("--video-length",   type=int, default=VIDEO_LENGTH, help=f"Frames per clip (1+4k pattern; default {VIDEO_LENGTH})")
    parser.add_argument("--fps",            type=int, default=FPS, help=f"Output fps; 24 or 16 (default {FPS})")
    parser.add_argument("--steps",          type=int, default=STEPS, help=f"Denoising steps (default {STEPS})")
    parser.add_argument("--gpu-memory-mode", default=GPU_MEMORY_MODE, help=f"DreamX --GPU_memory_mode; 'none' to skip (default {GPU_MEMORY_MODE})")
    parser.add_argument("--mirror-test", action="store_true",
                        help="Also generate mirror_test outputs (additive). One mp4 per first-frame PNG.")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only generate mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if not DREAMX_INFER.exists():
        print(f"FATAL: inference script not found at {DREAMX_INFER}", file=sys.stderr)
        return 2
    if not DREAMX_VENV_PY.exists():
        print(f"FATAL: python.exe not found at {DREAMX_VENV_PY}", file=sys.stderr)
        return 2
    for ckpt in (DREAMX_WAN, DREAMX_TRANSFORMER):
        if not ckpt.exists() and not args.dry_run:
            print(f"FATAL: checkpoint missing: {ckpt}\n  Run `python download_models.py` in {DREAMX_REPO} first.", file=sys.stderr)
            return 2

    work_dir = args.work_dir or (args.test_root / ".frames")
    work_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir = args.test_root / args.model_name / ".outputs"
    batch_output_dir.mkdir(parents=True, exist_ok=True)

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

    # Build combined eval.json — skip samples whose target already exists.
    pending: list[dict] = []
    pending_meta: list[dict] = []   # parallel: (sample, stem, action_seq, frame_png, target_mp4)
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
        action_seq, speeds = mind_actions_to_dreamx(action_json.get("data", []))

        pending.append({
            "image_path": str(frame_png).replace("\\", "/"),
            "caption": caption_for(s),
            "action_seq": action_seq,
            "action_speed_list": speeds,
        })
        pending_meta.append({
            "sample": s, "stem": stem, "action_seq": action_seq,
            "frame_png": frame_png, "target": target,
        })

    print(f"Will process {len(pending)} sample(s) (skipped {skipped} already-staged):")
    for m in pending_meta:
        s = m["sample"]
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}  action_seq={m['action_seq']}")
    print()
    if not pending:
        print("Nothing to do.")
        return 0

    combined_eval = work_dir / "combined.eval.json"
    with open(combined_eval, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
    print(f"Combined eval -> {combined_eval}\n")

    cmd = [
        str(DREAMX_VENV_PY), str(DREAMX_INFER),
        "--config_path",         str(DREAMX_CONFIG),
        "--model_name",          str(DREAMX_WAN),
        "--transformer_path",    str(DREAMX_TRANSFORMER),
        "--input_dir",           str(combined_eval),
        "--output_dir",          str(batch_output_dir),
        "--cam_method",          "prope",
        "--add_control_adapter",
        "--sample_size",         str(args.height), str(args.width),
        "--video_length",        str(args.video_length),
        "--fps",                 str(args.fps),
        "--guidance_scale",      str(GUIDANCE),
        "--num_inference_steps", str(args.steps),
        "--seed",                str(SEED),
        "--weight_dtype",        "bfloat16",
        "--ulysses_degree",      "1",
        "--ring_degree",         "1",
    ]
    # DreamX treats absence of --GPU_memory_mode as "no offload, full bf16 load".
    # Only append the flag when the wrapper actually wants a non-default mode.
    if args.gpu_memory_mode and args.gpu_memory_mode.lower() != "none":
        cmd += ["--GPU_memory_mode", args.gpu_memory_mode]
    print("Inference cmd:\n  " + " ".join(cmd) + "\n")

    if args.dry_run:
        print("[dry-run] not invoking inference.")
        return 0

    env = os.environ.copy()
    # Strip cross-venv pollution (host VIRTUAL_ENV may point at scope's uv-3.10 env).
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = str(DREAMX_REPO)
    env["VIRTUAL_ENV"] = str(DREAMX_REPO / ".venv")

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(DREAMX_REPO), env=env)
    elapsed = time.perf_counter() - t0
    print(f"\nInference rc={rc}  elapsed={elapsed:.1f}s  ({elapsed/max(1,len(pending)):.1f}s/sample amortized)\n")

    # Relocate mp4s into the MIND-tests layout.
    relocated = 0
    missing: list[str] = []
    for m in pending_meta:
        action_name = "_".join(m["action_seq"])
        produced = batch_output_dir / f"{m['stem']}_{action_name}.mp4"
        if produced.exists():
            m["target"].parent.mkdir(parents=True, exist_ok=True)
            produced.rename(m["target"])
            relocated += 1
            s = m["sample"]
            log_mp4(args.model_name, s["perspective"], s["test_type"], s["gt_name"], m["target"])
        else:
            missing.append(f"{m['stem']} -> {produced.name}")

    print(f"Relocated {relocated}/{len(pending_meta)} mp4(s) into MIND-tests layout.")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
