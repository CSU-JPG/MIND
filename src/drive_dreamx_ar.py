"""Drive DreamX-World-5B (AR / long-horizon) from MIND-Data into the MIND test layout.

Sibling of drive_dreamx.py — SAME staging (first-frame extract, MIND action ->
DreamX action_seq, combined eval.json, skip-if-exists, relocation), but it
cross-spawns the AUTOREGRESSIVE model (inference_ar_forcing.py + DreamX-World-5B)
instead of the Cam model (inference_dreamx5b.py + DreamX-World-5B-Cam).

Kept separate from drive_dreamx.py so the Cam and AR variants stage + score
under distinct model-name folders and never share a code path.

The Wan base + AR checkpoint paths are resolved by the wrapper bat (which has
huggingface_hub) and passed in via --wan-base / --base-checkpoint.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Reuse the shared staging helpers from the Cam driver (same MIND-Data parsing).
from drive_dreamx import (
    DREAMX_REPO,
    DREAMX_VENV_PY,
    PERSPECTIVES,
    TEST_TYPES,
    extract_first_frame,
    gather_samples,
    mind_actions_to_dreamx,
    output_path,
    unique_stem,
)
from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4

# AR-specific entry points (vs the Cam driver's inference_dreamx5b.py).
DREAMX_INFER = DREAMX_REPO / "inference_ar_forcing.py"
DREAMX_CONFIG = DREAMX_REPO / "configs" / "dreamx-ar" / "causal_camera_forcing_5b.yaml"
DREAMX_TRANSFORMER_DIR = DREAMX_REPO / "configs" / "dreamx-ar"  # holds config.json (attn_compress=4)

SEED = 42


def caption_for(sample: dict) -> str:
    """AR-only caption (overrides drive_dreamx.caption_for, used by the Cam model).
    Avoids the words 'virtual environment' / 'virtual reality', which made DreamX
    render every character wearing a VR headset; uses neutral game-world phrasing."""
    if sample["perspective"] == "1st_data":
        return "First-person view walking through a detailed 3D game world."
    return "Third-person view following a character as they walk through a detailed 3D game world."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-root", type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root", type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="dreamx-world_ar", help="Subfolder name under test-root")
    parser.add_argument("--wan-base", type=Path, required=True, help="Wan2.2-TI2V-5B base snapshot dir (text encoder/tokenizer/VAE)")
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="DreamX-World-5B (AR) model.safetensors")
    parser.add_argument("--work-dir", type=Path, default=None, help="Temp dir for frames + eval.json (default: <test-root>/.frames)")
    parser.add_argument("--only", nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type", choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit", type=int, help="Only run first N matched samples")
    parser.add_argument("--dry-run", action="store_true")
    # AR generation knobs. num_output_frames is LATENT frames; pixel = (N-1)*4+1.
    #   21 -> 81 px (~5s @16fps)  |  63 -> 249 px (~15s)  |  larger -> up to 1 min.
    parser.add_argument("--num-output-frames", type=int, default=21, help="Latent frames (default 21 -> 81 px)")
    parser.add_argument("--fps", type=int, default=16, help="Output fps (AR native is 16; default 16)")
    parser.add_argument("--color-correction-strength", type=float, default=0.3)
    parser.add_argument("--mirror-test", action="store_true", help="Also generate mirror_test outputs (additive).")
    parser.add_argument("--mirror-only", action="store_true", help="Only generate mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if not DREAMX_INFER.exists():
        print(f"FATAL: AR inference script not found at {DREAMX_INFER}", file=sys.stderr)
        return 2
    if not DREAMX_VENV_PY.exists():
        print(f"FATAL: python.exe not found at {DREAMX_VENV_PY}", file=sys.stderr)
        return 2
    if not args.dry_run:
        for need in (args.wan_base, args.base_checkpoint, DREAMX_CONFIG):
            if not Path(need).exists():
                print(f"FATAL: required path missing: {need}", file=sys.stderr)
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
    pending_meta: list[dict] = []
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
            "task_id": stem,  # inference_ar_forcing names output "<task_id>_<imgparent>_<imgbasename>.mp4"
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

    combined_eval = work_dir / "combined_ar.eval.json"
    with open(combined_eval, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
    print(f"Combined eval -> {combined_eval}\n")

    cmd = [
        str(DREAMX_VENV_PY), str(DREAMX_INFER),
        "--config_path", str(DREAMX_CONFIG),
        "--model_name", str(args.wan_base),
        "--transformer_path", str(DREAMX_TRANSFORMER_DIR),
        "--base_checkpoint_path", str(args.base_checkpoint),
        "--data_path", str(combined_eval),
        "--output_folder", str(batch_output_dir),
        "--num_output_frames", str(args.num_output_frames),
        "--fps", str(args.fps),
        "--seed", str(SEED),
        "--color_correction_strength", str(args.color_correction_strength),
        "--chunk_relative",
    ]
    print("Inference cmd:\n  " + " ".join(cmd) + "\n")

    if args.dry_run:
        print("[dry-run] not invoking inference.")
        return 0

    env = os.environ.copy()
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
    print(f"\nInference rc={rc}  elapsed={elapsed:.1f}s  ({elapsed / max(1, len(pending)):.1f}s/sample amortized)\n")

    # Relocate mp4s into the MIND-tests layout. inference_ar_forcing prefixes the
    # output with task_id (=stem), so glob "<stem>_*.mp4" to find each produced clip.
    relocated = 0
    missing: list[str] = []
    for m in pending_meta:
        hits = sorted(glob.glob(str(batch_output_dir / f"{m['stem']}_*.mp4")))
        if hits:
            produced = Path(hits[0])
            m["target"].parent.mkdir(parents=True, exist_ok=True)
            produced.rename(m["target"])
            relocated += 1
            s = m["sample"]
            log_mp4(args.model_name, s["perspective"], s["test_type"], s["gt_name"], m["target"])
        else:
            missing.append(f"{m['stem']} -> {m['stem']}_*.mp4")

    print(f"Relocated {relocated}/{len(pending_meta)} mp4(s) into MIND-tests layout.")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
