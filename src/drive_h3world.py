"""Drive H3-World (MiniMax-H3 LoRA) over MIND.

MIND action.json (ws/ad/ud/lr tri-state) -> H3-World's raw per-frame action
matrix (numpy [num_frames, abot_action.ACTION_DIM=17], WASD+IJKL layout --
same key semantics as drive_abot.py, since H3-World's LoRA was trained on the
same ABot-World-Explorer action schema). Only the 8 active key columns
(W/A/S/D/I/J/K/L) are set per MIND tick; the Q/E/Space and continuous
rotation/translation columns stay zero, matching how H3-World's own
--action-preset mode drives the model (key-only, no continuous deltas).

H3-World's code/abot/infer.py has no load-once/batch mode (unlike ABot's
--mind-batch): each sample is a fresh subprocess that reloads the ~33B
backbone + LoRA. This is slow (matches drive_evoke.py's per-sample pattern),
but it's the only inference entrypoint the repo exposes.

--num-frames must be 17k+5 (README's example uses 124); MIND action ticks are
truncated/padded (hold last tick) to that length. Run via H3-World's own
venv (see drive_h3world.sh).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import os

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples

H3_ROOT = Path(os.environ.get("H3_ROOT", str(Path(__file__).resolve().parent.parent.parent / "H3-World")))
H3_PY = Path(os.environ.get("H3_PY", str(H3_ROOT / ".venv" / "bin" / "python")))
INFER = H3_ROOT / "code" / "abot" / "infer.py"
CHECKPOINT = H3_ROOT / "checkpoints" / "H3-World" / "step-10000.safetensors"

PERSPECTIVES = ("1st_data", "3rd_data")
TEST_TYPES = ("action_space_test", "mem_test")
MODEL_NAME = "h3world"

NUM_FRAMES = 124  # 17*7 + 5, matches H3-World README's example
GENERIC_SCENE_PROMPT = "A person moves through the scene, stable forward motion, consistent environment."

# abot_action.ACTION_COLS layout: 11 binary keys, then 3 rotation deltas,
# then 3 translation deltas. Only the active WASD+IJKL columns are set here.
KEY_COLS = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"]
ACTION_DIM = len(KEY_COLS) + 3 + 3


def snap_mirror_frames(n_ticks: int) -> int:
    """Smallest valid 17k+5 frame count >= n_ticks."""
    return 5 + 17 * max(1, -(-(n_ticks - 5) // 17))


def mind_to_action_matrix(mind_data: list, num_frames: int = NUM_FRAMES,
                          resample: bool = False) -> np.ndarray:
    """MIND ticks -> H3-World's per-frame action matrix.

    Default (main tests): take the first num_frames ticks. Their action.json
    carries thousands of ticks against a ~124-frame clip, so this consumes the
    opening slice and never pads -- unchanged behaviour, so existing scores stay
    comparable.

    resample=True (mirror test): stretch the whole trajectory proportionally
    across num_frames. The mirror set is 48 ticks of 24-out/24-back and gsc
    splits the generated clip at its midpoint, so the turnaround has to land
    there. H3-World only accepts 17k+5 frame counts, so 48 ticks cannot map
    1:1 -- the nearest valid length is 56. Resampling puts tick 24 at frame 28
    = 56/2 exactly; truncating or holding the last tick does not.
    """
    mat = np.zeros((num_frames, ACTION_DIM), dtype=np.float32)
    n = num_frames if resample else min(len(mind_data), num_frames)
    for i in range(n):
        s = mind_data[i * len(mind_data) // num_frames] if resample else mind_data[i]
        if s.get("ws") == 1:
            mat[i, KEY_COLS.index("W")] = 1
        elif s.get("ws") == 2:
            mat[i, KEY_COLS.index("S")] = 1
        if s.get("ad") == 1:
            mat[i, KEY_COLS.index("A")] = 1
        elif s.get("ad") == 2:
            mat[i, KEY_COLS.index("D")] = 1
        if s.get("ud") == 1:
            mat[i, KEY_COLS.index("I")] = 1
        elif s.get("ud") == 2:
            mat[i, KEY_COLS.index("K")] = 1
        if s.get("lr") == 1:
            mat[i, KEY_COLS.index("J")] = 1
        elif s.get("lr") == 2:
            mat[i, KEY_COLS.index("L")] = 1
    # Any remaining frames stay zero (idle). Holding the last tick here is what
    # produced the broken mirror clips: 48 ticks held to 124 frames ran the
    # return leg for 100 frames instead of 24, so the camera ended far past its
    # start and gsc scored an overshoot rather than a return.
    return mat


def first_frame(video: Path, out_png: Path) -> bool:
    try:
        rd = imageio.get_reader(str(video))
        fr = rd.get_data(0)
        rd.close()
        imageio.imwrite(str(out_png), fr)
        return True
    except Exception as exc:
        print(f"[warn] first-frame failed {video}: {exc}")
        return False


def gather(gt_root: Path, perspectives) -> list[dict]:
    out = []
    for p in perspectives:
        for tt in TEST_TYPES:
            td = gt_root / p / "test" / tt
            if not td.is_dir():
                continue
            for sd in sorted(td.iterdir()):
                if (sd / "video.mp4").exists() and (sd / "action.json").exists():
                    out.append({"perspective": p, "test_type": tt, "gt_name": sd.name,
                                "video": sd / "video.mp4", "action": sd / "action.json"})
    return out


def run_one(png: Path, action_npy: Path, out_path: Path, args: argparse.Namespace,
            num_frames: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(H3_PY), str(INFER),
        "--checkpoint", str(args.checkpoint),
        "--first-frame", str(png),
        "--scene-prompt", args.scene_prompt,
        "--action-file", str(action_npy),
        "--num-frames", str(num_frames),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--cfg-scale", str(args.cfg_scale),
        "--out", str(out_path),
    ]
    r = subprocess.run(cmd, cwd=str(H3_ROOT))
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--perspective", default=None)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--scene-prompt", default=GENERIC_SCENE_PROMPT,
                     help="MIND has no per-sample caption; one generic scene prompt is used for all samples")
    ap.add_argument("--num-frames", type=int, default=None,
                     help=f"must be 17k+5 (124, 243, 481, ...); default {NUM_FRAMES}, or snapped to "
                          "the trajectory length under --mirror-test")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    # Additive and on by default: one invocation stages the main sets AND the
    # mirror set, each at its own frame count. This flag used to be exclusive
    # AND default-on, which meant a plain run generated only mirror clips and
    # silently skipped action_space_test / mem_test entirely.
    ap.add_argument("--mirror-test", action=argparse.BooleanOptionalAction, default=True,
                     help="also stage the mirror_test (go-then-return) set; default on")
    ap.add_argument("--mirror-only", action="store_true",
                     help="stage ONLY the mirror set, skipping action_space/mem")
    ap.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                     help="mirror trajectory (default 'w')")
    args = ap.parse_args()

    explicit_frames = args.num_frames is not None
    if explicit_frames and (args.num_frames - 5) % 17:
        ap.error(f"--num-frames must be 17k+5 (124, 243, 481, ...), got {args.num_frames}")
    if not explicit_frames:
        args.num_frames = NUM_FRAMES
    if not H3_PY.exists():
        raise SystemExit(f"H3-World venv not found: {H3_PY} -- see H3-World/README.md Setup")
    if not args.checkpoint.exists():
        raise SystemExit(f"H3-World checkpoint not found: {args.checkpoint} -- run download_models.sh first")

    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES

    # Each pass carries its own frame count: the main sets use --num-frames,
    # the mirror set is fitted to its trajectory so gsc's midpoint split lands
    # on the turnaround. That difference is why these used to be separate runs.
    samples: list[dict] = []
    if not args.mirror_only:
        for s in gather(args.gt_root, perspectives):
            s["num_frames"] = args.num_frames
            s["resample"] = False
            samples.append(s)
        print(f"[h3world-mind] main: {len(samples)} samples @ {args.num_frames} frames")

    if args.mirror_test or args.mirror_only:
        mirror = [s for s in gather_mirror_samples(args.gt_root, args.mirror_action)
                  if not args.perspective or s["perspective"] == args.perspective]
        mirror_frames = args.num_frames
        if mirror and not explicit_frames:
            ticks = len(json.load(open(mirror[0]["action"], encoding="utf-8"))["data"])
            mirror_frames = snap_mirror_frames(ticks)
        for s in mirror:
            s["num_frames"] = mirror_frames
            s["resample"] = True
        samples.extend(mirror)
        print(f"[h3world-mind] mirror action='{args.mirror_action}': {len(mirror)} samples "
              f"@ {mirror_frames} frames (split at {mirror_frames // 2})")

    samples = samples[args.start_index:]
    if args.limit:
        samples = samples[:args.limit]

    work = Path(tempfile.mkdtemp(prefix="h3world_mind_"))
    done = skipped = 0
    for s in samples:
        out_path = args.test_root / args.model_name / s["perspective"] / s["test_type"] / s["gt_name"] / "video.mp4"
        if out_path.exists():
            skipped += 1
            continue
        png = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}.png"
        src_png = s.get("frame_png_src")
        if src_png is not None:  # mirror: first frame is already a PNG
            import shutil
            shutil.copy(str(src_png), str(png))
        elif not first_frame(s["video"], png):
            continue
        mind = json.load(open(s["action"], encoding="utf-8"))["data"]
        mat = mind_to_action_matrix(mind, s["num_frames"], resample=s["resample"])
        action_npy = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}_action.npy"
        np.save(action_npy, mat)

        print(f"[h3world-mind] {s['perspective']}/{s['test_type']}/{s['gt_name']} -> {out_path}")
        rc = run_one(png, action_npy, out_path, args, s["num_frames"])
        if rc != 0:
            print(f"[h3world-mind] sample failed (rc={rc}): {s['gt_name']}")
            continue
        done += 1

    print(f"[h3world-mind] done: {done} generated, {skipped} already existed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
