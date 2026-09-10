"""Drive SolarWM's MiniMax-H3 over MIND with REAL camera-conditioned generation.

Two engines, selected via --engine:

  camera (default) -- h3_camera_infer.py: loads the REAL trained Stage0.5
    LoRA adapter and calls the REAL camera-conditioned H3Stage0p5Core.
    generate(). Each sample's real ws/ad/ud/lr action.json is converted
    directly into a [47,4,4] camera trajectory (mind_actions_to_camera_c2w
    below) -- genuine per-frame conditioning, not a text hint. UNTESTED end
    to end (h3_camera_infer.py itself is untested -- see its docstring for
    the unverified camera axis/sign caveat). Needs --adapter.

  text -- h3_infer.py: the ORIGINAL fallback. Drives the raw UNCONDITIONED
    base diffusers pipeline (no camera/action input at all -- confirmed: a
    "goes straight" prompt produced a left turn in manual testing). Each
    sample's action.json is only paraphrased into a text sentence
    (summarize_mind_actions), not real conditioning. Kept for comparison /
    in case the camera engine doesn't work on first try.

Consequence for scoring (both engines): lcm/visual/dino/avg_mse are
meaningful as a "does it hold scene identity/quality" check. `action` is
meaningless for the text engine (nothing to measure control against) and
UNVERIFIED for the camera engine (real conditioning exists, but whether it
actually measures as correct control is untested). `gsc` is unreliable for
both. Exclude both when scoring until camera-engine results are manually
inspected and confirmed to actually follow the intended direction:
  run_mind.sh solarwm lcm,visual,dino 1 both

Run via SolarWM's H3 venv (see drive_solarwm.sh) -- SOLARWM_ROOT/.venv-h3.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples

PERSPECTIVES = ("1st_data", "3rd_data")
TEST_TYPES = ("action_space_test", "mem_test")
BASE_PROMPT = "Third-person view moving through the scene, consistent environment, realistic camera motion."
# Used only by the text engine, only if a sample's action.json is missing/unparseable.
GENERIC_PROMPT = BASE_PROMPT + " Stable forward motion."

_WS = {0: "stays still", 1: "moves forward", 2: "moves backward"}
_LR = {0: "", 1: "turning left", 2: "turning right"}


def summarize_mind_actions(mind_data: list[dict], n_buckets: int = 3) -> str:
    """Coarse natural-language paraphrase of a real ws/ad/ud/lr trajectory.

    Text-engine only. Buckets the tick sequence into n_buckets equal spans,
    takes the majority ws (forward/back/still) and lr (turn left/right/
    neither) per span, then collapses consecutive identical spans into one
    phrase. This is NOT per-frame conditioning -- h3_infer.py's pipeline has
    no input for that -- it's a best-effort textual summary. See module
    docstring for the reliability caveat.
    """
    if not mind_data:
        return "It stays still throughout the clip."
    n = max(1, min(n_buckets, len(mind_data)))
    bounds = [round(i * len(mind_data) / n) for i in range(n + 1)]
    phrases = []
    for i in range(n):
        bucket = mind_data[bounds[i]:bounds[i + 1]] or mind_data[-1:]
        ws_counts: dict[int, int] = {}
        lr_counts: dict[int, int] = {}
        for tick in bucket:
            ws_counts[tick.get("ws", 0)] = ws_counts.get(tick.get("ws", 0), 0) + 1
            lr_counts[tick.get("lr", 0)] = lr_counts.get(tick.get("lr", 0), 0) + 1
        dominant_ws = max(ws_counts, key=ws_counts.get)
        dominant_lr = max(lr_counts, key=lr_counts.get)
        move = _WS.get(dominant_ws, "stays still")
        turn = _LR.get(dominant_lr, "")
        phrases.append(f"{move} while {turn}" if turn else move)
    collapsed = [phrases[0]]
    for p in phrases[1:]:
        if p != collapsed[-1]:
            collapsed.append(p)
    if len(collapsed) == 1:
        return f"It {collapsed[0]} throughout the clip."
    return "It " + ", then ".join(collapsed) + "."


def prompt_for(action_path: Path) -> str:
    """Text-engine only: a generic scene prompt + real-action paraphrase."""
    try:
        mind_data = json.loads(action_path.read_text(encoding="utf-8"))["data"]
    except Exception as exc:
        print(f"[warn] couldn't read/parse {action_path}, falling back to GENERIC_PROMPT: {exc}")
        return GENERIC_PROMPT
    return f"{BASE_PROMPT} {summarize_mind_actions(mind_data)}"


def mind_actions_to_camera_c2w(
    mind_data: list[dict],
    *,
    num_latents: int = 47,
    yaw_deg_per_frame: float = 1.2,
    forward_step: float = 0.05,
    backward_step: float = 0.03,
) -> np.ndarray:
    """Convert a real MIND ws/ad/ud/lr trajectory into a [num_latents,4,4]
    absolute C2W camera path -- genuine per-frame conditioning for the
    camera engine, not a text paraphrase.

    Buckets the tick sequence into num_latents (47, H3's latent-frame count)
    equal spans -- much finer than summarize_mind_actions' 3 buckets, since
    this drives real per-frame camera motion rather than one sentence. Per
    bucket, takes the majority ws/lr (same voting as summarize_mind_actions)
    and composes a small yaw+forward/backward step onto the running camera
    pose, same math as h3_camera_infer.py's build_camera_c2w. Only ws
    (forward/back) and lr (turn left/right) are used; ad/ud are ignored,
    same simplification as the text engine.

    UNVERIFIED axis/sign convention -- see h3_camera_infer.py's docstring.
    """
    if not mind_data:
        mind_data = [{"ws": 0, "lr": 0}]
    n = num_latents
    bounds = [round(i * len(mind_data) / n) for i in range(n + 1)]
    c2w = np.zeros((n, 4, 4), dtype=np.float32)
    current = np.eye(4, dtype=np.float32)
    for i in range(n):
        bucket = mind_data[bounds[i]:bounds[i + 1]] or mind_data[-1:]
        ws_counts: dict[int, int] = {}
        lr_counts: dict[int, int] = {}
        for tick in bucket:
            ws_counts[tick.get("ws", 0)] = ws_counts.get(tick.get("ws", 0), 0) + 1
            lr_counts[tick.get("lr", 0)] = lr_counts.get(tick.get("lr", 0), 0) + 1
        dominant_ws = max(ws_counts, key=ws_counts.get)
        dominant_lr = max(lr_counts, key=lr_counts.get)
        step = {0: 0.0, 1: forward_step, 2: -backward_step}.get(dominant_ws, 0.0)
        yaw = {0: 0.0, 1: -yaw_deg_per_frame, 2: yaw_deg_per_frame}.get(dominant_lr, 0.0)
        theta = np.deg2rad(yaw)
        rot = np.array(
            [[np.cos(theta), 0.0, np.sin(theta)],
             [0.0, 1.0, 0.0],
             [-np.sin(theta), 0.0, np.cos(theta)]],
            dtype=np.float32,
        )
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = rot
        delta[:3, 3] = [0.0, 0.0, -step]
        c2w[i] = current
        current = current @ delta
    return c2w


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
                    out.append({
                        "perspective": p, "test_type": tt, "gt_name": sd.name,
                        "video": sd / "video.mp4", "action": sd / "action.json",
                    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--solarwm-root", type=Path, required=True)
    ap.add_argument("--engine", choices=("camera", "text"), default="camera")
    ap.add_argument("--model-path", default=None,
                    help="base model path; defaults to each engine's own default if unset")
    ap.add_argument("--adapter-path", default=None,
                    help="camera engine only; defaults to sibling SolarWM-models/SolarWM-h3-33B-bid-stage0p5-158f")
    ap.add_argument("--model-name", default="solarwm")
    ap.add_argument("--perspective", default=None)
    ap.add_argument("--num-frames", type=int, default=124,
                    help="text engine only; rounded to a valid 17n+5 in [120,360] by h3_infer.py itself")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mirror-test", action="store_true", help="run the mirror_test (go-then-return)")
    ap.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS)
    args = ap.parse_args()

    py = args.solarwm_root / ".venv-h3" / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"SolarWM H3 venv not found: {py} -- run setup_env_h3.sh first")
    if args.engine == "camera":
        infer = args.solarwm_root / "h3_camera_infer.py"
        adapter_path = args.adapter_path or str(args.solarwm_root / ".." / "SolarWM-models" / "SolarWM-h3-33B-bid-stage0p5-158f")
    else:
        infer = args.solarwm_root / "h3_infer.py"
        adapter_path = None
    if not infer.exists():
        raise SystemExit(f"{infer.name} not found: {infer}")

    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES
    if args.mirror_test:
        samples = [s for s in gather_mirror_samples(args.gt_root, args.mirror_action)
                   if not args.perspective or s["perspective"] == args.perspective]
        print(f"[solarwm-mind] MIRROR action='{args.mirror_action}': {len(samples)} samples")
    else:
        samples = gather(args.gt_root, perspectives)
    samples = samples[args.start_index:]
    if args.limit:
        samples = samples[:args.limit]

    work = Path(tempfile.mkdtemp(prefix="solarwm_mind_"))
    manifest = []
    skipped = 0
    for s in samples:
        out_dir = args.test_root / args.model_name / s["perspective"] / s["test_type"] / s["gt_name"]
        out_path = out_dir / "video.mp4"
        if out_path.exists():
            skipped += 1
            continue

        png = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}.png"
        src_png = s.get("frame_png_src")
        if src_png is not None:
            shutil.copy(str(src_png), str(png))
        elif not first_frame(s["video"], png):
            continue

        if args.engine == "camera":
            try:
                mind_data = json.loads(s["action"].read_text(encoding="utf-8"))["data"]
            except Exception as exc:
                print(f"[warn] couldn't read/parse {s['action']}, skipping: {exc}")
                continue
            c2w_path = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}_c2w.npy"
            np.save(c2w_path, mind_actions_to_camera_c2w(mind_data))
            manifest.append({
                "image": str(png),
                "prompt": BASE_PROMPT,
                "camera_c2w": str(c2w_path),
                "steps": args.steps,
                "out": str(out_path),
            })
        else:
            manifest.append({
                "prompt": prompt_for(s["action"]),
                "image": str(png),
                "num_frames": args.num_frames,
                "steps": args.steps,
                "out_dir": str(out_dir),
                "name": "video",
            })

    if not manifest:
        print(f"[solarwm-mind] nothing to do ({skipped} already existed, or no samples)")
        return 0
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"[solarwm-mind] {len(manifest)} samples ({skipped} skipped) -> one load-once {infer.name} --mind-batch run ({args.engine} engine)")

    cmd = [str(py), str(infer), "--mind-batch", str(manifest_path)]
    if args.engine == "camera":
        cmd += ["--base-model", args.model_path or str(args.solarwm_root / ".." / "SolarWM-models" / "SolarWM-h3-33B-base")]
        cmd += ["--adapter", adapter_path]
    elif args.model_path:
        cmd += ["--model-path", args.model_path]
    r = subprocess.run(cmd, cwd=str(args.solarwm_root))
    print(f"[solarwm-mind] done (rc={r.returncode})")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
