"""Drive ABot-World over MIND -- LOAD-ONCE via inference.py --mind-batch.

Builds a manifest (first-frame PNG + converted-action JSON + out-path per sample),
then calls ABot's inference.py ONCE with --mind-batch. inference.py loops the manifest
in a single process, and get_pipeline is lru_cached, so the ~24GB model loads ONCE
(not per sample) -- the whole point.

MIND action.json (ws/ad/ud/lr tri-state) -> ABot keys (WASD move + IJKL look).
ABot is text-conditioned; MIND has no caption, so a generic forward-motion prompt is used.
Run via ABot's venv (see drive_abot.bat).
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio

# src/ is sys.path[0] when run as a script, so this resolves like the other drivers.
from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples

ABOT_ROOT = Path(__file__).resolve().parent.parent.parent / "ABot-World"
INFER = ABOT_ROOT / "scripts" / "inference.py"
ABOT_PY = ABOT_ROOT / ".venv" / "Scripts" / "python.exe"
PERSPECTIVES = ("1st_data", "3rd_data")
TEST_TYPES = ("action_space_test", "mem_test")
GENERIC_PROMPT = "Third-person view moving through the scene, stable forward motion, consistent environment."


def mind_to_abot(mind_data: list, fps: int = 16) -> dict:
    frames = []
    for i, s in enumerate(mind_data):
        frames.append({"keys": {
            "W": s.get("ws") == 1, "S": s.get("ws") == 2,
            "A": s.get("ad") == 1, "D": s.get("ad") == 2,
            "J": s.get("lr") == 1, "L": s.get("lr") == 2,
            "I": s.get("ud") == 1, "K": s.get("ud") == 2,
        }, "frame_id": f"{i:06d}"})
    return {"total_frames": len(frames), "fps": fps, "frames": frames}


def first_frame(video: Path, out_png: Path) -> bool:
    try:
        rd = imageio.get_reader(str(video)); fr = rd.get_data(0); rd.close()
        imageio.imwrite(str(out_png), fr)
        return True
    except Exception as exc:
        print(f"[warn] first-frame failed {video}: {exc}"); return False


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--model-name", default="abot")
    ap.add_argument("--perspective", default=None)
    ap.add_argument("--blocks", type=int, default=8, help="ABot fps-blocks (<=15; KV cache limit)")
    ap.add_argument("--quant", default="fp8-per-tensor",
                    help="quant tier for speed (default fp8-per-tensor, Blackwell). "
                         "Fallbacks if it errors: int8-torchao | none")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mirror-test", action="store_true", help="run the mirror_test (go-then-return)")
    ap.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                    help="mirror trajectory (default 'w')")
    args = ap.parse_args()

    if not ABOT_PY.exists():
        raise SystemExit(f"ABot venv not found: {ABOT_PY} -- run ABot-World\\setup_venv.bat")
    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES
    if args.mirror_test:
        samples = [s for s in gather_mirror_samples(args.gt_root, args.mirror_action)
                   if not args.perspective or s["perspective"] == args.perspective]
        print(f"[abot-mind] MIRROR action='{args.mirror_action}': {len(samples)} samples")
    else:
        samples = gather(args.gt_root, perspectives)
    samples = samples[args.start_index:]
    if args.limit:
        samples = samples[:args.limit]

    # Prep each sample: extract first frame, convert actions, build a manifest entry.
    work = Path(tempfile.mkdtemp(prefix="abot_mind_"))
    manifest = []
    for s in samples:
        out_path = args.test_root / args.model_name / s["perspective"] / s["test_type"] / s["gt_name"] / "video.mp4"
        if out_path.exists():
            continue
        png = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}.png"
        src_png = s.get("frame_png_src")
        if src_png is not None:                      # mirror: first frame is already a PNG
            shutil.copy(str(src_png), str(png))
        elif not first_frame(s["video"], png):
            continue
        mind = json.load(open(s["action"], encoding="utf-8"))["data"]
        aj = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}_abot.json"
        json.dump(mind_to_abot(mind), open(aj, "w"))
        manifest.append({"ref_image": str(png), "action_json": str(aj),
                         "prompt": GENERIC_PROMPT, "out_path": str(out_path), "fps_blocks": args.blocks})

    if not manifest:
        print("[abot-mind] nothing to do (all exist or no samples)"); return 0
    manifest_path = work / "manifest.json"
    json.dump(manifest, open(manifest_path, "w"))
    print(f"[abot-mind] {len(manifest)} samples -> one load-once inference.py --mind-batch run")

    # ONE inference.py call: pipeline loads once, loops the manifest.
    cmd = [str(ABOT_PY), str(INFER), "--quant-type", args.quant, "--mind-batch", str(manifest_path)]
    r = subprocess.run(cmd, cwd=str(ABOT_ROOT))
    print(f"[abot-mind] done (rc={r.returncode})")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
