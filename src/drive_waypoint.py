"""Drive Waypoint-1.5 (Overworld / world_engine) over the MIND benchmark.

Normal test types (action_space_test, mem_test): seed the GT video's first frame,
replay action.json's per-step ws/ad/ud/lr, write a matching-length mp4.

Mirror test (--mirror-test): flat MIND-Data/{persp}/test/mirror_test/ = 50 first-frame
PNGs + one -{action}.json (47-frame go-then-return). Seed each PNG, replay the chosen
mirror action (default "w" = forward-out / back-return); the gsc metric scores memory
consistency by time-flipping the second half against the first.

Output: test_root/waypoint/{persp}/{test_type}/<name>/video.mp4  (where run_mind.bat looks).
Run via scope-overworld's venv (world_engine + cv2 + imageio) -- see drive_waypoint.bat.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torchvision.transforms.v2.functional import resize as tv_resize
from world_engine import CtrlInput, WorldEngine

# src/ is sys.path[0] when run as a script, so this resolves like the other drivers.
from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples

REPO, CANVAS_H, CANVAS_W = "Overworld/Waypoint-1.5-1B", 720, 1280
AE_REPO, AE_FILE = "Overworld-Models/taehv1_5", "taehv1_5.pth"
VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44}
PERSPECTIVES = ("1st_data", "3rd_data")
NORMAL_TEST_TYPES = ("action_space_test", "mem_test")   # mirror_test handled separately
LOOK = 0.5


def action_to_ctrl(step: dict) -> CtrlInput:
    """MIND tri-state action -> Waypoint controller. ws=fwd/back, ad=strafe, lr/ud=look."""
    btn = []
    if step.get("ws") == 1: btn.append(VK["W"])
    elif step.get("ws") == 2: btn.append(VK["S"])
    if step.get("ad") == 1: btn.append(VK["A"])
    elif step.get("ad") == 2: btn.append(VK["D"])
    mx = {1: -LOOK, 2: LOOK}.get(step.get("lr", 0), 0.0)   # yaw: 1=left, 2=right
    my = {1: -LOOK, 2: LOOK}.get(step.get("ud", 0), 0.0)   # pitch: 1=up, 2=down
    return CtrlInput(button=btn, mouse=(mx, my))


def seed(engine: WorldEngine, frame_rgb: np.ndarray):
    """Center-crop to 16:9, resize to canvas, seed the 4-frame window."""
    t = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # CHW uint8
    _, ih, iw = t.shape
    target = CANVAS_W / CANVAS_H
    if iw / ih > target:
        nw = int(ih * target); left = (iw - nw) // 2; t = t[:, :, left:left + nw]
    elif iw / ih < target:
        nh = int(iw / target); top = (ih - nh) // 2; t = t[:, top:top + nh, :]
    t = tv_resize(t, [CANVAS_H, CANVAS_W], antialias=True).permute(1, 2, 0).contiguous()
    engine.append_frame(t.unsqueeze(0).expand(4, -1, -1, -1).contiguous())


def read_video(video: Path):
    cap = cv2.VideoCapture(str(video))
    ok, fr = cap.read()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return (cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None), n, fps


def prep(sample: dict):
    """(seed_rgb HWC uint8, n_frames, fps, action_data) for a normal OR mirror sample."""
    action_data = json.load(open(sample["action"], encoding="utf-8"))["data"]
    png = sample.get("frame_png_src")
    if png is not None:                                   # mirror: first frame is a PNG
        img = cv2.imread(str(png))
        seed_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        return seed_rgb, len(action_data), 24.0, action_data
    seed_rgb, n, fps = read_video(sample["video"])         # normal: extract from video.mp4
    return seed_rgb, n, fps, action_data


def gather_normal(gt_root: Path, perspectives) -> list[dict]:
    out = []
    for persp in perspectives:
        for tt in NORMAL_TEST_TYPES:
            td = gt_root / persp / "test" / tt
            if not td.is_dir():
                continue
            for sd in sorted(td.iterdir()):
                if (sd / "video.mp4").exists() and (sd / "action.json").exists():
                    out.append({"perspective": persp, "test_type": tt, "gt_name": sd.name,
                                "video": sd / "video.mp4", "action": sd / "action.json"})
    return out


def run_one(engine, sample, test_root, model_name, fps_override):
    out = test_root / model_name / sample["perspective"] / sample["test_type"] / sample["gt_name"] / "video.mp4"
    if out.exists():
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} (exists)")
        return
    seed_rgb, n_frames, gt_fps, data = prep(sample)
    if seed_rgb is None or n_frames == 0 or not data:
        print(f"[warn] no frames/actions: {sample['gt_name']}"); return
    fps = fps_override or gt_fps

    engine.reset()
    seed(engine, seed_rgb)
    n_chunks = math.ceil(n_frames / 4)
    frames = []
    for c in range(n_chunks):
        idx = min(int(c / max(n_chunks, 1) * len(data)), len(data) - 1)  # action along the clip
        fr = engine.gen_frame(ctrl=action_to_ctrl(data[idx]))
        for f in fr.cpu():
            frames.append(f.numpy())
    frames = frames[:n_frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=fps, codec="libx264", macro_block_size=None)
    print(f"[ok] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} -> {out}  ({len(frames)}f @ {fps:.0f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--model-name", default="waypoint")
    ap.add_argument("--fps", type=float, default=None, help="override output fps (default: GT fps / 24 for mirror)")
    ap.add_argument("--perspective", default=None, help="1st_data | 3rd_data (default: both)")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--mirror-test", action="store_true",
                    help="drive ONLY the mirror_test (go-then-return); default drives normal + mirror")
    ap.add_argument("--no-mirror", action="store_true",
                    help="skip the mirror_test (drive only action_space + mem)")
    ap.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                    help="mirror trajectory (default 'w' = forward-out/back-return)")
    args = ap.parse_args()

    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES
    mirror = [s for s in gather_mirror_samples(args.gt_root, args.mirror_action)
              if not args.perspective or s["perspective"] == args.perspective]
    if args.mirror_test:                       # mirror only (back-compat)
        samples = mirror
        print(f"[waypoint-mind] MIRROR test, action='{args.mirror_action}': {len(samples)} samples")
    else:                                      # DEFAULT: normal + mirror in one load-once session
        samples = gather_normal(args.gt_root, perspectives)
        print(f"[waypoint-mind] action_space + mem: {len(samples)} samples")
        if not args.no_mirror:
            samples = samples + mirror
            print(f"[waypoint-mind] + MIRROR (action='{args.mirror_action}'): {len(mirror)} -> {len(samples)} total")
    samples = samples[args.start_index:]
    if args.limit:
        samples = samples[:args.limit]

    print(f"[waypoint-mind] loading {REPO} (one-time compile) ...")
    model_dir = snapshot_download(REPO, allow_patterns=["model.safetensors", "config.yaml"])
    ae_path = hf_hub_download(AE_REPO, AE_FILE)
    engine = WorldEngine(model_dir, quant=args.quant,
                         model_config_overrides={"ae_uri": ae_path},
                         device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        engine.gen_frame(ctrl=CtrlInput())

    for s in samples:
        try:
            run_one(engine, s, args.test_root, args.model_name, args.fps)
        except Exception as exc:
            print(f"[err] {s['gt_name']}: {exc}")
    print("[waypoint-mind] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
