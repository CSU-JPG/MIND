"""Drive LingBot-World-2 (Wan-A14B, WSL fp8) over MIND. Runs INSIDE WSL via
lingbot-venv (needs numpy + the model). For each MIND sample, builds an action_path
dir (poses/intrinsics/wasd/ijkl npy + image + prompt) then calls generate.py.

lingbot-v2 is CAMERA-POSE driven (poses.npy 4x4 c2w REQUIRED). MIND gives per-frame
actor_pos + actor_rpy -> we synthesize c2w matrices from those. wasd/ijkl come from
ws/ad/ud/lr. intrinsics reused from examples/00. Prompt is generic (MIND has no caption).

SLOW: ~1 fps, 14B, per-sample generate.py reload -> smoke with --limit 1-2 first.
Paths are WSL paths (/mnt/c/...). Invoke via drive_lingbot_v2.bat.
"""
import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

LINGBOT = Path("/mnt/c/workspace/world/lingbot-world-v2")
GEN = LINGBOT / "generate.py"
PY = Path("/home/kschmid/lingbot-venv/bin/python")
GT_ROOT = Path("/mnt/c/workspace/world/MIND-Data")
TEST_ROOT = Path("/mnt/c/workspace/world/MIND-tests")
PERSPECTIVES = ("1st_data", "3rd_data")
TEST_TYPES = ("action_space_test", "mem_test")
PROMPT = "Third-person view moving through the scene, stable forward motion, consistent environment."


def euler_to_c2w(pos: dict, rpy: dict) -> np.ndarray:
    """actor_pos{x,y,z} + actor_rpy{x,y,z}(deg) -> 4x4 camera-to-world (OpenCV-ish)."""
    r, p, y = math.radians(rpy["x"]), math.radians(rpy["y"]), math.radians(rpy["z"])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32) / 100.0  # cm->m guess
    return M


def build_action_dir(mind_data: list, first_frame_src: Path, work: Path) -> Path:
    T = len(mind_data)
    poses = np.stack([euler_to_c2w(s["actor_pos"], s["actor_rpy"]) for s in mind_data])  # (T,4,4)
    wasd = np.zeros((T, 4), np.float32)   # W,A,S,D
    ijkl = np.zeros((T, 4), np.float32)   # I,J,K,L
    for i, s in enumerate(mind_data):
        if s.get("ws") == 1: wasd[i, 0] = 1     # W
        elif s.get("ws") == 2: wasd[i, 2] = 1   # S
        if s.get("ad") == 1: wasd[i, 1] = 1     # A
        elif s.get("ad") == 2: wasd[i, 3] = 1   # D
        if s.get("ud") == 1: ijkl[i, 0] = 1     # I (up)
        elif s.get("ud") == 2: ijkl[i, 2] = 1   # K (down)
        if s.get("lr") == 1: ijkl[i, 1] = 1     # J (left)
        elif s.get("lr") == 2: ijkl[i, 3] = 1   # L (right)
    np.save(work / "poses.npy", poses)
    np.save(work / "wasd_action.npy", wasd)
    np.save(work / "ijkl_action.npy", ijkl)
    np.save(work / "action.npy", np.zeros((T, 4), np.int32))
    # reuse the example intrinsics (fx,fy,cx,cy), tiled to T
    K = np.load(LINGBOT / "examples/00/intrinsics.npy")[0]
    np.save(work / "intrinsics.npy", np.tile(K, (T, 1)).astype(np.float32))
    # image.jpg already written into work by first_frame() before this call.
    (work / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    return work


def first_frame(video: Path, out_jpg: Path) -> bool:
    import imageio.v2 as imageio
    try:
        rd = imageio.get_reader(str(video)); fr = rd.get_data(0); rd.close()
        imageio.imwrite(str(out_jpg), fr)
        return True
    except Exception as exc:
        print(f"[warn] frame {video}: {exc}"); return False


def gather(perspectives):
    out = []
    for p in perspectives:
        for tt in TEST_TYPES:
            td = GT_ROOT / p / "test" / tt
            if not td.is_dir(): continue
            for sd in sorted(td.iterdir()):
                if (sd / "video.mp4").exists() and (sd / "action.json").exists():
                    out.append({"perspective": p, "test_type": tt, "gt_name": sd.name,
                                "video": sd / "video.mp4", "action": sd / "action.json"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perspective", default=None)
    ap.add_argument("--frame-num", type=int, default=81)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()
    persp = (args.perspective,) if args.perspective else PERSPECTIVES
    samples = gather(persp)[args.start_index:]
    if args.limit: samples = samples[:args.limit]
    print(f"[lingbot-mind] {len(samples)} samples")

    # Build every sample's action dir + a manifest, then ONE generate.py --mind_batch
    # call so the 14B model loads once (not per sample).
    root = Path(tempfile.mkdtemp(prefix="lingbot_mind_"))
    manifest = []
    for s in samples:
        out = TEST_ROOT / "lingbot-v2" / s["perspective"] / s["test_type"] / s["gt_name"] / "video.mp4"
        if out.exists(): print(f"[skip] {s['gt_name']}"); continue
        work = root / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}"
        work.mkdir(parents=True, exist_ok=True)
        if not first_frame(s["video"], work / "image.jpg"): continue
        mind = json.load(open(s["action"], encoding="utf-8"))["data"]
        build_action_dir(mind, work / "image.jpg", work)
        manifest.append({"image": str(work / "image.jpg"), "action_path": str(work),
                         "out_path": str(out), "prompt": PROMPT})

    if not manifest:
        print("[lingbot-mind] nothing to do"); return 0
    mpath = root / "manifest.json"
    json.dump(manifest, open(mpath, "w"))
    print(f"[lingbot-mind] {len(manifest)} samples -> ONE --mind_batch run (model loads once)")

    cmd = [str(PY), str(GEN), "--task", "i2v-A14B", "--infer_mode", "causal_fast",
           "--size", "320*576", "--ckpt_dir", str(LINGBOT),
           "--image", manifest[0]["image"], "--action_path", manifest[0]["action_path"],
           "--frame_num", str(args.frame_num), "--local_attn_size", "12", "--sink_size", "6",
           "--offload_model", "True", "--fp8", "--prompt", PROMPT, "--mind_batch", str(mpath)]
    r = subprocess.run(cmd, cwd=str(LINGBOT))
    print(f"[lingbot-mind] done (rc={r.returncode})")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
