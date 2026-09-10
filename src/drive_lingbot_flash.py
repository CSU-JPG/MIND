"""Drive flashdreams-lingbot (lingbot-world-fast / -fast-flash) into MIND test layout.

This driver runs *inside* the flashdreams-lingbot uv env (so ``lingbot.*`` and
``flashdreams.*`` import cleanly) and loads the 14B pipeline ONCE, then loops
over MIND-Data samples in-process. Subprocess-per-sample would reload the
model every iteration (~2-3 min each) — a non-starter for a ~100-sample sweep.

For each MIND sample:
  1. Extract first frame from video.mp4.
  2. Build dummy poses + intrinsics for the AR run length (flashdreams-lingbot
     is camera-controlled; MIND ships WASD actions, not camera matrices —
     Path A here fakes a slow forward-walk). Path B (TODO) wires action.json
     -> per-frame translation.
  3. Call ``pipeline.initialize_cache`` then loop ``pipeline.generate`` +
     ``pipeline.finalize`` for ``total_blocks`` AR steps.
  4. VAE-decode chunks, concat, write mp4 at 24 fps via PyAV.

Stage outputs to:
    test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4

Skip-if-exists. Output mp4 fps is fixed at 24 (matches MIND-Data ground truth).

Launch via the bat (which calls ``uv run --package flashdreams-lingbot
python drive_lingbot_flash.py``), not directly — this script needs the
flashdreams uv env on sys.path.
"""

import argparse
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

# Make MIND's utils importable when running from flashdreams uv env.
_MIND_SRC = Path(__file__).resolve().parent
if _MIND_SRC.exists() and str(_MIND_SRC) not in sys.path:
    sys.path.insert(0, str(_MIND_SRC))

import av
import numpy as np
import torch
from PIL import Image

from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_FAST,
    PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
)
from lingbot.encoder.camctrl import CamCtrlInput

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4


TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")

DEFAULT_W, DEFAULT_H = 832, 480
DEFAULT_FPS = 24

SLUG_TO_CFG = {
    "lingbot-world-fast": PIPELINE_LINGBOT_WORLD_FAST,
    "lingbot-world-fast-taehv-window15-sink3": PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
}


def extract_first_frame_pil(video_path: Path) -> Image.Image:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            return frame.to_image().convert("RGB")
    raise RuntimeError(f"No frames decoded from {video_path}")


def pil_to_pipeline_tensor(im: Image.Image, w: int, h: int, device: torch.device) -> torch.Tensor:
    """PIL -> [T=1, C=3, H, W] in [-1, 1] on the pipeline's device."""
    im = im.resize((w, h), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0   # [H, W, 3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return t.to(device=device, dtype=torch.bfloat16)


def make_dummy_intrinsics(num_frames: int, w: int = DEFAULT_W, h: int = DEFAULT_H,
                          device: torch.device | None = None) -> torch.Tensor:
    fx = fy = float(w) / 2.0
    cx, cy = float(w) / 2.0, float(h) / 2.0
    intr = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).expand(num_frames, 4).contiguous()
    if device is not None:
        intr = intr.to(device=device)
    return intr


def make_dummy_poses(num_frames: int, step: float = 0.02,
                     device: torch.device | None = None) -> torch.Tensor:
    """Identity rotation + linear forward (+z) translation per frame."""
    poses = torch.eye(4, dtype=torch.float32).expand(num_frames, 4, 4).clone()
    poses[:, 2, 3] = torch.arange(num_frames, dtype=torch.float32) * step
    if device is not None:
        poses = poses.to(device=device)
    return poses


def derive_caption(perspective: str) -> str:
    if perspective == "1st_data":
        return "First-person view exploring a 3D virtual environment, smooth camera motion."
    return "Third-person view of a character exploring a 3D virtual environment."


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


def write_video_24fps(frames_thwc_uint8: np.ndarray, out_path: Path, fps: int = DEFAULT_FPS) -> None:
    """frames_thwc_uint8: [T, H, W, 3] uint8. Writes H.264 mp4 at fps."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = frames_thwc_uint8.shape
    assert C == 3
    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "18"}
        for frame_np in frames_thwc_uint8:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def chunks_to_uint8(chunks: list[torch.Tensor]) -> np.ndarray:
    """chunks: list of [T_chunk, C, H, W] in [-1, 1] (bf16/fp32 on cpu/cuda).
    Returns [T_total, H, W, 3] uint8."""
    video = torch.cat([c.cpu().float() for c in chunks], dim=0)        # [T, C, H, W]
    video = ((video.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255)
    video = video.to(torch.uint8).permute(0, 2, 3, 1).contiguous()    # [T, H, W, C]
    return video.numpy()


def run_one(pipeline, sample: dict, test_root: Path, model_name: str, args, device: torch.device) -> int:
    out = output_path(test_root, model_name, sample)
    if out.exists() and not args.force:
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} -> {out} (exists)")
        return 0

    # First frame
    if sample.get("frame_png_src") is not None:
        first_im = Image.open(sample["frame_png_src"]).convert("RGB")
    else:
        first_im = extract_first_frame_pil(sample["video"])
    first_t = pil_to_pipeline_tensor(first_im, DEFAULT_W, DEFAULT_H, device)   # [1, 3, H, W]

    caption = derive_caption(sample["perspective"])
    sp = pipeline.decoder.spatial_compression_ratio
    cache = pipeline.initialize_cache(
        text=[caption],
        image=first_t,
        height=DEFAULT_H // sp,
        width=DEFAULT_W // sp,
    )

    # Per-AR-step dummy camera payload. Use a single global trajectory and
    # slice per block; flashdreams-lingbot expects T_chunk frames per call.
    num_frames_total = args.total_blocks * args.frames_per_block
    intr_full = make_dummy_intrinsics(num_frames_total, device=device)
    poses_full = make_dummy_poses(num_frames_total, step=args.pose_step, device=device)

    print(f"\n=== {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} ===")
    print(f"caption: {caption[:90]}{'...' if len(caption) > 90 else ''}")
    print(f"out:     {out}")
    if args.dry_run:
        print("  [dry-run] would generate", args.total_blocks, "AR blocks")
        return 0

    t0 = time.perf_counter()
    chunks: list[torch.Tensor] = []
    fpb = args.frames_per_block
    for i in range(args.total_blocks):
        sl = slice(i * fpb, (i + 1) * fpb)
        camctrl = CamCtrlInput(
            intrinsics=intr_full[sl],
            poses=poses_full[sl],
            world_scale=args.world_scale,
        )
        chunk = pipeline.generate(autoregressive_index=i, cache=cache, input=camctrl)
        pipeline.finalize(autoregressive_index=i, cache=cache)
        chunks.append(chunk.detach())

    frames_uint8 = chunks_to_uint8(chunks)
    write_video_24fps(frames_uint8, out, fps=args.fps)
    elapsed = time.perf_counter() - t0
    print(f"  wrote {frames_uint8.shape[0]} frames @ {args.fps}fps  elapsed={elapsed:.1f}s")

    if out.exists():
        log_mp4(model_name, sample["perspective"], sample["test_type"], sample["gt_name"], out)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-root", type=Path, default=Path(__file__).resolve().parent.parent.parent / "MIND-Data")
    p.add_argument("--test-root", type=Path, default=Path(__file__).resolve().parent.parent.parent / "MIND-tests")
    p.add_argument("--model-name", default="lingbot-flash")
    p.add_argument("--only", nargs="+")
    p.add_argument("--perspective", choices=PERSPECTIVES)
    p.add_argument("--test-type", choices=TEST_TYPES)
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS,
                   help="Output mp4 fps (default 24, matches MIND-Data ground truth).")

    # flashdreams-lingbot knobs
    p.add_argument("--slug", default="lingbot-world-fast", choices=list(SLUG_TO_CFG))
    p.add_argument("--total-blocks", type=int, default=21)
    p.add_argument("--frames-per-block", type=int, default=25,
                   help="Frames per AR block (used to size dummy trajectory; flashdreams example has ~25).")
    p.add_argument("--pose-step", type=float, default=0.02,
                   help="Forward translation per frame in the dummy camera trajectory.")
    p.add_argument("--world-scale", type=float, default=1.0,
                   help="Scalar applied to translations when normalizing world coords.")

    # mirror_test (matches drive_lingbot.py interface)
    p.add_argument("--mirror-test", action="store_true")
    p.add_argument("--mirror-only", action="store_true")
    p.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS)
    args = p.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    print(f"[drive_lingbot_flash] slug={args.slug}  model_name={args.model_name}  fps={args.fps}")
    print(f"[drive_lingbot_flash] loading pipeline ONCE (subsequent samples reuse it) ...")
    pipeline_config = SLUG_TO_CFG[args.slug]
    pipeline = pipeline_config.setup().to("cuda").eval()
    device = next(pipeline.parameters()).device if hasattr(pipeline, "parameters") else torch.device("cuda")
    print(f"[drive_lingbot_flash] pipeline ready on {device}.")

    samples = [] if args.mirror_only else gather_samples(args.gt_root)
    mirror_samples = gather_mirror_samples(args.gt_root, args.mirror_action) if args.mirror_test else []
    if args.perspective:
        samples = [s for s in samples if s["perspective"] == args.perspective]
        mirror_samples = [s for s in mirror_samples if s["perspective"] == args.perspective]
    if args.test_type:
        samples = [s for s in samples if s["test_type"] == args.test_type]
    if args.only:
        _match = lambda s: any(sub.lower() in s["gt_name"].lower() for sub in args.only)
        samples = [s for s in samples if _match(s)]
        mirror_samples = [s for s in mirror_samples if _match(s)]
    if args.limit:
        # Per-perspective cap -> N 1st + N 3rd (balanced). Mirror samples (needed for gsc)
        # are capped+kept SEPARATELY rather than cut by a flat samples[:N] -- they're
        # appended last, so a flat limit would drop them entirely (no gsc).
        def _cap_per_perspective(rows):
            out = []
            for _p in PERSPECTIVES:
                out += [s for s in rows if s["perspective"] == _p][: args.limit]
            return out
        samples = _cap_per_perspective(samples)
        mirror_samples = _cap_per_perspective(mirror_samples)
    samples += mirror_samples

    if not samples:
        print("No samples matched.")
        return 1

    print(f"Will process {len(samples)} sample(s) (pipeline loaded once):")
    for s in samples:
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}")
    print()

    failures: list[str] = []
    for s in samples:
        try:
            rc = run_one(pipeline, s, args.test_root, args.model_name, args, device)
        except Exception as e:
            print(f"  EXCEPTION on {s['gt_name']}: {type(e).__name__}: {e}")
            rc = 1
        if rc != 0:
            failures.append(f"{s['perspective']}/{s['test_type']}/{s['gt_name']}")
        # Reclaim fragmentation between samples (the 14B transformer stays resident;
        # this only frees per-sample activations/KV-cache, not the model itself).
        torch.cuda.empty_cache()

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
