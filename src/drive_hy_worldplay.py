"""Drive HY-WorldPlay from a MIND-Data tree into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and, for each sample:
  1. Extracts the first frame at action.json's mark_time -> a temp PNG
  2. Converts the per-frame WASD action timeline starting at mark_time into a
     HY-WorldPlay pose string (e.g. "w-3,d-2,w-6"). HY-WorldPlay's parser counts
     in LATENTS (4 frames) with a +1 seed-frame on the first command, so the
     pose string emits exactly video_length frames.
  3. Calls HY-WorldPlay's hyvideo/generate.py via the HY-WorldPlay venv python
     (MIND's venv lacks flash-attn / sageattention / the right torch wheel).
  4. Stages the produced video.mp4 at the standard MIND-tests path.

Differences from drive_matrix2.py:
  - HY-WorldPlay accepts a per-sample action trace (pose string) — matrix2 only
    accepts a yaml task config. So the action.json -> pose-string conversion
    here is the main piece of new logic.
  - HY-WorldPlay writes to <output_path>/gen.mp4 (built into generate.py); we
    point --output_path at the per-sample dir and rename gen.mp4 -> video.mp4.
  - HY-WorldPlay's argparse uses underscored flags (--video_length, --image_path),
    not dashed. Subprocess arg list reflects that.

Mirror-test mode: when --mirror-test is set, we additionally emit one mp4 per
first-frame PNG, driven by a single-action pose ("w-N", "s-N", etc.). Compound
mirror actions (wl, wr, sl, sr) collapse to their primary axis here because
HY-WorldPlay pose strings are single-axis-per-segment.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import av

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4


def _free_master_port() -> str:
    """Pick an OS-assigned free TCP port for c10d's TCPStore.

    Hardcoding MASTER_PORT=29500 fails intermittently on Windows: a lingering
    prior subprocess holds it (WSAEADDRINUSE 10048) or it lands in a Hyper-V/WSL
    reserved range (WSAEACCES 10013). Binding to port 0 lets the OS hand back a
    guaranteed-free, non-reserved port for each generate.py invocation.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])
    finally:
        s.close()

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")

# HY-WorldPlay temporal-latent compression: 4 frames per latent. First pose
# command also gets 1 extra "seed" frame (frame 0). See
# HY-WorldPlay/hyvideo/generate.py:parse_pose_string_to_actions for the canonical
# definition; this driver mirrors that.
HY_FRAMES_PER_LATENT = 4

# Valid video_length values: (L-1) must be divisible by 4 AND the resulting
# latent count ((L-1)/4 + 1) must be divisible by 4.  Smaller = lower VRAM, larger
# = more video. Default 45 = 12 latents = 3 AR blocks (~1.9s @ 24fps).
DEFAULT_VIDEO_LENGTH = 45


# MIND action.json schema (README says values are 0/1 but the data uses 0/1/2
# with 0 = neutral / no key pressed):
#   ws:  1 = move forward (W),  2 = move backward (S)
#   ad:  1 = strafe left  (A),  2 = strafe right  (D)
#   ud:  1 = look up,           2 = look down
#   lr:  1 = yaw left,          2 = yaw right
#
# HY-WorldPlay's pose-string parser (parse_pose_string_to_actions) accepts these
# tokens (see HY-WorldPlay/hyvideo/generate.py:413):
#   w/s -> forward +/-1     a/d -> strafe left +/-1
#   up/down -> pitch +/-1   left/right -> yaw +/-1
#
# Mapping is direct; only the axis priority for simultaneous keys is a choice.
_AXIS_TO_TOKEN: dict[tuple[str, int], str] = {
    ("ws", 1): "w",   ("ws", 2): "s",
    ("ad", 1): "a",   ("ad", 2): "d",
    ("ud", 1): "up",  ("ud", 2): "down",
    ("lr", 1): "left",("lr", 2): "right",
}
# Priority for resolving multi-axis presses to a single-axis pose token.
# Movement first (ws/ad); rotation second (lr/ud). Rationale: in MIND
# benchmarks, "did the model translate correctly?" is heavier than "did it
# rotate correctly?", and HY-WorldPlay pose strings can't express both at once.
_AXIS_PRIORITY = ("ws", "ad", "lr", "ud")
# Token to emit when an entire latent has no keypress at all. Forward is the
# closest analogue to "neutral camera continues moving" — the MIND ground-truth
# videos that have idle frames typically still pan forward slightly.
_IDLE_TOKEN = "w"


def _frame_to_token(event: dict) -> str:
    """Reduce a single MIND-Data action event to one HY-WorldPlay pose token."""
    for axis in _AXIS_PRIORITY:
        v = event.get(axis, 0)
        if v in (1, 2):
            return _AXIS_TO_TOKEN[(axis, v)]
    return _IDLE_TOKEN


def _latent_token_for_window(events: list[dict], start: int, end: int) -> str:
    """Pick the dominant pose token for events[start:end] (one latent's worth).

    "Dominant" = most-common non-idle token in the slice; falls back to idle
    if every frame in the slice is neutral.
    """
    counts: dict[str, int] = {}
    for ev in events[start:end]:
        tok = _frame_to_token(ev)
        counts[tok] = counts.get(tok, 0) + 1
    non_idle = {k: v for k, v in counts.items() if k != _IDLE_TOKEN}
    if non_idle:
        return max(non_idle.items(), key=lambda kv: kv[1])[0]
    return _IDLE_TOKEN


def actions_to_pose_string(events: list[dict], mark_time: int, video_length: int) -> str:
    """Convert MIND-Data action events into an HY-WorldPlay pose string.

    HY-WorldPlay pose string semantics (mirrors hyvideo/generate.py):
      - The string is a comma-separated list of <token>-<num_latents> commands.
      - First command emits 1 + num_latents*4 frames (the +1 is the seed frame).
      - Subsequent commands emit num_latents*4 frames each.
      - Total frames = 1 + total_latents * 4, so for video_length L (where
        (L-1) % 4 == 0) we need total_latents = (L - 1) / 4.

    Args:
        events: list of frame events from action.json["data"].
        mark_time: start index in events (action.json["mark_time"]).
        video_length: target total frame count for the generated video.

    Returns:
        Pose string like "w-3,d-2,w-6" whose latents sum to (video_length-1)/4.
    """
    if (video_length - 1) % 4 != 0:
        raise ValueError(
            f"video_length={video_length} invalid; (L-1) must be divisible by 4."
        )
    total_latents = (video_length - 1) // 4

    # Carve `events[mark_time:]` into total_latents windows of HY_FRAMES_PER_LATENT
    # frames each (the seed frame is conceptually part of the first window).
    base = mark_time
    if base + total_latents * HY_FRAMES_PER_LATENT > len(events):
        # Action.json shorter than expected — clamp by repeating last event.
        # Better than crashing on short clips; output ends up biased toward
        # the final action.
        pad = base + total_latents * HY_FRAMES_PER_LATENT - len(events)
        events = events + [events[-1]] * pad

    latent_tokens: list[str] = []
    for i in range(total_latents):
        win_start = base + i * HY_FRAMES_PER_LATENT
        win_end = win_start + HY_FRAMES_PER_LATENT
        latent_tokens.append(_latent_token_for_window(events, win_start, win_end))

    # Run-length encode adjacent identical tokens.
    rle: list[tuple[str, int]] = []
    for tok in latent_tokens:
        if rle and rle[-1][0] == tok:
            rle[-1] = (tok, rle[-1][1] + 1)
        else:
            rle.append((tok, 1))

    return ",".join(f"{tok}-{n}" for tok, n in rle)


def _mirror_pose_string(action: str, video_length: int) -> str:
    """Single-action pose for mirror_test (e.g. 'w-11' for w at L=45).

    Compound mirror actions (wl/wr/sl/sr) collapse to their movement axis only;
    HY-WorldPlay can't express simultaneous translate+rotate in a single segment.
    The "u"/"down" mirror actions map to HY's pitch tokens.
    """
    total_latents = (video_length - 1) // 4
    # MIRROR_ACTIONS = ("w", "s", "a", "d", "u", "down", "wl", "wr", "sl", "sr")
    token = {
        "w": "w", "s": "s", "a": "a", "d": "d",
        "u": "up", "down": "down",
        "wl": "w", "wr": "w", "sl": "s", "sr": "s",
    }.get(action, "w")
    return f"{token}-{total_latents}"


def extract_first_frame(video_path: Path, frame_index: int, out_path: Path) -> None:
    """Extract a single frame at `frame_index` from video_path into out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_index:
                frame.to_image().save(out_path, "PNG")
                return
            if i > frame_index:
                break
    # Fall back to frame 0 if we couldn't reach the requested index.
    with av.open(str(video_path)) as container:
        for frame in container.decode(container.streams.video[0]):
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


def run_one(sample: dict, args, hy_worldplay_repo: Path, hy_worldplay_py: Path,
            model_path: str, action_ckpt: str) -> int:
    out = output_path(args.test_root, args.model_name, sample)
    if out.exists():
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} (exists)")
        return 0

    work_dir = args.work_dir or (args.test_root / ".frames")
    frame_png = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.png"

    # Decide which frame to seed the world model with, and build the pose string.
    if sample.get("is_mirror"):
        # Mirror samples ship a pre-extracted first-frame PNG (frame 0) and a
        # single-action label. We don't need action.json.
        frame_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["frame_png_src"], frame_png)
        pose_string = _mirror_pose_string(sample.get("mirror_action", "w"), args.video_length)
    else:
        with open(sample["action"], "r", encoding="utf-8") as f:
            action_data = json.load(f)
        mark_time = action_data.get("mark_time", 0)
        events = action_data["data"]
        extract_first_frame(sample["video"], mark_time, frame_png)
        pose_string = actions_to_pose_string(events, mark_time, args.video_length)

    out.parent.mkdir(parents=True, exist_ok=True)
    prompt = sample.get(
        "prompt",
        _PROMPT_VARIANTS.get(getattr(args, "prompt_variant", "default"), _DEFAULT_PROMPT),
    )
    # generate.py writes gen.mp4 to the directory passed via --output_path.
    cmd = [
        str(hy_worldplay_py), "-X", "utf8",
        str(hy_worldplay_repo / "hyvideo" / "generate.py"),
        "--model_path", model_path,
        "--action_ckpt", action_ckpt,
        "--model_type", "ar",
        "--prompt", prompt,
        "--image_path", str(frame_png),
        "--resolution", args.resolution,
        "--aspect_ratio", args.aspect_ratio,
        "--width", str(args.width),
        "--height", str(args.height),
        "--video_length", str(args.video_length),
        "--seed", str(args.seed),
        "--rewrite", "false",
        "--sr", "false",
        "--save_pre_sr_video", "true",
        "--pose", pose_string,
        "--output_path", str(out.parent),
        "--few_step", "true",
        "--num_inference_steps", str(args.num_inference_steps),
        "--use_vae_parallel", "false",
        "--use_sageattn", "false",
        "--use_fp8_gemm", "false",
        "--transformer_resident_ar_rollout", "false",
        "--with-ui", "false",  # MIND scoring doesn't want UI overlay
    ]

    run_cwd = str(hy_worldplay_repo)
    if getattr(args, "backend", "generate") == "flashdreams":
        # Drive the flashdreams hy_worldplay runner instead of upstream generate.py.
        # Reuses MIND's extracted first frame + pose string; the runner self-resolves
        # the distilled Wan2.2-5B checkpoint (no HunyuanVideo-1.5 base / byT5 / siglip /
        # generate.py distributed mess). It writes <out.parent>/hy-worldplay-wan-i2v-5b.mp4.
        uv = os.environ.get("UV_EXE", r"C:\Users\kschmid\.local\bin\uv.exe")
        fd_repo = str(getattr(args, "flashdreams_repo", Path(__file__).resolve().parent.parent.parent / "flashdream_public"))
        cmd = [
            uv, "run", "--project", fd_repo, "--package", "flashdreams-hy_worldplay",
            "flashdreams-run", "hy-worldplay-wan-i2v-5b",
            "--image-path", str(frame_png),
            "--pose", pose_string,
            "--prompt", prompt,
            "--num-chunk", str(getattr(args, "num_chunk", 4)),
            "--pixel-height", str(args.height),
            "--pixel-width", str(args.width),
            "--fps", str(args.fps),
            "--seed", str(args.seed),
            "--output-dir", str(out.parent),
        ]
        run_cwd = fd_repo

    print(f"\n=== {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} ===")
    print(f"  pose:  {pose_string}")
    print(f"  img:   {frame_png}")
    print(f"  out:   {out}")

    if args.dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return 0

    env = os.environ.copy()
    # Strip MIND venv pollution before spawning the HY-WorldPlay venv python.
    # Same _sre.MAGIC mismatch trap that bites drive_matrix3 / drive_matrix2 if
    # the host shell's VIRTUAL_ENV / PYTHONHOME leaks into the child.
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # HY-WorldPlay's run.bat / setup.bat assumes these Windows stability knobs;
    # the venv python is the same one those scripts target, so copy the same
    # set here for parity.
    env.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("USE_LIBUV", "0")
    env.setdefault("TORCH_TCPSTORE_USE_LIBUV", "0")
    env.setdefault("GLOO_SOCKET_IFNAME", "Wi-Fi")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("RANK", "0")
    env.setdefault("LOCAL_RANK", "0")
    # Force (not setdefault): a stale MASTER_ADDR=kubernetes.docker.internal from a
    # Docker/k8s shell env makes c10d connect to a bogus host -> 0xC0000005 crash.
    env["MASTER_ADDR"] = "127.0.0.1"
    # Fresh free port per subprocess: 29500 intermittently fails on Windows with
    # WSAEADDRINUSE (10048, lingering process) or WSAEACCES (10013, reserved range).
    env["MASTER_PORT"] = _free_master_port()
    env["PYTHONPATH"] = str(hy_worldplay_repo)

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=run_cwd, env=env)
    elapsed = time.perf_counter() - t0
    print(f"  rc={rc}  elapsed={elapsed:.1f}s")

    # generate.py writes gen.mp4 (and optionally gen_sr.mp4) into --output_path.
    # MIND expects video.mp4 at the sample dir.
    gen = out.parent / ("hy-worldplay-wan-i2v-5b.mp4"
                        if getattr(args, "backend", "generate") == "flashdreams"
                        else "gen.mp4")
    if rc == 0 and gen.exists():
        gen.replace(out)
        # Drop the SR variant if it exists; MIND scores the base video.
        sr = out.parent / "gen_sr.mp4"
        if sr.exists():
            sr.unlink()
        # Fail if the staged mp4 is empty/truncated: generation can exit 0 but write
        # a 0-byte or header-only file (no frames). A real 480p clip is many KB, so
        # anything under 1 KB is treated as a failed/empty render and not logged.
        size = out.stat().st_size if out.exists() else 0
        if size < 1024:
            print(f"  WARN: staged mp4 is empty/truncated ({size} bytes): {out}")
            rc = 4
        else:
            log_mp4(args.model_name, sample["perspective"], sample["test_type"], sample["gt_name"], out)
    elif rc == 0 and not out.exists():
        print(f"  WARN: rc=0 but no gen.mp4 at {gen}; not staged.")
        rc = 3
    return rc


_DEFAULT_PROMPT = (
    "A first-person view of an indoor 3D environment, exploring the scene "
    "while the camera tracks forward through corridors and rooms."
)

# Selectable default-prompt styles (used when a sample has no explicit "prompt").
# Add new weather/style variants here; expose them via --prompt-variant.
_PROMPT_VARIANTS = {
    "default": _DEFAULT_PROMPT,
    "cartoony": (
        "A first-person view exploring a vibrant, colorful cartoon-style 3D "
        "environment, the camera tracking forward through whimsical stylized "
        "corridors and rooms as bright cartoon rain falls, with bold outlines, "
        "saturated candy colors, and a playful animated look."
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root", type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="hy-worldplay", help="Subfolder name under test-root")
    parser.add_argument("--backend", choices=["generate", "flashdreams"], default="generate",
                        help="'generate' = upstream hyvideo/generate.py (HunyuanVideo-1.5 base); "
                             "'flashdreams' = flashdreams-run hy-worldplay-wan-i2v-5b (distilled Wan2.2-5B, cleaner)")
    parser.add_argument("--flashdreams-repo", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "flashdream_public",
                        help="flashdreams repo root (for --backend flashdreams)")
    parser.add_argument("--num-chunk", type=int, default=4,
                        help="flashdreams AR chunks (4 latents each, ~16 frames/chunk)")
    parser.add_argument("--hy-worldplay-repo", type=Path, required=True,
                        help=r"Path to HY-WorldPlay repo (e.g. C:\workspace\world\HY-WorldPlay)")
    parser.add_argument("--hy-worldplay-py", type=Path, required=True,
                        help=r"Path to HY-WorldPlay's venv python.exe")
    parser.add_argument("--model-path", required=True,
                        help="HunyuanVideo-1.5 root (vae/, scheduler/, transformer/, text_encoder/, vision_encoder/)")
    parser.add_argument("--action-ckpt", required=True,
                        help="ar_distilled_action_model/diffusion_pytorch_model.safetensors")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Temp dir for extracted seed frames (default: <test-root>/.frames)")
    parser.add_argument("--only", nargs="+",
                        help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type", choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit", type=int, help="Only run first N matched samples")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N matched samples (applied AFTER filters, BEFORE --limit).")
    parser.add_argument("--video-length", type=int, default=DEFAULT_VIDEO_LENGTH,
                        help=f"Target video frame count (default {DEFAULT_VIDEO_LENGTH}). "
                        "Must satisfy ((L-1)//4 + 1) %% 4 == 0. Valid: 13, 29, 45, 61, 77, 93, 109, 125.")
    parser.add_argument("--resolution", default="480p", help="HY-WorldPlay resolution bucket (only 480p supported)")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--prompt-variant", choices=list(_PROMPT_VARIANTS), default="default",
                        help="Default prompt style when a sample has no explicit prompt "
                             "('cartoony' = colorful cartoon look with rain).")
    parser.add_argument("--width", type=int, default=624,
                        help="Frame width. 624x352 keeps a 5090 (32 GB) clear of the VRAM cliff at the default 4 AR blocks.")
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--num-inference-steps", type=int, default=4,
                        help="Distilled few-step inference; 4 is HY-WorldPlay's recommended floor.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fps", type=int, default=24,
                        help="Accepted for bat-script consistency; generate.py uses its own default.")
    parser.add_argument("--mirror-test", action="store_true",
                        help="Also generate mirror_test outputs (additive). One mp4 per first-frame PNG.")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only generate mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if (args.video_length - 1) % 4 != 0:
        print(f"FATAL: video_length={args.video_length} invalid; (L-1) must be divisible by 4.",
              file=sys.stderr)
        return 2
    if ((args.video_length - 1) // 4 + 1) % 4 != 0:
        print(f"FATAL: video_length={args.video_length} produces "
              f"{(args.video_length - 1) // 4 + 1} latents, must be divisible by 4. "
              "Valid: 13, 29, 45, 61, 77, 93, 109, 125.", file=sys.stderr)
        return 2

    hy_worldplay_repo = args.hy_worldplay_repo
    hy_worldplay_py = args.hy_worldplay_py
    if not (hy_worldplay_repo / "hyvideo" / "generate.py").exists():
        print(f"FATAL: generate.py not found at {hy_worldplay_repo}\\hyvideo\\generate.py",
              file=sys.stderr)
        return 2
    if not hy_worldplay_py.exists():
        print(f"FATAL: HY-WorldPlay venv python not found at {hy_worldplay_py}", file=sys.stderr)
        return 2

    work_dir = args.work_dir or (args.test_root / ".frames")
    work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = work_dir

    samples = [] if args.mirror_only else gather_samples(args.gt_root)
    if args.mirror_test:
        mirror_samples = gather_mirror_samples(args.gt_root, args.mirror_action)
        for s in mirror_samples:
            s["is_mirror"] = True
            s["mirror_action"] = args.mirror_action
        samples += mirror_samples
    if args.perspective:
        samples = [s for s in samples if s["perspective"] == args.perspective]
    if args.test_type:
        samples = [s for s in samples if s["test_type"] == args.test_type]
    if args.only:
        samples = [s for s in samples if any(sub.lower() in s["gt_name"].lower() for sub in args.only)]
    if args.start_index:
        if args.start_index >= len(samples):
            print(f"--start-index {args.start_index} is past the end of {len(samples)} matched "
                  f"sample(s); nothing to do.")
            return 0
        samples = samples[args.start_index:]
    if args.limit:
        # Cap per (perspective, test_type) instead of slicing the flat list:
        # mirror_test samples are appended last, so a flat samples[:limit] truncated
        # them off entirely (mirror_test was never generated). Now each test_type
        # (mem/action/mirror) gets up to --limit samples per perspective.
        _seen: dict = {}
        _capped = []
        for _s in samples:
            _k = (_s["perspective"], _s["test_type"])
            _n = _seen.get(_k, 0)
            if _n < args.limit:
                _capped.append(_s)
                _seen[_k] = _n + 1
        samples = _capped

    if not samples:
        print("No samples matched.")
        return 1

    print(f"Will process {len(samples)} sample(s):")
    for s in samples:
        kind = "[mirror]" if s.get("is_mirror") else ""
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']} {kind}")
    print()

    failures: list[str] = []
    for s in samples:
        rc = run_one(s, args, hy_worldplay_repo, hy_worldplay_py,
                     args.model_path, args.action_ckpt)
        if rc != 0:
            name = f"{s['perspective']}/{s['test_type']}/{s['gt_name']}"
            failures.append(name)
            # Skip this sample and continue rather than aborting the whole run: a
            # flaky per-video crash (e.g. 0xC0000005 access violation during the
            # per-video HunyuanVideo checkpoint load under memory pressure) must not
            # kill every remaining sample. The output mp4 isn't written, so a
            # restartable re-run retries only the skipped items.
            print(f"\n[skip-fail] no valid mp4 for {name} (rc={rc}); continuing to next sample.")
            continue

    print()
    if failures:
        # Report but DON'T fail the run: skipped items have no mp4 written, so a
        # restartable re-run retries exactly them. Returning 0 lets the caller
        # (drive_hy_worldplay.bat) proceed to the next perspective instead of
        # exiting on the first perspective that had any flaky crash.
        print(f"SKIPPED/FAILED ({len(failures)}) -- will be retried on re-run:")
        for name in failures:
            print(f"  {name}")
        print(f"Done. {len(samples) - len(failures)} produced, {len(failures)} skipped (retry via re-run).")
        return 0
    print(f"Done. {len(samples)} sample(s) produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
