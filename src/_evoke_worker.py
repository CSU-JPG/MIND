"""Persistent Evoke worker (file-based IPC), mirrors _helios_i2v_worker.py.

Loaded ONCE inside Evoke's venv (C:\\workspace\\world\\Evoke\\.venv). The MIND-side driver
(``src/drive_evoke.py``) writes a JSON-Lines manifest, then spawns this process pointed at it.
The worker loads EvokePipeline once and processes requests serially.

Manifest line format (one per line)::

    {"id": <int>, "tag": <str>, "image": <path>, "prompt": <str>,
     "pose_npz": <path>, "pose_source_resolution": [h, w], "target_path": <path>,
     "height": 256, "width": 448, "num_frames": 99, "seed": 42, "fps": 24}

Result line format (one per line, appended)::

    {"id": <int>, "ok": true,  "elapsed_s": 51.5, "target_path": <path>}
    {"id": <int>, "ok": false, "error": "<type>: <message>"}

Status file behavior and the mem-sampler thread match _helios_i2v_worker.py exactly.

VRAM note (found 2026-08-17 while wiring up Evoke's own run_examples_python.py): a plain
`.to("cuda")` load alone fills ~97.6% of a 32GB card (31842/32607 MiB) before any denoising,
leaving almost no headroom -- causing 100%-util/low-power VRAM-paging thrashing. This worker
uses `enable_model_cpu_offload()` instead, same fix applied there.

Correct inference recipe note (found 2026-08-17, cross-referenced against
scripts/inference/infer_post_distill.sh + infer_single.py in the Evoke repo): the post-distill
checkpoint needs the STAGE2 PYRAMID path, not the generic CFG path -- num_inference_steps alone
(with any value) runs the wrong branch. Required together:
  - VAE loaded separately in float32 (Wan-style VAEs are numerically unstable in fp16)
  - scheduler rebuilt with stages=3 (checkpoint's shipped scheduler_config.json has stages=1)
  - is_enable_stage2=True + stage2_num_inference_steps_list=[1,1,1] (3-stage pyramid, 1 step/stage
    = STAGE2_STEPS="1 1 1" in infer_post_distill.sh, NOT the argparse default [3,3,3] shown in
    infer_single.py, which is a dormant default only used when a caller overrides it)
  - guidance_scale=1.0 (CFG off; distillation removed the need for it)
Missing any of these -- including two earlier mistakes in this project (num_inference_steps=3
alone, then num_inference_steps=50 without the stage2 scheduler/kwargs) -- produces near-pure
noise output despite the pipeline running real, correctly-timed GPU compute.

Device-mismatch retry note: pipeline_evoke.py:1938-1945 reads self.vae.device for
latents_mean/latents_std BEFORE the VAE's cpu-offload hook has ever fired, while
self.vae.encode() moments later actually runs on cuda via its hook -- "Expected all tensors to
be on the same device" on whichever sample runs first in a process; later samples don't hit it
once the hook has fired once. Manually stripping the VAE's offload hook (tried first) "fixed"
this but broke a DIFFERENT op (`aten::slow_conv3d_forward` has no CUDA kernel in this build) --
so instead this worker just retries once on that specific error.
"""

import argparse
import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

faulthandler.enable()

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("GEO_HIST_MAX_FRAMES", "720")

_EVOKE_REPO = os.environ.get("EVOKE_REPO", str(Path(__file__).resolve().parent.parent.parent / "Evoke"))
if _EVOKE_REPO not in sys.path:
    sys.path.insert(0, _EVOKE_REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.models.modeling_utils import ModelMixin  # noqa: E402
from evoke.diffusers_version.scheduling_evoke_diffusers import EvokeScheduler  # noqa: E402
from evoke.modules.transformer_evoke import EvokeTransformer3DModel  # noqa: E402
from evoke.pipelines.pipeline_evoke import EvokePipeline  # noqa: E402
from evoke.utils.ev_validation import load_pose_for_v2v  # noqa: E402

# Same from_pretrained patch as Evoke's own run_examples_python.py.
_orig_from_pretrained = ModelMixin.from_pretrained.__func__
@classmethod
def _patched_from_pretrained(cls, *args, **kwargs):
    kwargs.pop("max_memory", None)
    kwargs.pop("_fast_init", None)
    return _orig_from_pretrained(cls, *args, **kwargs)
EvokeTransformer3DModel.from_pretrained = _patched_from_pretrained

STAGE2_NUM_STAGES = 3
STAGE2_STAGE_RANGE = [0.0, 1 / 3, 2 / 3, 1.0]
STAGE2_STEPS = [1, 1, 1]
GUIDANCE_SCALE = 1.0  # CFG off; distillation removed the need for it


def _write_status(path: Path, snapshot: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot), encoding="utf-8")
    last_err = None
    for _ in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.05)
    raise last_err


def _device_used_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 / 1024


def _start_mem_sampler(state: dict, interval: float = 10.0) -> threading.Thread:
    def _run() -> None:
        peak = 0.0
        while not state.get("stop"):
            try:
                used = _device_used_mib()
                resv = (torch.cuda.memory_reserved() / 1024 / 1024) if torch.cuda.is_available() else 0.0
                peak = max(peak, used)
                print(f"[mem] used={used:.0f}MiB reserved={resv:.0f}MiB peak={peak:.0f}MiB "
                      f"cur={state.get('tag', '-')}", flush=True)
            except Exception:
                pass
            time.sleep(interval)
    t = threading.Thread(target=_run, name="mem-sampler", daemon=True)
    t.start()
    return t


def _save_video(video_np, output_path, fps=24):
    import cv2
    h, w = video_np.shape[1], video_np.shape[2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(w), int(h)))
    for frame in video_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--model-path", required=True, help="HF repo or local snapshot dir")
    args = parser.parse_args()

    print(f"[env] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[env] gpu={p.name} total_vram={p.total_memory / 1024 / 1024:.0f}MiB", flush=True)

    mem_state = {"stop": False, "tag": "load"}
    _start_mem_sampler(mem_state)

    requests = []
    with open(args.manifest, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                requests.append(json.loads(line))

    total = len(requests)
    print(f"[worker] manifest: {total} samples", flush=True)
    _write_status(args.status_path, {"phase": "loading", "done": 0, "total": total})

    print(f"[worker] loading EvokePipeline from {args.model_path} ...", flush=True)
    # See module docstring for why vae/scheduler must be overridden explicitly.
    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = EvokeScheduler(
        num_train_timesteps=1000, shift=1.0, stages=STAGE2_NUM_STAGES, stage_range=STAGE2_STAGE_RANGE,
        gamma=1 / 3, scheduler_type="unipc", use_dynamic_shifting=False, time_shift_type="exponential",
    )
    # disable_mmap=True: Windows' auto-managed pagefile is too small for this transformer's mmap
    # virtual address space -- OSError: paging file is too small (os error 1455) otherwise. Same
    # fix as _helios_i2v_worker.py and run_examples_python.py.
    transformer = EvokeTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.float16, disable_mmap=True,
    )
    pipe = EvokePipeline.from_pretrained(
        args.model_path, transformer=transformer, vae=vae, scheduler=scheduler, torch_dtype=torch.float16,
    )
    # See module docstring: plain .to("cuda") fills ~97.6% VRAM before generation even starts.
    pipe.enable_model_cpu_offload()

    _write_status(args.status_path, {"phase": "ready", "done": 0, "total": total})
    print(f"[worker] model ready; processing {total} samples", flush=True)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "a", encoding="utf-8") as results_fh:
        for i, req in enumerate(requests):
            req_id = req.get("id", i)
            tag = req.get("tag", f"sample-{req_id}")
            mem_state["tag"] = f"{i + 1}/{total} {tag}"
            target = Path(req["target_path"])
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                print(f"[worker] {i + 1}/{total} skip {tag} (exists)", flush=True)
                results_fh.write(json.dumps({"id": req_id, "ok": True, "skipped": True,
                                              "target_path": str(target)}) + "\n")
                results_fh.flush()
                continue

            _write_status(args.status_path, {"phase": "running", "done": i, "total": total, "current": tag})
            print(f"[worker] {i + 1}/{total} {tag} -> {target}", flush=True)

            try:
                t0 = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                height, width = req.get("height", 256), req.get("width", 448)
                num_frames = req.get("num_frames", 99)

                kwargs = dict(
                    prompt=req["prompt"],
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    is_enable_stage2=True,
                    stage2_num_inference_steps_list=STAGE2_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    # Lower than the pipeline defaults (0.111/0.135) -- less noise mixed into the
                    # image latents means the model stays more anchored to the actual seed-image
                    # pixels instead of drifting toward whatever the (often-generic, MIND has no
                    # captions) text prompt describes. Per-sample, overridable via the manifest.
                    image_noise_sigma_min=req.get("image_noise_sigma_min", 0.02),
                    image_noise_sigma_max=req.get("image_noise_sigma_max", 0.05),
                    generator=torch.Generator(device="cuda").manual_seed(req.get("seed", 42)),
                )
                if req.get("image"):
                    # .convert("RGB") -- some source PNGs carry an alpha channel, which crashes
                    # the VAE's first conv (expects 3 channels, got 4).
                    kwargs["image"] = Image.open(req["image"]).convert("RGB")

                pose_npz = req.get("pose_npz")
                if pose_npz:
                    src_h, src_w = req.get("pose_source_resolution", (480, 832))
                    lingbot_Ks, lingbot_c2ws = load_pose_for_v2v(
                        pose_npz, target_height=height, target_width=width,
                        source_resolution=(src_h, src_w), pose_type="vipe",
                        num_target_frames=num_frames, target_fps=24,
                        source_fps=req.get("pose_fps", 24),
                    )
                    # enable_model_cpu_offload() moves submodules to cuda only during their own
                    # forward and evicts them back to cpu after -- hardcoding "cuda" here raced
                    # with that on whichever sample ran first (see run_examples_python.py for the
                    # same fix). pipe._execution_device is accelerate's own resolved device.
                    exec_device = pipe._execution_device
                    kwargs["lingbot_Ks"] = lingbot_Ks.to(exec_device)
                    kwargs["lingbot_c2ws"] = lingbot_c2ws.to(exec_device)

                try:
                    output = pipe(**kwargs)
                except RuntimeError as e:
                    # See module docstring: retry once for the VAE-offload-hook first-call race.
                    if "Expected all tensors to be on the same device" not in str(e):
                        raise
                    print(f"[worker] {i + 1}/{total} device-mismatch on first VAE use, retrying once", flush=True)
                    output = pipe(**kwargs)
                elapsed = time.perf_counter() - t0

                video_np = output.frames[0]
                if video_np.dtype != np.uint8:
                    video_np = (np.clip(video_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                _save_video(video_np, target, fps=req.get("fps", 24))

                peak_mib = (torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
                results_fh.write(json.dumps({
                    "id": req_id, "ok": True, "elapsed_s": round(elapsed, 2),
                    "actual_frames": int(video_np.shape[0]), "peak_vram_mib": round(peak_mib),
                    "target_path": str(target),
                }) + "\n")
                results_fh.flush()
                print(f"[worker] {i + 1}/{total} done in {elapsed:.1f}s "
                      f"(peak {peak_mib:.0f}MiB, {video_np.shape[0]} frames) -> {target}", flush=True)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                used = _device_used_mib()
                results_fh.write(json.dumps({"id": req_id, "ok": False, "error": err,
                                              "vram_used_mib": round(used), "traceback": tb}) + "\n")
                results_fh.flush()
                print(f"[worker] {i + 1}/{total} FAIL {tag}: {err}  (vram {used:.0f}MiB)", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    mem_state["stop"] = True
    _write_status(args.status_path, {"phase": "done", "done": total, "total": total})
    print(f"[worker] done: {total} samples processed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
