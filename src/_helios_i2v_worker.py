"""Persistent Helios i2v worker (file-based IPC).

Loaded ONCE inside Helios's venv. The MIND-side driver
(``src/drive_helios_i2v.py``) writes a JSON-Lines manifest of work items,
then spawns this process pointed at the manifest. The worker reads all
requests, loads ``HeliosPipeline`` once (~30-60s with cold disk cache, much
faster warm), and processes them serially.

Manifest line format (one per line)::

    {"id": <int>, "tag": <str>, "image": <path>, "prompt": <str>,
     "target_path": <path>, "height": 384, "width": 640, "num_frames": 99,
     "num_latent_frames_per_chunk": 3, "image_noise_sigma_min": 0.111,
     "image_noise_sigma_max": 0.135, "seed": 42, "fps": 24}

Result line format (one per line, appended)::

    {"id": <int>, "ok": true,  "elapsed_s": 51.5, "target_path": <path>}
    {"id": <int>, "ok": false, "error": "<type>: <message>"}

Status file ``--status-path`` is rewritten after every sample with a small
JSON snapshot ``{"phase": "loading"|"ready"|"running"|"done", "done": N,
"total": M, "current": "<tag>"}`` so the driver can show progress without
parsing the results file in real time. The replace is wrapped in a
Windows-safe retry loop (AV / search indexer can briefly hold ``.tmp``).

Why a worker process at all (vs in-process import): Helios's import side
effects monkey-patch globals (norms, RoPE, scheduler), so the driver must
run in a different venv (Python 3.10 for MIND, vs 3.11 for Helios). The
subprocess is the cleanest cross-venv handoff.
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

# Dump a Python traceback if the process receives a fatal native signal
# (segfault / abort / access violation). Without this a CUDA illegal-address
# or OOM-driven abort kills the worker with NO traceback at all -- exactly what
# masked the first-video crash (worker exit=1, empty results.jsonl, no error).
faulthandler.enable()

# Windows env tweaks. Mirror Helios's own infer_helios.py expectations so
# behavior matches what the driver previously got via per-sample spawn.
os.environ.setdefault("GLOO_SOCKET_IFNAME", "Wi-Fi")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("TORCH_TCPSTORE_USE_LIBUV", "0")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Insert the Helios repo on sys.path so `helios.diffusers_version.*` resolves.
# Python's sys.path[0] is the dir of THIS script (MIND/src/), not cwd; relying
# on cwd alone fails with ModuleNotFoundError. The driver passes the repo path
# via env (HELIOS_REPO) to keep the worker portable -- fallback to the
# canonical path if not set.
_HELIOS_REPO = os.environ.get("HELIOS_REPO", str(Path(__file__).resolve().parent.parent.parent / "Helios"))
if _HELIOS_REPO not in sys.path:
    sys.path.insert(0, _HELIOS_REPO)

import torch  # noqa: E402
from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.utils import export_to_video, load_image  # noqa: E402
from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline  # noqa: E402
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler  # noqa: E402
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel  # noqa: E402
from helios.modules.helios_kernels import (  # noqa: E402
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)


def _write_status(path: Path, snapshot: dict) -> None:
    """Atomic-replace status JSON with Windows-safe retry on PermissionError.

    AV / search indexer can momentarily hold the ``.tmp`` file open between
    write and rename. Retry ~1 sec; if still failing after that, surface the
    error (something more fundamental than transient AV is wrong).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot), encoding="utf-8")
    last_err: Exception | None = None
    for _ in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.05)
    raise last_err  # type: ignore[misc]


def _dtype_from_str(name: str) -> "torch.dtype":
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported weight_dtype: {name}")


def _device_used_mib() -> float:
    """Device-wide VRAM in use (MiB) -- matches what nvidia-smi reports."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 / 1024


def _start_mem_sampler(state: dict, interval: float = 10.0) -> threading.Thread:
    """Background thread: log VRAM every ``interval`` s, tagged with the
    in-flight sample. Ties the memory curve to a timestamp + sample in the SAME
    log, so a silent OOM crash shows the climb right up to the moment it dies.
    """
    def _run() -> None:
        peak = 0.0
        while not state.get("stop"):
            try:
                used = _device_used_mib()
                resv = (torch.cuda.memory_reserved() / 1024 / 1024) if torch.cuda.is_available() else 0.0
                peak = max(peak, used)
                print(f"[mem] used={used:.0f}MiB reserved={resv:.0f}MiB peak={peak:.0f}MiB "
                      f"cur={state.get('tag', '-')}", flush=True)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval)
    t = threading.Thread(target=_run, name="mem-sampler", daemon=True)
    t.start()
    return t


def main() -> int:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON-Lines file with one request per line.")
    parser.add_argument("--results-path", type=Path, required=True,
                        help="Worker appends one JSON line per processed item.")
    parser.add_argument("--status-path", type=Path, required=True,
                        help="Worker rewrites a small status JSON snapshot here.")
    parser.add_argument("--model-path", required=True,
                        help="HF repo (e.g. BestWishYsh/Helios-Base) or local snapshot dir.")
    parser.add_argument("--num-inference-steps", type=int, default=30,
                        help="30 for Base/Mid (full denoise), 4 for Distilled (pyramid).")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="5.0 for Base/Mid; 1.0 for Distilled (baked-in via DMD).")
    parser.add_argument("--is-enable-stage2", action="store_true",
                        help="Required for Distilled (pyramid scheduler); fatal on Base/Mid.")
    parser.add_argument("--low-vram", action="store_true",
                        help="Enable group offload (leaf_level + low_cpu_mem_usage=True).")
    parser.add_argument("--num-blocks-per-group", type=int, default=4)
    parser.add_argument("--weight-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype = _dtype_from_str(args.weight_dtype)

    # Startup diagnostics: torch / device / alloc config land in the log so a
    # post-mortem doesn't have to re-derive the environment.
    print(f"[env] torch={torch.__version__} cuda_avail={torch.cuda.is_available()} "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')} "
          f"launch_blocking={os.environ.get('CUDA_LAUNCH_BLOCKING', '<unset>')}", flush=True)
    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        print(f"[env] gpu={_p.name} total_vram={_p.total_memory / 1024 / 1024:.0f}MiB", flush=True)

    # Background VRAM sampler -- runs through load + every sample so the log
    # shows the memory climb tagged to whichever sample is in flight.
    mem_state: dict = {"stop": False, "tag": "load"}
    _start_mem_sampler(mem_state)

    requests: list[dict] = []
    with open(args.manifest, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                requests.append(json.loads(line))

    total = len(requests)
    print(f"[worker] manifest: {total} samples", flush=True)
    _write_status(args.status_path, {"phase": "loading", "done": 0, "total": total})

    # ----- ONE-TIME MODEL LOAD ------------------------------------------------
    print(f"[worker] loading transformer from {args.model_path} ...", flush=True)
    # disable_mmap=True bypasses safetensors mmap on Windows where the default
    # pagefile (~2-4 GB auto-managed) is too small to reserve the 54 GB virtual
    # address space the Helios transformer shards need. Without it:
    #   OSError: paging file is too small for this operation (os error 1455)
    # Trade-off: shards load into host RAM instead of mmap; needs 16+ GB free
    # RAM during load. Released after the model lands on GPU.
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=dtype,
        disable_mmap=True,
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()

    # Try local attention backends in preference order, falling through to SDPA.
    # Helios's helios_kernels uses flash_attn directly when present; if the
    # mjun0812 FA2 wheel is installed it triggers a torch op-registration bug
    # (empty op name -> RuntimeError). Keep flash_attn UNinstalled in this venv;
    # the loop here is for completeness if a future fixed wheel ships.
    for backend in ("flash", "sage", "flash_varlen", "sage_varlen"):
        try:
            transformer.set_attention_backend(backend)
            print(f"[worker] attention backend set: {backend}", flush=True)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] backend '{backend}' unavailable: {type(exc).__name__}", flush=True)
    else:
        print("[worker] no local backend; using diffusers default (SDPA)", flush=True)

    print("[worker] loading vae + scheduler + pipe ...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=torch.float32,
    )
    scheduler = HeliosScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(
        args.model_path, transformer=transformer, vae=vae, scheduler=scheduler,
        torch_dtype=dtype,
    )

    if args.low_vram:
        # low_cpu_mem_usage=True bypasses pin_memory() (cudaErrorAlreadyMapped
        # on consumer Blackwell BAR1). Matches Helios's run_helios.bat config.
        print(f"[worker] enabling group offload (leaf_level, blocks={args.num_blocks_per_group}) ...",
              flush=True)
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            num_blocks_per_group=args.num_blocks_per_group,
            use_stream=True,
            record_stream=True,
            low_cpu_mem_usage=True,
        )

    _write_status(args.status_path, {"phase": "ready", "done": 0, "total": total})
    print(f"[worker] model ready; processing {total} samples", flush=True)

    # ----- INFERENCE LOOP -----------------------------------------------------
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    # Open append so a re-run can continue past a partial results file.
    with open(args.results_path, "a", encoding="utf-8") as results_fh:
        for i, req in enumerate(requests):
            req_id = req.get("id", i)
            tag = req.get("tag", f"sample-{req_id}")
            mem_state["tag"] = f"{i + 1}/{total} {tag}"
            target = Path(req["target_path"])
            target.parent.mkdir(parents=True, exist_ok=True)

            # Skip if the target already exists (resume-safe).
            if target.exists():
                print(f"[worker] {i + 1}/{total} skip {tag} (exists)", flush=True)
                results_fh.write(json.dumps({
                    "id": req_id, "ok": True, "skipped": True,
                    "target_path": str(target),
                }) + "\n")
                results_fh.flush()
                continue

            _write_status(args.status_path, {
                "phase": "running", "done": i, "total": total, "current": tag,
            })
            print(f"[worker] {i + 1}/{total} {tag} -> {target}", flush=True)

            try:
                t0 = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                image = load_image(req["image"]).resize(
                    (req.get("width", 640), req.get("height", 384))
                )
                output = pipe(
                    prompt=req["prompt"],
                    height=req.get("height", 384),
                    width=req.get("width", 640),
                    num_frames=req.get("num_frames", 99),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(req.get("seed", 42)),
                    history_sizes=[16, 2, 1],
                    num_latent_frames_per_chunk=req.get("num_latent_frames_per_chunk", 3),
                    keep_first_frame=True,
                    is_enable_stage2=args.is_enable_stage2,
                    image=image,
                    image_noise_sigma_min=req.get("image_noise_sigma_min", 0.111),
                    image_noise_sigma_max=req.get("image_noise_sigma_max", 0.135),
                ).frames[0]
                elapsed = time.perf_counter() - t0

                export_to_video(output, str(target), fps=req.get("fps", 24))

                peak_mib = (torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
                results_fh.write(json.dumps({
                    "id": req_id, "ok": True,
                    "elapsed_s": round(elapsed, 2),
                    "peak_vram_mib": round(peak_mib),
                    "target_path": str(target),
                }) + "\n")
                results_fh.flush()
                print(f"[worker] {i + 1}/{total} done in {elapsed:.1f}s "
                      f"(peak {peak_mib:.0f}MiB) -> {target}", flush=True)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                used = _device_used_mib()
                results_fh.write(json.dumps({
                    "id": req_id, "ok": False,
                    "error": err,
                    "vram_used_mib": round(used),
                    "traceback": tb,
                }) + "\n")
                results_fh.flush()
                print(f"[worker] {i + 1}/{total} FAIL {tag}: {err}  (vram {used:.0f}MiB)", flush=True)
                # A CUDA OOM leaves the context wedged; release the cached
                # blocks so the next sample starts from a clean allocator
                # instead of cascading OOMs (or a hard abort) down the batch.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Keep going; one bad sample shouldn't kill the whole batch.

    mem_state["stop"] = True
    _write_status(args.status_path, {"phase": "done", "done": total, "total": total})
    print(f"[worker] done: {total} samples processed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
