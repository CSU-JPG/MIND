"""Persistent FastVideo Matrix-Game-3 Distilled worker (file-based IPC).

Loaded ONCE inside FastVideo's venv. The MIND-side driver
(``src/drive_matrix3_distilled.py``) writes a JSON-Lines manifest of work
items, then spawns this process pointed at the manifest. The worker reads
all requests, loads ``VideoGenerator`` once, processes them, and writes one
JSON result per line to ``--results-path``. Stdout/stderr are NOT used for
IPC because importing ``fastvideo`` monkey-patches ``sys.stdout`` to a
per-rank prefix wrapper that EINVALs on a piped stdout on Windows
(see fastvideo/utils.py:write_with_prefix).

Manifest line format (one per line)::

    {"id": <int>, "image": <path>, "prompt": <str>, "output_dir": <path>,
     "target_path": <path>, "height": 720, "width": 1280, "num_frames": 57,
     "num_inference_steps": 3, "guidance_scale": 1.0, "seed": 42}

Result line format::

    {"id": <int>, "ok": true,  "elapsed_s": 51.5, "target_path": <path>}
    {"id": <int>, "ok": false, "error": "<type>: <message>"}

Status file ``--status-path`` is rewritten after every sample with a small
JSON snapshot ``{"phase": "loading"|"ready"|"running"|"done", "done": N,
"total": M, "current": "<gt_name>"}`` so the driver can show progress
without parsing the results file in real time.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Windows env tweaks — duplicated from run_matrixgame3.bat because this script
# may be spawned with a stripped env (the MIND driver pops VIRTUAL_ENV/PYTHONHOME
# before exec to avoid SRE module mismatch on cross-venv spawn).
os.environ.setdefault("GLOO_SOCKET_IFNAME", "Wi-Fi")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("TORCH_TCPSTORE_USE_LIBUV", "0")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from fastvideo import VideoGenerator

MODEL_PATH = "FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers"


def _write_status(path: Path, snapshot: dict) -> None:
    # Direct write — atomic replace via tmp+rename hits WinError 5 on Windows
    # when the driver reads status.json at the same instant. Status snapshots
    # are advisory progress info; a partial line on a torn read is fine because
    # the driver json.loads in a try/except and just re-polls. PermissionError
    # also survivable: skip this tick.
    try:
        path.write_text(json.dumps(snapshot), encoding="utf-8")
    except (PermissionError, OSError):
        pass


def _find_produced_mp4(output_dir: Path, before_set: set[Path]) -> Path | None:
    after = {p for p in output_dir.glob("*.mp4")}
    new = sorted(after - before_set, key=lambda p: p.stat().st_mtime, reverse=True)
    if new:
        return new[0]
    if after:
        return max(after, key=lambda p: p.stat().st_mtime)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON-Lines file with one request per line.")
    parser.add_argument("--results-path", type=Path, required=True,
                        help="Worker appends one JSON line per processed item.")
    parser.add_argument("--status-path", type=Path, required=True,
                        help="Worker rewrites a small status JSON snapshot here.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--override-transformer-cls-name", default="MatrixGame3WanModel",
                        help="HF snapshot ships _class_name=WanModel in transformer/config.json which "
                             "isn't in FastVideo's registry; override to the registered name.")
    args = parser.parse_args()

    requests: list[dict] = []
    with open(args.manifest, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                requests.append(json.loads(line))

    total = len(requests)
    _write_status(args.status_path, {"phase": "loading", "done": 0, "total": total})

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        override_transformer_cls_name=args.override_transformer_cls_name,
    )
    _write_status(args.status_path, {"phase": "ready", "done": 0, "total": total})

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    # Once FastVideo's MultiprocExecutor dies (OOM, CUDA error, etc.) every
    # subsequent generate_video call raises the same "Forward execution thread
    # failed." instantly. Cascading hundreds of identical failures buries the
    # real cause. Bail out after the first executor death: stop the loop,
    # write a single summary failure to results, and exit non-zero so the
    # driver surfaces the actual error in the worker log.
    EXECUTOR_DEAD_MARKERS = (
        "Forward execution thread failed",
        "Worker.*shutdown",
        "MultiprocExecutor",
    )
    import re as _re
    executor_dead_re = _re.compile("|".join(EXECUTOR_DEAD_MARKERS))

    with open(args.results_path, "a", encoding="utf-8") as results_fh:
        for i, req in enumerate(requests):
            req_id = req.get("id", i)
            output_dir = Path(req["output_dir"])
            target_path = Path(req["target_path"])
            output_dir.mkdir(parents=True, exist_ok=True)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            before = {p for p in output_dir.glob("*.mp4")}

            _write_status(args.status_path, {
                "phase": "running", "done": i, "total": total,
                "current": target_path.parent.name,
            })

            try:
                t0 = time.perf_counter()
                generator.generate_video(
                    prompt=req["prompt"],
                    image_path=req["image"],
                    height=req.get("height", 720),
                    width=req.get("width", 1280),
                    num_frames=req.get("num_frames", 57),
                    num_inference_steps=req.get("num_inference_steps", 3),
                    guidance_scale=req.get("guidance_scale", 1.0),
                    seed=req.get("seed", 42),
                    output_path=str(output_dir),
                    save_video=True,
                )
                elapsed = time.perf_counter() - t0
            except Exception as e:
                tb = traceback.format_exc()
                results_fh.write(json.dumps({
                    "id": req_id, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": tb,
                }) + "\n")
                results_fh.flush()
                if executor_dead_re.search(f"{type(e).__name__}: {e}") or executor_dead_re.search(tb):
                    print(f"[worker] executor died on sample {req_id} "
                          f"({target_path.parent.name}); aborting batch so the "
                          "true cause (OOM / CUDA error) is visible upstream.",
                          flush=True)
                    _write_status(args.status_path, {
                        "phase": "aborted", "done": i, "total": total,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    return 1
                continue

            produced = _find_produced_mp4(output_dir, before)
            if produced is None:
                results_fh.write(json.dumps({
                    "id": req_id, "ok": False,
                    "error": "no mp4 produced in output_dir",
                    "output_dir": str(output_dir),
                }) + "\n")
                results_fh.flush()
                continue

            if produced.resolve() != target_path.resolve():
                if target_path.exists():
                    target_path.unlink()
                shutil.move(str(produced), str(target_path))

            results_fh.write(json.dumps({
                "id": req_id, "ok": True,
                "elapsed_s": elapsed,
                "target_path": str(target_path),
            }) + "\n")
            results_fh.flush()

    _write_status(args.status_path, {"phase": "done", "done": total, "total": total})
    return 0


if __name__ == "__main__":
    sys.exit(main())
