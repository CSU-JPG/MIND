"""Drive FastVideo Matrix-Game-3.0-Distilled from a MIND-Data tree.

Parallel to ``drive_matrix3.py`` but targets the FastVideo distilled
checkpoint (``FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers``). The
distilled model needs only 3 inference steps so the heavy cost is model
loading — therefore this driver spawns a SINGLE worker process inside
FastVideo's venv that processes the whole batch with one model load,
instead of re-spawning the model per sample like ``drive_matrix3.py`` does.

IPC is file-based, not stdout-based: importing ``fastvideo`` monkey-patches
``sys.stdout`` to a per-rank prefix wrapper that EINVALs on piped stdout on
Windows. So the driver writes a JSON-Lines manifest, redirects worker
stdout/stderr to a logfile, and tails the worker's results file for
progress.

Layout mirrors ``drive_matrix3.py``:
  - walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt>/
  - extracts first frame, reads ``action.json`` caption
  - hands work to the worker; worker writes ``<test_root>/<model>/<persp>/<test>/<gt>/video.mp4``
  - mirror_test supported (additive) for the ``gsc`` metric

Defaults match the FastVideo demo: 720x1280, 57 frames, 3 steps,
guidance_scale=1.0, seed=42.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import av

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4

# FastVideo venv is the only place ``from fastvideo import VideoGenerator``
# resolves with the matching torch + CUDA stack. Override with
# ``MATRIX3D_VENV_PY`` if a sibling FastVideo install is preferred.
DEFAULT_FASTVIDEO_VENV_PY = Path(__file__).resolve().parent.parent.parent / "FastVideo" / ".venv" / "Scripts" / "python.exe"
FASTVIDEO_VENV_PY = Path(os.environ.get("MATRIX3D_VENV_PY", str(DEFAULT_FASTVIDEO_VENV_PY)))

WORKER_SCRIPT = Path(__file__).resolve().parent / "_matrix3_distilled_worker.py"

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_image()
            img.save(out_path, "PNG")
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


def build_prompt(sample: dict) -> str:
    if sample["perspective"] == "1st_data":
        return "First-person view exploring a 3D virtual environment."
    return "Third-person view of a character exploring a 3D virtual environment."


def _spawn_worker(manifest_path: Path, results_path: Path, status_path: Path, log_path: Path) -> subprocess.Popen:
    """Spawn the FastVideo-venv worker with stdout/stderr -> logfile.

    Strips cross-venv pollution from os.environ so the FastVideo interpreter
    loads its own stdlib (otherwise _sre MAGIC mismatch crashes ``import re``
    on the very first import). Matches the env scrub used in drive_matrix3.py.

    Stdout and stderr go to ``log_path``; NOT piped, because FastVideo wraps
    sys.stdout in a prefix writer that EINVALs on a piped stdout on Windows.
    IPC is file-based instead (manifest + results + status).
    """
    env = os.environ.copy()
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        str(FASTVIDEO_VENV_PY), "-X", "utf8", str(WORKER_SCRIPT),
        "--manifest", str(manifest_path),
        "--results-path", str(results_path),
        "--status-path", str(status_path),
    ]
    log_fh = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(WORKER_SCRIPT.parent),
    )


def _tail_results(proc: subprocess.Popen, results_path: Path, status_path: Path,
                  log_path: Path, total: int, work_items: list[tuple[dict, Path, Path]],
                  model_name: str) -> list[str]:
    """Tail results JSONL while the worker runs. Returns failures list.

    Re-opens the results file periodically (creates lazily). Prints one line
    per completed sample. Also surfaces the worker's status snapshot so the
    user sees ``loading``/``ready``/``running`` transitions.
    """
    failures: list[str] = []
    item_by_id = {i: (s, frame_png, target) for i, (s, frame_png, target) in enumerate(work_items)}
    seen_ids: set[int] = set()
    last_status_phase: str | None = None
    last_progress_log = time.time()

    while True:
        # Read any new result lines.
        if results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                for line_no, raw in enumerate(f):
                    if line_no in seen_ids:
                        continue
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        resp = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = resp.get("id", line_no)
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    s, _frame_png, target = item_by_id[rid]
                    tag = f"{s['perspective']}/{s['test_type']}/{s['gt_name']}"
                    if resp.get("ok"):
                        print(f"  [{len(seen_ids)}/{total}] ok    {tag}   elapsed={resp.get('elapsed_s', 0):.1f}s",
                              flush=True)
                        log_mp4(model_name, s["perspective"], s["test_type"], s["gt_name"], target)
                    else:
                        err = resp.get("error", "unknown")
                        print(f"  [{len(seen_ids)}/{total}] FAIL  {tag}   {err}", flush=True)
                        failures.append(tag)

        # Surface worker status transitions.
        if status_path.exists():
            try:
                snapshot = json.loads(status_path.read_text(encoding="utf-8"))
                phase = snapshot.get("phase")
                if phase != last_status_phase:
                    print(f"[worker] phase={phase} done={snapshot.get('done', 0)}/"
                          f"{snapshot.get('total', total)}", flush=True)
                    last_status_phase = phase
                # Periodic heartbeat during long generation.
                if phase == "running" and time.time() - last_progress_log > 30:
                    print(f"[worker] processing {snapshot.get('current', '?')} "
                          f"({snapshot.get('done', 0)}/{snapshot.get('total', total)})", flush=True)
                    last_progress_log = time.time()
            except (json.JSONDecodeError, OSError):
                pass

        rc = proc.poll()
        if rc is not None:
            # Drain any final results that may have been written between the
            # last poll and the exit.
            if results_path.exists():
                with open(results_path, encoding="utf-8") as f:
                    for line_no, raw in enumerate(f):
                        if line_no in seen_ids:
                            continue
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            resp = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rid = resp.get("id", line_no)
                        if rid in seen_ids:
                            continue
                        seen_ids.add(rid)
                        s, _frame_png, target = item_by_id[rid]
                        tag = f"{s['perspective']}/{s['test_type']}/{s['gt_name']}"
                        if resp.get("ok"):
                            print(f"  [{len(seen_ids)}/{total}] ok    {tag}   "
                                  f"elapsed={resp.get('elapsed_s', 0):.1f}s", flush=True)
                            log_mp4(model_name, s["perspective"], s["test_type"], s["gt_name"], target)
                        else:
                            err = resp.get("error", "unknown")
                            print(f"  [{len(seen_ids)}/{total}] FAIL  {tag}   {err}", flush=True)
                            failures.append(tag)

            if rc != 0:
                print(f"\n[worker] exited rc={rc}. Last 40 lines of worker log:", flush=True)
                try:
                    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                    for ln in log_lines:
                        print(f"  | {ln}", flush=True)
                except OSError:
                    pass
            return failures

        time.sleep(2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root", type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="matrix-game-3-distilled",
                        help="Subfolder name under test-root")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Temp dir for extracted first frames + IPC files (default: <test-root>/.frames-distilled)")
    parser.add_argument("--only", nargs="+",
                        help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES,
                        help="Limit to one perspective (driver default = 1st_data via bat)")
    parser.add_argument("--test-type", choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit", type=int, help="Only run first N matched samples")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N matched samples after filters; used for mid-run resume.")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=57,
                        help="FastVideo distilled default is 57 frames @ 24fps ≈ 2.4s of video.")
    parser.add_argument("--num-inference-steps", type=int, default=3,
                        help="Distilled checkpoint converges in 3 steps; bump for ablations only.")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24,
                        help="Accepted for bat-script consistency; FastVideo controls fps internally.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mirror-test", action="store_true",
                        help="Also generate mirror_test outputs (additive).")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if not FASTVIDEO_VENV_PY.exists():
        print(f"FATAL: FastVideo venv python not found at {FASTVIDEO_VENV_PY}", file=sys.stderr)
        return 2
    if not WORKER_SCRIPT.exists():
        print(f"FATAL: worker script not found at {WORKER_SCRIPT}", file=sys.stderr)
        return 2

    work_dir = args.work_dir or (args.test_root / ".frames-distilled")
    work_dir.mkdir(parents=True, exist_ok=True)

    samples = [] if args.mirror_only else gather_samples(args.gt_root)
    if args.mirror_test:
        samples += gather_mirror_samples(args.gt_root, args.mirror_action)
    if args.perspective:
        samples = [s for s in samples if s["perspective"] == args.perspective]
    if args.test_type:
        samples = [s for s in samples if s["test_type"] == args.test_type]
    if args.only:
        samples = [s for s in samples if any(sub.lower() in s["gt_name"].lower() for sub in args.only)]
    if args.start_index:
        if args.start_index >= len(samples):
            print(f"--start-index {args.start_index} is past the end of {len(samples)} matched sample(s); "
                  "nothing to do.")
            return 0
        samples = samples[args.start_index:]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("No samples matched.")
        return 1

    print(f"Will process {len(samples)} sample(s) via {FASTVIDEO_VENV_PY.name}:")
    for s in samples:
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}")
    print()

    # Build per-sample work items first (extract frames, decide skip-if-exists).
    work_items: list[tuple[dict, Path, Path]] = []  # (sample, frame_png, target_mp4)
    for s in samples:
        target = output_path(args.test_root, args.model_name, s)
        if target.exists():
            print(f"[skip] {s['perspective']}/{s['test_type']}/{s['gt_name']} -> {target} (exists)")
            continue
        frame_png = work_dir / s["perspective"] / s["test_type"] / f"{s['gt_name']}.png"
        if s.get("frame_png_src") is not None:
            frame_png.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s["frame_png_src"], frame_png)
        else:
            extract_first_frame(s["video"], frame_png)
        work_items.append((s, frame_png, target))

    if not work_items:
        print("All requested samples already exist; nothing to do.")
        return 0

    if args.dry_run:
        for s, frame_png, target in work_items:
            print(f"[dry-run] {s['perspective']}/{s['test_type']}/{s['gt_name']}: "
                  f"frame={frame_png} -> {target}")
        return 0

    # File-based IPC paths. Fresh per run so the tailer's per-line index lines
    # up with manifest order without dealing with stale rows.
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    ipc_dir = work_dir / "ipc" / run_stamp
    ipc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ipc_dir / "manifest.jsonl"
    results_path = ipc_dir / "results.jsonl"
    status_path = ipc_dir / "status.json"
    log_path = ipc_dir / "worker.log"

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, (s, frame_png, target) in enumerate(work_items):
            req = {
                "id": i,
                "image": str(frame_png),
                "prompt": build_prompt(s),
                "output_dir": str(target.parent),
                "target_path": str(target),
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
            }
            mf.write(json.dumps(req) + "\n")

    print(f"[ipc] manifest : {manifest_path}")
    print(f"[ipc] results  : {results_path}")
    print(f"[ipc] worker log: {log_path}")
    print()

    proc = _spawn_worker(manifest_path, results_path, status_path, log_path)
    try:
        failures = _tail_results(proc, results_path, status_path, log_path,
                                  total=len(work_items), work_items=work_items,
                                  model_name=args.model_name)
    finally:
        if proc.poll() is None:
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for name in failures:
            print(f"  {name}")
        return 1
    print(f"Done. {len(work_items)} sample(s) produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
