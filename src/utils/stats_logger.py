"""Per-mp4 generation stats logger.

Each drive_*.py calls log_mp4(...) after a sample's mp4 has been written.
We take a single point snapshot of RAM (current process RSS) plus a
one-shot nvidia-smi sample of VRAM used/free + GPU utilization, and append
one row to stats_gen.csv. Continuous peak sampling is the responsibility of
an external monitor task, not this module.

Schema:
    timestamp, model, perspective, test_type, gt_name,
    size_mb, fps, ram_max_gb, vram_used_mib, vram_free_mib, gpu_util_pct

fps = playback fps read from the mp4 itself (rounded to 2 decimals); not a
generation-speed metric. Empty when av can't open the file.
"""

import csv
import datetime
import os
import subprocess
from pathlib import Path

import av
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV = Path(os.environ.get("MIND_STATS_CSV", REPO_ROOT / "stats_gen.csv"))
HEADER = [
    "timestamp", "model", "perspective", "test_type", "gt_name",
    "size_mb", "fps", "ram_max_gb", "vram_used_mib", "vram_free_mib", "gpu_util_pct",
]


def _mp4_fps(mp4_path: Path):
    try:
        with av.open(str(mp4_path)) as c:
            s = c.streams.video[0]
            if s.average_rate is None:
                return ""
            return f"{float(s.average_rate):.2f}"
    except Exception:
        return ""


def _nvidia_smi_snapshot(gpu_index: int = 0):
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i", str(gpu_index),
            ],
            timeout=4,
        ).decode().strip()
        used, free, util = (s.strip() for s in out.split(","))
        return int(used), int(free), int(util)
    except Exception:
        return None, None, None


def log_mp4(model: str, perspective: str, test_type: str, gt_name: str, mp4_path: Path) -> None:
    rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    vram_used, vram_free, gpu_util = _nvidia_smi_snapshot()
    size_mb = mp4_path.stat().st_size / (1024 ** 2) if mp4_path.exists() else 0.0

    row = [
        datetime.datetime.now().isoformat(timespec="seconds"),
        model, perspective, test_type, gt_name,
        f"{size_mb:.2f}",
        _mp4_fps(mp4_path),
        f"{rss_gb:.2f}",
        "" if vram_used is None else vram_used,
        "" if vram_free is None else vram_free,
        "" if gpu_util is None else gpu_util,
    ]

    write_header = not STATS_CSV.exists() or STATS_CSV.stat().st_size == 0
    with open(STATS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(HEADER)
        w.writerow(row)
