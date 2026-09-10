#!/usr/bin/env python3
"""Add GSC metric to existing MIND results without recalculating other metrics."""

import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add MIND src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from process import calculate_gsc_for_video


def add_gsc_to_results(result_file, test_root, num_gpus=1):
    """Load results, calculate GSC for each video, and save back."""

    # Load existing results
    with open(result_file) as f:
        results = json.load(f)

    test_root = Path(test_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process each sample
    for item in tqdm(results.get("data", []), desc="Adding GSC"):
        if "gsc" in item:
            # Already has GSC, skip
            continue

        # Build video path
        path = item["path"]
        perspective = item["perspective"]
        test_type = item["test_type"]
        video_path = test_root / perspective / test_type / path / "video.mp4"

        if not video_path.exists():
            print(f"  SKIP {path}: video not found at {video_path}")
            continue

        try:
            # Calculate GSC (this requires the MIND metric to be available)
            gsc_score = calculate_gsc_for_video(str(video_path), device=device)
            item["gsc"] = gsc_score
            print(f"  {path}: GSC = {gsc_score:.4f}")
        except Exception as e:
            print(f"  ERROR {path}: {e}")
            item["error"] = str(e)

    # Save updated results
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    result_file = "result_lingbot-v2_backup_2026-08-03.json"
    test_root = "C:\\workspace\\world\\MIND-tests\\lingbot-v2"

    if not Path(result_file).exists():
        print(f"ERROR: {result_file} not found")
        sys.exit(1)

    add_gsc_to_results(result_file, test_root)
