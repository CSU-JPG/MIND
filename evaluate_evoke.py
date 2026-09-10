#!/usr/bin/env python3
"""Evaluate Evoke-generated videos using MIND metrics."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def extract_frames(video_path, max_frames=None):
    """Extract frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stride = 1
    if max_frames and frame_count > max_frames:
        stride = frame_count // max_frames

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_idx += 1

    cap.release()
    return np.array(frames), fps


def compute_optical_flow_magnitude(frames):
    """Compute mean optical flow magnitude (motion metric)."""
    if len(frames) < 2:
        return 0.0

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flows = []

    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        flows.append(magnitude)

    return float(np.mean(flows)) if flows else 0.0


def compute_temporal_consistency(frames):
    """Compute temporal consistency (lower is more consistent)."""
    if len(frames) < 2:
        return 0.0

    diffs = []
    for i in range(len(frames) - 1):
        diff = np.mean(np.abs(frames[i].astype(float) - frames[i + 1].astype(float)))
        diffs.append(diff)

    return float(np.mean(diffs)) if diffs else 0.0


def compute_sharpness(frames):
    """Compute mean sharpness (Laplacian variance)."""
    sharpness_scores = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_scores.append(sharpness)

    return float(np.mean(sharpness_scores)) if sharpness_scores else 0.0


def evaluate_video(video_path, output_json=None):
    """Evaluate a single video and return metrics."""

    print(f"Evaluating: {video_path}")

    # Extract frames
    frames, fps = extract_frames(video_path, max_frames=128)

    if len(frames) == 0:
        print(f"ERROR: No frames extracted from {video_path}")
        return None

    print(f"  Frames: {len(frames)}, FPS: {fps:.1f}")

    # Compute metrics
    motion = compute_optical_flow_magnitude(frames)
    consistency = compute_temporal_consistency(frames)
    sharpness = compute_sharpness(frames)

    metrics = {
        "video": str(video_path),
        "frames": len(frames),
        "fps": float(fps),
        "motion_magnitude": motion,
        "temporal_consistency": consistency,
        "sharpness": sharpness,
    }

    print(f"  Motion: {motion:.2f}")
    print(f"  Temporal Consistency (diff): {consistency:.1f}")
    print(f"  Sharpness (Laplacian var): {sharpness:.1f}")

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved to: {output_json}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Evoke videos with MIND-like metrics")
    parser.add_argument("--video", type=Path, help="Single video to evaluate")
    parser.add_argument("--video-dir", type=Path, help="Directory of videos to evaluate")
    parser.add_argument("--output-json", type=Path, default=None, help="Save metrics JSON")
    parser.add_argument("--pattern", default="*.mp4", help="Video file pattern")

    args = parser.parse_args()

    if args.video:
        # Single video
        metrics = evaluate_video(args.video, args.output_json)
        if metrics:
            print(f"\n✓ Evaluation complete")

    elif args.video_dir:
        # Directory of videos
        video_dir = Path(args.video_dir)
        videos = sorted(video_dir.rglob(args.pattern))

        if not videos:
            print(f"No videos found in {video_dir} matching {args.pattern}")
            return

        print(f"Found {len(videos)} videos")
        results = []

        for video_path in videos:
            metrics = evaluate_video(video_path)
            if metrics:
                results.append(metrics)

        # Summary
        if results:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(f"Videos evaluated: {len(results)}")

            avg_motion = np.mean([r["motion_magnitude"] for r in results])
            avg_consistency = np.mean([r["temporal_consistency"] for r in results])
            avg_sharpness = np.mean([r["sharpness"] for r in results])

            print(f"Avg Motion: {avg_motion:.2f}")
            print(f"Avg Temporal Consistency: {avg_consistency:.1f}")
            print(f"Avg Sharpness: {avg_sharpness:.1f}")

            if args.output_json:
                summary = {
                    "num_videos": len(results),
                    "avg_motion": float(avg_motion),
                    "avg_temporal_consistency": float(avg_consistency),
                    "avg_sharpness": float(avg_sharpness),
                    "videos": results,
                }
                with open(args.output_json, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"\nSaved to: {args.output_json}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
