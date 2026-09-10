#!/bin/bash
# ==========================================================================
# Generate HY-WorldPlay videos for MIND scoring -- mem_test + action_space_test
# + mirror_test, BOTH perspectives. Restartable: drive_hy_worldplay.py skips any
# sample whose video.mp4 already exists, so re-running only fills the gaps
# (e.g. the missing mirror_test videos). Wraps drive_hy_worldplay.sh.
#
#   bash gen_hy_videos.sh            full run (HY_LIMIT=50/perspective, both, mirror on)
#   bash gen_hy_videos.sh 2          quick test: only 2 samples per perspective
#   MIND_PERSPECTIVE=3rd_data bash gen_hy_videos.sh   single perspective
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Mirror test ON (this is the coverage that was missing).
export MIND_MIRROR_TEST=1

# Optional first arg = HY_LIMIT (samples per perspective). Default 50 (full).
if [ -n "${1:-}" ]; then export HY_LIMIT="$1"; fi
export HY_LIMIT="${HY_LIMIT:-50}"

echo "============================================================"
echo "Generating HY-WorldPlay videos (restartable; skips existing)"
echo "  mirror_test : ON"
echo "  HY_LIMIT    : $HY_LIMIT per perspective"
echo "  perspective : ${MIND_PERSPECTIVE:-} (blank = both 1st_data + 3rd_data)"
echo "============================================================"

# Delegate to the working driver (resolves model paths, sets MASTER_ADDR=127.0.0.1,
# USE_LIBUV=0, etc). It loops perspectives x test_types and stages video.mp4.
bash "$HERE/drive_hy_worldplay.sh"
