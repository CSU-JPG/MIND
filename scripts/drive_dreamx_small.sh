#!/bin/bash
# Stage DreamX-World videos into MIND-tests/dreamx-world_small/ for run_mind.sh scoring.
#
# Identical to drive_dreamx.sh except --model-name is `dreamx-world_small`, so
# the staged videos land under a separate test-root subdir and don't collide
# with the full-resolution `dreamx-world` set.
#
# Usage:
#   drive_dreamx_small.sh                            stage all samples
#   drive_dreamx_small.sh --dry-run                  preview commands without running inference
#   drive_dreamx_small.sh --limit 5                  first 5 samples only
#   drive_dreamx_small.sh --perspective 1st_data     limit to first-person
#   drive_dreamx_small.sh --test-type mem_test       limit to memory tests
#
# All flags pass through to src/drive_dreamx.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

PY="$HERE/.venv/bin/python"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"
MODEL_NAME=dreamx-world_small
LOG="$HERE/drive_dreamx_small.log"

# DreamX-World's own .venv is missing on this box; reuse MIND's venv as the
# cross-spawned inference interpreter. Override DREAMX_VENV_PY beforehand
# to point elsewhere if you have a dedicated DreamX-World venv.
DREAMX_VENV_PY="${DREAMX_VENV_PY:-$PY}"

if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found: $PY" >&2
  exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

echo "============================================================"
echo "DreamX-World staging into MIND-tests (small)"
echo "============================================================"
echo "  gt_root   : $GT_ROOT"
echo "  test_root : $MIND_TESTS"
echo "  model     : $MODEL_NAME"
echo "  log       : $LOG"
echo "============================================================"

# Speed bundle: half-resolution, 30 steps, 121 frames @ 24fps (5s @ MIND-std fps),
# fp8 transformer weights. ~4-5x faster than the full-res defaults but matches
# MIND-Data's 24 fps so cropped action-metric comparisons are like-for-like.
# --perspective 1st_data: only stage first-person samples. Override with an
# extra `--perspective 3rd_data` arg (argparse last-wins).
MIND_FPS="${MIND_FPS:-24}"

# Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
# On by default; set MIND_MIRROR_TEST=0 to skip.
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

set +e
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_dreamx.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--height" "352" "--width" "640" "--video-length" "121" "--fps" "$MIND_FPS" "--steps" "30" "--gpu-memory-mode" "model_full_load_and_qfloat8" "--perspective" "1st_data" "${MIRROR_ARG[@]}" "$@"
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" != 0 ]; then
  echo
  echo "ERROR: drive_dreamx.py exited with $EXIT_CODE" >&2
  exit "$EXIT_CODE"
fi

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh $MODEL_NAME"
echo "============================================================"
# run_mind.sh defaults PERSON=1st, matching this script's 1st_data-only generation.
# gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,action,gsc
fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"
