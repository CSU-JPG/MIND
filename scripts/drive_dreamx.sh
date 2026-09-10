#!/bin/bash
# Stage DreamX-World videos into MIND-tests/dreamx-world/ and auto-score with run_mind.sh.
#
# Usage:
#   drive_dreamx.sh                            stage all samples then score
#   drive_dreamx.sh --dry-run                  preview commands without running inference
#   drive_dreamx.sh --limit 5                  first 5 samples only
#   drive_dreamx.sh --perspective 1st_data     limit to first-person
#   drive_dreamx.sh --test-type mem_test       limit to memory tests
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
MODEL_NAME=dreamx-world
LOG="$HERE/drive_dreamx.log"

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
echo "DreamX-World staging into MIND-tests"
echo "============================================================"
echo "  gt_root   : $GT_ROOT"
echo "  test_root : $MIND_TESTS"
echo "  model     : $MODEL_NAME"
echo "  log       : $LOG"
echo "============================================================"

# MIND standard fps (24) -- matches GT MIND-Data + scoring crop expectations.
MIND_FPS="${MIND_FPS:-24}"
# --perspective 1st_data: only stage first-person samples. Override with an
# extra `--perspective 3rd_data` arg (argparse last-wins).

# Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
# On by default; set MIND_MIRROR_TEST=0 to skip.
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

set +e
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_dreamx.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--fps" "$MIND_FPS" "--perspective" "1st_data" "${MIRROR_ARG[@]}" "$@"
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
