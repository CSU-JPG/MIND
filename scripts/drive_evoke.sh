#!/bin/bash
# Stage Evoke (i2v-style, camera-controlled) videos into MIND-tests/evoke/ for
# run_mind.sh scoring. Parallel to drive_helios_i2v.sh.
#
# Unlike Helios, Evoke takes a real camera pose trajectory (converted from MIND's
# action.json actor_pos/actor_rpy / camera_pos/camera_rpy ground truth -- see
# src/utils/evoke_pose.py), so both action_space_test AND mem_test run by default,
# across both 1st_data and 3rd_data.
#
# Usage:
#   drive_evoke.sh                            stage 1st + 3rd, score both
#   drive_evoke.sh --dry-run                  preview what would be staged
#   drive_evoke.sh --limit 5                  smoke test: first 5 samples
#   drive_evoke.sh --test-type mem_test       limit to memory tests
#   drive_evoke.sh --perspective 1st_data     override (default = both)
#
# Metric selection (forwarded to run_mind.sh after staging):
#   MIND_METRICS=lcm,visual                pick a subset
#   (unset)                                default = lcm,visual,dino,gsc (no action)
#   MIND_GPUS=2                            multi-GPU scoring
#   MIND_PERSON=1st                        person = 1st | 3rd | both (default both)
#   MIND_MIRROR_TEST=0                     disable mirror_test (default on)
#   MIND_START_INDEX=N                     resume mid-run
#
# Cross-venv knobs:
#   EVOKE_VENV_PY=<path>                   override Evoke venv python
#                                          (default $HERE/../Evoke/.venv/bin/python)
#   EVOKE_MODEL_PATH=<path>                override the local snapshot/evoke-base path
#
# All --flags pass through to src/drive_evoke.py; MIND_* env vars stay in this script.
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
LOG="$HERE/drive_evoke.log"

# NOTE: Evoke repo/venv location on this Linux box has not been verified --
# translated straight from the Windows path shape (sibling of MIND).
EVOKE_VENV_PY="${EVOKE_VENV_PY:-$HERE/../Evoke/.venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 2
fi
if [ ! -x "$EVOKE_VENV_PY" ]; then
  echo "ERROR: Evoke venv python not found: $EVOKE_VENV_PY" >&2
  echo "Run: $HERE/../Evoke/setup_evoke.sh" >&2
  exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

echo "============================================================"
echo "Evoke (camera-controlled i2v) staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : evoke"
echo "  evoke_py     : $EVOKE_VENV_PY"
echo "  log          : $LOG"
echo "============================================================"

MIND_START_INDEX="${MIND_START_INDEX:-0}"
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

_T_START=$(date +%s)

set +e
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_evoke.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--start-index" "$MIND_START_INDEX" "${MIRROR_ARG[@]}" "$@"
EXIT_CODE=$?
set -e

_T_END=$(date +%s)
_T_ELAPSED_SEC=$((_T_END - _T_START))
_T_ELAPSED=$(printf '%02d:%02d:%02d' $((_T_ELAPSED_SEC/3600)) $((_T_ELAPSED_SEC%3600/60)) $((_T_ELAPSED_SEC%60)))
echo
echo "--- staging elapsed: $_T_ELAPSED ---"

if [ "$EXIT_CODE" != 0 ]; then
  echo
  echo "ERROR: drive_evoke.py exited with $EXIT_CODE" >&2
  exit "$EXIT_CODE"
fi

MIND_PERSON="${MIND_PERSON:-both}"
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,gsc
fi
MIND_GPUS="${MIND_GPUS:-1}"

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh evoke \"$MIND_METRICS\" $MIND_GPUS $MIND_PERSON"
echo "============================================================"
bash "$HERE/run_mind.sh" evoke "$MIND_METRICS" "$MIND_GPUS" "$MIND_PERSON"
