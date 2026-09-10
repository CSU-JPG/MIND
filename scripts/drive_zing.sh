#!/bin/bash
# Stage Zing-0.5 (action-conditioned ti2v) videos into MIND-tests/zing/ for
# run_mind.sh scoring. Parallel to drive_evoke.sh/drive_zing.bat.
#
# Zing consumes a reference first frame + per-frame keyboard actions, which map
# directly onto MIND's action.json ws/ad/ud/lr ticks (see src/drive_zing.py),
# so both action_space_test AND mem_test run by default, across both
# 1st_data and 3rd_data.
#
# Unlike Evoke, zing_v0_5 is natively batch-oriented: one JSONL is built for
# every sample and the checkpoint loads exactly once.
#
# Cross-venv: this script runs src/drive_zing.py under MIND's OWN venv (it
# needs MIND's mirror_test_utils etc.); drive_zing.py then subprocess-calls
# ZING_VENV_PY (Zing's own venv) to actually run zing_v0_5 generation.
#
#   drive_zing.sh                             stage 1st + 3rd, score both
#   drive_zing.sh --dry-run                   preview what would be staged
#   drive_zing.sh --limit 5                   smoke test: first 5 samples
#   drive_zing.sh --test-type mem_test        limit to memory tests
#   drive_zing.sh --perspective 1st_data      override (default = both)
#   drive_zing.sh --num-frames 121            longer rollouts (default 97)
#
# Metric selection (forwarded to run_mind.sh after staging):
#   MIND_METRICS=lcm,visual ./drive_zing.sh   pick a subset
#   (unset)                                   default = lcm,visual,dino,gsc (no action)
#   MIND_GPUS=2                               multi-GPU scoring
#   MIND_PERSON=1st                           person = 1st | 3rd | both (default both)
#   MIND_MIRROR_TEST=0                        disable mirror_test (default on)
#   MIND_START_INDEX=N                        resume mid-run
#
# Cross-venv knobs:
#   ZING_VENV_PY=<path>                       override zing venv python
#                                              (default $ZING_REPO/.venv/bin/python)
#   ZING_REPO=<path>                          override the zing checkout
#
# All --flags pass through to src/drive_zing.py; MIND_* env vars stay in this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"

PY="$HERE/.venv/bin/python"
GT_ROOT="$(realpath -m "$HERE/../MIND-Data")"
MIND_TESTS="$(realpath -m "$HERE/../MIND-tests")"
LOG="$HERE/drive_zing.log"

ZING_REPO="$(realpath -m "${ZING_REPO:-$HERE/../zing-world-model}")"
ZING_VENV_PY="${ZING_VENV_PY:-$ZING_REPO/.venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 2
fi
if [ ! -x "$ZING_VENV_PY" ]; then
  echo "ERROR: zing venv python not found: $ZING_VENV_PY" >&2
  exit 2
fi
if [ ! -d "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

export ZING_REPO
export ZING_VENV_PY

echo "============================================================"
echo "Zing-0.5 (action-conditioned ti2v) staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : zing"
echo "  zing_repo    : $ZING_REPO"
echo "  zing_py      : $ZING_VENV_PY"
echo "  log          : $LOG"
echo "============================================================"

MIND_START_INDEX="${MIND_START_INDEX:-0}"
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

T_START=$(date +%s)

"$PY" "$HERE/src/drive_zing.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" \
  --start-index "$MIND_START_INDEX" "${MIRROR_ARG[@]}" "$@" 2>&1 | tee "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
printf '\n--- staging elapsed: %02d:%02d:%02d ---\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))

if [ "$EXIT_CODE" != "0" ]; then
  echo
  echo "ERROR: drive_zing.py exited with $EXIT_CODE" >&2
  exit "$EXIT_CODE"
fi

MIND_PERSON="${MIND_PERSON:-both}"
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS="lcm,visual,dino,gsc"
fi
MIND_GPUS="${MIND_GPUS:-1}"

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh zing \"$MIND_METRICS\" $MIND_GPUS $MIND_PERSON"
echo "============================================================"
"$HERE/run_mind.sh" zing "$MIND_METRICS" "$MIND_GPUS" "$MIND_PERSON"
