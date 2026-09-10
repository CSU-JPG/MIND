#!/bin/bash
# Stage FastVideo Matrix-Game-3.0-Distilled videos into MIND-tests/matrix-game-3-distilled/
# for run_mind.sh scoring. Parallel to drive_matrix3.sh, but targets the
# distilled checkpoint via FastVideo's pipeline (3 inference steps, fast).
#
# Usage:
#   drive_matrix3_distilled.sh                       stage all 1st-person + mirror
#   drive_matrix3_distilled.sh --dry-run             preview commands
#   drive_matrix3_distilled.sh --limit 5             first 5 samples only
#   drive_matrix3_distilled.sh --test-type mem_test  limit to memory tests
#   drive_matrix3_distilled.sh --perspective 3rd_data  override (default = 1st_data)
#
# Metric selection (forwarded to run_mind.sh after staging):
#   MIND_METRICS=lcm,visual                pick a subset
#   (unset)                                default = lcm,visual,dino,action,gsc
#   MIND_GPUS=2                            multi-GPU scoring
#   MIND_PERSON=1st                        person = 1st | 3rd | both (default 1st)
#   MIND_MIRROR_TEST=0                     disable mirror_test (default on)
#   MIND_START_INDEX=N                     resume mid-run
#
# Cross-venv knobs:
#   MATRIX3D_VENV_PY=<path>                override FastVideo venv python
#                                           (default $HERE/../FastVideo/.venv/bin/python)
#
# All --flags pass through to src/drive_matrix3_distilled.py; MIND_* env vars
# stay in this script.
#
# NOTE (action metric): "action" needs ViPE, whose build chain is Windows/cp310
# only (prebuilt flash_attn wheel, MSVC) -- no aarch64 wheel exists, so treat
# "action" as likely unavailable on this box even though it's still requested
# below by default (matches setup_mind_venv.sh's note).
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
LOG="$HERE/drive_matrix3_distilled.log"

# FastVideo's venv has the matching torch + CUDA stack for the distilled model.
# NOTE: FastVideo's own venv location/setup on this Linux box has not been
# verified -- translated straight from the Windows path shape
# (C:\workspace\world\FastVideo\.venv\Scripts\python.exe), i.e. a sibling of
# MIND under workspace/world. Adjust below (or export MATRIX3D_VENV_PY) if
# FastVideo lives elsewhere on this box.
: "${MATRIX3D_VENV_PY:=$HERE/../FastVideo/.venv/bin/python}"

# Distilled checkpoint defaults -- 57 frames @ 24fps standard for MIND-Data.
: "${MIND_FPS:=24}"

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 2
fi
if [ ! -x "$MATRIX3D_VENV_PY" ]; then
  echo "ERROR: FastVideo venv python not found: $MATRIX3D_VENV_PY" >&2
  echo "Set MATRIX3D_VENV_PY to point at the FastVideo .venv, or install it." >&2
  exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

echo "============================================================"
echo "Matrix-Game-3 DISTILLED (FastVideo) staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : matrix-game-3-distilled"
echo "  fastvideo_py : $MATRIX3D_VENV_PY"
echo "  log          : $LOG"
echo "============================================================"

# Defaults: 1st-person only, mirror_test on (matches drive_matrix3.sh).
: "${MIND_START_INDEX:=0}"
: "${MIND_MIRROR_TEST:=1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_matrix3_distilled.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" --fps "$MIND_FPS" --perspective 1st_data --start-index "$MIND_START_INDEX" "${MIRROR_ARG[@]}" "$@"

: "${MIND_PERSON:=1st}"
: "${MIND_METRICS:=lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,action,gsc
fi
: "${MIND_GPUS:=1}"

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh matrix-game-3-distilled \"$MIND_METRICS\" $MIND_GPUS $MIND_PERSON"
echo "============================================================"
bash "$HERE/run_mind.sh" matrix-game-3-distilled "$MIND_METRICS" "$MIND_GPUS" "$MIND_PERSON"
