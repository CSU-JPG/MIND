#!/bin/bash
# Stage Matrix-Game-2 videos into MIND-tests/matrix-game-2/ for run_mind.sh scoring.
#
# Usage:
#   drive_matrix2.sh                            stage all samples
#   drive_matrix2.sh --dry-run                  preview commands without running inference
#   drive_matrix2.sh --limit 5                  first 5 samples only
#   drive_matrix2.sh --perspective 1st_data     limit to first-person
#   drive_matrix2.sh --test-type mem_test       limit to memory tests
#   drive_matrix2.sh --config-path /path/to/inference_gta_drive.yaml
#
# Env knobs (set before running):
#   MATRIX2_VENV_PY    python for the matrix2 venv (defaults to MIND's venv)
#   MATRIX2_PRETRAINED pretrained_model_path dir holding Wan2.1_VAE.pth
#                      (default: $HERE/../Matrix-Game/Matrix-Game-2/Matrix-Game-2.0)
#   MIND_MIRROR_TEST=0 skip the mirror_test pass (default on, drives the gsc metric)
#   MIND_START_INDEX   skip the first N matched samples (resume mid-run)
#
# Metric selection (forwarded to run_mind.sh after staging):
#   MIND_METRICS=lcm,visual         pick a subset
#   (unset)                         default = lcm,visual,dino,action,gsc
#   MIND_GPUS=2                     multi-GPU scoring
#   MIND_PERSON=1st                 person = 1st | 3rd | both
#
# All --flags pass through to src/drive_matrix2.py; MIND_* env vars stay in this script.
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
LOG="$HERE/drive_matrix2.log"

# matrix2 venv defaults to MIND's venv; override via MATRIX2_VENV_PY env var
# (drive_matrix2.py reads MATRIX2_VENV_PY directly, so we just inherit it).
: "${MATRIX2_VENV_PY:=$PY}"
export MATRIX2_VENV_PY

# MIND-Data is 24 fps; matrix2 inference doesn't expose --fps. Recorded for traceability.
: "${MIND_FPS:=24}"

if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found: $PY" >&2
  exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

echo "============================================================"
echo "Matrix-Game-2 staging into MIND-tests"
echo "============================================================"
echo "  gt_root   : $GT_ROOT"
echo "  test_root : $MIND_TESTS"
echo "  model     : matrix-game-2"
echo "  venv_py   : $MATRIX2_VENV_PY"
echo "  log       : $LOG"
echo "============================================================"

: "${MIND_START_INDEX:=0}"

# Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
# On by default; set MIND_MIRROR_TEST=0 to skip.
: "${MIND_MIRROR_TEST:=1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_matrix2.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" --fps "$MIND_FPS" --perspective 1st_data --start-index "$MIND_START_INDEX" "${MIRROR_ARG[@]}" "$@"

: "${MIND_PERSON:=1st}"
: "${MIND_METRICS:=lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,action,gsc
fi
: "${MIND_GPUS:=1}"

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh matrix-game-2 \"$MIND_METRICS\" $MIND_GPUS $MIND_PERSON"
echo "============================================================"
bash "$HERE/run_mind.sh" matrix-game-2 "$MIND_METRICS" "$MIND_GPUS" "$MIND_PERSON"
