#!/bin/bash
# Stage worldfm videos into MIND-tests/worldfm/.
# TODO: needs src/drive_worldfm.py; no .venv (set WORLDFM_PY env).
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
: "${WORLDFM_PY:=python3}"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"
MODEL_NAME="worldfm"
# NOTE: worldfm's own location/setup on this Linux box has not been
# verified -- translated straight from the Windows path shape.
WORLDFM_REPO="$HERE/../worldfm"
LOG="$HERE/drive_worldfm.log"
: "${MIND_FPS:=24}"

if [ ! -x "$PY" ]; then echo "ERROR: venv python not found: $PY" >&2; exit 2; fi
if [ ! -e "$WORLDFM_REPO" ]; then echo "ERROR: repo not found: $WORLDFM_REPO" >&2; exit 2; fi
if [ ! -e "$HERE/src/drive_worldfm.py" ]; then echo "ERROR: src/drive_worldfm.py not yet written" >&2; exit 2; fi

echo "=== worldfm staging into MIND-tests  |  model=$MODEL_NAME ==="
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_worldfm.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" --model-name "$MODEL_NAME" --repo "$WORLDFM_REPO" --py "$WORLDFM_PY" --fps "$MIND_FPS" --perspective 1st_data "$@"
# gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
: "${MIND_METRICS:=lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,action,gsc
fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"
