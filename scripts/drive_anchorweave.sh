#!/bin/bash
# Stage AnchorWeave videos into MIND-tests/anchorweave/.
# TODO: needs src/drive_anchorweave.py; no .venv (set ANCHORWEAVE_PY env).
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
ANCHORWEAVE_PY="${ANCHORWEAVE_PY:-python3}"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"
MODEL_NAME=anchorweave
# NOTE: AnchorWeave repo location on this Linux box has not been verified --
# translated straight from the Windows path shape (sibling of MIND).
ANCHORWEAVE_REPO="$HERE/../AnchorWeave"
LOG="$HERE/drive_anchorweave.log"
MIND_FPS="${MIND_FPS:-24}"

if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found: $PY" >&2
  exit 2
fi
if [ ! -e "$ANCHORWEAVE_REPO" ]; then
  echo "ERROR: repo not found: $ANCHORWEAVE_REPO" >&2
  exit 2
fi
if [ ! -e "$HERE/src/drive_anchorweave.py" ]; then
  echo "ERROR: src/drive_anchorweave.py not yet written" >&2
  exit 2
fi

echo "=== AnchorWeave staging into MIND-tests  |  model=$MODEL_NAME ==="
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_anchorweave.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--repo" "$ANCHORWEAVE_REPO" "--py" "$ANCHORWEAVE_PY" "--fps" "$MIND_FPS" "--perspective" "1st_data" "$@"

# gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,action,gsc
fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"
