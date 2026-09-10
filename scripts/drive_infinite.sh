#!/bin/bash
# Stage Infinite-World videos into MIND-tests/infinite-world/.
# TODO: needs src/drive_infinite.py; no venv of its own (set INFINITE_PY env).
#
# NOTE (Linux port): MIND_METRICS below defaults to including "action" (ViPE).
# Per setup_mind_venv.sh's note, ViPE's build chain is Windows/cp310-specific
# (no aarch64 flash_attn wheel), so "action" is likely unavailable on this box.
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
INFINITE_PY="${INFINITE_PY:-python}"
GT_ROOT="${GT_ROOT:-$HERE/../MIND-Data}"
MIND_TESTS="${MIND_TESTS:-$HERE/../MIND-tests}"
MODEL_NAME="infinite-world"
# Cross-project path, sibling of MIND under the same world/ root (not
# verified to exist/be set up on this Linux box).
INFINITE_REPO="${INFINITE_REPO:-$HERE/../Infinite-World}"
LOG="$HERE/drive_infinite.log"
MIND_FPS="${MIND_FPS:-24}"

if [ ! -x "$PY" ]; then echo "ERROR: venv python not found: $PY" >&2; exit 2; fi
if [ ! -e "$INFINITE_REPO" ]; then echo "ERROR: repo not found: $INFINITE_REPO" >&2; exit 2; fi
if [ ! -e "$HERE/src/drive_infinite.py" ]; then echo "ERROR: src/drive_infinite.py not yet written" >&2; exit 2; fi

echo "=== Infinite-World staging into MIND-tests  |  model=$MODEL_NAME ==="
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_infinite.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--repo" "$INFINITE_REPO" "--py" "$INFINITE_PY" "--fps" "$MIND_FPS" "--perspective" "1st_data" "$@"

# gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then MIND_METRICS="lcm,visual,dino,action,gsc"; fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"
