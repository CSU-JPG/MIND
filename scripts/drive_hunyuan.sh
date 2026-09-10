#!/bin/bash
# Stage Hunyuan-GameCraft videos into MIND-tests/hunyuan-gamecraft/ for run_mind.sh scoring.
#
# Hunyuan-GameCraft has no dedicated venv; its run_low_mem script uses bare
# `python`. Provide a python env (env var HUNYUAN_PY or pip-install
# requirements.txt in the MIND venv).
#
# TODO: needs src/drive_hunyuan.py that:
#   - Walks MIND-Data first frames + action.json (WASD -> --action-list / --action-speed-list)
#   - Invokes Hunyuan-GameCraft-1.0/hymm_sp/sample_batch.py
#   - Stages output to MIND-tests/hunyuan-gamecraft/<perspective>/<test_type>/<gt_name>/video.mp4
#
# Defaults (from run_low_mem.bat): 704x1216, 33 frames, 8 steps, fp8.
#
# NOTE (Linux port): MIND_METRICS below defaults to including "action", which
# uses ViPE. Per setup_mind_venv.sh's note, ViPE's build chain is Windows/
# cp310-specific (no aarch64 flash_attn wheel), so "action" is likely
# unavailable on this box -- the other 4 metrics (lcm,visual,dino,gsc) still work.
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
HUNYUAN_PY="${HUNYUAN_PY:-python}"
GT_ROOT="${GT_ROOT:-$HERE/../MIND-Data}"
MIND_TESTS="${MIND_TESTS:-$HERE/../MIND-tests}"
MODEL_NAME="hunyuan-gamecraft"
# Cross-project path, sibling of MIND under the same world/ root (not
# verified to exist/be set up on this Linux box).
HUNYUAN_REPO="${HUNYUAN_REPO:-$HERE/../Hunyuan-GameCraft-1.0}"
LOG="$HERE/drive_hunyuan.log"
MIND_FPS="${MIND_FPS:-24}"

if [ ! -x "$PY" ]; then echo "ERROR: venv python not found: $PY" >&2; exit 2; fi
if [ ! -e "$GT_ROOT" ]; then echo "ERROR: gt_root not found: $GT_ROOT" >&2; exit 2; fi
if [ ! -e "$HUNYUAN_REPO/hymm_sp/sample_batch.py" ]; then
    echo "ERROR: Hunyuan-GameCraft/hymm_sp/sample_batch.py not found" >&2
    exit 2
fi
if [ ! -e "$HERE/src/drive_hunyuan.py" ]; then
    echo "ERROR: src/drive_hunyuan.py not yet written" >&2
    echo "This script is a stub; create the driver script first." >&2
    exit 2
fi

echo "============================================================"
echo "Hunyuan-GameCraft staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : $MODEL_NAME"
echo "  hunyuan_repo : $HUNYUAN_REPO"
echo "  hunyuan_py   : $HUNYUAN_PY"
echo "  log          : $LOG"
echo "============================================================"

"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_hunyuan.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--hunyuan-repo" "$HUNYUAN_REPO" "--hunyuan-py" "$HUNYUAN_PY" "--fps" "$MIND_FPS" "--perspective" "1st_data" "$@"

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh $MODEL_NAME"
echo "============================================================"
# gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then MIND_METRICS="lcm,visual,dino,action,gsc"; fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"
