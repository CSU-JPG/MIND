#!/bin/bash
# score_subset.sh -- bounded action scoring: N 1st-person, then N 3rd-person. Sequential (one GPU).
#
# FIX: before each pass, PARK existing result_dreamx-world_ar_*.json into results_bak/ so
# process.py's auto-resume can't pre-fill result_list. Previously the resumed entries
# satisfied the --limit quota, so the monitor fired the stop-event immediately and 0 new
# samples were scored (the 0s "1200 video/s" instant-skip). Parking = each pass scores fresh.
#
#   bash score_subset.sh                       do 1st then 3rd
#   SKIP_1ST=1 bash score_subset.sh             skip the 1st pass (1st already scored)
#   N=30 bash score_subset.sh                   change the per-perspective count
#
# After it runs: 1st result is parked in results_bak/, 3rd is the newest
# result_dreamx-world_ar_*.json in this folder. (Stop any other scoring first -- one GPU.)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export PYTHONIOENCODING=utf-8
# CUDA toolkit for this box (GB300, CUDA 13.2) -- deviates from the Windows
# original's hardcoded v13.0 path.
if [ -z "${CUDA_HOME:-}" ]; then
    CUDA_HOME="/usr/local/cuda-13.2"
fi
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

PY="$HERE/.venv/bin/python"
# Sibling dirs at the same level as MIND itself. UNVERIFIED on this box.
PARENT="$(dirname "$HERE")"
GT="$PARENT/MIND-Data"
TEST="$PARENT/MIND-tests/dreamx-world_ar"
METRICS="lcm,visual,dino,action,gsc"
N="${N:-50}"

if [ ! -x "$PY" ]; then echo "ERROR: venv python not found: $PY" >&2; exit 2; fi
mkdir -p "$HERE/results_bak"

if [ -z "${SKIP_1ST:-}" ]; then
    echo "============================================================"
    echo "[1/2] First $N 1st-person (FRESH -- parking old JSONs first)"
    echo "============================================================"
    if compgen -G "$HERE/result_dreamx-world_ar_*.json" >/dev/null; then
        mv -f "$HERE"/result_dreamx-world_ar_*.json "$HERE/results_bak/"
    fi
    "$PY" src/process.py --gt_root "$GT" --test_root "$TEST" --metrics "$METRICS" --num_gpus 1 --perspectives 1st_data --limit "$N"
fi

echo "============================================================"
echo "[2/2] First $N 3rd-person (FRESH -- parking old JSONs first)"
echo "============================================================"
if compgen -G "$HERE/result_dreamx-world_ar_*.json" >/dev/null; then
    mv -f "$HERE"/result_dreamx-world_ar_*.json "$HERE/results_bak/"
fi
"$PY" src/process.py --gt_root "$GT" --test_root "$TEST" --metrics "$METRICS" --num_gpus 1 --perspectives 3rd_data --limit "$N"

echo "============================================================"
echo "Done. 3rd -> newest result_dreamx-world_ar_*.json; 1st parked in results_bak/."
echo "(Each ~$N samples, with action.)"
echo "============================================================"
