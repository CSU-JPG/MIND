#!/bin/bash
# score_3rd.sh -- score first N 3rd-person samples WITH action, FRESH.
# Parks existing result_dreamx-world_ar_*.json into results_bak/ first so process.py's
# auto-resume can't pre-fill result_list (that double-counts --limit -> 0s instant-skip).
# 1st-person results stay safe in results_bak/. One GPU -- stop any other scoring first.
#   bash score_3rd.sh            first 50 3rd-person
#   N=30 bash score_3rd.sh       change the count

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

echo "Parking existing result JSONs (so 3rd scores fresh, no resume double-count)..."
if compgen -G "$HERE/result_dreamx-world_ar_*.json" >/dev/null; then
    mv -f "$HERE"/result_dreamx-world_ar_*.json "$HERE/results_bak/"
fi

echo "============================================================"
echo "Scoring first $N 3rd-person (action included)"
echo "============================================================"
"$PY" src/process.py --gt_root "$GT" --test_root "$TEST" --metrics "$METRICS" --num_gpus 1 --perspectives 3rd_data --limit "$N"

echo "============================================================"
echo "Done. 3rd-person -> newest result_dreamx-world_ar_*.json (~$N, with action)."
echo "1st-person results preserved in results_bak/."
echo "============================================================"
