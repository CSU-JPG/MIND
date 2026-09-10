#!/bin/bash
# Smoke test: render ONE sample per model (--limit 1) across every driver, in
# sequence (parallel GPU use crushes throughput). Each drive_<model>.sh scores
# its single sample at the end (existing behavior), so you get one video + a
# result_<model>_*.json per model -- a quick "does every model still run?" pass
# and a 1-clip side-by-side.
#
# Missing/broken drivers (no script, missing venv/weights) exit non-zero; we
# log and continue. Summary at the end lists which succeeded.
#
# Usage:
#   bash drive_one_each.sh                      one sample from every model below
#   bash drive_one_each.sh --dry-run            preview each driver's command only
#   bash drive_one_each.sh --perspective 1st_data
#
# Override the model list (space-separated drive_<NAME>.sh stems):
#   MIND_RUN_MODELS="dreamx matrix3 sana_wm" bash drive_one_each.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# Full set of models with real drivers on this box. Edit / override via
# MIND_RUN_MODELS. Stems must match drive_<stem>.sh.
MIND_RUN_MODELS="${MIND_RUN_MODELS:-dreamx dreamx_ar matrix2 matrix3 matrix3_distilled sana_wm lingbot lingbot_flash deepverse helios_i2v hy_worldplay}"

echo "============================================================"
echo "MIND: one sample per model (--limit 1)"
echo "  models    : $MIND_RUN_MODELS"
echo "  extra args: $*"
echo "============================================================"

OK_LIST=""
FAIL_LIST=""

for M in $MIND_RUN_MODELS; do
    echo
    echo "============================================================"
    echo "=== drive_${M}.sh --limit 1 $*"
    echo "============================================================"
    if [ -e "$HERE/drive_${M}.sh" ]; then
        if bash "$HERE/drive_${M}.sh" --limit 1 "$@"; then
            OK_LIST="$OK_LIST $M"
        else
            RC=$?
            FAIL_LIST="$FAIL_LIST $M"
            echo "--- $M FAILED (rc=$RC), continuing ---"
        fi
    else
        FAIL_LIST="$FAIL_LIST ${M}(missing-script)"
        echo "--- drive_${M}.sh NOT FOUND, skipping ---"
    fi
done

echo
echo "============================================================"
echo "Summary (one sample per model)"
echo "============================================================"
echo "  OK   :$OK_LIST"
echo "  FAIL :$FAIL_LIST"
echo
