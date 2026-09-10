#!/bin/bash
# One-shot runner: each active driver with --mirror-test (additive: regular + mirror).
#
# Runs in sequence (NOT parallel) -- GPU contention crushes throughput when
# heavy generators overlap. Each driver scores itself at the end (existing
# behavior), so result_<model>_*.json drops as each finishes.
#
# Skips any driver whose prereqs are missing (venv, weights, etc.) -- that
# driver exits non-zero and we move on. Summary at end lists which succeeded.
#
# Usage:
#   bash drive_all_with_mirror.sh                       all 5 drivers, default args
#   bash drive_all_with_mirror.sh --limit 5             pass-through to each driver
#   bash drive_all_with_mirror.sh --perspective 1st_data
#
# Override the model list with MIND_RUN_MODELS (space-separated):
#   MIND_RUN_MODELS="dreamx_small matrix3" bash drive_all_with_mirror.sh
#
# To run mirror-only (skip the regular action_space + mem_test passes), pass
# --mirror-only via "$@" -- it propagates to every driver.

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

MIND_RUN_MODELS="${MIND_RUN_MODELS:-dreamx_small matrix3 sana_wm lingbot deepverse}"

echo "============================================================"
echo "MIND: drive all (with --mirror-test) | models: $MIND_RUN_MODELS"
echo "extra args: $*"
echo "============================================================"

OK_LIST=""
FAIL_LIST=""

for M in $MIND_RUN_MODELS; do
    echo
    echo "============================================================"
    echo "=== drive_${M}.sh --mirror-test $*"
    echo "============================================================"
    if [ -e "$HERE/drive_${M}.sh" ]; then
        if bash "$HERE/drive_${M}.sh" --mirror-test "$@"; then
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
echo "Summary"
echo "============================================================"
echo "  OK   :$OK_LIST"
echo "  FAIL :$FAIL_LIST"
echo
