#!/bin/bash
# ==========================================================================
# Score EVERY driven test set in MIND-tests against MIND-Data GT (via
# run_mind.sh per subdir), then print the combined scores table. Skips the
# .frames cache. run_mind.sh resumes from prior result_*.json, so re-running
# only scores new samples. Backends aren't needed -- this scores videos
# already staged.
#
#   bash score_all.sh                                    lcm,visual,dino,action,gsc  both  (default)
#   bash score_all.sh "lcm,visual,dino,action,gsc" 1 both  full metrics (action=ViPE)
#   bash score_all.sh "lcm,visual" 1 1st                  quick, first-person only
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$HERE/.venv/bin/python"
# Sibling dir at the same level as MIND itself (mirrors the Windows layout
# C:\workspace\world\MIND-tests). UNVERIFIED on this box -- adjust if
# MIND-tests actually lives elsewhere here.
TESTS="$(dirname "$HERE")/MIND-tests"

METRICS="${1:-}"
if [ -z "$METRICS" ]; then METRICS="lcm,visual,dino,action,gsc"; fi
GPUS="${2:-}"
if [ -z "$GPUS" ]; then GPUS="1"; fi
PERSON="${3:-}"
if [ -z "$PERSON" ]; then PERSON="both"; fi

if [ ! -x "$PY" ]; then echo "ERROR: venv missing -- run setup_mind_venv.sh" >&2; exit 1; fi
if [ ! -d "$TESTS" ]; then echo "ERROR: no MIND-tests dir at $TESTS" >&2; exit 1; fi

echo "Scoring all sets under $TESTS   metrics=$METRICS  person=$PERSON"
echo

for D in "$TESTS"/*/; do
    [ -d "$D" ] || continue
    NAME="$(basename "$D")"
    if [ "$NAME" = ".frames" ]; then continue; fi
    echo "============================================================"
    echo "=== $NAME"
    echo "============================================================"
    bash "$HERE/run_mind.sh" "$NAME" "$METRICS" "$GPUS" "$PERSON"
done

echo
echo "============================================================"
echo "=== combined scores table"
echo "============================================================"
JSONS=()
for J in "$HERE"/result_*.json; do
    [ -e "$J" ] && JSONS+=("$J")
done
if [ "${#JSONS[@]}" -gt 0 ]; then
    "$PY" "$HERE/_scores_table.py" "${JSONS[@]}"
else
    echo "  no result_*.json yet"
fi
