#!/bin/bash
# rerun_all.sh -- re-score dreamx-world_ar from scratch, then AUTO-RESUME on crashes.
# Clears stale result JSONs ONCE (fresh start), scores both perspectives / all metrics.
# process.py writes the result JSON every 0.5s; if scoring dies partway (ViPE/action
# can crash), this re-runs run_mind.sh WITHOUT clearing -- so it auto-resumes from the
# partial JSON, skipping already-scored samples -- up to MAX_TRIES times until complete.
#
#   bash rerun_all.sh                    fresh start + auto-resume retries (default 20)
#   MAX_TRIES=50 bash rerun_all.sh       raise the retry cap

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

METRICS="lcm,visual,dino,action,gsc"
MAX_TRIES="${MAX_TRIES:-20}"

# Fresh start: move stale result JSONs aside so the FIRST pass scores from scratch.
mkdir -p "$HERE/results_bak"
if compgen -G "$HERE/result_dreamx-world_ar_*.json" >/dev/null; then
    echo "Moving stale result JSONs to results_bak/ ..."
    mv -f "$HERE"/result_dreamx-world_ar_*.json "$HERE/results_bak/"
fi

TRY=0
while true; do
    TRY=$((TRY + 1))
    echo "============================================================"
    echo "Scoring attempt $TRY/$MAX_TRIES  (auto-resumes from partial JSON after pass 1)"
    echo "============================================================"
    bash "$HERE/run_mind.sh" dreamx-world_ar "$METRICS" 1 both
    RC=$?
    if [ "$RC" = "0" ]; then
        break
    fi

    echo "Attempt $TRY exited with code $RC (likely a ViPE/action crash)."
    if [ "$TRY" -ge "$MAX_TRIES" ]; then
        echo "Reached MAX_TRIES=$MAX_TRIES; giving up. Partial results are in the newest result JSON."
        exit "$RC"
    fi
    echo "Resuming in 5s -- run_mind will skip already-scored samples..."
    sleep 5
done

echo "============================================================"
echo "Done. Newest result_dreamx-world_ar_*.json holds the full results."
echo "============================================================"
exit 0
