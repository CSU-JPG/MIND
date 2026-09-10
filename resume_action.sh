#!/bin/bash
# resume_action.sh -- RESUME dreamx-world_ar scoring after a crash (action runs are
# ~15h and ViPE can die partway). process.py saves the result JSON every 0.5s and
# auto-resumes from the newest result_dreamx-world_ar_*.json, skipping already-scored
# samples and continuing with the rest. Re-run this as many times as needed.
#
#   rerun_all.sh      = FRESH start (clears JSONs, no resume)
#   resume_action.sh  = CONTINUE from where it died (keeps JSONs, auto-resume)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! compgen -G "$HERE/result_dreamx-world_ar_*.json" >/dev/null; then
    echo "No result_dreamx-world_ar_*.json found -- nothing to resume."
    echo "Use rerun_all.sh for a fresh start instead."
    exit 1
fi

echo "Resuming dreamx-world_ar scoring -- auto-resume from newest result JSON,"
echo "skipping already-scored samples."
bash "$HERE/run_mind.sh" dreamx-world_ar "lcm,visual,dino,action,gsc" 1 both
