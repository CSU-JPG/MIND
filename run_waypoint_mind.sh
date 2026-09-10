#!/bin/bash
# ==========================================================================
# End-to-end Waypoint-1.5 on MIND: drive -> score -> print table, in one go.
# Handles the comma-metrics quoting internally (script-to-script). Drive args
# pass through (e.g. --limit 5, --perspective 3rd_data). Metrics/person
# overridable via MIND_METRICS / MIND_PERSON env vars.
#
#   bash run_waypoint_mind.sh                 full 1st-person: drive + score + table
#   bash run_waypoint_mind.sh --limit 5       quick smoke
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MIND_METRICS="${MIND_METRICS:-lcm,visual,dino}"
MIND_PERSON="${MIND_PERSON:-1st}"

echo "============================================================"
echo "[1/3] DRIVE Waypoint-1.5 over MIND"
echo "============================================================"
if ! bash "$HERE/drive_waypoint.sh" "$@"; then
    echo "drive step failed"
    exit 1
fi

echo
echo "Done driving -> MIND-tests/waypoint/. Scoring removed (run separately when ready):"
echo "  run_mind.sh waypoint lcm,visual,dino 1 both"
echo "  scores.sh waypoint"
