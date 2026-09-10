#!/bin/bash
# ==========================================================================
# Print the combined MIND scores table from the EXISTING result_*.json files.
# No scoring is run -- this just reads what's already there.
#   bash scores.sh                    all result_*.json
#   bash scores.sh matrix-game-3      only matching result files
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$HERE/.venv/bin/python"
PAT="result_*.json"
if [ -n "${1:-}" ]; then PAT="result_${1}*.json"; fi

# _scores_table.py prints ONE file; pass the NEWEST match (by mtime) so you
# see the latest run, not the oldest.
NEWEST="$(ls -t $HERE/$PAT 2>/dev/null | head -n 1 || true)"
if [ -z "$NEWEST" ]; then
    echo "  no result files matching $PAT"
    exit 1
fi
"$PY" "$HERE/_scores_table.py" "$NEWEST"

echo
echo "================= by perspective ================="
for P in 1st_data 3rd_data; do
    "$PY" "$HERE/_split_persp.py" "$NEWEST" "$P" "${TMPDIR:-/tmp}/_mind_${P}.json" >/dev/null 2>&1 || true
    echo "--- $P ---"
    "$PY" "$HERE/_scores_table.py" "${TMPDIR:-/tmp}/_mind_${P}.json"
    rm -f "${TMPDIR:-/tmp}/_mind_${P}.json" 2>/dev/null || true
done
