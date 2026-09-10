#!/bin/bash
# Drive ABot-World over MIND: seed each sample's first frame, convert its MIND
# actions (ws/ad/ud/lr) -> ABot keys (WASD+IJKL), run ABot inference, write to
# MIND-tests/abot/. Runs through ABot-World's own venv. Then score with:
#   run_mind.sh abot lcm,visual,dino 1 1st
#
#   drive_abot.sh --limit 3          smoke (SLOW: reloads ~24GB model per sample)
#   drive_abot.sh                    all 1st-person
#   drive_abot.sh --blocks 12        longer rollout (<=15)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"

ABOT="$HERE/../ABot-World"
PY="$ABOT/.venv/bin/python"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"

# NOTE: ABot-World's own venv location/setup on this Linux box has not been
# verified -- translated straight from the Windows path shape
# (C:\workspace\world\ABot-World\.venv\Scripts\python.exe), i.e. a sibling
# of MIND under workspace/world. Adjust ABOT above (or export it) if
# ABot-World lives elsewhere on this box.
if [ ! -x "$PY" ]; then
  echo "ERROR: ABot venv missing -- run $ABOT/setup_venv.sh first" >&2
  exit 1
fi

# no --perspective -> both 1st_data + 3rd_data. Restrict via: drive_abot.sh --perspective 3rd_data
"$PY" "$HERE/src/drive_abot.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" "$@"

echo
echo "Videos -> $MIND_TESTS/abot/"
echo "Now score:  run_mind.sh abot lcm,visual,dino 1 1st"
