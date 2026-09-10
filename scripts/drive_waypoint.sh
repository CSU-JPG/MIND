#!/bin/bash
# ==========================================================================
# Drive Waypoint-1.5 over the MIND benchmark: seed each sample's first frame,
# replay its per-frame ws/ad/ud/lr actions, write videos to MIND-tests/waypoint/.
# Runs through scope-overworld's venv (has world_engine + cv2 + imageio); shares
# its compile cache. Then score with:  run_mind.sh waypoint lcm,visual,dino 1 1st
#
#   drive_waypoint.sh                       1st-person, all samples
#   drive_waypoint.sh --limit 5             quick smoke
#   drive_waypoint.sh --perspective 3rd_data
#
# Deviates from drive_waypoint.bat: that version launches via `uv run --no-sync
# --project` into scope-overworld's uv-managed venv. This repo (MIND) is not
# to use uv for its own tooling, so here we invoke scope-overworld's venv
# python directly instead -- functionally equivalent (no-sync just meant
# "don't resolve/install first", which a direct venv python call also does).
#
# NOTE (action/VK codes): src/drive_waypoint.py converts MIND ws/ad/ud/lr
# actions to Windows virtual-key codes (VK["W"]=0x57 etc.) that world_engine
# consumes -- that's Python-level logic inside drive_waypoint.py, unrelated
# to this shell script, so it is NOT translated here. Whether world_engine's
# input backend even accepts/interprets Windows VK codes on Linux has not
# been verified; flagging per the mind-benchmark skill's documented gotcha.
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"

# NOTE: scope-overworld's own venv location/setup on this Linux box has not
# been verified -- translated straight from the Windows path shape
# (C:\workspace\world\scope-overworld), i.e. a sibling of MIND under
# workspace/world. Adjust SCOPE below (or export it) if scope-overworld
# lives elsewhere on this box.
SCOPE="$HERE/../scope-overworld"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$SCOPE/.inductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR="$SCOPE/.triton_cache"

if [ ! -e "$SCOPE/.venv" ]; then
  echo "ERROR: scope-overworld venv missing" >&2
  exit 1
fi

SCOPE_PY="$SCOPE/.venv/bin/python"
if [ ! -x "$SCOPE_PY" ]; then
  echo "ERROR: scope-overworld venv python missing: $SCOPE_PY" >&2
  exit 1
fi

# no --perspective -> both 1st_data + 3rd_data (matches drive_abot). Restrict via --perspective 3rd_data
"$SCOPE_PY" "$HERE/src/drive_waypoint.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" "$@"

echo
echo "Videos -> $MIND_TESTS/waypoint/"
echo "Now score:  run_mind.sh waypoint lcm,visual,dino 1 1st"
