#!/bin/bash
# ==========================================================================
# Drive LingBot-World-2 (Wan-A14B) over MIND.
# Synthesizes poses.npy from MIND actor_pos/rpy + wasd/ijkl from ws/ad/ud/lr,
# runs generate.py per sample, moves output to MIND-tests/lingbot-v2/.
# SLOW: ~1 fps, 14B, reload per sample -> smoke with --limit 1 FIRST.
#
#   bash drive_lingbot_v2.sh --limit 1        smoke (one sample)
#   bash drive_lingbot_v2.sh --perspective 1st_data
#
# NOTE (Linux port): the Windows original shells out to `wsl -e bash -lc
# ...` to run lingbot-world-v2 in WSL with a fixed venv path
# (/home/kschmid/lingbot-venv). This box IS already Linux, so no WSL hop is
# needed -- run the lingbot-world-v2 venv's python directly. Path shape below
# assumes lingbot-world-v2 is a sibling of MIND under the same world/ root
# with its own .venv (mirroring this repo's own .venv layout); this has NOT
# been verified to exist/be set up on this box.
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
LINGBOT_V2_REPO="${LINGBOT_V2_REPO:-$HERE/../lingbot-world-v2}"
LINGBOT_V2_PY="${LINGBOT_V2_PY:-$LINGBOT_V2_REPO/.venv/bin/python}"

if [ ! -e "$LINGBOT_V2_REPO" ]; then echo "ERROR: lingbot-world-v2 repo not found: $LINGBOT_V2_REPO" >&2; exit 2; fi
if [ ! -x "$LINGBOT_V2_PY" ]; then echo "ERROR: lingbot-world-v2 venv python not found: $LINGBOT_V2_PY" >&2; exit 2; fi

(
  cd "$LINGBOT_V2_REPO"
  "$LINGBOT_V2_PY" "$HERE/src/drive_lingbot_v2.py" "$@"
)

echo
echo "Videos -> $HERE/../MIND-tests/lingbot-v2/"
echo 'Then score:  bash run_mind.sh lingbot-v2 "lcm,visual,dino,action,gsc" 1 both'
