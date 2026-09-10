#!/bin/bash
# Drive SolarWM's MiniMax-H3 over MIND. Default engine (camera) uses REAL
# camera-conditioned generation: the trained Stage0.5 LoRA adapter +
# h3_camera_infer.py, with each sample's real ws/ad/ud/lr action.json
# converted directly into a [47,4,4] camera trajectory (genuine per-frame
# conditioning, not a text hint). --engine text falls back to the original
# uncoditioned base pipeline + text-paraphrase approach. Writes to
# MIND-tests/solarwm/. Runs through SolarWM's own .venv-h3 venv.
#
# UNTESTED end to end (camera engine) -- see h3_camera_infer.py's own
# docstring for the unverified camera axis/sign caveat, and
# src/drive_solarwm.py's docstring for the full engine comparison. Only
# lcm/visual/dino/avg_mse are trustworthy right now; `action` AND `gsc`
# should be treated as noise until camera-engine output is manually
# inspected and confirmed to follow the intended direction. Score with:
#   run_mind.sh solarwm lcm,visual,dino 1 both
#
# LOAD-ONCE: builds a manifest and makes one --mind-batch call -- the model
# loads once and loops every sample, same idea as drive_abot.sh.
#
#   drive_solarwm.sh --limit 2                       smoke (camera engine)
#   drive_solarwm.sh                                 all 1st+3rd person
#   drive_solarwm.sh --engine text --limit 2          fallback (no real conditioning)
#   drive_solarwm.sh --mirror-test                    mirror clips (gsc not meaningful, see above)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"

# NOTE: SolarWM's location on this box varies by checkout -- on the GB300 dev
# box it's been seen at /localhome/kschmid/SolarWM (not a sibling of MIND).
# Override with SOLARWM_ROOT= if the sibling-of-MIND default below is wrong.
SOLARWM_ROOT="${SOLARWM_ROOT:-$HERE/../SolarWM}"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"

PY="$SOLARWM_ROOT/.venv-h3/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: SolarWM H3 venv missing at $PY" >&2
  echo "  -- run 'bash setup_env_h3.sh' in \$SOLARWM_ROOT first, or set" >&2
  echo "  SOLARWM_ROOT=/path/to/SolarWM if it's not at $SOLARWM_ROOT" >&2
  exit 1
fi

# no --perspective -> both 1st_data + 3rd_data. Restrict via: drive_solarwm.sh --perspective 3rd_data
"$PY" "$HERE/src/drive_solarwm.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" \
  --solarwm-root "$SOLARWM_ROOT" "$@"

echo
echo "Videos -> $MIND_TESTS/solarwm/"
echo "Now score (action AND gsc are unreliable here, exclude them):"
echo "  run_mind.sh solarwm lcm,visual,dino 1 both"
