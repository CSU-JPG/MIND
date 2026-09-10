#!/bin/bash
# Drive Echo-WM (JoyAI-Echo, LTX-based) over MIND: seed each sample's first
# frame, convert its MIND actions (ws/ad/ud/lr) -> Echo's WASD/IJKL action
# string DSL, run echo_wm/inference_wm.py, write to MIND-tests/echo/.
# Runs through Echo's own venv.
#
# SLOW: inference_wm.py has no load-once/batch mode -- each sample reloads the
# ~47.8 GB checkpoint plus the Gemma text encoder (same per-sample pattern as
# drive_h3world.sh / drive_evoke.sh).
#
#   drive_echo.sh --dry-run --limit 5   print action strings, no model load
#   drive_echo.sh --limit 2             smoke test
#   drive_echo.sh                       all 1st+3rd person
#   drive_echo.sh --mirror-test         mirror clips (needed for gsc)
#   drive_echo.sh --causal              512x288 few-step entrypoint
#
# Then score with:
#   run_mind.sh echo "lcm,visual,dino,action,gsc" 1 both --resume none
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"

ECHO_ROOT="$(realpath -m "${ECHO_ROOT:-$HERE/../JoyAI-Echo}")"
GT_ROOT="$(realpath -m "$HERE/../MIND-Data")"
MIND_TESTS="$(realpath -m "$HERE/../MIND-tests")"

PY="$HERE/.venv/bin/python"
ECHO_PY="${ECHO_PY:-$ECHO_ROOT/echo_wm/.venv/bin/python}"

# --dry-run only needs the MIND venv and the dataset; skip the Echo checks so
# the action-string mapping can be inspected on a box without the checkpoint.
DRY_RUN=0
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then DRY_RUN=1; fi
done

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 2
fi
if [ ! -d "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi
if [ "$DRY_RUN" = "0" ] && [ ! -x "$ECHO_PY" ]; then
  echo "ERROR: Echo-WM venv missing at $ECHO_PY -- run $ECHO_ROOT/echo_wm/setup_and_run.sh" >&2
  exit 2
fi

echo "============================================================"
echo "Echo-WM staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : echo"
echo "  echo_root    : $ECHO_ROOT"
echo "  echo_py      : $ECHO_PY"
echo "  dry_run      : $DRY_RUN"
echo "============================================================"

export ECHO_ROOT
export ECHO_PY

"$PY" "$HERE/src/drive_echo.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" "$@"

echo
echo "Videos -> $MIND_TESTS/echo/"
echo "Now score:  bash run_mind.sh echo \"lcm,visual,dino,action,gsc\" 1 both --resume none"
