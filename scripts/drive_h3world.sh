#!/bin/bash
# Drive H3-World (MiniMax-H3 LoRA) over MIND: seed each sample's first frame,
# convert its MIND actions (ws/ad/ud/lr) -> H3-World's raw per-frame action
# matrix (WASD+IJKL, same schema as ABot), run H3-World's code/abot/infer.py,
# write to MIND-tests/h3world/. Runs through H3-World's own venv.
#
# SLOW: infer.py has no load-once/batch mode -- each sample reloads the ~33B
# backbone + LoRA (matches drive_evoke.sh's per-sample pattern).
#
#   drive_h3world.sh --limit 2          smoke test
#   drive_h3world.sh                    all 1st+3rd person
#   drive_h3world.sh --mirror-test      mirror clips
#
# Then score with:
#   run_mind.sh h3world lcm,visual,dino 1 both
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"

H3_ROOT="$(realpath -m "${H3_ROOT:-$HERE/../H3-World}")"
GT_ROOT="$(realpath -m "$HERE/../MIND-Data")"
MIND_TESTS="$(realpath -m "$HERE/../MIND-tests")"

PY="$HERE/.venv/bin/python"
H3_PY="$H3_ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 2
fi
if [ ! -x "$H3_PY" ]; then
  echo "ERROR: H3-World venv missing at $H3_PY -- see H3-World/README.md Setup" >&2
  exit 2
fi
if [ ! -d "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

echo "============================================================"
echo "H3-World (MiniMax-H3 LoRA) staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : h3world"
echo "  h3_root      : $H3_ROOT"
echo "  h3_py        : $H3_PY"
echo "============================================================"

export H3_ROOT
export H3_PY

"$PY" "$HERE/src/drive_h3world.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" "$@"

echo
echo "Videos -> $MIND_TESTS/h3world/"
echo "Now score:  run_mind.sh h3world lcm,visual,dino 1 both"
