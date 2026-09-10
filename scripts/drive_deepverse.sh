#!/bin/bash
# Stage DeepVerse videos into MIND-tests/deepverse/ for run_mind.sh scoring.
#
# TODO: needs src/drive_deepverse.py that:
#   - Walks MIND-Data first frames + action.json
#   - Invokes DeepVerse/run.py (uses DeepVerse/.venv/bin/python)
#   - Stages output to MIND-tests/deepverse/<perspective>/<test_type>/<gt_name>/video.mp4
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# These scripts live in scripts/ but every path below is written relative to
# the repository root -- .venv, src/, and the sibling MIND-Data / MIND-tests.
# Resolve the root rather than assuming this file sits in it, so the script
# works from either location.
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
cd "$HERE"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

PY="$HERE/.venv/bin/python"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"
MODEL_NAME=deepverse
# NOTE: DeepVerse repo location on this Linux box has not been verified --
# translated straight from the Windows path shape (sibling of MIND).
DEEPVERSE_REPO="$HERE/../DeepVerse"
# Cross-spawn python for DeepVerse inference. Resolution order:
#   1. DEEPVERSE_VENV_PY env var (if set, used as-is -- no existence check)
#   2. DeepVerse/.venv/bin/python (if present)
#   3. plain `python3` on PATH (whatever's active in the calling shell)
if [ -z "${DEEPVERSE_VENV_PY:-}" ]; then
  if [ -x "$DEEPVERSE_REPO/.venv/bin/python" ]; then
    DEEPVERSE_VENV_PY="$DEEPVERSE_REPO/.venv/bin/python"
  else
    DEEPVERSE_VENV_PY=python3
  fi
fi
LOG="$HERE/drive_deepverse.log"
MIND_FPS="${MIND_FPS:-24}"

if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found: $PY" >&2
  exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi
if [ ! -e "$DEEPVERSE_REPO/run.py" ]; then
  echo "ERROR: DeepVerse/run.py not found at $DEEPVERSE_REPO/run.py" >&2
  exit 2
fi
if [ ! -e "$HERE/src/drive_deepverse.py" ]; then
  echo "ERROR: src/drive_deepverse.py not yet written" >&2
  echo "This script is a stub; create the driver script first." >&2
  exit 2
fi

echo "============================================================"
echo "DeepVerse staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : $MODEL_NAME"
echo "  deepverse    : $DEEPVERSE_REPO"
echo "  deepverse_py : $DEEPVERSE_VENV_PY"
echo "  log          : $LOG"
echo "============================================================"

# drive_deepverse.py PERSPECTIVES tuple now defaults to ("3rd_data","1st_data"),
# so omitting --perspective walks both with 3rd-person first. Pass --perspective
# <p> on the CLI to restrict to one. CLI args after "$@" override the defaults.

# Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
# On by default; set MIND_MIRROR_TEST=0 to skip.
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

set +e
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_deepverse.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--deepverse-repo" "$DEEPVERSE_REPO" "--deepverse-py" "$DEEPVERSE_VENV_PY" "--fps" "$MIND_FPS" "${MIRROR_ARG[@]}" "$@"
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" != 0 ]; then
  echo
  echo "ERROR: drive_deepverse.py exited with $EXIT_CODE" >&2
  exit "$EXIT_CODE"
fi

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh $MODEL_NAME"
echo "============================================================"
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,action,gsc
fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"
