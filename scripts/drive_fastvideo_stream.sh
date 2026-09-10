#!/bin/bash
# Launch the FastVideo MatrixGame streaming demo from MIND.
#
# Unlike the other drive_*.sh scripts, this is a LIVE PREVIEW launcher --
# game-streaming-poc writes an MJPEG HTTP stream, not MP4 files, so the
# output cannot feed MIND's offline metrics. Useful for visually sanity-
# checking the Matrix-Game-2.0 FastVideo variants alongside MIND runs.
#
# Usage:
#   drive_fastvideo_stream.sh                       prompted for actions
#   drive_fastvideo_stream.sh "wu wu ai dl"          one-shot
#   drive_fastvideo_stream.sh "wa wa wd sq" --loops 3 --variant gta_distilled_model
#
# Args after the action string pass through to game_streaming.py unchanged.
# Open http://localhost:8080/ in a browser to view, or:
#   ffplay http://localhost:8080/stream.mjpg
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
# NOTE: fastvideo-dynamo repo location on this Linux box has not been
# verified -- translated straight from the Windows path shape (sibling of
# MIND under workspace/world).
POC="$HERE/../fastvideo-dynamo/game-streaming-poc/game_streaming.py"

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 2
fi
if [ ! -e "$POC" ]; then
  echo "ERROR: game_streaming.py not found at $POC" >&2
  echo "Did you clone fastvideo-dynamo? Run its setup.sh to install fastvideo first." >&2
  exit 2
fi

# Verify fastvideo is installed (setup.sh in game-streaming-poc handles this).
if ! "$PY" -c "import fastvideo" 2>/dev/null; then
  echo "ERROR: fastvideo not importable in MIND's venv." >&2
  echo "Run: $HERE/../fastvideo-dynamo/game-streaming-poc/setup.sh" >&2
  exit 2
fi

ACTIONS="${1:-}"
if [ -n "$ACTIONS" ]; then
  shift
fi

echo "============================================================"
echo "FastVideo MatrixGame stream"
echo "============================================================"
echo "  poc       : $POC"
echo "  python    : $PY"
if [ -n "$ACTIONS" ]; then
  echo "  actions   : $ACTIONS"
fi
echo "  browser   : http://localhost:8080/"
echo "============================================================"

if [ -n "$ACTIONS" ]; then
  "$PY" "$POC" --actions "$ACTIONS" "$@"
else
  "$PY" "$POC" "$@"
fi
