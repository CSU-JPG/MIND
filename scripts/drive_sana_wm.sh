#!/bin/bash
# Stage Sana world-model videos into MIND-tests/sana-wm/, then run MIND scoring.
# src/drive_sana_wm.py walks MIND-Data first frames + action.json, invokes
# Sana/inference_video_scripts/inference_sana_wm.py via Sana/.venv-wm/..., and
# stages output to MIND-tests/sana-wm/<perspective>/<test_type>/<gt_name>/video.mp4
# (one mp4 per mirror_test sample). Runs both perspectives by default; set
# MIND_SKIP_3RD=1 to skip the slower 3rd-person pass.
#
# NOTE (action metric): "action" needs ViPE, whose build chain is Windows/cp310
# only (prebuilt flash_attn wheel, MSVC) -- no aarch64 wheel exists, so treat
# "action" as likely unavailable on this box even though it's still requested
# below by default (matches setup_mind_venv.sh's note).
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
# Sana-WM uses a dedicated venv (.venv-wm) created by Sana/environment_setup_sana_wm.sh
# -- distinct from Sana/.venv (the older Sana env, may not exist).
# NOTE: Sana's own venv location/setup on this Linux box has not been
# verified -- translated straight from the Windows path shape
# (C:\workspace\world\Sana\.venv-wm\Scripts\python.exe), i.e. a sibling of
# MIND under workspace/world. Adjust below if Sana lives elsewhere on this box.
SANA_PY="$HERE/../Sana/.venv-wm/bin/python"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"
MODEL_NAME="sana-wm"
SANA_REPO="$HERE/../Sana"
SANA_ENTRY="$SANA_REPO/inference_video_scripts/inference_sana_video.py"
LOG="$HERE/drive_sana_wm.log"
: "${MIND_FPS:=24}"

if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found: $PY" >&2
  exit 2
fi
if [ ! -e "$SANA_REPO" ]; then
  echo "ERROR: Sana repo not found: $SANA_REPO" >&2
  exit 2
fi
if [ ! -e "$SANA_ENTRY" ]; then
  echo "ERROR: Sana entry not found: $SANA_ENTRY" >&2
  exit 2
fi
if [ ! -e "$HERE/src/drive_sana_wm.py" ]; then
  echo "ERROR: src/drive_sana_wm.py not yet written" >&2
  exit 2
fi

echo "=== Sana-WM staging into MIND-tests  |  model=$MODEL_NAME ==="
# drive_sana_wm.py PERSPECTIVES tuple is ordered ("3rd_data","1st_data"). We
# run both perspectives by default so the staged set is complete for run_mind.
# 1st_data runs first because it's faster and historically the higher-value
# pass; 3rd_data follows. To opt out of 3rd_data (slower pass), set
# MIND_SKIP_3RD=1 in the environment before launch. To override the
# perspective entirely, pass an extra `--perspective <name>` on the CLI
# (argparse last-wins) -- this short-circuits the dual-pass loop.

: "${MIND_SKIP_3RD:=0}"

# Mirror-test generation drives the gsc metric (one mp4 per mirror_test
# sample dir). On by default; set MIND_MIRROR_TEST=0 to skip.
: "${MIND_MIRROR_TEST:=1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

echo "--- pass 1/2: --perspective 1st_data ---"
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_sana_wm.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" --model-name "$MODEL_NAME" --sana-repo "$SANA_REPO" --sana-py "$SANA_PY" --fps "$MIND_FPS" --perspective 1st_data "${MIRROR_ARG[@]}" "$@"

if [ "$MIND_SKIP_3RD" = "1" ]; then
  echo "--- pass 2/2 skipped: MIND_SKIP_3RD=1 ---"
else
  echo "--- pass 2/2: --perspective 3rd_data ---"
  "$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_sana_wm.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" --model-name "$MODEL_NAME" --sana-repo "$SANA_REPO" --sana-py "$SANA_PY" --fps "$MIND_FPS" --perspective 3rd_data "${MIRROR_ARG[@]}" "$@"
fi

# Metric list including gsc. drive_sana_wm.py emits one video.mp4 per
# mirror_test sample dir, which is what the current gsc metric consumes.
# Override by setting MIND_METRICS before calling this script.
: "${MIND_METRICS:=lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,action,gsc
fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"
