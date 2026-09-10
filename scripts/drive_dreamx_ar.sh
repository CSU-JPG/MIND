#!/bin/bash
# Stage DreamX-World-5B (AR / long-horizon) videos into MIND-tests/dreamx-world_ar/
# and auto-score with run_mind.sh. SEPARATE from drive_dreamx.sh, which drives the
# Cam (bidirectional, inference_dreamx5b.py) model. This one drives the autoregressive
# model (inference_ar_forcing.py + GD-ML/DreamX-World-5B).
#
# Usage:
#   drive_dreamx_ar.sh                       stage all 1st-person samples then score
#   drive_dreamx_ar.sh --dry-run             preview without running inference
#   drive_dreamx_ar.sh --limit 5             first 5 samples only
#   drive_dreamx_ar.sh --num-output-frames 63   ~15s long-horizon clips (default 21 -> 81px)
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
MODEL_NAME=dreamx-world_ar
LOG="$HERE/drive_dreamx_ar.log"

# DreamX-World's own .venv now exists and is the correct interpreter for the AR
# model (cu130 torch + sageattention + the inference fixes). Use it for inference
# AND for resolving the HF checkpoints (it has huggingface_hub).
# NOTE: DreamX-World repo/venv location on this Linux box has not been
# verified -- translated straight from the Windows path shape (sibling of
# MIND under workspace/world).
DREAMX_REPO="$HERE/../DreamX-World"
DREAMX_VENV_PY="${DREAMX_VENV_PY:-$DREAMX_REPO/.venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  echo "The MIND venv runs staging + scoring. Create it (separate from DreamX-World) first." >&2
  exit 2
fi
if [ ! -x "$DREAMX_VENV_PY" ]; then
  echo "ERROR: DreamX-World venv python not found: $DREAMX_VENV_PY" >&2
  exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

# Resolve the Wan2.2 base + DreamX-World-5B (AR) checkpoint from the HF cache,
# using the DreamX venv (it has huggingface_hub).
echo "Resolving Wan2.2-TI2V-5B base from HF cache..."
WAN_BASE="$("$DREAMX_VENV_PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))")"
echo "Resolving DreamX-World-5B (AR) checkpoint from HF cache..."
AR_CKPT="$("$DREAMX_VENV_PY" -c "import glob,os; from huggingface_hub import snapshot_download; d=snapshot_download('GD-ML/DreamX-World-5B'); print(glob.glob(os.path.join(d,'**','*.safetensors'),recursive=True)[0])")"

if [ -z "$WAN_BASE" ]; then
  echo "ERROR: could not resolve Wan base" >&2
  exit 2
fi
if [ -z "$AR_CKPT" ]; then
  echo "ERROR: could not resolve DreamX-World-5B AR checkpoint - run: python download_models.py --only dreamx_ar" >&2
  exit 2
fi

# AR model is native 16fps; mirror-test on by default (drives the gsc metric).
MIND_FPS="${MIND_FPS:-16}"
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

# Perspective: set MIND_PERSPECTIVE=1st_data (or 3rd_data) to limit to one.
# Unset (default) processes BOTH 1st-person and 3rd-person samples.
PERSP_ARG=()
if [ -n "${MIND_PERSPECTIVE:-}" ]; then
  PERSP_ARG=(--perspective "$MIND_PERSPECTIVE")
fi

echo "============================================================"
echo "DreamX-World-5B (AR / long-horizon) staging into MIND-tests"
echo "============================================================"
echo "  gt_root   : $GT_ROOT"
echo "  test_root : $MIND_TESTS"
echo "  model     : $MODEL_NAME"
echo "  wan_base  : $WAN_BASE"
echo "  ar_ckpt   : $AR_CKPT"
echo "  log       : $LOG"
echo "============================================================"

set +e
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_dreamx_ar.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--wan-base" "$WAN_BASE" "--base-checkpoint" "$AR_CKPT" "--fps" "$MIND_FPS" "${PERSP_ARG[@]}" "${MIRROR_ARG[@]}" "$@"
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" != 0 ]; then
  echo "ERROR: drive_dreamx_ar.py exited with $EXIT_CODE" >&2
  exit "$EXIT_CODE"
fi

echo "Generation done. Running scoring: run_mind.sh $MODEL_NAME"
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,action,gsc
fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"
