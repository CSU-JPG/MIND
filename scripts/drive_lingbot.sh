#!/bin/bash
# Stage LingBot-World base-cam-nf4 videos into MIND-tests/lingbot-base-cam-nf4/ for run_mind.sh scoring.
#
# Uses lingbot-world's generate.py with the MIND action.json passed through
# --action_path, so per-frame WASD/ud/lr conditioning flows end-to-end. That
# makes the `action` MIND metric meaningful for this model (unlike the
# matrix3 / dreamx drivers which stub a single placeholder action).
#
# Usage:
#   bash drive_lingbot.sh                            stage all samples then score
#   bash drive_lingbot.sh --dry-run                  preview commands without running inference
#   bash drive_lingbot.sh --limit 5                  first 5 samples only
#   bash drive_lingbot.sh --perspective 1st_data     limit to first-person
#   bash drive_lingbot.sh --test-type mem_test       limit to memory tests
#
# All flags pass through to src/drive_lingbot.py.
#
# NOTE (Linux port): MIND_METRICS below defaults to including "action" (ViPE).
# Per setup_mind_venv.sh's note, ViPE's build chain is Windows/cp310-specific
# (no aarch64 flash_attn wheel), so "action" is likely unavailable on this box.
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

# Pin CUDA toolkit for triton to match lingbot-world's torch build (Windows
# original pinned cu128 explicitly since v13.0 was also installed there and
# triton's JIT would otherwise pick v13.0 via CUDA_PATH, mismatching the
# cu128-built torch ABI). This box's actual CUDA toolkit layout/version for
# lingbot-world's torch has NOT been verified -- /usr/local/cuda-12.8 is the
# conventional Linux location if that toolkit is installed side-by-side with
# others; adjust if lingbot-world here actually uses a different CUDA build.
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_PATH="/usr/local/cuda-12.8"
    export CUDA_HOME="/usr/local/cuda-12.8"
    export PATH="/usr/local/cuda-12.8/bin:$PATH"
fi

# Disable torch.compile/dynamo -- triton + inductor was flaky on the Windows
# box; kept off here too pending verification on this box.
export TORCHDYNAMO_DISABLE=1

# Keep huggingface_hub from re-checking HF for tokenizer/model updates on every
# run; rely on local cache. Stops corp-proxy stalls during AutoTokenizer.from_pretrained.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY="$HERE/.venv/bin/python"
GT_ROOT="${GT_ROOT:-$HERE/../MIND-Data}"
MIND_TESTS="${MIND_TESTS:-$HERE/../MIND-tests}"
# Path A: use the fast-mini-cam ckpt + generate_fast.py. The previous
# base-cam-nf4 + generate.py path failed on every sample (OSError: missing
# config.json in the NF4 release). fast-mini-cam ships a complete layout
# and produces mp4s out of the box; action-metric quality will be loose
# since action_path is dropped (see drive_lingbot.py for the Path B plan).
MODEL_NAME="lingbot-fast"
# Cross-project path, sibling of MIND under the same world/ root (not
# verified to exist/be set up on this Linux box).
CKPT_DIR="${CKPT_DIR:-$HERE/../lingbot-world/fast-mini-cam}"
LOG="$HERE/drive_lingbot.log"

# lingbot fps is set via wan/configs/shared_config.py (sample_fps=24 patched in).
# MIND_FPS here is for traceability only; lingbot doesn't expose a CLI --fps flag.
MIND_FPS="${MIND_FPS:-24}"

if [ ! -x "$PY" ]; then
    echo "ERROR: venv python not found: $PY" >&2
    exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
    echo "ERROR: gt_root not found: $GT_ROOT" >&2
    exit 2
fi
if [ ! -e "$CKPT_DIR" ]; then
    echo "WARNING: ckpt_dir not present: $CKPT_DIR"
    echo "Run download_fast.sh in lingbot-world first (unless using --dry-run)."
    echo
fi

echo "============================================================"
echo "LingBot-World-Fast staging into MIND-tests"
echo "============================================================"
echo "  gt_root   : $GT_ROOT"
echo "  test_root : $MIND_TESTS"
echo "  model     : $MODEL_NAME"
echo "  ckpt_dir  : $CKPT_DIR"
echo "  log       : $LOG"
echo "============================================================"

# --perspective 1st_data: lingbot-fast only stages first-person samples.
# To include 3rd_data, pass an overriding `--perspective 3rd_data` as an
# extra arg -- argparse's "last wins" rule lets the user override.

# Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
# On by default; set MIND_MIRROR_TEST=0 to skip.
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then MIRROR_ARG=(--mirror-test); fi

"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_lingbot.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--ckpt-dir" "$CKPT_DIR" "--fps" "$MIND_FPS" "--perspective" "1st_data" "${MIRROR_ARG[@]}" "$@"

echo
echo "============================================================"
echo "Staging done. Scoring with run_mind.sh $MODEL_NAME ..."
echo "============================================================"
echo

# run_mind.sh defaults PERSON=1st, matching this script's 1st_data-only generation.
# gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then MIND_METRICS="lcm,visual,dino,action,gsc"; fi
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS"

echo
echo "Done. See result_${MODEL_NAME}_*.json in this directory."
