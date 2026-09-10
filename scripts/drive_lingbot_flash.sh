#!/bin/bash
# Stage flashdreams-lingbot videos into MIND-tests/lingbot-flash/ for run_mind.sh scoring.
#
# Uses the new flashdreams plugin and runs ENTIRELY in-process: the 14B
# lingbot pipeline loads ONCE in src/drive_lingbot_flash.py and stays
# resident across all MIND samples. Subprocess-per-sample would reload the
# model every iteration -- a non-starter at ~100 samples.
#
# Two flashdreams-lingbot slugs are supported:
#   lingbot-world-fast                       (Wan VAE decoder, 4-step) -- default
#   lingbot-world-fast-taehv-window15-sink3  (LightTAE decoder, tighter streaming window)
#
# Set the slug via LINGBOT_SLUG env var:
#   LINGBOT_SLUG=lingbot-world-fast-taehv-window15-sink3
#
# Output mp4 fps is fixed at 24 (matches MIND-Data ground truth).
#
# Usage:
#   bash drive_lingbot_flash.sh                              stage all samples then score
#   bash drive_lingbot_flash.sh --dry-run                    preview commands
#   bash drive_lingbot_flash.sh --limit 5                    first 5 samples only
#   bash drive_lingbot_flash.sh --perspective 1st_data       limit to first-person
#   bash drive_lingbot_flash.sh --test-type mem_test         limit to memory tests
#   bash drive_lingbot_flash.sh --total-blocks 32            longer video
#
# All flags pass through to src/drive_lingbot_flash.py.
#
# NOTE: action metric will be loose. MIND's action.json (WASD) is NOT
# converted -- we synthesize a dummy forward-walk camera trajectory.
# lcm/visual/dino/gsc are meaningful.
#
# NOTE (Linux port): this script's `uv` usage is NOT for MIND's own venv (that
# stays plain python3 -m venv per setup_mind_venv.sh) -- it's calling into the
# flashdream_public sibling project's own uv-based workspace, a separate
# concern from this repo's venv policy.
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

# Pin CUDA toolkit for triton to match the flashdreams-lingbot torch build.
# Windows original pinned cu128 explicitly; this box's actual CUDA toolkit
# layout for this stack has NOT been verified -- adjust if it differs.
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_PATH="/usr/local/cuda-12.8"
    export CUDA_HOME="/usr/local/cuda-12.8"
    export PATH="/usr/local/cuda-12.8/bin:$PATH"
fi

export TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# lingbot-world-fast (~74GB) is already cached; skip flashdreams' 200GB disk preflight
# (it checks free space before reusing the cache and would otherwise abort on a full disk).
export FLASHDREAMS_MIN_CACHE_FREE_GB=0

# Serialize shard download. Parallel processes each loading torch segfaulted on
# the Windows box (access violation); 1 worker = serial = safe. Kept as-is here.
export FLASHDREAMS_HF_SHARD_DOWNLOAD_WORKERS=1

# Disable hf-xet (Rust downloader) -- it access-violated on Windows during shard
# resolution, which still crashed even with the in-process serial download.
export HF_HUB_DISABLE_XET=1
export HF_XET_HIGH_PERFORMANCE=0

# Strip ambient venv state -- we're spawning into flashdreams's uv env, not MIND's.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH UV_PYTHON UV_PROJECT_ENVIRONMENT

# Pin lingbot to Python 3.10 (uv defaults the workspace to a newer version; we
# avoid 3.12/3.13 per this repo's Windows-side gotcha notes). First run on 3.10
# rebuilds the env (one-time, heavy). If a workspace member requires >3.10 and
# resolution fails, drop this line -- the in-process shard patch already makes
# newer Python work, so the pin is optional.
export UV_PYTHON=3.10

UV_EXE="${UV_EXE:-uv}"
# Cross-project path, sibling of MIND under the same world/ root (not
# verified to exist/be set up on this Linux box).
FLASHDREAMS="${FLASHDREAMS:-$HERE/../flashdream_public}"
DRIVE_PY="$HERE/src/drive_lingbot_flash.py"

GT_ROOT="${GT_ROOT:-$HERE/../MIND-Data}"
MIND_TESTS="${MIND_TESTS:-$HERE/../MIND-tests}"
LINGBOT_SLUG="${LINGBOT_SLUG:-lingbot-world-fast}"

MODEL_NAME="lingbot-flash"
if [ "$LINGBOT_SLUG" = "lingbot-world-fast-taehv-window15-sink3" ]; then MODEL_NAME="lingbot-flash-taehv"; fi

LOG="$HERE/drive_lingbot_flash.log"
MIND_FPS="${MIND_FPS:-24}"

if ! command -v "$UV_EXE" >/dev/null 2>&1 && [ ! -x "$UV_EXE" ]; then
    echo "ERROR: uv not found at $UV_EXE" >&2
    exit 2
fi
if [ ! -e "$FLASHDREAMS/integrations/lingbot" ]; then
    echo "ERROR: flashdreams-lingbot plugin not found at $FLASHDREAMS/integrations/lingbot" >&2
    exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
    echo "ERROR: gt_root not found: $GT_ROOT" >&2
    exit 2
fi

echo "============================================================"
echo "flashdreams-lingbot (in-process, model loads ONCE)"
echo "============================================================"
echo "  gt_root   : $GT_ROOT"
echo "  test_root : $MIND_TESTS"
echo "  model     : $MODEL_NAME"
echo "  slug      : $LINGBOT_SLUG"
echo "  fps       : $MIND_FPS"
echo "  log       : $LOG"
echo "============================================================"

# Mirror-test generation drives the gsc metric.
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then MIRROR_ARG=(--mirror-test); fi

# Perspective: default BOTH (1st_data + 3rd_data). Override a single perspective
# with MIND_PERSPECTIVE=1st_data (or 3rd_data). The driver's gather_samples
# iterates both perspectives when --perspective is omitted.
PERSP_ARG=()
SCORE_PERSON="both"
MIND_PERSPECTIVE_LC="$(echo "${MIND_PERSPECTIVE:-}" | tr '[:upper:]' '[:lower:]')"
if [ "$MIND_PERSPECTIVE_LC" = "1st_data" ]; then PERSP_ARG=(--perspective 1st_data); SCORE_PERSON="1st"; fi
if [ "$MIND_PERSPECTIVE_LC" = "3rd_data" ]; then PERSP_ARG=(--perspective 3rd_data); SCORE_PERSON="3rd"; fi

# Run the driver INSIDE flashdreams's uv env so lingbot.* + flashdreams.* import.
# cd into the flashdreams repo so uv resolves the workspace correctly.
LINGBOT_LIMIT="${LINGBOT_LIMIT:-30}"
EXIT_CODE=0
(
    cd "$FLASHDREAMS"
    "$UV_EXE" run --with psutil --package flashdreams-lingbot python "$DRIVE_PY" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" --model-name "$MODEL_NAME" --slug "$LINGBOT_SLUG" --fps "$MIND_FPS" --limit "$LINGBOT_LIMIT" "${PERSP_ARG[@]}" "${MIRROR_ARG[@]}" "$@"
) || EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo
    echo "ERROR: drive_lingbot_flash.py exited with $EXIT_CODE" >&2
    exit "$EXIT_CODE"
fi

echo
echo "============================================================"
echo "Staging done. Scoring with run_mind.sh $MODEL_NAME ..."
echo "============================================================"
echo

MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then MIND_METRICS="lcm,visual,dino,action,gsc"; fi
MIND_GPUS="${MIND_GPUS:-1}"
# Score the same perspective(s) we generated (default both 1st + 3rd).
SCORE_EXIT=0
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS" "$MIND_GPUS" "$SCORE_PERSON" || SCORE_EXIT=$?

if [ "$SCORE_EXIT" -ne 0 ]; then
    echo
    echo "ERROR: run_mind.sh exited with $SCORE_EXIT" >&2
    exit "$SCORE_EXIT"
fi
echo
echo "Done. See result_${MODEL_NAME}_*.json in this directory."
