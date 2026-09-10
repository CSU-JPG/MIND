#!/bin/bash
# Stage HY-WorldPlay videos into MIND-tests/hy-worldplay/.
#
# HY-WorldPlay = Tencent's autoregressive-distilled world model living at
# (sibling of MIND) HY-WorldPlay/. Distinct from HY-World 2.0 (driven by
# drive_hy_world.sh) and HY-Playground (driven by drive_hy_playground.sh).
#
# TODO: needs src/drive_hy_worldplay.py that:
#   - Walks MIND-Data first frames + action.json
#   - Converts per-frame ws/ad/ud/lr -> HY-WorldPlay pose string (see
#     hyvideo/generate.py:parse_pose_string -- "w-N,d-N,up-N,..." comma-sep,
#     duration is in latents not frames).
#   - Invokes HY-WorldPlay/hyvideo/generate.py via torch.distributed.run
#     (mirrors HY-WorldPlay/run.bat's Linux equivalent). Per-sample cmd is
#     heavy -- ideally the driver keeps the model resident across samples or
#     batches them.
#   - Sources MODEL_PATH + AR_DISTILL_ACTION_MODEL_PATH from
#     HY-WorldPlay/paths.sh (written by setup.sh).
#   - Stages mp4 to MIND-tests/hy-worldplay/<perspective>/<test_type>/<gt_name>/video.mp4
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

PY="$HERE/.venv/bin/python"
# HY-WorldPlay has its own venv (Python 3.10 + flash-attn + sageattention).
# MIND's .venv does NOT satisfy those deps. Cross-project path, sibling of
# MIND under the same world/ root -- not verified to exist/be set up on this box.
HY_WORLDPLAY_REPO="${HY_WORLDPLAY_REPO:-$HERE/../HY-WorldPlay}"
HY_WORLDPLAY_PY="${HY_WORLDPLAY_PY:-$HY_WORLDPLAY_REPO/.venv/bin/python}"
GT_ROOT="${GT_ROOT:-$HERE/../MIND-Data}"
MIND_TESTS="${MIND_TESTS:-$HERE/../MIND-tests}"
MODEL_NAME="hy-worldplay"
LOG="$HERE/drive_hy_worldplay.log"
MIND_FPS="${MIND_FPS:-24}"

if [ ! -x "$PY" ]; then echo "ERROR: MIND venv python not found: $PY" >&2; exit 2; fi
if [ ! -e "$HY_WORLDPLAY_REPO" ]; then echo "ERROR: HY-WorldPlay repo not found: $HY_WORLDPLAY_REPO" >&2; exit 2; fi
if [ ! -x "$HY_WORLDPLAY_PY" ]; then
    echo "ERROR: HY-WorldPlay venv python not found: $HY_WORLDPLAY_PY   Run HY-WorldPlay/setup.sh first." >&2
    exit 2
fi

# paths.sh is written by HY-WorldPlay/setup.sh after download_models.py runs.
# It defines MODEL_PATH + AR_DISTILL_ACTION_MODEL_PATH (caller-set env wins).
if [ -e "$HY_WORLDPLAY_REPO/paths.sh" ]; then
    # shellcheck disable=SC1091
    source "$HY_WORLDPLAY_REPO/paths.sh"
fi
if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH not set.  Expected $HY_WORLDPLAY_REPO/paths.sh from setup.sh" >&2
    echo "OR explicitly: MODEL_PATH=<path to HunyuanVideo-1.5>" >&2
    exit 2
fi
if [ -z "${AR_DISTILL_ACTION_MODEL_PATH:-}" ]; then
    echo "ERROR: AR_DISTILL_ACTION_MODEL_PATH not set.  Expected $HY_WORLDPLAY_REPO/paths.sh from setup.sh" >&2
    echo "OR explicitly: AR_DISTILL_ACTION_MODEL_PATH=<path to ar_distilled_action_model/diffusion_pytorch_model.safetensors>" >&2
    exit 2
fi

if [ ! -e "$HERE/src/drive_hy_worldplay.py" ]; then echo "ERROR: src/drive_hy_worldplay.py not yet written" >&2; exit 2; fi

# Mirror-test generation drives the gsc metric (per-gt_name mirror_test mp4s).
# On by default; set MIND_MIRROR_TEST=0 to skip.
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then MIRROR_ARG=(--mirror-test); fi

# Default prompt style when a sample has no explicit prompt (default|cartoony).
# Override: MIND_PROMPT_VARIANT=cartoony
MIND_PROMPT_VARIANT="${MIND_PROMPT_VARIANT:-default}"

echo "============================================================"
echo "HY-WorldPlay staging into MIND-tests  |  model=$MODEL_NAME  |  log=$LOG"
echo "============================================================"
echo "  MIND py             : $PY"
echo "  HY-WorldPlay py     : $HY_WORLDPLAY_PY"
echo "  HY-WorldPlay repo   : $HY_WORLDPLAY_REPO"
echo "  MODEL_PATH          : $MODEL_PATH"
echo "  AR_DISTILL_ACTION   : $AR_DISTILL_ACTION_MODEL_PATH"
echo "  fps                 : $MIND_FPS"
echo "  mirror_test         : $MIND_MIRROR_TEST"
echo "  prompt_variant      : $MIND_PROMPT_VARIANT"
echo "============================================================"

# Perspective(s) + per-perspective sample cap. Default: BOTH (1st_data + 3rd_data),
# 50 samples each. drive_hy_worldplay.py's --limit slices AFTER the perspective
# filter, so a single both-perspective pass with --limit 50 would yield 50 total
# (all 1st). Run one pass per perspective to get 50 EACH. Override:
#   MIND_PERSPECTIVE=1st_data (single perspective)   HY_LIMIT=N
HY_LIMIT="${HY_LIMIT:-50}"
PERSP_LIST=(1st_data 3rd_data)
SCORE_PERSON="both"
MIND_PERSPECTIVE_LC="$(echo "${MIND_PERSPECTIVE:-}" | tr '[:upper:]' '[:lower:]')"
if [ "$MIND_PERSPECTIVE_LC" = "1st_data" ]; then PERSP_LIST=(1st_data); SCORE_PERSON="1st"; fi
if [ "$MIND_PERSPECTIVE_LC" = "3rd_data" ]; then PERSP_LIST=(3rd_data); SCORE_PERSON="3rd"; fi

for P in "${PERSP_LIST[@]}"; do
    echo
    echo "=== staging perspective $P (limit $HY_LIMIT) ==="
    echo
    "$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_hy_worldplay.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--hy-worldplay-repo" "$HY_WORLDPLAY_REPO" "--hy-worldplay-py" "$HY_WORLDPLAY_PY" "--model-path" "$MODEL_PATH" "--action-ckpt" "$AR_DISTILL_ACTION_MODEL_PATH" "--fps" "$MIND_FPS" "--perspective" "$P" "--limit" "$HY_LIMIT" "--prompt-variant" "$MIND_PROMPT_VARIANT" "${MIRROR_ARG[@]}" "$@" || {
        echo
        echo "ERROR: drive_hy_worldplay.py failed for $P" >&2
        exit 1
    }
done

echo
echo "=== Running scoring: run_mind.sh $MODEL_NAME $SCORE_PERSON ==="
echo
# Explicit metric list including gsc (mirror_test mp4s needed -- enabled above) and
# action (ViPE). Score the SAME perspective(s) we staged.
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then MIND_METRICS="lcm,visual,dino,action,gsc"; fi
MIND_GPUS="${MIND_GPUS:-1}"
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS" "$MIND_GPUS" "$SCORE_PERSON"
