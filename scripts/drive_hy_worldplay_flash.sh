#!/bin/bash
# Stage HY-WorldPlay videos via the FLASHDREAMS runner (distilled Wan2.2-5B)
# into MIND-tests/hy-worldplay-flash/.
#
# Unlike drive_hy_worldplay.sh (upstream hyvideo/generate.py + HunyuanVideo-1.5
# base + byT5 Glyph + siglip + torchrun), this drives
#   flashdreams-run hy-worldplay-wan-i2v-5b
# via src/drive_hy_worldplay.py --backend flashdreams. The runner self-resolves
# HY-WorldPlay's distilled WAN-5B checkpoint, so NO MODEL_PATH / action ckpt /
# HunyuanVideo-1.5 base are needed -- it sidesteps the byT5/siglip/distributed
# failures entirely. The MIND pose string maps straight to the runner's --pose.
#
# Prereq: flashdreams workspace synced (uv sync in <world>/flashdream_public
# so the flashdreams-hy_worldplay package + flashdreams-run entry-point exist).
#
# NOTE (Linux port): this script's `uv` usage is NOT for MIND's own venv (that
# stays plain python3 -m venv per setup_mind_venv.sh) -- it's calling into the
# flashdream_public sibling project's own uv-based workspace/runner, which is
# a separate concern from this repo's venv policy. UV_EXE defaults to `uv` on
# PATH here rather than a hardcoded Windows path; override if needed.
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
GT_ROOT="${GT_ROOT:-$HERE/../MIND-Data}"
MIND_TESTS="${MIND_TESTS:-$HERE/../MIND-tests}"
MODEL_NAME="hy-worldplay-flash"
# Cross-project paths, siblings of MIND under the same world/ root -- not
# verified to exist/be set up on this box.
HY_WORLDPLAY_REPO="${HY_WORLDPLAY_REPO:-$HERE/../HY-WorldPlay}"
FLASHDREAMS_REPO="${FLASHDREAMS_REPO:-$HERE/../flashdream_public}"
LOG="$HERE/drive_hy_worldplay_flash.log"
MIND_FPS="${MIND_FPS:-24}"
HY_NUM_CHUNK="${HY_NUM_CHUNK:-4}"
UV_EXE="${UV_EXE:-uv}"

if [ ! -x "$PY" ]; then echo "ERROR: MIND venv python not found: $PY" >&2; exit 2; fi
if [ ! -e "$FLASHDREAMS_REPO" ]; then echo "ERROR: flashdreams repo not found: $FLASHDREAMS_REPO" >&2; exit 2; fi
if ! command -v "$UV_EXE" >/dev/null 2>&1 && [ ! -x "$UV_EXE" ]; then
    echo "ERROR: uv not found: $UV_EXE" >&2
    exit 2
fi
if [ ! -e "$HERE/src/drive_hy_worldplay.py" ]; then echo "ERROR: src/drive_hy_worldplay.py missing" >&2; exit 2; fi

# Mirror-test generation drives the gsc metric.
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then MIRROR_ARG=(--mirror-test); fi

# Default prompt style when a sample has no explicit prompt (default|cartoony).
# Override: MIND_PROMPT_VARIANT=cartoony
MIND_PROMPT_VARIANT="${MIND_PROMPT_VARIANT:-default}"

echo "============================================================"
echo "HY-WorldPlay (FLASHDREAMS backend) -> MIND-tests  |  model=$MODEL_NAME"
echo "  MIND py        : $PY"
echo "  flashdreams    : $FLASHDREAMS_REPO"
echo "  uv             : $UV_EXE"
echo "  num_chunk      : $HY_NUM_CHUNK   fps: $MIND_FPS   log: $LOG"
echo "  prompt_variant : $MIND_PROMPT_VARIANT"
echo "============================================================"

# Perspective(s) + per-perspective cap. Default BOTH (1st_data + 3rd_data), 50 each.
# Override: MIND_PERSPECTIVE=1st_data   HY_LIMIT=N
HY_LIMIT="${HY_LIMIT:-50}"
PERSP_LIST=(1st_data 3rd_data)
SCORE_PERSON="both"
MIND_PERSPECTIVE_LC="$(echo "${MIND_PERSPECTIVE:-}" | tr '[:upper:]' '[:lower:]')"
if [ "$MIND_PERSPECTIVE_LC" = "1st_data" ]; then PERSP_LIST=(1st_data); SCORE_PERSON="1st"; fi
if [ "$MIND_PERSPECTIVE_LC" = "3rd_data" ]; then PERSP_LIST=(3rd_data); SCORE_PERSON="3rd"; fi

# --hy-worldplay-repo/-py + --model-path/--action-ckpt are required by the driver's
# argparse but UNUSED on the flashdreams path; pass existing placeholders to satisfy it.
for P in "${PERSP_LIST[@]}"; do
    echo
    echo "=== staging perspective $P (limit $HY_LIMIT, flashdreams) ==="
    echo
    "$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_hy_worldplay.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--model-name" "$MODEL_NAME" "--backend" "flashdreams" "--flashdreams-repo" "$FLASHDREAMS_REPO" "--num-chunk" "$HY_NUM_CHUNK" "--hy-worldplay-repo" "$HY_WORLDPLAY_REPO" "--hy-worldplay-py" "$PY" "--model-path" "unused" "--action-ckpt" "unused" "--fps" "$MIND_FPS" "--perspective" "$P" "--limit" "$HY_LIMIT" "--prompt-variant" "$MIND_PROMPT_VARIANT" "${MIRROR_ARG[@]}" "$@" || {
        echo
        echo "ERROR: drive_hy_worldplay.py --backend flashdreams failed for $P" >&2
        exit 1
    }
done

echo
echo "=== Running scoring: run_mind.sh $MODEL_NAME $SCORE_PERSON ==="
echo
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,action,gsc}"
if [ -z "$MIND_METRICS" ]; then MIND_METRICS="lcm,visual,dino,action,gsc"; fi
MIND_GPUS="${MIND_GPUS:-1}"
bash "$HERE/run_mind.sh" "$MODEL_NAME" "$MIND_METRICS" "$MIND_GPUS" "$SCORE_PERSON"
