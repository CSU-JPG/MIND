#!/bin/bash
# Stage PKU-YuanGroup/Helios (i2v) videos into MIND-tests/helios-i2v/ for
# run_mind.sh scoring. Parallel to drive_matrix3.sh / drive_matrix3_distilled.sh.
#
# Usage:
#   drive_helios_i2v.sh                            stage 1st + 3rd, score both
#   drive_helios_i2v.sh --dry-run                  preview commands
#   drive_helios_i2v.sh --limit 5                  first 5 samples per perspective
#   drive_helios_i2v.sh --test-type mem_test       limit to memory tests
#   drive_helios_i2v.sh --perspective 1st_data     override (default = both)
#   drive_helios_i2v.sh --low-vram                 pass --enable_low_vram_mode to Helios
#
# Metric selection (forwarded to run_mind.sh after staging):
#   MIND_METRICS=lcm,visual                pick a subset
#   (unset)                                default = lcm,visual,dino,gsc (no action)
#   MIND_GPUS=2                            multi-GPU scoring
#   MIND_PERSON=1st                        person = 1st | 3rd | both (default both)
#   MIND_MIRROR_TEST=0                     disable mirror_test (default on)
#   MIND_START_INDEX=N                     resume mid-run
#   MIND_FPS=16                            override generation fps (default 24)
#
# Cross-venv knobs:
#   HELIOS_VENV_PY=<path>                  override Helios venv python
#                                          (default $HERE/../Helios/.venv/bin/python)
#
# All --flags pass through to src/drive_helios_i2v.py; MIND_* env vars stay in this script.
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
LOG="$HERE/drive_helios_i2v.log"

# NOTE: Helios repo/venv location on this Linux box has not been verified --
# translated straight from the Windows path shape (sibling of MIND).
HELIOS_VENV_PY="${HELIOS_VENV_PY:-$HERE/../Helios/.venv/bin/python}"

MIND_FPS="${MIND_FPS:-24}"

# Force variant to base (single-stage full denoising). Mirrors run_helios.sh's
# default. Override with `HELIOS_VARIANT=distilled` BEFORE invoking this
# script. Without this explicit default, an env-inherited value could be
# wrong (we saw a session env still carrying "distilled" from a prior
# run_helios.bat run on the Windows box).
HELIOS_VARIANT="${HELIOS_VARIANT:-base}"

# Per-variant num_inference_steps default to MATCH run_helios.bat's variant
# logic: Distilled uses ~4 pyramid-distilled steps with stage2 ON; Base/Mid use
# ~30 full denoise steps with stage2 OFF. drive_helios_i2v.py defaults to 4
# (Distilled-tuned); we override here when the variant is Base/Mid.
if [ -z "${MIND_NUM_INFERENCE_STEPS:-}" ]; then
  if [ "$(echo "$HELIOS_VARIANT" | tr '[:upper:]' '[:lower:]')" = "distilled" ]; then
    MIND_NUM_INFERENCE_STEPS=4
  else
    MIND_NUM_INFERENCE_STEPS=30
  fi
fi

# Default to --low-vram (group offload). Matches run_helios.bat's default and
# is required for Helios-Base on a 32 GB 5090 -- without it the transformer
# + activations exhaust VRAM and OOM during the first denoise step. To opt
# out (e.g. on an A6000 / L40S / this GB300's larger HBM), pass --high-vram
# on the cmd line.
_LOW_VRAM_DEFAULT=""
if [[ " $* " != *" --high-vram "* ]] && [[ " $* " != *" --low-vram "* ]]; then
  _LOW_VRAM_DEFAULT="--low-vram"
fi

# Prompt suffix gets appended to every per-sample prompt (after the
# action.json caption or perspective default). The MIND benchmark's
# action_space_test samples don't bake motion descriptions into their
# captions, so feeding Helios "moving forward" gives a steering signal
# that improves i2v motion coherence. Override via MIND_PROMPT_SUFFIX env
# (set to empty string to suppress entirely).
MIND_PROMPT_SUFFIX="${MIND_PROMPT_SUFFIX-moving forward}"

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 2
fi
if [ ! -x "$HELIOS_VENV_PY" ]; then
  echo "ERROR: Helios venv python not found: $HELIOS_VENV_PY" >&2
  echo "Run: $HERE/../Helios/run_helios.sh --setup --skip-run" >&2
  exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

echo "============================================================"
echo "Helios (i2v) staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : helios-i2v"
echo "  helios_py    : $HELIOS_VENV_PY"
echo "  variant      : $HELIOS_VARIANT  (matches run_helios.sh --variant)"
echo "  steps        : $MIND_NUM_INFERENCE_STEPS  (per-variant default; override with MIND_NUM_INFERENCE_STEPS=N)"
echo "  prompt suffix: $MIND_PROMPT_SUFFIX  (override with MIND_PROMPT_SUFFIX=...)"
echo "  fps          : $MIND_FPS"
echo "  log          : $LOG"
echo "============================================================"

MIND_START_INDEX="${MIND_START_INDEX:-0}"
MIND_MIRROR_TEST="${MIND_MIRROR_TEST:-1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

# ---- start wall-clock + RAM/GPU sampling ---------------------------------
_T_START=$(date +%s)

# Snapshot pre-staging .mp4 count + frame count assumption so we can compute
# real-fps (frames generated per wall-clock second) after staging finishes.
# Frames-per-video defaults to 99 (Helios default); override via MIND_NUM_FRAMES
# or by passing --num_frames N in "$@" (the latter is forwarded to Python).
MIND_NUM_FRAMES="${MIND_NUM_FRAMES:-99}"
_MP4_COUNT_BEFORE=$(find "$MIND_TESTS/helios-i2v" -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')

_METRICS_FILE="$(mktemp -t helios_drive_ram.XXXXXX)"
_GPU_FILE="$(mktemp -t helios_drive_gpu.XXXXXX)"
echo "0 0" > "$_METRICS_FILE"
echo "0" > "$_GPU_FILE"

# Background RAM sampler: track peak RSS (KB) across all python processes.
(
  peak=0
  while true; do
    m=$(ps -C python -C python3 -o rss= 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [ -n "$m" ] && [ "$m" -gt "$peak" ] 2>/dev/null; then
      peak=$m
      awk -v p="$peak" 'BEGIN{printf "%d %.2f\n", p, p/1024/1024}' > "$_METRICS_FILE"
    fi
    sleep 2
  done
) &
_RAM_SAMPLER_PID=$!

# Background GPU sampler: track peak used VRAM (MiB) via nvidia-smi.
(
  peak=0
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    if [ -n "$m" ] && [ "$m" -gt "$peak" ] 2>/dev/null; then
      peak=$m
      echo "$peak" > "$_GPU_FILE"
    fi
    sleep 2
  done
) &
_GPU_SAMPLER_PID=$!

# ---- staging: ONE worker process handles BOTH perspectives --------------
# drive_helios_i2v.py spawns a single persistent Helios worker that loads the
# model once and iterates the whole manifest with no further reloads. With no
# --perspective it stages both 1st_data + 3rd_data in that one process; if the
# caller passes --perspective in "$@" the driver filters to it. Either way the
# model loads exactly once. Do NOT loop perspectives here -- that respawns the
# worker and reloads the model per perspective.
set +e
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_helios_i2v.py" "--gt-root" "$GT_ROOT" "--test-root" "$MIND_TESTS" "--fps" "$MIND_FPS" "--start-index" "$MIND_START_INDEX" "--num-inference-steps" "$MIND_NUM_INFERENCE_STEPS" "--prompt-suffix" "$MIND_PROMPT_SUFFIX" $_LOW_VRAM_DEFAULT "${MIRROR_ARG[@]}" "$@"
EXIT_CODE=$?
set -e

# Stop samplers regardless of success/failure
kill "$_RAM_SAMPLER_PID" "$_GPU_SAMPLER_PID" 2>/dev/null || true
wait "$_RAM_SAMPLER_PID" "$_GPU_SAMPLER_PID" 2>/dev/null || true

_T_END=$(date +%s)
_T_ELAPSED_SEC=$((_T_END - _T_START))
_T_ELAPSED=$(printf '%02d:%02d:%02d' $((_T_ELAPSED_SEC/3600)) $((_T_ELAPSED_SEC%3600/60)) $((_T_ELAPSED_SEC%60)))

_RAM_PEAK_GB=$(awk '{print $2}' "$_METRICS_FILE")
[ -z "$_RAM_PEAK_GB" ] && _RAM_PEAK_GB="?"
_GPU_PEAK_MIB=$(cat "$_GPU_FILE")
[ -z "$_GPU_PEAK_MIB" ] && _GPU_PEAK_MIB="?"
rm -f "$_METRICS_FILE" "$_GPU_FILE"

# Compute real fps = (videos_generated_this_run * frames_per_video) / elapsed_seconds
_MP4_COUNT_AFTER=$(find "$MIND_TESTS/helios-i2v" -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
_MP4_DELTA=$((_MP4_COUNT_AFTER - _MP4_COUNT_BEFORE))
if [ "$_T_ELAPSED_SEC" -gt 0 ] && [ "$_MP4_DELTA" -gt 0 ]; then
  _REAL_FPS=$(awk -v d="$_MP4_DELTA" -v f="$MIND_NUM_FRAMES" -v s="$_T_ELAPSED_SEC" 'BEGIN{printf "%.2f", (d*f)/s}')
  _SEC_PER_VIDEO=$(awk -v d="$_MP4_DELTA" -v s="$_T_ELAPSED_SEC" 'BEGIN{printf "%.1f", s/d}')
else
  _REAL_FPS=0
  _SEC_PER_VIDEO=0
fi

echo
echo "--- staging elapsed: $_T_ELAPSED  | peak python RAM: $_RAM_PEAK_GB GB  | peak GPU VRAM: $_GPU_PEAK_MIB MiB ---"
echo "--- videos generated: $_MP4_DELTA  | ~$_SEC_PER_VIDEO s/video  | real fps: $_REAL_FPS frames/sec wall-clock ---"

if [ "$EXIT_CODE" != 0 ]; then
  echo
  echo "ERROR: staging failed with exit code $EXIT_CODE" >&2
  exit "$EXIT_CODE"
fi

MIND_PERSON="${MIND_PERSON:-both}"
MIND_METRICS="${MIND_METRICS:-lcm,visual,dino,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,gsc
fi
MIND_GPUS="${MIND_GPUS:-1}"

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh helios-i2v \"$MIND_METRICS\" $MIND_GPUS $MIND_PERSON"
echo "============================================================"
bash "$HERE/run_mind.sh" helios-i2v "$MIND_METRICS" "$MIND_GPUS" "$MIND_PERSON"
