#!/bin/bash
# Score a test video set against MIND-Data ground truth.
#
# Usage:
#   bash run_mind.sh                                          scores matrix-game-3, all metrics, both perspectives
#   bash run_mind.sh <test_subdir>                            scores MIND-tests/<test_subdir>
#   bash run_mind.sh <test_subdir> <metrics>                  custom metrics
#   bash run_mind.sh <test_subdir> <metrics> <gpus>           multi-GPU
#   bash run_mind.sh <test_subdir> <metrics> <gpus> <person>  person = 1st | 3rd | both (default: both)
#
# Examples:
#   bash run_mind.sh matrix-game-3 lcm,visual
#   bash run_mind.sh dreamx-world lcm,visual,dino,action,gsc 1
#   bash run_mind.sh dreamx-world lcm,visual,dino,gsc 1 1st
#
# NOTE: the original .bat's PowerShell comma-splitting gotcha ("--%" + quoting
# trick) does not apply in bash -- just quote the metrics arg normally.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export PYTHONIOENCODING=utf-8
# Strip cross-venv pollution. The shell that launched this script often has
# VIRTUAL_ENV / PYTHONHOME / PYTHONPATH set from another project (e.g.
# scope's venv). MIND uses its own venv, so if those leak through Python can
# try to graft the wrong site-packages onto MIND's stdlib path.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP VIRTUAL_ENV VIRTUAL_ENV_PROMPT UV_PYTHON UV_PROJECT_ENVIRONMENT || true

# action metric runs ViPE, which JIT-compiles a CUDA C++ extension on first
# call. On Windows that needed cl.exe (MSVC) + CUDA_HOME. On Linux there is
# no vcvars equivalent; we just make sure CUDA_HOME points at a toolkit if
# one isn't already set, and rely on gcc/nvcc being on PATH. Per the
# mind-benchmark skill's gotchas, ViPE/action is likely unavailable on this
# box (GB300, aarch64) regardless -- see build_vipe.sh's header.
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda-13.2" ]; then
        CUDA_HOME="/usr/local/cuda-13.2"
    elif [ -d "/usr/local/cuda-12.8" ]; then
        CUDA_HOME="/usr/local/cuda-12.8"
    elif [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
    fi
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_HOME
    PATH="$CUDA_HOME/bin:$PATH"
fi
export PATH

PY="$HERE/.venv/bin/python"
# Put the venv's bin FIRST on PATH. The action metric shells out to the `vipe`
# CLI by name, and moreutils ships an unrelated /usr/bin/vipe ("edit pipe in
# editor") that rejects every ViPE flag. Because action catches per-sample, the
# run then completes with no action metric rather than failing -- so this is
# silent unless you grep the log for "skip action". vipe_utils resolves the venv
# copy directly now; this keeps PATH honest for anything else that shells out.
export PATH="$HERE/.venv/bin:$PATH"
# Sibling dirs at the same level as MIND itself (mirrors the Windows layout
# C:\workspace\world\{MIND,MIND-Data,MIND-tests}). UNVERIFIED on this box --
# adjust if MIND-Data / MIND-tests actually live elsewhere here.
PARENT="$(dirname "$HERE")"
GT_ROOT="$PARENT/MIND-Data"
MIND_TESTS="$PARENT/MIND-tests"

TEST_SUBDIR="${1:-}"
METRICS="${2:-}"
NUM_GPUS="${3:-}"
PERSON="${4:-}"
# Anything after the four positionals is forwarded to process.py untouched, so
# its own flags are reachable through this wrapper. --resume none is the one
# that matters in practice: 'auto' silently picks up the most recent
# result_<name>_*.json and skips every sample already in it, so re-scoring to
# add a metric reports "0 to score" and changes nothing.
EXTRA_ARGS=()
if [ "$#" -gt 4 ]; then
    shift 4
    EXTRA_ARGS=("$@")
fi

if [ -z "$TEST_SUBDIR" ]; then TEST_SUBDIR="matrix-game-3"; fi
if [ -z "$METRICS" ]; then METRICS="lcm,visual,dino,action,gsc"; fi
if [ -z "$NUM_GPUS" ]; then NUM_GPUS="1"; fi
# 4th positional defaults to "1st" -- driver scripts (drive_dreamx, drive_lingbot,
# drive_matrix3) all generate 1st_data only. Override with "3rd" or "both".
if [ -z "$PERSON" ]; then PERSON="1st"; fi

# 4th positional = perspective filter. Accepts: 1st, 3rd, both.
PERSPECTIVES=""
case "${PERSON,,}" in
    1st) PERSPECTIVES="1st_data" ;;
    3rd) PERSPECTIVES="3rd_data" ;;
    both) PERSPECTIVES="" ;;
    *)
        echo "ERROR: 4th arg must be 1st, 3rd, or both (got: $PERSON)" >&2
        exit 2
        ;;
esac

# Accept either a bare subdir name (resolved under MIND_TESTS) or an absolute
# path. Detection: contains "/" at the start (absolute) -- simplified from
# the Windows ":\" / "\\" UNC detection since Linux paths just start with "/".
if [[ "$TEST_SUBDIR" == /* ]]; then
    TEST_ROOT="$TEST_SUBDIR"
else
    TEST_ROOT="$MIND_TESTS/$TEST_SUBDIR"
fi

if [ ! -x "$PY" ]; then
    echo "ERROR: venv python not found: $PY" >&2
    echo "Make sure MIND is set up." >&2
    exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
    echo "ERROR: gt_root not found: $GT_ROOT" >&2
    exit 2
fi
if [ ! -e "$TEST_ROOT" ]; then
    echo "ERROR: test_root not found: $TEST_ROOT" >&2
    echo "Stage videos there first via drive_matrix3.py / drive_mind.py / score_matrix3.py." >&2
    exit 2
fi

LOG="$HERE/run_mind_${TEST_SUBDIR}.log"

echo "============================================================"
echo "MIND scoring"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $TEST_ROOT"
echo "  metrics      : $METRICS"
echo "  gpus         : $NUM_GPUS"
echo "  person       : $PERSON (perspectives=$PERSPECTIVES)"
echo "  log          : $LOG"
echo "============================================================"

# Use run_dreamx.py to tee scoring output to both terminal AND a timestamped
# log file. Same wrapper that drive_*.sh use for generation logs.
set +e
if [ -n "$PERSPECTIVES" ]; then
    "$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/process.py" "--gt_root" "$GT_ROOT" "--test_root" "$TEST_ROOT" "--metrics" "$METRICS" "--num_gpus" "$NUM_GPUS" "--perspectives" "$PERSPECTIVES" "${EXTRA_ARGS[@]}"
else
    "$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/process.py" "--gt_root" "$GT_ROOT" "--test_root" "$TEST_ROOT" "--metrics" "$METRICS" "--num_gpus" "$NUM_GPUS" "${EXTRA_ARGS[@]}"
fi
EXIT_CODE=$?
set -e

# Resolve the newest result_<test>_*.json in $HERE so the Done block can
# print an exact path. process.py mints the timestamp at runtime.
RESULT_PATH="(none found)"
NEWEST_RESULT="$(ls -t "$HERE"/result_"${TEST_SUBDIR}"_*.json 2>/dev/null | head -n 1 || true)"
if [ -n "$NEWEST_RESULT" ]; then
    RESULT_PATH="$NEWEST_RESULT"
fi

if [ "$EXIT_CODE" = "0" ]; then
    echo
    echo "============================================================"
    echo "Done."
    echo "  log    : $LOG"
    echo "  result : $RESULT_PATH"
    echo "============================================================"
else
    echo
    echo "ERROR: process.py exited with $EXIT_CODE"
    echo "  log : $LOG"
fi
exit "$EXIT_CODE"
