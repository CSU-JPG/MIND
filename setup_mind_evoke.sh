#!/bin/bash
# MIND SETUP -- Video Generation Evaluation for Evoke.
#
# No uv (per repo convention) -- uses python3 -m venv + pip instead of
# setup_mind_evoke.bat's uv venv / uv pip.
#
# Deviates from the Windows original's cu128 torch index (tuned for the
# Windows box's RTX 5090, and pinned to torch==2.7.0/torchvision==0.22.0/
# torchaudio==2.7.0): this box (GB300, aarch64, CUDA 13.2) needs cu132, and
# torch/torchvision/torchaudio are left UNPINNED rather than guessing exact
# version numbers that may not exist on that index -- a pinned version can
# silently not exist on a given CUDA index and pip falls back to something
# wrong instead of erroring (same lesson learned setting up H3-World).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo
echo "================================================================================"
echo "MIND SETUP -- Video Generation Evaluation for Evoke"
echo "================================================================================"
echo

# Check if running from MIND dir
if [ ! -e "mind" ] && [ ! -e ".git" ]; then
    echo "ERROR: Run this script from the MIND directory" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH" >&2
    exit 1
fi

VENV="$HERE/.venv"
PY="$VENV/bin/python"

echo "[1/2] Setting up virtual environment ($(python3 --version 2>&1))..."
if [ ! -e "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "     Created .venv"
fi
if [ ! -x "$PY" ]; then
    echo "ERROR: venv create failed" >&2
    exit 1
fi
"$PY" -m pip install --upgrade pip

echo "[2/2] Installing dependencies..."
echo "     Installing torch (cu132, unpinned -- see script header)..."
"$PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132

echo "     Installing MIND requirements..."
if [ -e "requirements.txt" ]; then
    "$PY" -m pip install -r requirements.txt
else
    echo "     Installing core MIND packages..."
    "$PY" -m pip install opencv-python pillow numpy einops scipy
    "$PY" -m pip install clip-benchmark clip-interrogator
fi

echo
echo "================================================================================"
echo "SETUP COMPLETE"
echo "================================================================================"
echo
echo "To evaluate Evoke videos with MIND:"
echo "  .venv/bin/python evaluate_evoke.py --video outputs/t2v/geo_pred.mp4"
echo
echo "For batch evaluation:"
echo "  .venv/bin/python evaluate_evoke.py --video-dir <path-to-Evoke-outputs>"
echo
exit 0
