#!/bin/bash
# Build MIND's scoring venv (Linux + this box's GPU/CUDA -- see note below).
# Runs staging (drive_*.py) + scoring (process.py). DreamX-World inference is
# cross-spawned in its OWN venv via DREAMX_VENV_PY, so this venv does NOT need
# the DreamX stack. ViPE (action metric) is built separately (see
# setup_mind_evoke.bat's Windows equivalent, or build the venv the same way).
#
# Deviates from setup_mind_venv.bat's cu130 index: that's tuned for the
# Windows box's RTX 5090 (sm_120). This box (pmgb300ws-0304) is a GB300
# (aarch64, CUDA 13.2) -- confirmed earlier the same day (echo_wm's setup)
# that this exact box needs cu132, not cu130/cu128. torch/torchvision left
# unpinned from cu132 rather than guessing an exact version number that may
# not exist on that index (same lesson learned the hard way setting up
# H3-World tonight -- a pinned version can silently not exist on a given
# CUDA index and pip falls back to something wrong instead of erroring).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"
PY="$VENV/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH" >&2
  exit 1
fi

echo "[mind-setup] creating venv ($(python3 --version 2>&1))..."
python3 -m venv "$VENV"
if [ ! -x "$PY" ]; then
  echo "ERROR: venv create failed" >&2
  exit 1
fi
"$PY" -m pip install --upgrade pip

echo "[mind-setup] torch + torchvision (cu132, unpinned -- see script header)..."
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

echo "[mind-setup] MIND requirements (transformers 4.56, torchmetrics, lpips, pyiqa, clip, modelscope, av)..."
"$PY" -m pip install -r envs/requirements.txt

echo
echo "============================================================"
echo "[mind-setup] DONE. venv: $VENV"
echo "============================================================"
echo "Metrics ready now: lcm, visual, dino, gsc."
echo "The 'action' metric needs ViPE, and per the mind-benchmark skill's own"
echo "documented gotchas, its build chain is torch==2.10.0+cu128 (Windows) +"
echo "a Windows/cp310-specific prebuilt flash_attn wheel + MSVC/vcvars to"
echo "build vipe.exe -- almost certainly does NOT port to this box (GB300,"
echo "aarch64), matching tonight's repeated finding that flash-attn/xformers"
echo "have no aarch64 wheels at all. Treat 'action' as likely unavailable"
echo "here; the other 4 metrics (lcm,visual,dino,gsc) don't need ViPE."
echo "Then run download_mind_dataset.sh / process.py as usual (DreamX inference uses its own venv)."
