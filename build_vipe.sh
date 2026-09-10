#!/bin/bash
# Build ViPE's native extension into the MIND venv (Linux).
#
# Not a translation of build_vipe.bat. That script's core step is loading MSVC
# via vcvars64.bat so cl.exe is on PATH; there is no equivalent here, and an
# earlier line-by-line port failed on exactly that. On Linux the toolchain is
# already on PATH, so the work is different: make sure the *build* dependencies
# are inside the venv, because --no-build-isolation means pip uses the venv's
# packages rather than a fresh isolated environment.
#
#   bash build_vipe.sh
#   MAX_JOBS=4 bash build_vipe.sh      # fewer parallel compiles, less memory
#   VIPE_DIR=/path/to/vipe bash build_vipe.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HERE/.venv/bin/python}"
PIP="$(dirname "$PY")/pip"
VIPE_DIR="${VIPE_DIR:-$HERE/vipe}"
# Parallel compile jobs. The default is one per core, which on this box is
# enough to exhaust memory during the CUDA compiles.
export MAX_JOBS="${MAX_JOBS:-8}"

echo "=========================================="
echo "ViPE native build"
echo "  python: $PY"
echo "  source: $VIPE_DIR"
echo "  jobs:   $MAX_JOBS"
echo "=========================================="
echo

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 1
fi
if [ ! -f "$VIPE_DIR/setup.py" ]; then
  echo "ERROR: $VIPE_DIR has no setup.py." >&2
  echo "  The vipe submodule may not be checked out: git submodule update --init vipe" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc is not on PATH; ViPE's csrc extension needs the CUDA toolkit." >&2
  echo "  Try: export PATH=/usr/local/cuda/bin:\$PATH" >&2
  exit 1
fi

echo "--- toolchain ---"
nvcc --version | tail -2
"$PY" -c "import torch; print(f'torch {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}')"
echo

# Built for whatever GPU is actually in this box rather than a hardcoded list.
# build_vipe.bat pins 8.0;8.6;9.0;10.0;12.0 for the Windows RTX 5090; the Grace
# Blackwell parts here report a different capability, and compiling for
# architectures that are not present only lengthens the build.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  DETECTED="$("$PY" -c "import torch; major, minor = torch.cuda.get_device_capability(); print(f'{major}.{minor}')" 2>/dev/null || true)"
  if [ -n "$DETECTED" ]; then
    export TORCH_CUDA_ARCH_LIST="$DETECTED"
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (detected)"
  else
    echo "WARNING: could not detect the GPU capability; letting torch choose." >&2
  fi
else
  echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (from environment)"
fi
echo

# The reason a plain `pip install -e . --no-build-isolation` fails here. Without
# isolation pip builds using the venv's own packages, so a missing `wheel` shows
# up as "error: invalid command 'bdist_wheel'" -- which reads like a problem with
# the package rather than a missing build dependency. ninja is not required but
# the fallback is distutils, which compiles the CUDA sources serially.
echo "--- build dependencies (needed because of --no-build-isolation) ---"
"$PIP" install --quiet setuptools wheel ninja
"$PY" -c "import wheel, setuptools; print('wheel and setuptools present')"
command -v ninja >/dev/null 2>&1 && ninja --version || echo "ninja not on PATH; distutils fallback will be slow"
echo

echo "--- building (this compiles CUDA sources; several minutes) ---"
"$PIP" install -e "$VIPE_DIR" --no-build-isolation

echo
echo "--- verifying ---"
"$PY" -c "import vipe; print('vipe imported from', vipe.__file__)"

echo
echo "=========================================="
echo "Done. The 'action' metric is now available:"
echo "  bash run_mind.sh <model> \"lcm,visual,dino,action,gsc\" 1 both"
echo "=========================================="
