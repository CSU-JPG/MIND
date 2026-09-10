@echo off
REM build_vipe.bat - build ViPE in-place inside MIND's existing venv.
REM
REM What this does:
REM   1) Loads MSVC build env via vcvars64.bat (so cl.exe is on PATH).
REM   2) Resolves MIND's .venv python (torch 2.10.0+cu128, sm_120 in arch_list).
REM   3) Ensures matching flash-attn prebuilt wheel is installed (skipped if present).
REM   4) Runs `pip install --no-build-isolation -e .` from MIND\vipe to compile
REM      and link ViPE's native CUDA extensions in-place with sm_120 support.
REM
REM Key compile-flag detail: ViPE's specs.py passes -DUSE_CUDA so torch >= 2.10's
REM compiled_autograd.h Windows guard fires and skips an if-constexpr block that
REM otherwise triggers MSVC C2872 'std: ambiguous symbol' when Eigen is also in
REM the include chain.
REM
REM ViPE's setup.py downloads Eigen 3.4 automatically (USE_SYSTEM_EIGEN=0 by
REM default), so we don't need a separate Eigen install.
REM
REM Prerequisites:
REM   - Visual Studio 2022 Community/Pro with "Desktop development with C++".
REM   - CUDA Toolkit 12.x on PATH (nvcc.exe). torch 2.10.0+cu128 needs CUDA 12.x.
REM
REM Usage:
REM   .\build_vipe.bat              (build)
REM   .\build_vipe.bat /clean       (clear ViPE build cache then build)

setlocal enableextensions enabledelayedexpansion

set "PY=%~dp0.venv\Scripts\python.exe"
set "VIPE_DIR=%~dp0vipe"
set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
REM Pin CUDA Toolkit to 13.0 (matches torch 2.12+cu130 now in the MIND venv).
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
REM TORCH_CUDA_ARCH_LIST: include sm_120 so RTX 5090 SASS is baked into vipe_ext.pyd.
set "TORCH_CUDA_ARCH_LIST=8.0;8.6;9.0;10.0;12.0"

if not exist "%PY%" (
    echo ERROR: MIND venv python not found: %PY%
    echo Create the venv with `uv venv` from this directory first.
    exit /b 1
)
if not exist "%VIPE_DIR%\setup.py" (
    echo ERROR: ViPE source not found at %VIPE_DIR%
    exit /b 1
)
if not exist "%VS_VCVARS%" (
    echo ERROR: vcvars64.bat not found at %VS_VCVARS%
    echo Adjust VS_VCVARS at the top of this script to your VS 2022 install path.
    exit /b 1
)
if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo ERROR: CUDA 12.8 nvcc not found at %CUDA_HOME%\bin\nvcc.exe
    echo Adjust CUDA_HOME at the top of this script if your toolkit is elsewhere.
    exit /b 1
)

if /i "%~1"=="/clean" (
    echo Clearing ViPE build cache...
    if exist "%VIPE_DIR%\build" rmdir /s /q "%VIPE_DIR%\build"
    if exist "%VIPE_DIR%\vipe.egg-info" rmdir /s /q "%VIPE_DIR%\vipe.egg-info"
    for /f "delims=" %%D in ('dir /s /b /ad "%VIPE_DIR%\__pycache__" 2^>nul') do rmdir /s /q "%%D"
    shift
)

echo.
echo === Loading MSVC build env (vcvars64.bat) ===
call "%VS_VCVARS%" >nul
if errorlevel 1 (
    echo [build_vipe] vcvars64.bat failed.
    exit /b %ERRORLEVEL%
)
where cl >nul 2>&1 || (
    echo [build_vipe] cl.exe still not on PATH after vcvars; MSVC env load failed.
    exit /b 1
)

echo.
echo === Tool sanity ===
where nvcc 2>nul && nvcc --version | findstr /R "release"
"%PY%" --version
"%PY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('arch_list:', torch.cuda.get_arch_list())"
"%PY%" -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || (
    echo [build_vipe] flash_attn import failed.
    echo Install a prebuilt wheel matching torch + cu128 + cp310 + win, e.g.:
    echo   uv pip install --python "%PY%" --no-build-isolation https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-win_amd64.whl
    exit /b 1
)

echo.
echo === Ensuring in-venv build deps (wheel/setuptools/ninja) for --no-build-isolation ===
uv pip install --python "%PY%" wheel setuptools ninja

echo.
echo === Building ViPE editable into MIND venv ===
echo (Compiles native CUDA extensions with sm_120 + USE_CUDA; first build ~5-10 min.)
pushd "%VIPE_DIR%"
uv pip install --python "%PY%" --no-build-isolation -e .
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo [build_vipe] ViPE native build failed with exit code %RC%.
    echo Common causes:
    echo   - MSVC C2872 'std: ambiguous' = USE_CUDA define missing ^(see specs.py^).
    echo   - flash_attn wheel ABI mismatch with installed torch.
    echo   - Out of disk space in TEMP during Eigen download.
    exit /b %RC%
)

echo.
echo === Verifying ViPE import + CLI ===
"%PY%" -c "import vipe; print('vipe package:', vipe.__file__)" || (
    echo [build_vipe] post-build import of vipe failed.
    exit /b 1
)

echo.
echo ============================================================
echo build_vipe: done. ViPE installed editable into MIND venv:
echo   %PY%
echo You can now run scoring with action enabled:
echo   .\run_mind.bat ^<test_subdir^> lcm,visual,dino,action,gsc
echo ============================================================
exit /b 0
