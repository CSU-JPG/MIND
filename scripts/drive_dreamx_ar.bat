@echo off
:: Stage DreamX-World-5B (AR / long-horizon) videos into MIND-tests\dreamx-world_ar\
:: and auto-score with run_mind.bat. SEPARATE from drive_dreamx.bat, which drives the
:: Cam (bidirectional, inference_dreamx5b.py) model. This one drives the autoregressive
:: model (inference_ar_forcing.py + GD-ML/DreamX-World-5B).
::
:: Usage:
::   drive_dreamx_ar.bat                       stage all 1st-person samples then score
::   drive_dreamx_ar.bat --dry-run             preview without running inference
::   drive_dreamx_ar.bat --limit 5             first 5 samples only
::   drive_dreamx_ar.bat --num-output-frames 63   ~15s long-horizon clips (default 21 -> 81px)

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=dreamx-world_ar
set LOG=%~dp0drive_dreamx_ar.log

:: DreamX-World's own .venv now exists and is the correct interpreter for the AR
:: model (cu130 torch + sageattention + the inference fixes). Use it for inference
:: AND for resolving the HF checkpoints (it has huggingface_hub).
set DREAMX_REPO=C:\workspace\world\DreamX-World
if not defined DREAMX_VENV_PY set DREAMX_VENV_PY=%DREAMX_REPO%\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo ERROR: MIND venv python not found: %PY%
    echo The MIND venv runs staging + scoring. Create it ^(separate from DreamX-World^) first.
    exit /b 2
)
if not exist "%DREAMX_VENV_PY%" (
    echo ERROR: DreamX-World venv python not found: %DREAMX_VENV_PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

:: Resolve the Wan2.2 base + DreamX-World-5B (AR) checkpoint from the HF cache,
:: using the DreamX venv (it has huggingface_hub).
echo Resolving Wan2.2-TI2V-5B base from HF cache...
for /f "delims=" %%i in ('%DREAMX_VENV_PY% -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))"') do set "WAN_BASE=%%i"
echo Resolving DreamX-World-5B ^(AR^) checkpoint from HF cache...
for /f "delims=" %%i in ('%DREAMX_VENV_PY% -c "import glob,os; from huggingface_hub import snapshot_download; d=snapshot_download('GD-ML/DreamX-World-5B'); print(glob.glob(os.path.join(d,'**','*.safetensors'),recursive=True)[0])"') do set "AR_CKPT=%%i"

if not defined WAN_BASE ( echo ERROR: could not resolve Wan base & exit /b 2 )
if not defined AR_CKPT ( echo ERROR: could not resolve DreamX-World-5B AR checkpoint - run: python download_models.py --only dreamx_ar & exit /b 2 )

:: AR model is native 16fps; mirror-test on by default (drives the gsc metric).
if not defined MIND_FPS set MIND_FPS=16
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

:: Perspective: set MIND_PERSPECTIVE=1st_data (or 3rd_data) to limit to one.
:: Unset (default) processes BOTH 1st-person and 3rd-person samples.
set PERSP_ARG=
if defined MIND_PERSPECTIVE set PERSP_ARG=--perspective %MIND_PERSPECTIVE%

echo ============================================================
echo DreamX-World-5B ^(AR / long-horizon^) staging into MIND-tests
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : %MODEL_NAME%
echo   wan_base  : !WAN_BASE!
echo   ar_ckpt   : !AR_CKPT!
echo   log       : %LOG%
echo ============================================================

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_dreamx_ar.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--wan-base" "!WAN_BASE!" "--base-checkpoint" "!AR_CKPT!" "--fps" "%MIND_FPS%" %PERSP_ARG% %MIRROR_ARG% %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
    echo ERROR: drive_dreamx_ar.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo Generation done. Running scoring: run_mind.bat %MODEL_NAME%
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
