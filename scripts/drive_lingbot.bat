@echo off
:: Stage LingBot-World base-cam-nf4 videos into MIND-tests\lingbot-base-cam-nf4\ for run_mind.bat scoring.
::
:: Uses lingbot-world's generate.py with the MIND action.json passed through
:: --action_path, so per-frame WASD/ud/lr conditioning flows end-to-end. That
:: makes the `action` MIND metric meaningful for this model (unlike the
:: matrix3 / dreamx drivers which stub a single placeholder action).
::
:: Usage:
::   drive_lingbot.bat                            stage all samples then score
::   drive_lingbot.bat --dry-run                  preview commands without running inference
::   drive_lingbot.bat --limit 5                  first 5 samples only
::   drive_lingbot.bat --perspective 1st_data     limit to first-person
::   drive_lingbot.bat --test-type mem_test       limit to memory tests
::
:: All flags pass through to src\drive_lingbot.py.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Pin triton-windows to CUDA 12.8 toolkit (matches cu128 torch wheels). With
:: v13.0 also installed, triton's JIT otherwise picks v13.0 via CUDA_PATH and
:: fails to link kernels against the cu128-built torch ABI.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

:: Disable torch.compile/dynamo — triton-windows + inductor on Windows is flaky.
set TORCHDYNAMO_DISABLE=1

:: Keep huggingface_hub from re-checking HF for tokenizer/model updates on every
:: run; rely on local cache. Stops corp-proxy stalls during AutoTokenizer.from_pretrained.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

set PY=%~dp0.venv\Scripts\python.exe
if not defined GT_ROOT    set GT_ROOT=C:\workspace\world\MIND-Data
if not defined MIND_TESTS set MIND_TESTS=C:\workspace\world\MIND-tests
:: Path A: use the fast-mini-cam ckpt + generate_fast.py. The previous
:: base-cam-nf4 + generate.py path failed on every sample (OSError: missing
:: config.json in the NF4 release). fast-mini-cam ships a complete layout
:: and produces mp4s out of the box; action-metric quality will be loose
:: since action_path is dropped (see drive_lingbot.py for the Path B plan).
set MODEL_NAME=lingbot-fast
set CKPT_DIR=C:\workspace\world\lingbot-world\fast-mini-cam
set LOG=%~dp0drive_lingbot.log

:: lingbot fps is set via wan/configs/shared_config.py (sample_fps=24 patched in).
:: MIND_FPS here is for traceability only; lingbot doesn't expose a CLI --fps flag.
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)
if not exist "%CKPT_DIR%" (
    echo WARNING: ckpt_dir not present: %CKPT_DIR%
    echo Run download_fast.bat in lingbot-world first ^(unless using --dry-run^).
    echo.
)

echo ============================================================
echo LingBot-World-Fast staging into MIND-tests
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : %MODEL_NAME%
echo   ckpt_dir  : %CKPT_DIR%
echo   log       : %LOG%
echo ============================================================

:: --perspective 1st_data: lingbot-fast only stages first-person samples.
:: To include 3rd_data, pass an overriding `--perspective 3rd_data` as an
:: extra arg — argparse's "last wins" rule lets the user override.

:: Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
:: On by default; set MIND_MIRROR_TEST=0 to skip.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_lingbot.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--ckpt-dir" "%CKPT_DIR%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" %MIRROR_ARG% %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_lingbot.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo.
echo ============================================================
echo Staging done. Scoring with run_mind.bat %MODEL_NAME% ...
echo ============================================================
echo.

:: run_mind.bat defaults PERSON=1st, matching this bat's 1st_data-only generation.
:: gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" %MODEL_NAME% "%MIND_METRICS%"

set SCORE_EXIT=%ERRORLEVEL%
if not %SCORE_EXIT%==0 (
    echo.
    echo ERROR: run_mind.bat exited with %SCORE_EXIT%
    exit /b %SCORE_EXIT%
)
echo.
echo Done. See result_%MODEL_NAME%_*.json in this directory.
endlocal
