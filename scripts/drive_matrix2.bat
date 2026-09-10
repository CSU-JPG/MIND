@echo off
:: Stage Matrix-Game-2 videos into MIND-tests\matrix-game-2\ for run_mind.bat scoring.
::
:: Usage:
::   drive_matrix2.bat                            stage all samples
::   drive_matrix2.bat --dry-run                  preview commands without running inference
::   drive_matrix2.bat --limit 5                  first 5 samples only
::   drive_matrix2.bat --perspective 1st_data     limit to first-person
::   drive_matrix2.bat --test-type mem_test       limit to memory tests
::   drive_matrix2.bat --config-path C:\path\to\inference_gta_drive.yaml
::
:: Env knobs (set before running):
::   MATRIX2_VENV_PY    python.exe for the matrix2 venv (defaults to MIND's venv)
::   MATRIX2_PRETRAINED pretrained_model_path dir holding Wan2.1_VAE.pth
::                      (default: C:\workspace\world\Matrix-Game\Matrix-Game-2\Matrix-Game-2.0)
::   MIND_MIRROR_TEST=0 skip the mirror_test pass (default on, drives the gsc metric)
::   MIND_START_INDEX   skip the first N matched samples (resume mid-run)
::
:: Metric selection (forwarded to run_mind.bat after staging):
::   set MIND_METRICS=lcm,visual         pick a subset
::   (unset)                             default = lcm,visual,dino,action,gsc
::   set MIND_GPUS=2                     multi-GPU scoring
::   set MIND_PERSON=1st                 person = 1st | 3rd | both
::
:: All --flags pass through to src\drive_matrix2.py; MIND_* env vars stay in this bat.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set LOG=%~dp0drive_matrix2.log

:: matrix2 venv defaults to MIND's venv; override via MATRIX2_VENV_PY env var
:: (drive_matrix2.py reads MATRIX2_VENV_PY directly, so we just inherit it).
if not defined MATRIX2_VENV_PY set MATRIX2_VENV_PY=%PY%

:: MIND-Data is 24 fps; matrix2 inference doesn't expose --fps. Recorded for traceability.
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo Matrix-Game-2 staging into MIND-tests
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : matrix-game-2
echo   venv_py   : %MATRIX2_VENV_PY%
echo   log       : %LOG%
echo ============================================================

if not defined MIND_START_INDEX set MIND_START_INDEX=0

:: Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
:: On by default; set MIND_MIRROR_TEST=0 to skip.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_matrix2.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" "--start-index" "%MIND_START_INDEX%" %MIRROR_ARG% %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_matrix2.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

if not defined MIND_PERSON  set MIND_PERSON=1st
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
if not defined MIND_GPUS    set MIND_GPUS=1

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat matrix-game-2 "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
echo ============================================================
:: Quote MIND_METRICS — CMD splits unquoted comma-bearing args, which would
:: shove `dino` into the 4th positional (PERSON) and trigger an arg error.
call "%~dp0run_mind.bat" matrix-game-2 "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
exit /b %ERRORLEVEL%
