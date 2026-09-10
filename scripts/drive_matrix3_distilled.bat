@echo off
:: Stage FastVideo Matrix-Game-3.0-Distilled videos into MIND-tests\matrix-game-3-distilled\
:: for run_mind.bat scoring. Parallel to drive_matrix3.bat, but targets the
:: distilled checkpoint via FastVideo's pipeline (3 inference steps, fast).
::
:: Usage:
::   drive_matrix3_distilled.bat                       stage all 1st-person + mirror
::   drive_matrix3_distilled.bat --dry-run             preview commands
::   drive_matrix3_distilled.bat --limit 5             first 5 samples only
::   drive_matrix3_distilled.bat --test-type mem_test  limit to memory tests
::   drive_matrix3_distilled.bat --perspective 3rd_data  override (default = 1st_data)
::
:: Metric selection (forwarded to run_mind.bat after staging):
::   set MIND_METRICS=lcm,visual                pick a subset
::   (unset)                                    default = lcm,visual,dino,action,gsc
::   set MIND_GPUS=2                            multi-GPU scoring
::   set MIND_PERSON=1st                        person = 1st | 3rd | both (default 1st)
::   set MIND_MIRROR_TEST=0                     disable mirror_test (default on)
::   set MIND_START_INDEX=N                     resume mid-run
::
:: Cross-venv knobs:
::   set MATRIX3D_VENV_PY=<path>                override FastVideo venv python
::                                              (default C:\workspace\world\FastVideo\.venv\Scripts\python.exe)
::
:: All --flags pass through to src\drive_matrix3_distilled.py; MIND_* env vars
:: stay in this bat.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set LOG=%~dp0drive_matrix3_distilled.log

:: FastVideo's venv has the matching torch + CUDA stack for the distilled model.
if not defined MATRIX3D_VENV_PY set MATRIX3D_VENV_PY=C:\workspace\world\FastVideo\.venv\Scripts\python.exe

:: Distilled checkpoint defaults — 57 frames @ 24fps standard for MIND-Data.
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" (
    echo ERROR: MIND venv python not found: %PY%
    exit /b 2
)
if not exist "%MATRIX3D_VENV_PY%" (
    echo ERROR: FastVideo venv python not found: %MATRIX3D_VENV_PY%
    echo Set MATRIX3D_VENV_PY to point at the FastVideo .venv, or install it.
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo Matrix-Game-3 DISTILLED (FastVideo) staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : matrix-game-3-distilled
echo   fastvideo_py : %MATRIX3D_VENV_PY%
echo   log          : %LOG%
echo ============================================================

:: Defaults: 1st-person only, mirror_test on (matches drive_matrix3.bat).
if not defined MIND_START_INDEX set MIND_START_INDEX=0
if not defined MIND_MIRROR_TEST  set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_matrix3_distilled.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" "--start-index" "%MIND_START_INDEX%" %MIRROR_ARG% %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_matrix3_distilled.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

if not defined MIND_PERSON  set MIND_PERSON=1st
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
if not defined MIND_GPUS    set MIND_GPUS=1

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat matrix-game-3-distilled "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
echo ============================================================
:: Quote MIND_METRICS — CMD splits unquoted comma-bearing args, which would
:: shove `dino` into the 4th positional (PERSON) and trigger an arg error.
call "%~dp0run_mind.bat" matrix-game-3-distilled "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
exit /b %ERRORLEVEL%
