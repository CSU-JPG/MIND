@echo off
:: Stage Matrix-Game-3 videos into MIND-tests\matrix-game-3\ for run_mind.bat scoring.
::
:: Usage:
::   drive_matrix3.bat                            stage all samples
::   drive_matrix3.bat --dry-run                  preview commands without running inference
::   drive_matrix3.bat --limit 5                  first 5 samples only
::   drive_matrix3.bat --perspective 1st_data     limit to first-person
::   drive_matrix3.bat --test-type mem_test       limit to memory tests
::
:: Metric selection (forwarded to run_mind.bat after staging):
::   set MIND_METRICS=lcm,visual         pick a subset
::   set MIND_METRICS=lcm,visual,dino    multiple
::   (unset)                             default = lcm,visual,dino,action,gsc
::   set MIND_GPUS=2                     multi-GPU scoring
::   set MIND_PERSON=1st                 person = 1st | 3rd | both
::
:: All --flags pass through to src\drive_matrix3.py; MIND_* env vars stay in this bat.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set LOG=%~dp0drive_matrix3.log

:: matrix3's previous cross-spawn target C:\workspace\world\DeepVerse\.venv is gone.
:: Default MATRIX3_VENV_PY to MIND's venv; caller can override beforehand to point
:: at a dedicated DeepVerse / matrix3 env if one is rebuilt.
if not defined MATRIX3_VENV_PY set MATRIX3_VENV_PY=%PY%

:: matrix3 generate.py defaults to 24 fps (MIND-Data standard). No --fps flag exposed;
:: documented here for traceability. Override would require patching matrix3 generate.py.
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
echo Matrix-Game-3 staging into MIND-tests
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : matrix-game-3
echo   log       : %LOG%
echo ============================================================

:: --perspective 1st_data: only stage first-person samples. Override with an
:: extra `--perspective 3rd_data` arg (argparse last-wins).
:: MIND_START_INDEX: skip first N matched samples (resume mid-run). Default 0.
if not defined MIND_START_INDEX set MIND_START_INDEX=0

:: Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
:: On by default; set MIND_MIRROR_TEST=0 to skip.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_matrix3.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" "--start-index" "%MIND_START_INDEX%" %MIRROR_ARG% %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_matrix3.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

if not defined MIND_PERSON  set MIND_PERSON=1st
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
if not defined MIND_GPUS    set MIND_GPUS=1

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat matrix-game-3 "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
echo ============================================================
:: Quote MIND_METRICS — CMD splits unquoted comma-bearing args, which would
:: shove `dino` into the 4th positional (PERSON) and trigger an arg error.
call "%~dp0run_mind.bat" matrix-game-3 "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
exit /b %ERRORLEVEL%
