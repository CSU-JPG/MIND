@echo off
:: Stage Echo-WM (JoyAI-Echo, LTX-based action-conditioned i2v) videos into
:: MIND-tests\echo\ for run_mind.bat scoring. Parallel to drive_h3world / drive_evoke.
::
:: Echo consumes a first-frame image + a WASD/IJKL action-string DSL
:: ("w-60,a-30,none-12") whose segment durations are counted in FRAMES, so
:: MIND's action.json ws/ad/ud/lr ticks run-length-encode straight onto it
:: (see src\drive_echo.py). Both action_space_test AND mem_test run by default,
:: across 1st_data and 3rd_data.
::
:: SLOW: inference_wm.py has no load-once/batch mode -- every sample reloads the
:: ~47.8 GB checkpoint plus the Gemma text encoder.
::
:: Usage:
::   drive_echo.bat                             stage 1st + 3rd, score both
::   drive_echo.bat --dry-run --limit 5         print action strings, no model load
::   drive_echo.bat --limit 2                   smoke test
::   drive_echo.bat --mirror-test               mirror clips (needed for gsc)
::   drive_echo.bat --causal                    512x288 few-step entrypoint
::   drive_echo.bat --num-frames 193            longer rollouts (must be 8k+1)
::
:: Metric selection (forwarded to run_mind.bat after staging):
::   set MIND_METRICS=lcm,visual                pick a subset
::   (unset)                                    default = lcm,visual,dino,action,gsc
::   set MIND_GPUS=2                            multi-GPU scoring
::   set MIND_PERSON=1st                        person = 1st | 3rd | both (default both)
::   set MIND_START_INDEX=N                     resume mid-run
::
:: Cross-venv knobs:
::   set ECHO_ROOT=<path>                       override the JoyAI-Echo checkout
::   set ECHO_PY=<path>                         override the echo_wm venv python
::
:: All --flags pass through to src\drive_echo.py; MIND_* env vars stay in this bat.

setlocal enableextensions enabledelayedexpansion

:: This script lives in scripts\ but every path below is relative to the repo
:: root, so resolve the root rather than assuming this file sits in it.
set "ROOT=%~dp0"
if not exist "%ROOT%src\" for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set "PY=%ROOT%.venv\Scripts\python.exe"
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set "LOG=%ROOT%drive_echo.log"

if not defined ECHO_ROOT set ECHO_ROOT=C:\workspace\world\JoyAI-Echo
if not defined ECHO_PY set "ECHO_PY=%ECHO_ROOT%\echo_wm\.venv\Scripts\python.exe"

:: --dry-run only needs the MIND venv and the dataset, so the Echo checks are
:: skipped for it -- lets the action-string mapping be inspected without the
:: 47.8 GB checkpoint on disk.
set DRY_RUN=0
for %%A in (%*) do if "%%~A"=="--dry-run" set DRY_RUN=1

if not exist "%PY%" (
    echo ERROR: MIND venv python not found: %PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)
if "%DRY_RUN%"=="0" if not exist "%ECHO_PY%" (
    echo ERROR: Echo-WM venv python not found: %ECHO_PY%
    echo Run %ECHO_ROOT%\echo_wm\setup_and_run.sh first.
    exit /b 2
)

echo ============================================================
echo Echo-WM staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : echo
echo   echo_root    : %ECHO_ROOT%
echo   echo_py      : %ECHO_PY%
echo   dry_run      : %DRY_RUN%
echo   log          : %LOG%
echo ============================================================

if not defined MIND_START_INDEX set MIND_START_INDEX=0

set "_T_START="
for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_START=%%T"

"%PY%" "%ROOT%run_dreamx.py" "%LOG%" "%PY%" "src\drive_echo.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--start-index" "%MIND_START_INDEX%" %*
set "EXIT_CODE=%ERRORLEVEL%"

for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_END=%%T"
for /f %%E in ('powershell -NoProfile -Command "$d=([TimeSpan]::FromTicks(!_T_END! - !_T_START!)); '{0:00}:{1:00}:{2:00}' -f $d.Hours,$d.Minutes,$d.Seconds"') do set "_T_ELAPSED=%%E"
echo.
echo --- staging elapsed: !_T_ELAPSED! ---

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ERROR: drive_echo.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

if "%DRY_RUN%"=="1" (
    echo.
    echo Dry run -- nothing staged, skipping scoring.
    exit /b 0
)

if not defined MIND_PERSON  set MIND_PERSON=both
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
if not defined MIND_GPUS    set MIND_GPUS=1

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat echo "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
echo ============================================================
:: Quote MIND_METRICS -- CMD splits unquoted comma-bearing args.
call "%ROOT%run_mind.bat" echo "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
exit /b %ERRORLEVEL%
