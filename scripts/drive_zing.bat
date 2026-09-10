@echo off
:: Stage Zing-0.5 (action-conditioned ti2v) videos into MIND-tests\zing\ for
:: run_mind.bat scoring. Parallel to drive_evoke.bat.
::
:: Zing consumes a reference first frame + per-frame keyboard actions, which map directly
:: onto MIND's action.json ws/ad/ud/lr ticks (see src\drive_zing.py), so both
:: action_space_test AND mem_test run by default, across both 1st_data and 3rd_data.
::
:: Unlike Evoke, zing_v0_5 is natively batch-oriented: one JSONL is built for every sample
:: and the checkpoint loads exactly once.
::
:: Usage:
::   drive_zing.bat                             stage 1st + 3rd, score both
::   drive_zing.bat --dry-run                   preview what would be staged
::   drive_zing.bat --limit 5                   smoke test: first 5 samples
::   drive_zing.bat --test-type mem_test        limit to memory tests
::   drive_zing.bat --perspective 1st_data      override (default = both)
::   drive_zing.bat --num-frames 121            longer rollouts (default 97)
::
:: Metric selection (forwarded to run_mind.bat after staging):
::   set MIND_METRICS=lcm,visual                pick a subset
::   (unset)                                    default = lcm,visual,dino,gsc (no action)
::   set MIND_GPUS=2                            multi-GPU scoring
::   set MIND_PERSON=1st                        person = 1st | 3rd | both (default both)
::   set MIND_MIRROR_TEST=0                     disable mirror_test (default on)
::   set MIND_START_INDEX=N                     resume mid-run
::
:: Cross-venv knobs:
::   set ZING_VENV_PY=<path>                    override zing venv python
::                                              (default C:\workspace\world\zing-world-model\.venv\Scripts\python.exe)
::   set ZING_REPO=<path>                       override the zing checkout
::
:: All --flags pass through to src\drive_zing.py; MIND_* env vars stay in this bat.

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set LOG=%~dp0drive_zing.log

if not defined ZING_REPO set ZING_REPO=C:\workspace\world\zing-world-model
if not defined ZING_VENV_PY set ZING_VENV_PY=%ZING_REPO%\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo ERROR: MIND venv python not found: %PY%
    exit /b 2
)
if not exist "%ZING_VENV_PY%" (
    echo ERROR: zing venv python not found: %ZING_VENV_PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo Zing-0.5 (action-conditioned ti2v) staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : zing
echo   zing_repo    : %ZING_REPO%
echo   zing_py      : %ZING_VENV_PY%
echo   log          : %LOG%
echo ============================================================

if not defined MIND_START_INDEX set MIND_START_INDEX=0
if not defined MIND_MIRROR_TEST  set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

set "_T_START="
for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_START=%%T"

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_zing.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--start-index" "%MIND_START_INDEX%" %MIRROR_ARG% %*
set "EXIT_CODE=%ERRORLEVEL%"

for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_END=%%T"
for /f %%E in ('powershell -NoProfile -Command "$d=([TimeSpan]::FromTicks(!_T_END! - !_T_START!)); '{0:00}:{1:00}:{2:00}' -f $d.Hours,$d.Minutes,$d.Seconds"') do set "_T_ELAPSED=%%E"
echo.
echo --- staging elapsed: !_T_ELAPSED! ---

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ERROR: drive_zing.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

if not defined MIND_PERSON  set MIND_PERSON=both
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,gsc
if not defined MIND_GPUS    set MIND_GPUS=1

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat zing "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
echo ============================================================
:: Quote MIND_METRICS -- CMD splits unquoted comma-bearing args.
call "%~dp0run_mind.bat" zing "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
exit /b %ERRORLEVEL%
