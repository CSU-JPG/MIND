@echo off
:: Stage Evoke (i2v-style, camera-controlled) videos into MIND-tests\evoke\ for
:: run_mind.bat scoring. Parallel to drive_helios_i2v.bat.
::
:: Unlike Helios, Evoke takes a real camera pose trajectory (converted from MIND's
:: action.json actor_pos/actor_rpy / camera_pos/camera_rpy ground truth -- see
:: src\utils\evoke_pose.py), so both action_space_test AND mem_test run by default,
:: across both 1st_data and 3rd_data.
::
:: Usage:
::   drive_evoke.bat                            stage 1st + 3rd, score both
::   drive_evoke.bat --dry-run                  preview what would be staged
::   drive_evoke.bat --limit 5                  smoke test: first 5 samples
::   drive_evoke.bat --test-type mem_test       limit to memory tests
::   drive_evoke.bat --perspective 1st_data     override (default = both)
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
::   set EVOKE_VENV_PY=<path>                   override Evoke venv python
::                                              (default C:\workspace\world\Evoke\.venv\Scripts\python.exe)
::   set EVOKE_MODEL_PATH=<path>                override the local snapshot/evoke-base path
::
:: All --flags pass through to src\drive_evoke.py; MIND_* env vars stay in this bat.

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set LOG=%~dp0drive_evoke.log

if not defined EVOKE_VENV_PY set EVOKE_VENV_PY=C:\workspace\world\Evoke\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo ERROR: MIND venv python not found: %PY%
    exit /b 2
)
if not exist "%EVOKE_VENV_PY%" (
    echo ERROR: Evoke venv python not found: %EVOKE_VENV_PY%
    echo Run: C:\workspace\world\Evoke\setup_evoke.bat
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo Evoke (camera-controlled i2v) staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : evoke
echo   evoke_py     : %EVOKE_VENV_PY%
echo   log          : %LOG%
echo ============================================================

if not defined MIND_START_INDEX set MIND_START_INDEX=0
if not defined MIND_MIRROR_TEST  set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

set "_T_START_EPOCH="
for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_START=%%T"

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_evoke.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--start-index" "%MIND_START_INDEX%" %MIRROR_ARG% %*
set "EXIT_CODE=%ERRORLEVEL%"

for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_END=%%T"
for /f %%E in ('powershell -NoProfile -Command "$d=([TimeSpan]::FromTicks(!_T_END! - !_T_START!)); '{0:00}:{1:00}:{2:00}' -f $d.Hours,$d.Minutes,$d.Seconds"') do set "_T_ELAPSED=%%E"
echo.
echo --- staging elapsed: !_T_ELAPSED! ---

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ERROR: drive_evoke.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

if not defined MIND_PERSON  set MIND_PERSON=both
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,gsc
if not defined MIND_GPUS    set MIND_GPUS=1

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat evoke "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
echo ============================================================
:: Quote MIND_METRICS -- CMD splits unquoted comma-bearing args.
call "%~dp0run_mind.bat" evoke "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
exit /b %ERRORLEVEL%
