@echo off
:: Stage DreamX-World videos into MIND-tests\dreamx-world\ and auto-score with run_mind.bat.
::
:: Usage:
::   drive_dreamx.bat                            stage all samples then score
::   drive_dreamx.bat --dry-run                  preview commands without running inference
::   drive_dreamx.bat --limit 5                  first 5 samples only
::   drive_dreamx.bat --perspective 1st_data     limit to first-person
::   drive_dreamx.bat --test-type mem_test       limit to memory tests
::
:: All flags pass through to src\drive_dreamx.py.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=dreamx-world
set LOG=%~dp0drive_dreamx.log

:: DreamX-World's own .venv is missing on this box; reuse MIND's venv as the
:: cross-spawned inference interpreter. Override DREAMX_VENV_PY beforehand
:: to point elsewhere if you have a dedicated DreamX-World venv.
if not defined DREAMX_VENV_PY set DREAMX_VENV_PY=%PY%

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo DreamX-World staging into MIND-tests
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : %MODEL_NAME%
echo   log       : %LOG%
echo ============================================================

:: MIND standard fps (24) — matches GT MIND-Data + scoring crop expectations.
if not defined MIND_FPS set MIND_FPS=24
:: --perspective 1st_data: only stage first-person samples. Override with an
:: extra `--perspective 3rd_data` arg (argparse last-wins).

:: Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
:: On by default; set MIND_MIRROR_TEST=0 to skip.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_dreamx.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" %MIRROR_ARG% %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_dreamx.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat %MODEL_NAME%
echo ============================================================
:: run_mind.bat defaults PERSON=1st, matching this bat's 1st_data-only generation.
:: gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
