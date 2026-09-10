@echo off
:: Stage Warp-as-History videos into MIND-tests\warp-history\.
:: TODO: needs src\drive_warp_history.py; no .venv (set WARP_HISTORY_PY env).

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
if not defined WARP_HISTORY_PY set WARP_HISTORY_PY=python
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=warp-history
set WARP_HISTORY_REPO=C:\workspace\world\Warp-as-History
set LOG=%~dp0drive_warp_history.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%WARP_HISTORY_REPO%" ( echo ERROR: repo not found: %WARP_HISTORY_REPO% & exit /b 2 )
if not exist "%~dp0src\drive_warp_history.py" ( echo ERROR: src\drive_warp_history.py not yet written & exit /b 2 )

echo === Warp-as-History staging into MIND-tests  ^|  model=%MODEL_NAME% ===
"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_warp_history.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--repo" "%WARP_HISTORY_REPO%" "--py" "%WARP_HISTORY_PY%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )
:: gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
