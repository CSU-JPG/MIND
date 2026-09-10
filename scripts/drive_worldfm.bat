@echo off
:: Stage worldfm videos into MIND-tests\worldfm\.
:: TODO: needs src\drive_worldfm.py; no .venv (set WORLDFM_PY env).

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
if not defined WORLDFM_PY set WORLDFM_PY=python
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=worldfm
set WORLDFM_REPO=C:\workspace\world\worldfm
set LOG=%~dp0drive_worldfm.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%WORLDFM_REPO%" ( echo ERROR: repo not found: %WORLDFM_REPO% & exit /b 2 )
if not exist "%~dp0src\drive_worldfm.py" ( echo ERROR: src\drive_worldfm.py not yet written & exit /b 2 )

echo === worldfm staging into MIND-tests  ^|  model=%MODEL_NAME% ===
"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_worldfm.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--repo" "%WORLDFM_REPO%" "--py" "%WORLDFM_PY%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )
:: gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
