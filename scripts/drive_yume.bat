@echo off
:: Stage YUME videos into MIND-tests\yume\.
:: TODO: needs src\drive_yume.py; no .venv (set YUME_PY env).

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
if not defined YUME_PY set YUME_PY=python
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=yume
set YUME_REPO=C:\workspace\world\YUME
set LOG=%~dp0drive_yume.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%YUME_REPO%" ( echo ERROR: repo not found: %YUME_REPO% & exit /b 2 )
if not exist "%~dp0src\drive_yume.py" ( echo ERROR: src\drive_yume.py not yet written & exit /b 2 )

echo === YUME staging into MIND-tests  ^|  model=%MODEL_NAME% ===
"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_yume.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--repo" "%YUME_REPO%" "--py" "%YUME_PY%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )
:: gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
