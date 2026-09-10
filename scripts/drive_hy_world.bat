@echo off
:: Stage HY-World 2.0 videos into MIND-tests\hy-world\.
:: TODO: needs src\drive_hy_world.py; HY-World-2.0 has no .venv (set HY_WORLD_PY env).

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
if not defined HY_WORLD_PY set HY_WORLD_PY=python
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=hy-world
set HY_WORLD_REPO=C:\workspace\world\HY-World-2.0
set LOG=%~dp0drive_hy_world.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%HY_WORLD_REPO%" ( echo ERROR: HY-World-2.0 not found at %HY_WORLD_REPO% & exit /b 2 )
if not exist "%~dp0src\drive_hy_world.py" ( echo ERROR: src\drive_hy_world.py not yet written & exit /b 2 )

echo ============================================================
echo HY-World-2.0 staging into MIND-tests  ^|  model=%MODEL_NAME%  ^|  log=%LOG%
echo ============================================================

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_hy_world.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--hy-world-repo" "%HY_WORLD_REPO%" "--hy-world-py" "%HY_WORLD_PY%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( echo. & echo ERROR: drive_hy_world.py exited with %EXIT_CODE% & exit /b %EXIT_CODE% )

echo. & echo === Running scoring: run_mind.bat %MODEL_NAME% === & echo.
:: gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
