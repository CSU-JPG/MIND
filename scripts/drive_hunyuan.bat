@echo off
:: Stage Hunyuan-GameCraft videos into MIND-tests\hunyuan-gamecraft\ for run_mind.bat scoring.
::
:: Hunyuan-GameCraft has no dedicated venv; run_low_mem.bat uses bare `python`.
:: Provide a python env (env var HUNYUAN_PY or pip-install requirements.txt in MIND venv).
::
:: TODO: needs src\drive_hunyuan.py that:
::   - Walks MIND-Data first frames + action.json (WASD -> --action-list / --action-speed-list)
::   - Invokes Hunyuan-GameCraft-1.0\hymm_sp\sample_batch.py
::   - Stages output to MIND-tests\hunyuan-gamecraft\<perspective>\<test_type>\<gt_name>\video.mp4
::
:: Defaults (from run_low_mem.bat): 704x1216, 33 frames, 8 steps, fp8.

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set HUNYUAN_PY=%HUNYUAN_PY%
if not defined HUNYUAN_PY set HUNYUAN_PY=python
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=hunyuan-gamecraft
set HUNYUAN_REPO=C:\workspace\world\Hunyuan-GameCraft-1.0
set LOG=%~dp0drive_hunyuan.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%GT_ROOT%" ( echo ERROR: gt_root not found: %GT_ROOT% & exit /b 2 )
if not exist "%HUNYUAN_REPO%\hymm_sp\sample_batch.py" (
    echo ERROR: Hunyuan-GameCraft\hymm_sp\sample_batch.py not found
    exit /b 2
)
if not exist "%~dp0src\drive_hunyuan.py" (
    echo ERROR: src\drive_hunyuan.py not yet written
    echo This bat is a stub; create the driver script first.
    exit /b 2
)

echo ============================================================
echo Hunyuan-GameCraft staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : %MODEL_NAME%
echo   hunyuan_repo : %HUNYUAN_REPO%
echo   hunyuan_py   : %HUNYUAN_PY%
echo   log          : %LOG%
echo ============================================================

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_hunyuan.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--hunyuan-repo" "%HUNYUAN_REPO%" "--hunyuan-py" "%HUNYUAN_PY%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( echo. & echo ERROR: drive_hunyuan.py exited with %EXIT_CODE% & exit /b %EXIT_CODE% )

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat %MODEL_NAME%
echo ============================================================
:: gsc requires per-gt_name mirror_test mp4s; override via MIND_METRICS env to subset.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
