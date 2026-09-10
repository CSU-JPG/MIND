@echo off
:: Stage DeepVerse videos into MIND-tests\deepverse\ for run_mind.bat scoring.
::
:: TODO: needs src\drive_deepverse.py that:
::   - Walks MIND-Data first frames + action.json
::   - Invokes DeepVerse\run.py (uses DeepVerse\.venv\Scripts\python.exe)
::   - Stages output to MIND-tests\deepverse\<perspective>\<test_type>\<gt_name>\video.mp4

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=deepverse
set DEEPVERSE_REPO=C:\workspace\world\DeepVerse
:: Cross-spawn python for DeepVerse inference. Resolution order:
::   1. DEEPVERSE_VENV_PY env var (if set, used as-is — no existence check)
::   2. DeepVerse\.venv\Scripts\python.exe (if present)
::   3. plain `python` on PATH (whatever's active in the calling shell)
if not defined DEEPVERSE_VENV_PY (
    if exist "%DEEPVERSE_REPO%\.venv\Scripts\python.exe" (
        set DEEPVERSE_VENV_PY=%DEEPVERSE_REPO%\.venv\Scripts\python.exe
    ) else (
        set DEEPVERSE_VENV_PY=python
    )
)
set LOG=%~dp0drive_deepverse.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%GT_ROOT%" ( echo ERROR: gt_root not found: %GT_ROOT% & exit /b 2 )
if not exist "%DEEPVERSE_REPO%\run.py" (
    echo ERROR: DeepVerse\run.py not found at %DEEPVERSE_REPO%\run.py
    exit /b 2
)
if not exist "%~dp0src\drive_deepverse.py" (
    echo ERROR: src\drive_deepverse.py not yet written
    echo This bat is a stub; create the driver script first.
    exit /b 2
)

echo ============================================================
echo DeepVerse staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : %MODEL_NAME%
echo   deepverse    : %DEEPVERSE_REPO%
echo   deepverse_py : %DEEPVERSE_VENV_PY%
echo   log          : %LOG%
echo ============================================================

:: drive_deepverse.py PERSPECTIVES tuple now defaults to ("3rd_data","1st_data"),
:: so omitting --perspective walks both with 3rd-person first. Pass --perspective
:: <p> on the CLI to restrict to one. CLI args after %* override the defaults.

:: Mirror-test generation drives the gsc metric (per-sample mirror_test mp4s).
:: On by default; set MIND_MIRROR_TEST=0 to skip.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_deepverse.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--deepverse-repo" "%DEEPVERSE_REPO%" "--deepverse-py" "%DEEPVERSE_VENV_PY%" "--fps" "%MIND_FPS%" %MIRROR_ARG% %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( echo. & echo ERROR: drive_deepverse.py exited with %EXIT_CODE% & exit /b %EXIT_CODE% )

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat %MODEL_NAME%
echo ============================================================
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
