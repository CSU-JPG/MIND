@echo off
:: Stage Sana world-model videos into MIND-tests\sana-wm\, then run MIND scoring.
:: src\drive_sana_wm.py walks MIND-Data first frames + action.json, invokes
:: Sana\inference_video_scripts\inference_sana_wm.py via Sana\.venv-wm\..., and
:: stages output to MIND-tests\sana-wm\<perspective>\<test_type>\<gt_name>\video.mp4
:: (one mp4 per mirror_test sample). Runs both perspectives by default; set
:: MIND_SKIP_3RD=1 to skip the slower 3rd-person pass.

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
:: Sana-WM uses a dedicated venv (.venv-wm) created by Sana\environment_setup_sana_wm.bat
:: — distinct from Sana\.venv (the older Sana env, may not exist).
set SANA_PY=C:\workspace\world\Sana\.venv-wm\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=sana-wm
set SANA_REPO=C:\workspace\world\Sana
set SANA_ENTRY=%SANA_REPO%\inference_video_scripts\inference_sana_video.py
set LOG=%~dp0drive_sana_wm.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%SANA_REPO%" ( echo ERROR: Sana repo not found: %SANA_REPO% & exit /b 2 )
if not exist "%SANA_ENTRY%" ( echo ERROR: Sana entry not found: %SANA_ENTRY% & exit /b 2 )
if not exist "%~dp0src\drive_sana_wm.py" ( echo ERROR: src\drive_sana_wm.py not yet written & exit /b 2 )

echo === Sana-WM staging into MIND-tests  ^|  model=%MODEL_NAME% ===
:: drive_sana_wm.py PERSPECTIVES tuple is ordered ("3rd_data","1st_data"). We
:: run both perspectives by default so the staged set is complete for run_mind.
:: 1st_data runs first because it's faster and historically the higher-value
:: pass; 3rd_data follows. To opt out of 3rd_data (slower pass), set
:: MIND_SKIP_3RD=1 in the environment before launch. To override the
:: perspective entirely, pass an extra `--perspective <name>` on the CLI
:: (argparse last-wins) — this short-circuits the dual-pass loop.

if not defined MIND_SKIP_3RD set MIND_SKIP_3RD=0

:: Mirror-test generation drives the gsc metric (one mp4 per mirror_test
:: sample dir). On by default; set MIND_MIRROR_TEST=0 to skip.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

echo --- pass 1/2: --perspective 1st_data ---
"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_sana_wm.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--sana-repo" "%SANA_REPO%" "--sana-py" "%SANA_PY%" "--fps" "%MIND_FPS%" "--perspective" "1st_data" %MIRROR_ARG% %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )

if "%MIND_SKIP_3RD%"=="1" (
    echo --- pass 2/2 skipped: MIND_SKIP_3RD=1 ---
) else (
    echo --- pass 2/2: --perspective 3rd_data ---
    "%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_sana_wm.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--sana-repo" "%SANA_REPO%" "--sana-py" "%SANA_PY%" "--fps" "%MIND_FPS%" "--perspective" "3rd_data" %MIRROR_ARG% %*
    set EXIT_CODE=%ERRORLEVEL%
    if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )
)

:: Metric list including gsc. drive_sana_wm.py emits one video.mp4 per
:: mirror_test sample dir, which is what the current gsc metric consumes.
:: Override by setting MIND_METRICS before calling this bat.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
