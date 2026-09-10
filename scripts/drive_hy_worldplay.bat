@echo off
:: Stage HY-WorldPlay videos into MIND-tests\hy-worldplay\.
::
:: HY-WorldPlay = Tencent's autoregressive-distilled world model living at
:: C:\workspace\world\HY-WorldPlay. Distinct from HY-World 2.0 (driven by
:: drive_hy_world.bat) and HY-Playground (driven by drive_hy_playground.bat).
::
:: TODO: needs src\drive_hy_worldplay.py that:
::   - Walks MIND-Data first frames + action.json
::   - Converts per-frame ws/ad/ud/lr → HY-WorldPlay pose string (see
::     hyvideo/generate.py:parse_pose_string -- "w-N,d-N,up-N,..." comma-sep,
::     duration is in latents not frames).
::   - Invokes HY-WorldPlay\hyvideo\generate.py via torch.distributed.run
::     (mirrors HY-WorldPlay\run.bat). Per-sample cmd is heavy -- ideally the
::     driver keeps the model resident across samples or batches them.
::   - Sources MODEL_PATH + AR_DISTILL_ACTION_MODEL_PATH from
::     HY-WorldPlay\paths.bat (written by setup.bat).
::   - Stages mp4 to MIND-tests\hy-worldplay\<perspective>\<test_type>\<gt_name>\video.mp4

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
:: HY-WorldPlay has its own venv (Python 3.10 + flash-attn + sageattention).
:: MIND's .venv does NOT satisfy those deps.
if not defined HY_WORLDPLAY_PY set HY_WORLDPLAY_PY=C:\workspace\world\HY-WorldPlay\.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=hy-worldplay
set HY_WORLDPLAY_REPO=C:\workspace\world\HY-WorldPlay
set LOG=%~dp0drive_hy_worldplay.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: MIND venv python not found: %PY% & exit /b 2 )
if not exist "%HY_WORLDPLAY_REPO%" ( echo ERROR: HY-WorldPlay repo not found: %HY_WORLDPLAY_REPO% & exit /b 2 )
if not exist "%HY_WORLDPLAY_PY%" ( echo ERROR: HY-WorldPlay venv python not found: %HY_WORLDPLAY_PY%   Run HY-WorldPlay\setup.bat first. & exit /b 2 )

:: paths.bat is written by HY-WorldPlay\setup.bat after download_models.py runs.
:: It defines MODEL_PATH + AR_DISTILL_ACTION_MODEL_PATH (caller-set env wins).
if exist "%HY_WORLDPLAY_REPO%\paths.bat" call "%HY_WORLDPLAY_REPO%\paths.bat"
if not defined MODEL_PATH (
    echo ERROR: MODEL_PATH not set.  Expected %HY_WORLDPLAY_REPO%\paths.bat from setup.bat
    echo OR explicitly: set MODEL_PATH=^<path to HunyuanVideo-1.5^>
    exit /b 2
)
if not defined AR_DISTILL_ACTION_MODEL_PATH (
    echo ERROR: AR_DISTILL_ACTION_MODEL_PATH not set.  Expected %HY_WORLDPLAY_REPO%\paths.bat from setup.bat
    echo OR explicitly: set AR_DISTILL_ACTION_MODEL_PATH=^<path to ar_distilled_action_model\diffusion_pytorch_model.safetensors^>
    exit /b 2
)

if not exist "%~dp0src\drive_hy_worldplay.py" ( echo ERROR: src\drive_hy_worldplay.py not yet written & exit /b 2 )

:: Mirror-test generation drives the gsc metric (per-gt_name mirror_test mp4s).
:: On by default; set MIND_MIRROR_TEST=0 to skip.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

:: Default prompt style when a sample has no explicit prompt (default|cartoony).
:: Override: set MIND_PROMPT_VARIANT=cartoony
if not defined MIND_PROMPT_VARIANT set MIND_PROMPT_VARIANT=default

echo ============================================================
echo HY-WorldPlay staging into MIND-tests  ^|  model=%MODEL_NAME%  ^|  log=%LOG%
echo ============================================================
echo   MIND py             : %PY%
echo   HY-WorldPlay py     : %HY_WORLDPLAY_PY%
echo   HY-WorldPlay repo   : %HY_WORLDPLAY_REPO%
echo   MODEL_PATH          : %MODEL_PATH%
echo   AR_DISTILL_ACTION   : %AR_DISTILL_ACTION_MODEL_PATH%
echo   fps                 : %MIND_FPS%
echo   mirror_test         : %MIND_MIRROR_TEST%
echo   prompt_variant      : %MIND_PROMPT_VARIANT%
echo ============================================================

:: Perspective(s) + per-perspective sample cap. Default: BOTH (1st_data + 3rd_data),
:: 50 samples each. drive_hy_worldplay.py's --limit slices AFTER the perspective
:: filter, so a single both-perspective pass with --limit 50 would yield 50 total
:: (all 1st). Run one pass per perspective to get 50 EACH. Override:
::   set MIND_PERSPECTIVE=1st_data   (single perspective)   set HY_LIMIT=N
if not defined HY_LIMIT set HY_LIMIT=50
set PERSP_LIST=1st_data 3rd_data
set SCORE_PERSON=both
if /I "%MIND_PERSPECTIVE%"=="1st_data" ( set PERSP_LIST=1st_data & set SCORE_PERSON=1st )
if /I "%MIND_PERSPECTIVE%"=="3rd_data" ( set PERSP_LIST=3rd_data & set SCORE_PERSON=3rd )

for %%P in (%PERSP_LIST%) do (
    echo. & echo === staging perspective %%P ^(limit %HY_LIMIT%^) === & echo.
    "%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_hy_worldplay.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--hy-worldplay-repo" "%HY_WORLDPLAY_REPO%" "--hy-worldplay-py" "%HY_WORLDPLAY_PY%" "--model-path" "%MODEL_PATH%" "--action-ckpt" "%AR_DISTILL_ACTION_MODEL_PATH%" "--fps" "%MIND_FPS%" "--perspective" "%%P" "--limit" "%HY_LIMIT%" "--prompt-variant" "%MIND_PROMPT_VARIANT%" %MIRROR_ARG% %*
    if errorlevel 1 ( echo. & echo ERROR: drive_hy_worldplay.py failed for %%P & exit /b 1 )
)

echo. & echo === Running scoring: run_mind.bat %MODEL_NAME% %SCORE_PERSON% === & echo.
:: Explicit metric list including gsc (mirror_test mp4s needed -- enabled above) and
:: action (ViPE). Score the SAME perspective(s) we staged.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
if not defined MIND_GPUS set MIND_GPUS=1
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%" %MIND_GPUS% %SCORE_PERSON%
exit /b %ERRORLEVEL%
