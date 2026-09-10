@echo off
:: Stage HY-WorldPlay videos via the FLASHDREAMS runner (distilled Wan2.2-5B)
:: into MIND-tests\hy-worldplay-flash\.
::
:: Unlike drive_hy_worldplay.bat (upstream hyvideo/generate.py + HunyuanVideo-1.5
:: base + byT5 Glyph + siglip + torchrun), this drives
::   flashdreams-run hy-worldplay-wan-i2v-5b
:: via src\drive_hy_worldplay.py --backend flashdreams. The runner self-resolves
:: HY-WorldPlay's distilled WAN-5B checkpoint, so NO MODEL_PATH / action ckpt /
:: HunyuanVideo-1.5 base are needed -- it sidesteps the byT5/siglip/distributed
:: failures entirely. The MIND pose string maps straight to the runner's --pose.
::
:: Prereq: flashdreams workspace synced (uv sync in C:\workspace\world\flashdream_public
:: so the flashdreams-hy_worldplay package + flashdreams-run entry-point exist).

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=hy-worldplay-flash
set HY_WORLDPLAY_REPO=C:\workspace\world\HY-WorldPlay
set FLASHDREAMS_REPO=C:\workspace\world\flashdream_public
set LOG=%~dp0drive_hy_worldplay_flash.log
if not defined MIND_FPS set MIND_FPS=24
if not defined HY_NUM_CHUNK set HY_NUM_CHUNK=4
:: uv for the flashdreams runner (the driver spawns `uv run --project ...`).
if not defined UV_EXE set UV_EXE=C:\Users\kschmid\.local\bin\uv.exe

if not exist "%PY%" ( echo ERROR: MIND venv python not found: %PY% & exit /b 2 )
if not exist "%FLASHDREAMS_REPO%" ( echo ERROR: flashdreams repo not found: %FLASHDREAMS_REPO% & exit /b 2 )
if not exist "%UV_EXE%" ( echo ERROR: uv not found: %UV_EXE% & exit /b 2 )
if not exist "%~dp0src\drive_hy_worldplay.py" ( echo ERROR: src\drive_hy_worldplay.py missing & exit /b 2 )

:: Mirror-test generation drives the gsc metric.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

:: Default prompt style when a sample has no explicit prompt (default|cartoony).
:: Override: set MIND_PROMPT_VARIANT=cartoony
if not defined MIND_PROMPT_VARIANT set MIND_PROMPT_VARIANT=default

echo ============================================================
echo HY-WorldPlay (FLASHDREAMS backend) -^> MIND-tests  ^|  model=%MODEL_NAME%
echo   MIND py        : %PY%
echo   flashdreams    : %FLASHDREAMS_REPO%
echo   uv             : %UV_EXE%
echo   num_chunk      : %HY_NUM_CHUNK%   fps: %MIND_FPS%   log: %LOG%
echo   prompt_variant : %MIND_PROMPT_VARIANT%
echo ============================================================

:: Perspective(s) + per-perspective cap. Default BOTH (1st_data + 3rd_data), 50 each.
:: Override: set MIND_PERSPECTIVE=1st_data   set HY_LIMIT=N
if not defined HY_LIMIT set HY_LIMIT=50
set PERSP_LIST=1st_data 3rd_data
set SCORE_PERSON=both
if /I "%MIND_PERSPECTIVE%"=="1st_data" ( set PERSP_LIST=1st_data & set SCORE_PERSON=1st )
if /I "%MIND_PERSPECTIVE%"=="3rd_data" ( set PERSP_LIST=3rd_data & set SCORE_PERSON=3rd )

:: --hy-worldplay-repo/-py + --model-path/--action-ckpt are required by the driver's
:: argparse but UNUSED on the flashdreams path; pass existing placeholders to satisfy it.
for %%P in (%PERSP_LIST%) do (
    echo. & echo === staging perspective %%P ^(limit %HY_LIMIT%, flashdreams^) === & echo.
    "%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_hy_worldplay.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--backend" "flashdreams" "--flashdreams-repo" "%FLASHDREAMS_REPO%" "--num-chunk" "%HY_NUM_CHUNK%" "--hy-worldplay-repo" "%HY_WORLDPLAY_REPO%" "--hy-worldplay-py" "%PY%" "--model-path" "unused" "--action-ckpt" "unused" "--fps" "%MIND_FPS%" "--perspective" "%%P" "--limit" "%HY_LIMIT%" "--prompt-variant" "%MIND_PROMPT_VARIANT%" %MIRROR_ARG% %*
    if errorlevel 1 ( echo. & echo ERROR: drive_hy_worldplay.py --backend flashdreams failed for %%P & exit /b 1 )
)

echo. & echo === Running scoring: run_mind.bat %MODEL_NAME% %SCORE_PERSON% === & echo.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
if not defined MIND_GPUS set MIND_GPUS=1
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%" %MIND_GPUS% %SCORE_PERSON%
exit /b %ERRORLEVEL%
