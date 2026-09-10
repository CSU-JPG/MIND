@echo off
setlocal enableextensions
cd /d "%~dp0"
REM ==========================================================================
REM Generate HY-WorldPlay videos for MIND scoring -- mem_test + action_space_test
REM + mirror_test, BOTH perspectives. Restartable: drive_hy_worldplay.py skips any
REM sample whose video.mp4 already exists, so re-running only fills the gaps
REM (e.g. the missing mirror_test videos). Wraps drive_hy_worldplay.bat.
REM
REM   gen_hy_videos.bat            full run (HY_LIMIT=50/perspective, both, mirror on)
REM   gen_hy_videos.bat 2          quick test: only 2 samples per perspective
REM   set MIND_PERSPECTIVE=3rd_data & gen_hy_videos.bat   single perspective
REM ==========================================================================

REM Mirror test ON (this is the coverage that was missing).
set "MIND_MIRROR_TEST=1"

REM Optional first arg = HY_LIMIT (samples per perspective). Default 50 (full).
if not "%~1"=="" set "HY_LIMIT=%~1"
if not defined HY_LIMIT set "HY_LIMIT=50"

echo ============================================================
echo Generating HY-WorldPlay videos (restartable; skips existing)
echo   mirror_test : ON
echo   HY_LIMIT    : %HY_LIMIT% per perspective
echo   perspective : %MIND_PERSPECTIVE% (blank = both 1st_data + 3rd_data)
echo ============================================================

REM Delegate to the working driver (resolves model paths, sets MASTER_ADDR=127.0.0.1,
REM USE_LIBUV=0, etc). It loops perspectives x test_types and stages video.mp4.
call "%~dp0drive_hy_worldplay.bat"
exit /b %ERRORLEVEL%
