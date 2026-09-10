@echo off
setlocal enableextensions
REM ==========================================================================
REM Drive Waypoint-1.5 over the MIND benchmark: seed each sample's first frame,
REM replay its per-frame ws/ad/ud/lr actions, write videos to MIND-tests\waypoint\.
REM Runs through scope-overworld's venv (has world_engine + cv2 + imageio); shares
REM its compile cache. Then score with:  run_mind.bat waypoint lcm,visual,dino 1 1st
REM
REM   drive_waypoint.bat                       1st-person, all samples
REM   drive_waypoint.bat --limit 5             quick smoke
REM   drive_waypoint.bat --perspective 3rd_data
REM ==========================================================================
set "SCOPE=C:\workspace\world\scope-overworld"
set "UVX=C:\Users\kschmid\.local\bin\uv.exe"
set "GT_ROOT=C:\workspace\world\MIND-Data"
set "MIND_TESTS=C:\workspace\world\MIND-tests"
set "TORCHINDUCTOR_COMPILE_THREADS=1"
set "TORCHINDUCTOR_CACHE_DIR=%SCOPE%\.inductor_cache"
set "TORCHINDUCTOR_FX_GRAPH_CACHE=1"
set "TRITON_CACHE_DIR=%SCOPE%\.triton_cache"
cd /d "%~dp0"

if not exist "%SCOPE%\.venv" ( echo ERROR: scope-overworld venv missing & exit /b 1 )

REM no --perspective -> both 1st_data + 3rd_data (matches drive_abot). Restrict via --perspective 3rd_data
"%UVX%" run --no-sync --project "%SCOPE%" python "%~dp0src\drive_waypoint.py" --gt-root "%GT_ROOT%" --test-root "%MIND_TESTS%" %*

echo.
echo Videos -^> %MIND_TESTS%\waypoint\
echo Now score:  run_mind.bat waypoint lcm,visual,dino 1 1st
endlocal
