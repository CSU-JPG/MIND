@echo off
setlocal enableextensions
REM ==========================================================================
REM Drive ABot-World over MIND: seed each sample's first frame, convert its MIND
REM actions (ws/ad/ud/lr) -> ABot keys (WASD+IJKL), run ABot inference, write to
REM MIND-tests\abot\. Runs through ABot-World's own venv. Then score with:
REM   run_mind.bat abot lcm,visual,dino 1 1st
REM
REM   drive_abot.bat --limit 3          smoke (SLOW: reloads ~24GB model per sample)
REM   drive_abot.bat                    all 1st-person
REM   drive_abot.bat --blocks 12        longer rollout (<=15)
REM ==========================================================================
set "ABOT=C:\workspace\world\ABot-World"
set "PY=%ABOT%\.venv\Scripts\python.exe"
set "GT_ROOT=C:\workspace\world\MIND-Data"
set "MIND_TESTS=C:\workspace\world\MIND-tests"
cd /d "%~dp0"

if not exist "%PY%" ( echo ERROR: ABot venv missing -- run %ABOT%\setup_venv.bat & exit /b 1 )

REM no --perspective -> both 1st_data + 3rd_data. Restrict via: drive_abot.bat --perspective 3rd_data
"%PY%" "%~dp0src\drive_abot.py" --gt-root "%GT_ROOT%" --test-root "%MIND_TESTS%" %*

echo.
echo Videos -^> %MIND_TESTS%\abot\
echo Now score:  run_mind.bat abot lcm,visual,dino 1 1st
endlocal
