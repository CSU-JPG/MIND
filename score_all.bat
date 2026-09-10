@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Score EVERY driven test set in MIND-tests against MIND-Data GT (via run_mind.bat
REM per subdir), then print the combined scores table. Skips the .frames cache.
REM run_mind.bat resumes from prior result_*.json, so re-running only scores new
REM samples. Backends aren't needed -- this scores videos already staged.
REM
REM   score_all.bat                              lcm,visual,dino  both  (default)
REM   score_all.bat lcm,visual,dino,action,gsc 1 both   full metrics (action=ViPE)
REM   score_all.bat lcm,visual 1 1st            quick, first-person only
REM ==========================================================================
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "TESTS=C:\workspace\world\MIND-tests"

set "METRICS=%~1"
if not defined METRICS set "METRICS=lcm,visual,dino,action,gsc"
set "GPUS=%~2"
if not defined GPUS set "GPUS=1"
set "PERSON=%~3"
if not defined PERSON set "PERSON=both"

if not exist "%PY%"    ( echo ERROR: venv missing -- run setup_mind_venv.bat & exit /b 1 )
if not exist "%TESTS%" ( echo ERROR: no MIND-tests dir at %TESTS% & exit /b 1 )

echo Scoring all sets under %TESTS%   metrics=%METRICS%  person=%PERSON%
echo.
for /d %%D in ("%TESTS%\*") do (
    if /I not "%%~nxD"==".frames" (
        echo ============================================================
        echo === %%~nxD
        echo ============================================================
        call "%~dp0run_mind.bat" "%%~nxD" "%METRICS%" "%GPUS%" "%PERSON%"
    )
)

echo.
echo ============================================================
echo === combined scores table
echo ============================================================
set "JSONS="
for %%J in ("%~dp0result_*.json") do set "JSONS=!JSONS! "%%~fJ""
if defined JSONS ( "%PY%" "%~dp0_scores_table.py" !JSONS! ) else ( echo   no result_*.json yet )
endlocal
