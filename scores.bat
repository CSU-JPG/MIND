@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Print the combined MIND scores table from the EXISTING result_*.json files.
REM No scoring is run -- this just reads what's already there.
REM   scores.bat                    all result_*.json
REM   scores.bat matrix-game-3      only matching result files
REM ==========================================================================
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PAT=result_*.json"
if not "%~1"=="" set "PAT=result_%~1*.json"

REM _scores_table.py prints ONE file; pass the NEWEST match (by mtime) so you see the
REM latest run, not the oldest (which the glob returns first).
set "NEWEST="
for /f "delims=" %%J in ('dir /b /o-d "%~dp0%PAT%" 2^>nul') do if not defined NEWEST set "NEWEST=%~dp0%%J"
if not defined NEWEST ( echo   no result files matching %PAT% & exit /b 1 )
"%PY%" "%~dp0_scores_table.py" "%NEWEST%"

echo.
echo ================= by perspective =================
for %%P in (1st_data 3rd_data) do (
    "%PY%" "%~dp0_split_persp.py" "%NEWEST%" %%P "%TEMP%\_mind_%%P.json" >nul 2>nul
    echo --- %%P ---
    "%PY%" "%~dp0_scores_table.py" "%TEMP%\_mind_%%P.json"
    del "%TEMP%\_mind_%%P.json" 2>nul
)
endlocal
