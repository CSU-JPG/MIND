@echo off
:: Launch the FastVideo MatrixGame streaming demo from MIND.
::
:: Unlike the other drive_*.bat scripts, this is a LIVE PREVIEW launcher —
:: game-streaming-poc writes an MJPEG HTTP stream, not MP4 files, so the
:: output cannot feed MIND's offline metrics. Useful for visually sanity-
:: checking the Matrix-Game-2.0 FastVideo variants alongside MIND runs.
::
:: Usage:
::   drive_fastvideo_stream.bat                       prompted for actions
::   drive_fastvideo_stream.bat "wu wu ai dl"          one-shot
::   drive_fastvideo_stream.bat "wa wa wd sq" --loops 3 --variant gta_distilled_model
::
:: Args after the action string pass through to game_streaming.py unchanged.
:: Open http://localhost:8080/ in a browser to view, or:
::   ffplay http://localhost:8080/stream.mjpg

setlocal EnableExtensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set POC=C:\workspace\world\fastvideo-dynamo\game-streaming-poc\game_streaming.py

if not exist "%PY%" (
    echo ERROR: MIND venv python not found: %PY%
    exit /b 2
)
if not exist "%POC%" (
    echo ERROR: game_streaming.py not found at %POC%
    echo Did you clone fastvideo-dynamo? Run its setup.bat to install fastvideo first.
    exit /b 2
)

:: Verify fastvideo is installed (setup.bat in game-streaming-poc handles this).
"%PY%" -c "import fastvideo" 2>nul
if errorlevel 1 (
    echo ERROR: fastvideo not importable in MIND's venv.
    echo Run: C:\workspace\world\fastvideo-dynamo\game-streaming-poc\setup.bat
    exit /b 2
)

set ACTIONS=%~1
if defined ACTIONS shift

echo ============================================================
echo FastVideo MatrixGame stream
echo ============================================================
echo   poc       : %POC%
echo   python    : %PY%
if defined ACTIONS echo   actions   : %ACTIONS%
echo   browser   : http://localhost:8080/
echo ============================================================

if defined ACTIONS (
    "%PY%" "%POC%" --actions "%ACTIONS%" %*
) else (
    "%PY%" "%POC%" %*
)
exit /b %ERRORLEVEL%
