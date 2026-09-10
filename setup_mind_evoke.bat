@echo off
setlocal enableextensions enabledelayedexpansion

echo.
echo ================================================================================
echo MIND SETUP -- Video Generation Evaluation for Evoke
echo ================================================================================
echo.

REM Check if running from MIND dir
if not exist "mind" (
    if not exist ".git" (
        echo ERROR: Run this script from C:\workspace\world\MIND
        exit /b 1
    )
)

where uv >nul 2>&1
if errorlevel 1 (
    set "UV_EXE=C:\Users\kschmid\.local\bin\uv.exe"
) else (
    set "UV_EXE=uv"
)
if not exist "!UV_EXE!" (
    if not "!UV_EXE!"=="uv" (
        echo ERROR: uv not found at !UV_EXE! and not on PATH.
        exit /b 1
    )
)

REM Virtual environment
echo [1/2] Setting up virtual environment (uv, Python 3.10)...
if not exist ".venv" (
    "!UV_EXE!" venv --python 3.10 .venv
    echo      Created .venv
)

REM Install dependencies
echo [2/2] Installing dependencies...
echo      Installing torch 2.7.0 with CUDA 12.8...
"!UV_EXE!" pip install --python .venv\Scripts\python.exe torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

echo      Installing MIND requirements...
if exist "requirements.txt" (
    "!UV_EXE!" pip install --python .venv\Scripts\python.exe -r requirements.txt
) else (
    echo      Installing core MIND packages...
    "!UV_EXE!" pip install --python .venv\Scripts\python.exe opencv-python pillow numpy einops scipy
    "!UV_EXE!" pip install --python .venv\Scripts\python.exe clip-benchmark clip-interrogator
)

echo.
echo ================================================================================
echo SETUP COMPLETE
echo ================================================================================
echo.
echo To evaluate Evoke videos with MIND:
echo   .venv\Scripts\python.exe evaluate_evoke.py --video outputs/t2v/geo_pred.mp4
echo.
echo For batch evaluation:
echo   .venv\Scripts\python.exe evaluate_evoke.py --video-dir C:\workspace\world\Evoke\outputs
echo.
exit /b 0
