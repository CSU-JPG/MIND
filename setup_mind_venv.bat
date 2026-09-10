@echo off
:: Build MIND's scoring venv (Python 3.10, Windows + Blackwell sm_120 / cu130).
:: Runs staging (drive_*.py) + scoring (process.py). DreamX-World inference is
:: cross-spawned in its OWN venv via DREAMX_VENV_PY, so this venv does NOT need
:: the DreamX stack. ViPE (action metric) is built separately via build_vipe.bat.
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set "UV=C:\Users\kschmid\.local\bin\uv.exe"
set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

if not exist "%UV%" ( echo ERROR: uv not found at %UV% & exit /b 1 )

echo [mind-setup] creating venv (Python 3.10)...
"!UV!" venv --python 3.10 "%VENV%"
if not exist "%PY%" ( echo ERROR: venv create failed & exit /b 1 )

echo [mind-setup] torch + torchvision (cu130, Blackwell sm_120)...
"!UV!" pip install --python "%PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 ( echo ERROR: torch cu130 install failed & exit /b 1 )

echo [mind-setup] MIND requirements (transformers 4.56, torchmetrics, lpips, pyiqa, clip, modelscope, av)...
"!UV!" pip install --python "%PY%" -r envs\requirements.txt
if errorlevel 1 ( echo ERROR: requirements install failed & exit /b 1 )

echo.
echo ============================================================
echo [mind-setup] DONE. venv: %VENV%
echo ============================================================
echo Metrics ready now: lcm, visual, dino, gsc.
echo The 'action' metric needs ViPE - build it separately: build_vipe.bat
echo Then run:  drive_dreamx_ar.bat --limit 2   (DreamX inference uses its own venv)
exit /b 0
