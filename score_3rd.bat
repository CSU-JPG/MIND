@echo off
:: score_3rd.bat -- score first N 3rd-person samples WITH action, FRESH.
:: Parks existing result_dreamx-world_ar_*.json into results_bak\ first so process.py's
:: auto-resume can't pre-fill result_list (that double-counts --limit -> 0s instant-skip).
:: 1st-person results stay safe in results_bak\. One GPU -- stop any other scoring first.
::   score_3rd.bat            first 50 3rd-person
::   set N=30 & score_3rd.bat change the count

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

set "PY=%~dp0.venv\Scripts\python.exe"
set "GT=C:\workspace\world\MIND-Data"
set "TEST=C:\workspace\world\MIND-tests\dreamx-world_ar"
set "METRICS=lcm,visual,dino,action,gsc"
if not defined N set "N=50"

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%~dp0results_bak" mkdir "%~dp0results_bak"

echo Parking existing result JSONs (so 3rd scores fresh, no resume double-count)...
if exist "%~dp0result_dreamx-world_ar_*.json" move /y "%~dp0result_dreamx-world_ar_*.json" "%~dp0results_bak\" >nul

echo ============================================================
echo Scoring first %N% 3rd-person (action included)
echo ============================================================
"%PY%" src\process.py --gt_root "%GT%" --test_root "%TEST%" --metrics %METRICS% --num_gpus 1 --perspectives 3rd_data --limit %N%
if errorlevel 1 ( echo ERROR: 3rd-person pass failed & exit /b 1 )

echo ============================================================
echo Done. 3rd-person -> newest result_dreamx-world_ar_*.json (~%N%, with action).
echo 1st-person results preserved in results_bak\.
echo ============================================================
