@echo off
:: score_subset.bat -- bounded action scoring: N 1st-person, then N 3rd-person. Sequential (one GPU).
::
:: FIX: before each pass, PARK existing result_dreamx-world_ar_*.json into results_bak\ so
:: process.py's auto-resume can't pre-fill result_list. Previously the resumed entries
:: satisfied the --limit quota, so the monitor fired the stop-event immediately and 0 new
:: samples were scored (the 0s "1200 video/s" instant-skip). Parking = each pass scores fresh.
::
::   score_subset.bat                  do 1st then 3rd
::   set SKIP_1ST=1 & score_subset.bat skip the 1st pass (1st already scored)
::   set N=30 & score_subset.bat       change the per-perspective count
::
:: After it runs: 1st result is parked in results_bak\, 3rd is the newest
:: result_dreamx-world_ar_*.json in this folder. (Stop any other scoring first -- one GPU.)

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

if defined SKIP_1ST goto third
echo ============================================================
echo [1/2] First %N% 1st-person (FRESH -- parking old JSONs first)
echo ============================================================
if exist "%~dp0result_dreamx-world_ar_*.json" move /y "%~dp0result_dreamx-world_ar_*.json" "%~dp0results_bak\" >nul
"%PY%" src\process.py --gt_root "%GT%" --test_root "%TEST%" --metrics %METRICS% --num_gpus 1 --perspectives 1st_data --limit %N%
if errorlevel 1 ( echo ERROR: 1st-person pass failed & exit /b 1 )

:third
echo ============================================================
echo [2/2] First %N% 3rd-person (FRESH -- parking old JSONs first)
echo ============================================================
if exist "%~dp0result_dreamx-world_ar_*.json" move /y "%~dp0result_dreamx-world_ar_*.json" "%~dp0results_bak\" >nul
"%PY%" src\process.py --gt_root "%GT%" --test_root "%TEST%" --metrics %METRICS% --num_gpus 1 --perspectives 3rd_data --limit %N%
if errorlevel 1 ( echo ERROR: 3rd-person pass failed & exit /b 1 )

echo ============================================================
echo Done. 3rd -> newest result_dreamx-world_ar_*.json; 1st parked in results_bak\.
echo (Each ~%N% samples, with action.)
echo ============================================================
