@echo off
setlocal enableextensions
REM ==========================================================================
REM Download the MIND benchmark GT dataset (first-person + third-person) from HF
REM (CSU-JPG/MIND) via the venv's huggingface_hub (no hf CLI needed). Lands under
REM ..\MIND-Data\, a sibling of this repository, which is what run_mind.sh sets
REM as gt_root and what the drive_* scripts resolve:
REM   ..\MIND-Data\{perspective}\test\{test_type}\...
REM This previously landed in .\mind_data, which nothing reads -- running both
REM this and `src\download_models.py --dataset` produced two 35 GB copies.
REM HF_TOKEN read from env if gated. Skips already-downloaded files (resumable).
REM
REM   download_mind_dataset.bat                 full dataset (both perspectives)
REM   download_mind_dataset.bat first_person    only that subfolder
REM ==========================================================================
set "PY=%~dp0.venv\Scripts\python.exe"
for %%I in ("%~dp0..\MIND-Data") do set "DEST=%%~fI"
if not exist "%PY%" ( echo ERROR: %PY% missing -- run setup_mind_venv.bat first & exit /b 1 )
if "%HF_TOKEN%"=="" echo NOTE: HF_TOKEN not set (ok if public)

echo Downloading CSU-JPG/MIND %~1 -^> %DEST%
"%PY%" -c "import sys; from huggingface_hub import snapshot_download; sub=sys.argv[1] if len(sys.argv)>1 else None; p=snapshot_download('CSU-JPG/MIND', repo_type='dataset', local_dir=r'%DEST%', allow_patterns=[sub+'/*'] if sub else None); print('downloaded ->', p)" %~1
if errorlevel 1 ( echo download failed & exit /b 1 )

echo.
echo Done. gt_root for process.py = %DEST%
endlocal
