@echo off
:: Score a test video set against MIND-Data ground truth.
::
:: Usage:
::   run_mind.bat                                          scores matrix-game-3, all metrics, both perspectives
::   run_mind.bat <test_subdir>                            scores MIND-tests\<test_subdir>
::   run_mind.bat <test_subdir> <metrics>                  custom metrics
::   run_mind.bat <test_subdir> <metrics> <gpus>           multi-GPU
::   run_mind.bat <test_subdir> <metrics> <gpus> <person>  person = 1st | 3rd | both (default: both)
::
:: Examples:
::   run_mind.bat matrix-game-3 lcm,visual
::   run_mind.bat dreamx-world lcm,visual,dino,action,gsc 1
::   run_mind.bat dreamx-world lcm,visual,dino,gsc 1 1st

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
:: Strip cross-venv pollution. The PowerShell session that launched this bat
:: often has VIRTUAL_ENV / PYTHONHOME / PYTHONPATH set from another project
:: (e.g. scope's uv-managed 3.12 venv). MIND uses python 3.10, so if those
:: leak through they make Python try to graft scope's 3.12 site-packages onto
:: MIND's 3.10 stdlib path, causing _sre.MAGIC mismatch at startup.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONSTARTUP="
set "VIRTUAL_ENV="
set "VIRTUAL_ENV_PROMPT="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="

:: action metric runs ViPE, which JIT-compiles a CUDA C++ extension on first
:: call. That needs (a) cl.exe on PATH (MSVC) and (b) CUDA_HOME pointing at
:: the toolkit install. Set both up here so process.py doesn't crash on
:: `vipe infer ... --pipeline default`.
if not defined CUDA_HOME (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" (
        set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" (
        set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    )
)
:: Bring in MSVC's cl.exe + INCLUDE/LIB. vcvars64.bat appends to PATH; idempotent
:: enough to re-run, but we check VSINSTALLDIR to avoid double-init noise.
if not defined VSINSTALLDIR (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    )
)
if defined CUDA_HOME set "PATH=%CUDA_HOME%\bin;%PATH%"

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests

set TEST_SUBDIR=%~1
set METRICS=%~2
set NUM_GPUS=%~3
set PERSON=%~4

if not defined TEST_SUBDIR set TEST_SUBDIR=matrix-game-3
:: Default METRICS to all 5 if missing OR explicitly empty (passing "" still
:: defines METRICS=<empty>, which would skip the `if not defined` guard).
if not defined METRICS set METRICS=lcm,visual,dino,action,gsc
if "%METRICS%"=="" set METRICS=lcm,visual,dino,action,gsc
if not defined NUM_GPUS set NUM_GPUS=1
:: 4th positional defaults to "1st" — driver bats (drive_dreamx, drive_lingbot,
:: drive_matrix3) all generate 1st_data only. Override with "3rd" or "both".
if not defined PERSON set PERSON=1st

:: 4th positional = perspective filter. Accepts: 1st, 3rd, both.
set PERSPECTIVES=
if /I "%PERSON%"=="1st"  set PERSPECTIVES=1st_data
if /I "%PERSON%"=="3rd"  set PERSPECTIVES=3rd_data
if /I "%PERSON%"=="both" set PERSPECTIVES=
if defined PERSON if not defined PERSPECTIVES if /I not "%PERSON%"=="both" (
    echo ERROR: 4th arg must be 1st, 3rd, or both ^(got: %PERSON%^)
    exit /b 2
)

:: Accept either a bare subdir name (resolved under MIND_TESTS) or an absolute
:: path. Detection: an absolute path contains ":\" or starts with "\\" (UNC).
:: This lets you tab-complete MIND-tests\dreamx-world_small in the shell without
:: tripping the prefix-prepend below.
echo %TEST_SUBDIR% | findstr /R /C:":\\" /C:"^\\\\" >nul
if errorlevel 1 (
    set TEST_ROOT=%MIND_TESTS%\%TEST_SUBDIR%
) else (
    set TEST_ROOT=%TEST_SUBDIR%
)

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    echo Make sure MIND is set up.
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)
if not exist "%TEST_ROOT%" (
    echo ERROR: test_root not found: %TEST_ROOT%
    echo Stage videos there first via drive_matrix3.py / drive_mind.py / score_matrix3.py.
    exit /b 2
)

set LOG=%~dp0run_mind_%TEST_SUBDIR%.log

echo ============================================================
echo MIND scoring
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %TEST_ROOT%
echo   metrics      : %METRICS%
echo   gpus         : %NUM_GPUS%
echo   person       : %PERSON% (perspectives=%PERSPECTIVES%)
echo   log          : %LOG%
echo ============================================================

:: Use run_dreamx.ps1 to tee scoring output to both terminal AND a timestamped
:: log file. Same wrapper that drive_*.bat use for generation logs.
if defined PERSPECTIVES (
    "%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\process.py" "--gt_root" "%GT_ROOT%" "--test_root" "%TEST_ROOT%" "--metrics" "%METRICS%" "--num_gpus" "%NUM_GPUS%" "--perspectives" "%PERSPECTIVES%"
) else (
    "%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\process.py" "--gt_root" "%GT_ROOT%" "--test_root" "%TEST_ROOT%" "--metrics" "%METRICS%" "--num_gpus" "%NUM_GPUS%"
)

set EXIT_CODE=%ERRORLEVEL%

:: Resolve the newest result_<test>_*.json in %~dp0 so the Done block can
:: print an exact path. process.py mints the timestamp at runtime so the bat
:: doesn't know it up front. Run the FOR loop BEFORE the if/else so we don't
:: have `goto` inside parens (cmd treats that as falling out of the block,
:: which causes the else branch to also fire — that's the bug we hit before).
set "RESULT_PATH=(none found)"
for /f "delims=" %%F in ('dir /b /o-d "%~dp0result_%TEST_SUBDIR%_*.json" 2^>nul') do (
    if not defined _RESULT_PICKED (
        set "RESULT_PATH=%~dp0%%F"
        set "_RESULT_PICKED=1"
    )
)
set "_RESULT_PICKED="

if %EXIT_CODE%==0 (
    echo.
    echo ============================================================
    echo Done.
    echo   log    : %LOG%
    echo   result : %RESULT_PATH%
    echo ============================================================
) else (
    echo.
    echo ERROR: process.py exited with %EXIT_CODE%
    echo   log : %LOG%
)
exit /b %EXIT_CODE%
