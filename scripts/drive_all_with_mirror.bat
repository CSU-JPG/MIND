@echo off
:: One-shot runner: each active driver with --mirror-test (additive: regular + mirror).
::
:: Runs in sequence (NOT parallel) — GPU contention crushes throughput when
:: heavy generators overlap. Each driver scores itself at the end (existing
:: behavior), so result_<model>_*.json drops as each finishes.
::
:: Skips any driver whose prereqs are missing (venv, weights, etc.) — that
:: driver exits non-zero and we move on. Summary at end lists which succeeded.
::
:: Usage:
::   drive_all_with_mirror.bat                       all 5 drivers, default args
::   drive_all_with_mirror.bat --limit 5             pass-through to each driver
::   drive_all_with_mirror.bat --perspective 1st_data
::
:: Override the model list with MIND_RUN_MODELS (space-separated):
::   set MIND_RUN_MODELS=dreamx_small matrix3
::   drive_all_with_mirror.bat
::
:: To run mirror-only (skip the regular action_space + mem_test passes), pass
:: --mirror-only via %* — it propagates to every driver.

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

if not defined MIND_RUN_MODELS set MIND_RUN_MODELS=dreamx_small matrix3 sana_wm lingbot deepverse

echo ============================================================
echo MIND: drive all (with --mirror-test) ^| models: %MIND_RUN_MODELS%
echo extra args: %*
echo ============================================================

set OK_LIST=
set FAIL_LIST=

for %%M in (%MIND_RUN_MODELS%) do (
    echo.
    echo ============================================================
    echo === drive_%%M.bat --mirror-test %*
    echo ============================================================
    if exist "%~dp0drive_%%M.bat" (
        call "%~dp0drive_%%M.bat" --mirror-test %*
        if errorlevel 1 (
            set FAIL_LIST=!FAIL_LIST! %%M
            echo --- %%M FAILED ^(rc=!ERRORLEVEL!^), continuing ---
        ) else (
            set OK_LIST=!OK_LIST! %%M
        )
    ) else (
        set FAIL_LIST=!FAIL_LIST! %%M^(missing-bat^)
        echo --- drive_%%M.bat NOT FOUND, skipping ---
    )
)

echo.
echo ============================================================
echo Summary
echo ============================================================
echo   OK   :%OK_LIST%
echo   FAIL :%FAIL_LIST%
echo.
endlocal
