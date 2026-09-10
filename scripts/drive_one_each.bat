@echo off
:: Smoke test: render ONE sample per model (--limit 1) across every driver, in
:: sequence (parallel GPU use crushes throughput). Each drive_<model>.bat scores
:: its single sample at the end (existing behavior), so you get one video + a
:: result_<model>_*.json per model -- a quick "does every model still run?" pass
:: and a 1-clip side-by-side.
::
:: Missing/broken drivers (no .bat, missing venv/weights) exit non-zero; we log
:: and continue. Summary at the end lists which succeeded.
::
:: Usage:
::   drive_one_each.bat                      one sample from every model below
::   drive_one_each.bat --dry-run            preview each driver's command only
::   drive_one_each.bat --perspective 1st_data
::
:: Override the model list (space-separated drive_<NAME>.bat stems):
::   set MIND_RUN_MODELS=dreamx matrix3 sana_wm
::   drive_one_each.bat

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Full set of models with real drivers on this box. Edit / override via
:: MIND_RUN_MODELS. Stems must match drive_<stem>.bat.
if not defined MIND_RUN_MODELS set MIND_RUN_MODELS=dreamx dreamx_ar matrix2 matrix3 matrix3_distilled sana_wm lingbot lingbot_flash deepverse helios_i2v hy_worldplay

echo ============================================================
echo MIND: one sample per model ^(--limit 1^)
echo   models    : %MIND_RUN_MODELS%
echo   extra args: %*
echo ============================================================

set OK_LIST=
set FAIL_LIST=

for %%M in (%MIND_RUN_MODELS%) do (
    echo.
    echo ============================================================
    echo === drive_%%M.bat --limit 1 %*
    echo ============================================================
    if exist "%~dp0drive_%%M.bat" (
        call "%~dp0drive_%%M.bat" --limit 1 %*
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
echo Summary ^(one sample per model^)
echo ============================================================
echo   OK   :%OK_LIST%
echo   FAIL :%FAIL_LIST%
echo.
endlocal
