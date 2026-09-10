@echo off
:: rerun_all.bat -- re-score dreamx-world_ar from scratch, then AUTO-RESUME on crashes.
:: Clears stale result JSONs ONCE (fresh start), scores both perspectives / all metrics.
:: process.py writes the result JSON every 0.5s; if scoring dies partway (ViPE/action
:: can crash), this re-runs run_mind WITHOUT clearing -- so it auto-resumes from the
:: partial JSON, skipping already-scored samples -- up to MAX_TRIES times until complete.
::
::   rerun_all.bat                 fresh start + auto-resume retries (default 20)
::   set MAX_TRIES=50 & rerun_all  raise the retry cap

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set "METRICS=lcm,visual,dino,action,gsc"
if not defined MAX_TRIES set "MAX_TRIES=20"

:: Fresh start: move stale result JSONs aside so the FIRST pass scores from scratch.
if not exist "%~dp0results_bak" mkdir "%~dp0results_bak"
if exist "%~dp0result_dreamx-world_ar_*.json" (
    echo Moving stale result JSONs to results_bak\ ...
    move /y "%~dp0result_dreamx-world_ar_*.json" "%~dp0results_bak\" >nul
)

set /a TRY=0
:retry
set /a TRY+=1
echo ============================================================
echo Scoring attempt !TRY!/%MAX_TRIES%  (auto-resumes from partial JSON after pass 1)
echo ============================================================
call "%~dp0run_mind.bat" dreamx-world_ar "%METRICS%" 1 both
set "RC=!ERRORLEVEL!"
if "!RC!"=="0" goto done

echo Attempt !TRY! exited with code !RC! ^(likely a ViPE/action crash^).
if !TRY! GEQ %MAX_TRIES% (
    echo Reached MAX_TRIES=%MAX_TRIES%; giving up. Partial results are in the newest result JSON.
    exit /b !RC!
)
echo Resuming in 5s -- run_mind will skip already-scored samples...
timeout /t 5 /nobreak >nul
goto retry

:done
echo ============================================================
echo Done. Newest result_dreamx-world_ar_*.json holds the full results.
echo ============================================================
endlocal
exit /b 0
