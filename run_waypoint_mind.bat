@echo off
setlocal enableextensions
REM ==========================================================================
REM End-to-end Waypoint-1.5 on MIND: drive -> score -> print table, in one go.
REM Handles the comma-metrics quoting internally (bat-to-bat, no PowerShell trap).
REM Drive args pass through (e.g. --limit 5, --perspective 3rd_data). Metrics/person
REM overridable via MIND_METRICS / MIND_PERSON env vars.
REM
REM   run_waypoint_mind.bat                 full 1st-person: drive + score + table
REM   run_waypoint_mind.bat --limit 5       quick smoke
REM ==========================================================================
cd /d "%~dp0"
if not defined MIND_METRICS set "MIND_METRICS=lcm,visual,dino"
if not defined MIND_PERSON  set "MIND_PERSON=1st"

echo ============================================================
echo [1/3] DRIVE Waypoint-1.5 over MIND
echo ============================================================
call "%~dp0drive_waypoint.bat" %*
if errorlevel 1 ( echo drive step failed & exit /b 1 )

echo.
echo Done driving -^> MIND-tests\waypoint\. Scoring removed (run separately when ready):
echo   run_mind.bat --%% waypoint lcm,visual,dino 1 both
echo   scores.bat waypoint
endlocal
