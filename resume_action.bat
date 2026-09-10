@echo off
:: resume_action.bat -- RESUME dreamx-world_ar scoring after a crash (action runs are
:: ~15h and ViPE can die partway). process.py saves the result JSON every 0.5s and
:: auto-resumes from the newest result_dreamx-world_ar_*.json, skipping already-scored
:: samples and continuing with the rest. Re-run this as many times as needed.
::
::   rerun_all.bat      = FRESH start (clears JSONs, no resume)
::   resume_action.bat  = CONTINUE from where it died (keeps JSONs, auto-resume)

setlocal enableextensions
cd /d "%~dp0"

if not exist "%~dp0result_dreamx-world_ar_*.json" (
    echo No result_dreamx-world_ar_*.json found -- nothing to resume.
    echo Use rerun_all.bat for a fresh start instead.
    exit /b 1
)

echo Resuming dreamx-world_ar scoring -- auto-resume from newest result JSON,
echo skipping already-scored samples. Metrics quoted so cmd keeps them as one arg.
call "%~dp0run_mind.bat" dreamx-world_ar "lcm,visual,dino,action,gsc" 1 both
