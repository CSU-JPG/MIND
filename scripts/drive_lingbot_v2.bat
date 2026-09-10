@echo off
setlocal enableextensions
REM ==========================================================================
REM Drive LingBot-World-2 (Wan-A14B) over MIND, in WSL (fp8, single 5090).
REM Synthesizes poses.npy from MIND actor_pos/rpy + wasd/ijkl from ws/ad/ud/lr,
REM runs generate.py per sample, moves output to MIND-tests\lingbot-v2\.
REM SLOW: ~1 fps, 14B, reload per sample -> smoke with --limit 1 FIRST.
REM
REM   drive_lingbot_v2.bat --limit 1        smoke (one sample)
REM   drive_lingbot_v2.bat --perspective 1st_data
REM ==========================================================================
wsl -e bash -lc "cd /mnt/c/workspace/world/lingbot-world-v2; /home/kschmid/lingbot-venv/bin/python /mnt/c/workspace/world/MIND/src/drive_lingbot_v2.py %*"
echo.
echo Videos -^> C:\workspace\world\MIND-tests\lingbot-v2\
echo Then score:  run_mind.bat --%% lingbot-v2 "lcm,visual,dino,action,gsc" 1 both
endlocal
