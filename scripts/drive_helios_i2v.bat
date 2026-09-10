@echo off
:: Stage PKU-YuanGroup/Helios (i2v) videos into MIND-tests\helios-i2v\ for
:: run_mind.bat scoring. Parallel to drive_matrix3.bat / drive_matrix3_distilled.bat.
::
:: Usage:
::   drive_helios_i2v.bat                            stage 1st + 3rd, score both
::   drive_helios_i2v.bat --dry-run                  preview commands
::   drive_helios_i2v.bat --limit 5                  first 5 samples per perspective
::   drive_helios_i2v.bat --test-type mem_test       limit to memory tests
::   drive_helios_i2v.bat --perspective 1st_data     override (default = both)
::   drive_helios_i2v.bat --low-vram                 pass --enable_low_vram_mode to Helios
::
:: Metric selection (forwarded to run_mind.bat after staging):
::   set MIND_METRICS=lcm,visual                pick a subset
::   (unset)                                    default = lcm,visual,dino,gsc (no action)
::   set MIND_GPUS=2                            multi-GPU scoring
::   set MIND_PERSON=1st                        person = 1st | 3rd | both (default both)
::   set MIND_MIRROR_TEST=0                     disable mirror_test (default on)
::   set MIND_START_INDEX=N                     resume mid-run
::   set MIND_FPS=16                            override generation fps (default 24)
::
:: Cross-venv knobs:
::   set HELIOS_VENV_PY=<path>                  override Helios venv python
::                                              (default C:\workspace\world\Helios\.venv\Scripts\python.exe)
::
:: All --flags pass through to src\drive_helios_i2v.py; MIND_* env vars stay in this bat.

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set LOG=%~dp0drive_helios_i2v.log

if not defined HELIOS_VENV_PY set HELIOS_VENV_PY=C:\workspace\world\Helios\.venv\Scripts\python.exe

if not defined MIND_FPS set MIND_FPS=24

:: Force variant to base (single-stage full denoising). Mirrors run_helios.bat's
:: default. Override with `set HELIOS_VARIANT=distilled` BEFORE invoking this
:: bat. Without this explicit set, the env-inherited value could be wrong (we
:: saw a session env still carrying "distilled" from a prior run_helios.bat).
if not defined HELIOS_VARIANT set "HELIOS_VARIANT=base"

:: Per-variant num_inference_steps default to MATCH run_helios.bat's variant
:: logic: Distilled uses ~4 pyramid-distilled steps with stage2 ON; Base/Mid use
:: ~30 full denoise steps with stage2 OFF. drive_helios_i2v.py defaults to 4
:: (Distilled-tuned); we override here when the variant is Base/Mid.
if not defined MIND_NUM_INFERENCE_STEPS (
    if /I "!HELIOS_VARIANT!"=="distilled" (
        set "MIND_NUM_INFERENCE_STEPS=4"
    ) else (
        set "MIND_NUM_INFERENCE_STEPS=30"
    )
)

:: Default to --low-vram (group offload). Matches run_helios.bat's default and
:: is required for Helios-Base on a 32 GB 5090 -- without it the transformer
:: + activations exhaust VRAM and OOM during the first denoise step. To opt
:: out (e.g. on an A6000 / L40S), pass --high-vram on the cmd line.
echo %* | findstr /I /C:"--high-vram" >nul
if errorlevel 1 (
    echo %* | findstr /I /C:"--low-vram" >nul
    if errorlevel 1 set "_LOW_VRAM_DEFAULT=--low-vram"
)

:: Prompt suffix gets appended to every per-sample prompt (after the
:: action.json caption or perspective default). The MIND benchmark's
:: action_space_test samples don't bake motion descriptions into their
:: captions, so feeding Helios "moving forward" gives a steering signal
:: that improves i2v motion coherence. Override via MIND_PROMPT_SUFFIX env
:: (set to empty string to suppress entirely).
if not defined MIND_PROMPT_SUFFIX set "MIND_PROMPT_SUFFIX=moving forward"

if not exist "%PY%" (
    echo ERROR: MIND venv python not found: %PY%
    exit /b 2
)
if not exist "%HELIOS_VENV_PY%" (
    echo ERROR: Helios venv python not found: %HELIOS_VENV_PY%
    echo Run: C:\workspace\world\Helios\run_helios.bat --setup --skip-run
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo Helios (i2v) staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : helios-i2v
echo   helios_py    : %HELIOS_VENV_PY%
echo   variant      : %HELIOS_VARIANT%  (matches run_helios.bat --variant)
echo   steps        : %MIND_NUM_INFERENCE_STEPS%  (per-variant default; override with MIND_NUM_INFERENCE_STEPS=N)
echo   prompt suffix: %MIND_PROMPT_SUFFIX%  (override with MIND_PROMPT_SUFFIX=...)
echo   fps          : %MIND_FPS%
echo   log          : %LOG%
echo ============================================================

if not defined MIND_START_INDEX set MIND_START_INDEX=0
if not defined MIND_MIRROR_TEST  set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

REM ---- start wall-clock + RAM/GPU sampling ---------------------------------
for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_START=%%T"

REM Snapshot pre-staging .mp4 count + frame count assumption so we can compute
REM real-fps (frames generated per wall-clock second) after staging finishes.
REM Frames-per-video defaults to 99 (Helios default); override via MIND_NUM_FRAMES
REM or by passing --num_frames N in %* (the latter is forwarded to Python).
if not defined MIND_NUM_FRAMES set "MIND_NUM_FRAMES=99"
for /f %%C in ('powershell -NoProfile -Command "(Get-ChildItem -LiteralPath '%MIND_TESTS%\helios-i2v' -Recurse -Filter *.mp4 -File -ErrorAction SilentlyContinue | Measure-Object).Count"') do set "_MP4_COUNT_BEFORE=%%C"

set "_METRICS_FILE=%TEMP%\helios_drive_ram_%RANDOM%.txt"
set "_GPU_FILE=%TEMP%\helios_drive_gpu_%RANDOM%.txt"
echo 0 0 > "%_METRICS_FILE%"
echo 0 > "%_GPU_FILE%"
start /b "" powershell -NoProfile -WindowStyle Hidden -Command "$peak=0; while ($true) { try { $m=(Get-Process python -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Maximum).Maximum; if ($m -and $m -gt $peak) { $peak=$m; '{0} {1:N2}' -f $peak, ($peak/1GB) | Out-File -LiteralPath '%_METRICS_FILE%' -Force -Encoding ascii } } catch {}; Start-Sleep -Seconds 2 }"
start /b "" powershell -NoProfile -WindowStyle Hidden -Command "$peak=0; while ($true) { try { $m=(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | ForEach-Object { [int]$_.Trim() } | Measure-Object -Maximum).Maximum; if ($m -and $m -gt $peak) { $peak=$m; $peak | Out-File -LiteralPath '%_GPU_FILE%' -Force -Encoding ascii } } catch {}; Start-Sleep -Seconds 2 }"

REM ---- staging: ONE worker process handles BOTH perspectives --------------
REM drive_helios_i2v.py spawns a single persistent Helios worker that loads the
REM model once and iterates the whole manifest with no further reloads. With no
REM --perspective it stages both 1st_data + 3rd_data in that one process; if the
REM caller passes --perspective in %* the driver filters to it. Either way the
REM model loads exactly once. Do NOT loop perspectives here -- that respawns the
REM worker and reloads the model per perspective.
"%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\drive_helios_i2v.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--fps" "%MIND_FPS%" "--start-index" "%MIND_START_INDEX%" "--num-inference-steps" "%MIND_NUM_INFERENCE_STEPS%" "--prompt-suffix" "%MIND_PROMPT_SUFFIX%" %_LOW_VRAM_DEFAULT% %MIRROR_ARG% %*
if errorlevel 1 (
    echo ERROR: drive_helios_i2v.py exited with !ERRORLEVEL!
    goto :stop_samplers
)
set "EXIT_CODE=0"
goto :after_staging

:stop_samplers
set "EXIT_CODE=!ERRORLEVEL!"
if "!EXIT_CODE!"=="0" set "EXIT_CODE=1"

:after_staging
REM Stop samplers regardless of success/failure
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'powershell.exe' -and ($_.CommandLine -match 'helios_drive_ram_' -or $_.CommandLine -match 'helios_drive_gpu_') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_END=%%T"
for /f %%E in ('powershell -NoProfile -Command "$d=([TimeSpan]::FromTicks(!_T_END! - !_T_START!)); '{0:00}:{1:00}:{2:00}' -f $d.Hours,$d.Minutes,$d.Seconds"') do set "_T_ELAPSED=%%E"
for /f %%S in ('powershell -NoProfile -Command "[Math]::Round(([TimeSpan]::FromTicks(!_T_END! - !_T_START!)).TotalSeconds, 1)"') do set "_T_ELAPSED_SEC=%%S"

set "_RAM_PEAK_GB=?"
for /f "tokens=2" %%R in ('type "!_METRICS_FILE!" 2^>nul') do set "_RAM_PEAK_GB=%%R"
set "_GPU_PEAK_MIB=?"
for /f %%G in ('type "!_GPU_FILE!" 2^>nul') do set "_GPU_PEAK_MIB=%%G"
del /q "!_METRICS_FILE!" "!_GPU_FILE!" >nul 2>nul

REM Compute real fps = (videos_generated_this_run * frames_per_video) / elapsed_seconds
for /f %%C in ('powershell -NoProfile -Command "(Get-ChildItem -LiteralPath '%MIND_TESTS%\helios-i2v' -Recurse -Filter *.mp4 -File -ErrorAction SilentlyContinue | Measure-Object).Count"') do set "_MP4_COUNT_AFTER=%%C"
set /a "_MP4_DELTA=!_MP4_COUNT_AFTER! - !_MP4_COUNT_BEFORE!"
for /f %%F in ('powershell -NoProfile -Command "if (!_T_ELAPSED_SEC! -gt 0 -and !_MP4_DELTA! -gt 0) { [Math]::Round((!_MP4_DELTA! * !MIND_NUM_FRAMES!) / !_T_ELAPSED_SEC!, 2) } else { 0 }"') do set "_REAL_FPS=%%F"
for /f %%V in ('powershell -NoProfile -Command "if (!_MP4_DELTA! -gt 0) { [Math]::Round(!_T_ELAPSED_SEC! / !_MP4_DELTA!, 1) } else { 0 }"') do set "_SEC_PER_VIDEO=%%V"

echo.
echo --- staging elapsed: !_T_ELAPSED!  ^| peak python RAM: !_RAM_PEAK_GB! GB  ^| peak GPU VRAM: !_GPU_PEAK_MIB! MiB ---
echo --- videos generated: !_MP4_DELTA!  ^| ~!_SEC_PER_VIDEO! s/video  ^| real fps: !_REAL_FPS! frames/sec wall-clock ---

if not "!EXIT_CODE!"=="0" (
    echo.
    echo ERROR: staging failed with exit code !EXIT_CODE!
    exit /b !EXIT_CODE!
)

if not defined MIND_PERSON  set MIND_PERSON=both
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,gsc
if not defined MIND_GPUS    set MIND_GPUS=1

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat helios-i2v "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
echo ============================================================
:: Quote MIND_METRICS — CMD splits unquoted comma-bearing args.
call "%~dp0run_mind.bat" helios-i2v "%MIND_METRICS%" %MIND_GPUS% %MIND_PERSON%
exit /b %ERRORLEVEL%
