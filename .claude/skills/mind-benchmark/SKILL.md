---
name: mind-benchmark
description: "Benchmark an interactive world model (Waypoint, ABot, matrix, dreamx, sana, ...) on the MIND benchmark at C:\\workspace\\world\\MIND. Use when driving a model over MIND, scoring memory-consistency/action-control metrics, adding a new model driver, running the mirror test, or comparing models. Covers the drive->score->compare flow and its Windows/PowerShell gotchas."
---

# MIND world-model benchmarking

MIND (`C:\workspace\world\MIND`) scores how well an interactive world model keeps **memory consistency** and **action control**. GT is at `C:\workspace\world\MIND-Data` (`{1st_data,3rd_data}/test/{action_space_test,mem_test,mirror_test}`). Driven videos go to `C:\workspace\world\MIND-tests\<model>\...`. Scores are `result_<model>_<timestamp>.json` in the MIND root.

## The pipeline (3 stages)
1. **Drive** a model on MIND actions -> videos in `MIND-tests\<model>`.
2. **Score** vs GT -> `result_<model>_*.json`.
3. **Compare** via the scores table.

```
.\drive_<model>.bat --limit 5        :: smoke first
.\drive_<model>.bat                  :: full (both perspectives)
.\drive_<model>.bat --mirror-test    :: mirror clips (separately)
.\run_mind.bat --% <model> "" 1 both :: score all 5 metrics ("" = default)
.\scores.bat <model>                 :: view newest result table
```

## Metrics (arrows = better direction)
`lcm`↑ (memory consistency, PSNR-like), `visual`↑ (quality), `dino`↓ (semantic drift), `avg_mse`↓ (pixel err), `gsc`↓ (mirror consistency — splits clip, time-flips return vs outbound; lower=you returned to the same view), `action` (control). Default = all 5. `gsc` needs mirror clips; `action` needs ViPE (JIT-compiles a CUDA ext first use; `run_mind.bat` sets up vcvars + CUDA_HOME) and is SLOW (~300 s/video vs ~15 s for the other 4).
- **`action` often writes nothing (n/a) even when requested** — a known scoring bug; the other 5 metrics still give a clear verdict. Don't block a comparison on it.
- Result JSON = `{data:[{path, perspective, test_type, lcm{}, visual_quality{}, dino{}, ...}], ...}`; metrics are nested dicts, so reduce via `_scores_table.py`, not by hand.

## Reading / splitting results
`scores.bat <model>` shows the newest result (aggregate over both perspectives). To split by perspective, filter `data` to `perspective=="1st_data"`/`"3rd_data"` into a temp JSON and run `_scores_table.py` on each. The `150/100` sample split in a full run = 150 action_space+mem (lcm/visual/dino/mse) + 100 mirror (gsc).

## Gotchas (these bite every time)
- **PowerShell comma trap:** `run_mind.bat waypoint lcm,visual,dino ...` fails ("4th arg ... got: dino") because cmd splits commas. Fix: `run_mind.bat --% <model> "lcm,visual,dino" 1 both` (`--%` AND quotes), or just pass `""` to get the all-5 default.
- **Resume adds samples, NOT metrics:** if a model was scored with fewer metrics, `run_mind` **skips** those samples forever. To add `action`/`gsc`, `del result_<model>_*.json` then re-score fresh.
- **`.bat` in PowerShell needs `.\`**; `.bat` files must be CRLF.
- **CPU torch:** the MIND venv can silently get `torch+cpu` -> `process.py` sees 0 GPUs -> `mp.Pool(0)` crash. Reinstall the **ViPE-compatible** CUDA build (see below), NOT an arbitrary one.
- **Free the GPU** before scoring/driving (shared 32 GB); check `nvidia-smi` first.

## action / ViPE (the fragile metric)
`action` shells out to a **`vipe` CLI** (built into the MIND venv via `build_vipe.bat`). It is PINNED to a torch/ABI:
- MIND venv must be **torch 2.10.0 + cu128** (has CUDA + sm_120 for the 5090) + **flash_attn 2.8.3+cu128torch2.10-cp310** (mjun0812 prebuilt wheel, `--no-build-isolation`). Then `build_vipe.bat` (needs MSVC/vcvars + cl.exe) produces `.venv\Scripts\vipe.exe`.
- **`build_vipe.bat`'s `CUDA_HOME` must match torch's CUDA.** For torch cu128 set `CUDA_HOME=...\CUDA\v12.8` (NOT v13.0). Wrong version -> `RuntimeError: detected CUDA version mismatches` during the native build. The bat had v13.0 hardcoded — check line ~33 matches your torch.
- **Reinstalling a different torch (e.g. cu130 / 2.13) WIPES the ViPE build** -> `action` crashes per-sample with `ViPE extract_traj: FileNotFoundError [WinError 2]` (the `vipe` exe is gone) and silently writes n/a. This exact mistake cost hours. If you must fix CPU-torch, reinstall **torch==2.10.0 --index-url .../cu128**, re-add the flash_attn wheel, then re-run `build_vipe.bat`.
- The editable build does NOT pull ViPE's runtime deps -> `vipe.exe` crashes with `ModuleNotFoundError: No module named 'hydra'`. Fix: `uv pip install --python .venv\Scripts\python.exe hydra-core` (install any other missing runtime deps the CLI reports too).
- Verify before scoring with action: `.venv\Scripts\vipe.exe --help` should list `infer`/`visualize` (no traceback).
- Full fix chain: torch 2.10/cu128 -> flash_attn wheel (`--no-build-isolation`) -> CUDA_HOME=v12.8 -> `build_vipe.bat` -> `hydra-core` -> `vipe.exe --help` works.
- `action` is ALSO slow (~300 s/video vs ~15 s for the other 4). For a fast comparison, exclude it: `"lcm,visual,dino,gsc"`.

## Adding a new model driver
Model runs in ITS OWN venv/project. Write `src\drive_<model>.py` + `drive_<model>.bat` (runs via that model's python). Pattern:
- `gather()` walks `{persp}/test/{action_space_test,mem_test}/<name>/{video.mp4,action.json}`; output `MIND-tests\<model>\{persp}\{test_type}\<name>\video.mp4`; skip-if-exists.
- **Convert MIND actions** (`ws/ad/ud/lr` tri-state: 0=neutral,1/2=dirs) to the model's control format. Examples: Waypoint (world_engine) -> Windows VK codes (W=0x57,S=0x53,A=0x41,D=0x44) + mouse for lr/ud. ABot -> `{keys:{W,A,S,D,I,J,K,L}}` (WASD move, IJKL look; lr->J/L, ud->I/K).
- **Mirror test:** `from utils.mirror_test_utils import gather_mirror_samples, MIRROR_DEFAULT_ACTION`; seed each `data-NN.png` (a PNG, not a video) + replay the `-w.json` go-then-return; output under `mirror_test`.
- **LOAD-ONCE for speed:** never subprocess-per-sample (reloads the model each time). Either loop in-process (Waypoint keeps the engine loaded), or build a manifest + one call to a batch mode (ABot's `inference.py --mind-batch`, which loops the lru_cached pipeline). Quantize for more speed (ABot `--quant fp8-per-tensor` on Blackwell).

## Existing drivers
`drive_waypoint.bat` (world_engine, scope-overworld venv), `drive_abot.bat` (ABot venv, `--mind-batch` load-once + FP8), plus `drive_matrix3/dreamx/sana/lingbot/deepverse.bat` (each needs its backend installed). `drive_all_with_mirror.bat` runs the default set. `scores.bat` shows the newest result; `scores.bat` (no arg) = all models.
