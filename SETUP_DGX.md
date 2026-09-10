# MIND on the GB300 (pmgb300ws-0304)

What it took to get MIND generating and scoring on the aarch64 Grace Blackwell
box, and the failures worth not rediscovering. Paths assume `~/MIND` with
`MIND-Data` and `MIND-tests` as siblings.

## Prerequisites

| | Location | Size |
| --- | --- | --- |
| Dataset | `~/MIND-Data` | 35 GB |
| DINOv3 | `~/MIND/dinov3_vitb16` | ~1 GB |
| MUSIQ / Aesthetic / CLIP | `~/.cache/mind` | ~1 GB |
| ViPE | `~/MIND/vipe`, installed editable | built from source |

`src/download_models.py` fetches all of the model artifacts and knows the right
repo ids — there is no need to find them by hand:

```bash
~/MIND/.venv/bin/python ~/MIND/src/download_models.py
```

DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) is **gated by Meta**.
Without access it fails with a 401 rather than a missing-file error, and the
download partially succeeds first — README and LICENSE come down before the
weights are refused, which makes it look like a network problem. Request access
on the model page and export `HF_TOKEN` before running.

Only `MIND-Data/*/test/` is read during scoring. The `train/` split is dead
weight for benchmarking.

## Building ViPE

Required for the `action` metric, which is the one that measures whether the
model followed the commanded trajectory. Three separate failures stack up here,
each hidden behind the previous one:

**1. `error: invalid command 'bdist_wheel'`.** `--no-build-isolation` means pip
builds using the venv's own packages, so `wheel` has to be installed there.
Without `ninja` the build also falls back to distutils and compiles the CUDA
sources serially.

```bash
~/MIND/.venv/bin/pip install setuptools wheel ninja
```

**2. `math_util.h:1059: 'float lerp(float, float, float)' conflicts with a
previous declaration`.** The conflict is with `std::lerp`, a C++20 addition, at
`/usr/include/c++/13/cmath:3642` — not with anything in CUDA. `vipe/setup.py`
sets no `-std=` flag, so nvcc follows GCC 13's C++20 default and ViPE's own
scalar `lerp` becomes a redeclaration. Only that overload collides; the
`float2`/`float4` ones use ViPE's own types.

Fixed by pinning the standard in `setup.py`'s existing `cpp_flags` /
`cuda_flags` rather than editing the header, so it survives a submodule update.
Committed on the fork as `dbd37e3`.

**3. `ModuleNotFoundError: No module named 'hydra'`.** ViPE's CLI entry point
imports hydra, which is not among its declared dependencies — `omegaconf`
installs but `hydra-core` does not. The package installs cleanly and
`import vipe` works, so this only surfaces when `vipe infer` is first executed:

```bash
~/MIND/.venv/bin/pip install hydra-core
```

Verify with `~/MIND/.venv/bin/vipe --help`, which exercises the entry point. A
successful `import vipe` does **not** prove the CLI works, and the scoring path
shells out to the CLI rather than importing.

`build_vipe.sh` performs all of this. Note it is a rewrite rather than a
translation of `build_vipe.bat`: that script's central step is loading MSVC via
`vcvars64.bat`, which has no Linux equivalent, and an earlier line-by-line port
failed on exactly that.

## Running

```bash
bash run_mind.sh <model> <metrics> <num_gpus> <perspective> [extra args]
```

Defaults: `matrix-game-3`, `lcm,visual,dino,action,gsc`, `1`, `1st`.

The fourth positional is a perspective filter — `1st`, `3rd` or `both`. It
defaults to **`1st`**, so a plain invocation silently scores only half the data.

### The resume trap

`process.py --resume` defaults to `auto`, which picks up the most recent
`result_<name>_*.json` beside it and skips every sample already listed. Adding a
metric and re-running therefore reports:

```
[resume] skipping 150 already-scored sample(s); 0 to score
No data found!
```

and changes nothing. Pass `--resume none` to force a full re-score. `run_mind.sh`
forwards anything after the four positionals to `process.py`, so:

```bash
bash run_mind.sh zing "lcm,visual,dino,action,gsc" 1 both --resume none
```

### Failure modes are all-or-nothing

Metrics load their models inside the multiprocessing pool, so a missing artifact
kills the run at 0% rather than scoring what it can and reporting the rest as
unavailable. A missing DINOv3 loses the whole pass.

The exception is `action`, which catches per-sample (`[skip action] ViPE
extract_traj crashed`) and continues. That is worse in one respect: the run
completes, takes roughly twice as long because it crops videos and attempts
ViPE for every sample, and produces a result file with **no `action` key at
all** rather than an error. Check for the key before trusting a result.

## Driver scripts

`scripts/drive_*.sh` were moved into `scripts/` without updating their path
logic — each sets `HERE` to its own directory and then treats it as the
repository root, so `.venv`, `src/` and the sibling `MIND-Data` / `MIND-tests`
all resolve one level too deep. Symptom:

```
ERROR: MIND venv python not found: /localhome/kschmid/MIND/scripts/.venv/bin/python
```

All 31 now resolve the root instead of assuming it:

```bash
[ -d "$HERE/src" ] || HERE="$(cd "$HERE/.." && pwd)"
```

which works from either location.

### Frame counts

zing rejects most frame counts with a message that names neither the offending
value nor a valid one:

```
ValueError: the total frame count must map exactly to VAE latent frames
```

The constraint is `(reference_frame_count + frames - 1) % temporal_scale == 0`.
`drive_zing.py` always sends `reference_frame_count=1` and the VAE's temporal
scale is 4, so **`--num-frames` must be a multiple of 4**. The shipped default
of 97 could never work; it is now 96, and the driver checks the value up front
and suggests two valid ones.

H3-World has a different rule — `17k+5`, so 124, 243, 481 — and already
validated it.

## Mirror test

50 first-frame PNGs and 10 directional go-then-return action trajectories, 47
frames each. `gsc` scores it by splitting a clip in half and time-flipping the
second half against the first, so it measures spatial memory rather than image
quality.

The clips have to be **generated** before they can be scored. If
`MIND-tests/<model>/<perspective>/mirror_test/` is absent, scoring logs
`test_dir missing` and skips it:

```bash
bash ~/MIND/scripts/drive_h3world.sh --mirror-test
```

Each driver produces one canonical mirror clip per first frame, deliberately, so
numbers compare across models. `--mirror-action` overrides the trajectory
(default `w`) but breaks that comparability.

## Echo-WM

`src/drive_echo.py` (+ `scripts/drive_echo.sh` / `.bat`) drives Echo-WM from
`JoyAI-Echo/echo_wm`. Two things differ from the other drivers.

**Actions are a string, not a matrix.** Echo takes a WASD/IJKL DSL —
`w-24,s-24,none-1` — which `helpers/action_camera.py:parse_action_string`
expands with `frames.extend([keys] * n)`, so durations are counted in *frames*.
MIND's per-tick key sets therefore run-length-encode straight onto it with no
resampling, and simultaneous keys concatenate (`ajkw-13`). Key semantics are
the same WASD+IJKL layout as H3-World and ABot. All 170 action files in
MIND-Data round-trip through Echo's parser.

**Mirror clips are fitted to the trajectory.** The other drivers pad a short
trajectory by holding the last tick. That never triggers on the main tests —
their `action.json` carries thousands of ticks against a ~100-frame clip — but
the mirror set is 48 ticks of 24-out/24-back, and holding to 97 frames yields
`w-24,s-73`: the camera reverses three times past the origin. Since `gsc`
splits the clip at its midpoint and expects the turnaround there, that scores
as a failed return. `drive_echo.py` pads with idle instead and snaps
`--num-frames` to the trajectory length under `--mirror-test` (48 → 49),
producing `w-24,s-24,none-1`.

**H3-World had this defect and is now fixed.** It held the last tick and
generated 124 frames from the same 48 ticks, so its `gsc` measured an overshoot
rather than a return. `drive_h3world.py` now pads with idle and, under
`--mirror-test`, snaps to 56 frames (17*3+5) and resamples the trajectory across
them so the turnaround lands at frame 28 = 56/2. H3-World cannot use Echo's
exact 1:1 mapping because its frame count must be 17k+5, so 48 ticks have no
1:1 target; resampling is what makes the midpoint land correctly.

The main-test path is deliberately untouched — those `action.json` files carry
thousands of ticks against a ~124-frame clip, so they truncate and never pad,
and existing scores stay comparable.

The 100 already-generated clips still use the old mapping and need regenerating
(`bash scripts/drive_h3world.sh --mirror-test`) before H3-World's `gsc` is
meaningful.

Frame count must be 8k+1 (LTX VAE temporal scale, enforced in
`ltx-pipelines/src/ltx_pipelines/retake.py:451`); the driver defaults to 97 and
rejects anything else with the two nearest valid values.

There is no load-once/batch mode: each sample is a fresh subprocess reloading
the ~47.8 GB checkpoint plus the Gemma text encoder, the same per-sample
pattern as H3-World and Evoke. `--dry-run` prints the action string per sample
without loading anything, and works on a box with no checkpoint on disk.

`--action-overlay` defaults to **on** upstream; the driver turns it off. It
writes the HUD to a second file (`video_action.mp4`) rather than burning it
into `video.mp4`, so scores were never at risk, but it costs time and leaves a
stray mp4 in MIND-tests.

```bash
bash scripts/drive_echo.sh --dry-run --limit 5     # inspect the mapping
bash scripts/drive_echo.sh --limit 2               # smoke test
bash scripts/drive_echo.sh                         # all 1st+3rd
bash scripts/drive_echo.sh --mirror-test           # mirror clips, for gsc
bash run_mind.sh echo "lcm,visual,dino,action,gsc" 1 both --resume none
```

## Comparability

MIND numbers only compare across models when every run uses the same prompt
text, the same metric set and the same ViPE revision. Three things in this
repository can silently break that:

- **Prompts.** `--enhance-prompt` (currently Evoke only) substitutes longer
  scene-anchoring text. Off by default for this reason.
- **Metric set.** A result file missing `action` or `dino` is not comparable
  with one that has them.
- **The ViPE submodule.** It tracks a commit; if the pointer moves, `action`
  numbers come from a different revision than earlier scores did.

## Two checkouts, two remotes

The Windows checkout and `~/MIND` on the box are separate clones with different
remotes, and the same is true of the `vipe` submodule — Windows points at the
`3a1b2c3` fork on branch `win`, the box at `nv-tlabs` upstream in detached HEAD.
A `git pull` on one will never see commits pushed from the other, and the box
cannot reach the C++17 fix at all until the fork is added as a remote:

```bash
cd ~/MIND/vipe && git remote add fork https://github.com/3a1b2c3/vipe.git && git fetch fork && git checkout win
```

Until then the box only builds because `setup.py` was copied across directly,
which any `git submodule update` there will silently revert.
