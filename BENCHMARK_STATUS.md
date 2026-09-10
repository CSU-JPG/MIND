# MIND benchmark: coverage, gaps, and what to run

Companion to `SETUP_DGX.md` (setup + gotchas) and `results/SPREADSHEET_ISSUES.md`
(why the comparison table cannot yet be read as a ranking).

## Only four models are staged on this box

`~/MIND-tests/` holds **echo, h3world, solarwm, zing** and nothing else. Other
models (helios-i2v, hy-worldplay, sana-wm, dreamx-*, matrix-game-3) have
`result_*.json` files here from earlier work, but **no video data on this
machine** -- they cannot be re-scored or debugged without restaging first.

An earlier version of this file listed a ten-model coverage table built from the
result JSONs. That table described scoring history, not what is on disk, and led
to a plan full of steps that could not run. Read staging, not results, when
deciding what to do next.

| model | staged | scored | state |
| --- | --- | --- | --- |
| zing | 250 clips, both perspectives | 141 rows, all metrics, 50 mirror gsc | **done** |
| h3world | 1st: mem 50 / act 25 / mirror rebuilding; 3rd: mem 35 / act 25 / mirror 50 | partial | mirror regen + re-score |
| echo | 1st: act 24, mirror 2 | never | needs full generation |
| solarwm | 1st: act 3 | never | no driver in `scripts/`; dropped |

zing's finished result is `result_zing_2026-09-09-23-18-50.json`: 141 rows, zero
errors, 91 main samples with lcm/visual_quality/dino/action/gsc, and 50 mirror
rows carrying real go-then-return gsc.

## Scoring is parallel by default

`process.py` used to size its worker pool by GPU count (`mp.Pool(processes=
num_gpus)`), so a single-GPU box scored one video at a time: 175s/video, load
average 2 on 72 cores, GPU 95% idle. `--num_workers` now decouples the two and
defaults to `min(cores // 8, 8)`.

Measured on this GB300: **175s/video -> ~24s/video**, identical results.

**Do not raise the cap without measuring.** Each worker holds ~21GB of VRAM at
steady state (8 workers = 171GB of 256GB). A brief experiment with a cap of 16
was based on a ramp-up snapshot that showed ~10.6GB; the steady-state figure is
double that, 16 workers needs ~340GB, and it OOM'd.

The default is per-process and cannot see other jobs. **Two concurrent runs both
taking the default will ask for 2x the VRAM** -- pass an explicit
`--num_workers 4` when deliberately scoring two models at once.

## An OOM silently produces rows with no action

This is the most dangerous failure mode in the pipeline and it has now happened
twice from different causes.

When ViPE dies -- whether from a bad binary or from VRAM pressure -- `action` is
caught per-sample and the row is written with `error: None` and no action value.
The run completes, reports no failures, and produces a result file that looks
finished. `result_h3world_2026-09-09-22-21-31.json` is such a file: 136 rows,
zero errors, zero action.

That file is also a `--resume` trap. Resume builds its skip-set from
`(perspective, test_type, path)` without checking *what* was scored, so
resuming onto it marks 136 samples as done and carries them forward
permanently unscored. `--resume auto` picks the newest file, which is exactly
the poisoned one.

Two habits until `action` failures mark the row:

- verify before resuming: `python3 -c "import json;r=json.load(open(F))['data'];
  print(len(r), sum(1 for x in r if x.get('action')))"`
- prefer an explicit `--resume <path>` over `auto`, or `--resume none`

## Mirror gsc lives under `video_results`, not on the row

Mirror rows store their score at `row['video_results'][i]['gsc']`. A top-level
`row['gsc']` is empty for every mirror row **by design**. Querying the top level
makes it look like mirror scoring is broken across every model; it is not.

Verified working: h3world 49/49 and zing 50/50 nested gsc, all tagged
`source: 'mirror'`, with populated `motion_mse`.

## Two things to know about `gsc`

**It is computed by two different code paths into one column.** `process.py:131`
scores real mirror_test go-then-return clips; `process.py:231` scores ordinary
clips split and time-flipped, which on footage that never returns anywhere is
first-half/second-half self-similarity, not spatial memory. Each result carries
`source` (`mirror` or `self`) to tell them apart.

**A static clip scores perfectly.** If the model barely moves, the two halves
are near-identical. `motion_mse` (error between the first frame and the midpoint
frame) exposes this: near-zero means the clip never travelled and its score is
meaningless.

zing's mirror set is robust to this -- unfiltered mean avg_mse 0.00716, versus
0.00697 after dropping the four lowest-motion clips. matrix-game-3's older
numbers were not: avg_mse 0.0015 with infinite PSNR, best in the column purely
by not moving.

## Mirror clips must be fitted to their trajectory

gsc splits the clip at its midpoint and expects the turnaround there. The mirror
trajectory is 48 ticks: 24 out, 24 back. A driver generating its usual clip
length puts the turnaround in the first quarter and scores an overshoot.

| driver | mirror frames | turnaround | mapping |
| --- | --- | --- | --- |
| zing | 48 | 24 | 1:1, no padding |
| echo | 49 | 24 | 1:1 + one idle frame (verified on real output) |
| h3world | 56 | 28 | resampled (17k+5 cannot hit 48) |
| h3world OLD | 124 | 24 | **broken** |
| matrix-game-3 | 336 | ? | **almost certainly broken** |

`fit_mirror_frames(n_ticks, scale, offset)` in `utils/mirror_test_utils.py`
implements sizing for any `scale*k + offset` constraint. Pad short trajectories
with idle, never by repeating the last key -- repeating runs the return leg past
the origin.

**The drivers skip samples that already have output.** Regenerating requires
moving the old directory aside first; the scorer only walks the exact names
`mem_test`, `action_space_test`, `mirror_test`, so a renamed directory is
invisible to scoring but stays on disk.

## Echo's load-once worker is opt-in

`drive_echo.py`'s `--worker` used to default to on. It holds the pipeline and
every sample's intermediates resident and OOM'd at 249 GiB on a 249 GiB card,
where the per-sample path (fresh process per clip) completed. It now defaults
off, so no flag is needed for a working run.

`--worker` still exists and saves ~25-30h of checkpoint reloads on a full pass
if it is ever fixed to free between samples.

## ViPE dominates the per-video cost

A single video costs 123-659s inside ViPE, and parallelism overlaps that cost
rather than reducing it. ViPE is spawned fresh per video (`vipe infer`, one
video per invocation -- the CLI has no batch mode), so model construction is
paid on every clip.

Reusing one pipeline per worker is feasible: `run()` computes its output paths
from `self.out_path` at call time, so that attribute can be reassigned between
videos. But `run()` also constructs `SLAMSystem` internally on every call, so
reuse would save process spawn, `import torch` and CUDA init (~30-60s of the
123s floor) and **not** the model construction. Measure the split before
building it.

There is already a GT-trajectory cache (`images.txt` under
`~/.cache/mind/vipe/`), which is why some videos return in seconds.

## The GPU can wedge, and signals will not clear it

Symptoms seen here: ViPE processes running 20x their normal duration, each
pinning one CPU core with **GPU utilization at 0%**, log output stopping
entirely, and SSH briefly unreachable with the card at 103 C.

`kill -9` turns those processes into unreapable `<defunct>` zombies -- each has
a thread stuck in `D` state inside the driver, and the dead processes keep
holding VRAM (72GB in one incident). `nvidia-smi -r` fails while contexts are
held. **Only a reboot clears it.**

Diagnosing: `ps -L -p <pid> -o stat=` shows per-thread state; a `D` there means
driver-wedged, not busy. New processes can still use the card afterwards, but
the leaked VRAM does not come back.

## What to run

One GPU job at a time. Check first:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

### h3world -- finish mirror regeneration, then re-score

`1st_data/mirror_test` was moved aside and its old 124-frame clips deleted, so
it regenerates from empty. `3rd_data` is already correct at 56 frames.

```bash
bash scripts/drive_h3world.sh --mirror-only
bash run_mind.sh h3world "lcm,visual,dino,action,gsc" 1 both --resume none
```

`--resume none` is deliberate: it avoids the poisoned newest result file.
Note `3rd_data/mem_test` has 35 clips, not 50 -- short, and worth checking.

### echo -- the largest remaining job

```bash
bash scripts/drive_echo.sh --width 512 --height 288 --steps 10 --mirror-only
bash scripts/drive_echo.sh --width 512 --height 288 --steps 10 --no-mirror-test
bash run_mind.sh echo "lcm,visual,dino,action,gsc" 1 both --resume none
```

No `--no-worker` needed any more. Use `PYTORCH_ALLOC_CONF`; the
`PYTORCH_CUDA_ALLOC_CONF` spelling is deprecated and ignored. The smoke test
confirmed the mirror mapping produces 49 frames at 512x288.

Echo defaults to 1280x704 and 30 steps; MIND scores at low resolution
regardless, so most of that budget is never measured. Drop the size flags only
if you want echo comparable at native resolution.

## Evoke is not part of this benchmark

`evaluate_evoke.py` computes three proxy metrics -- optical-flow motion
magnitude, frame-to-frame pixel difference, Laplacian sharpness -- with no
ground truth and none of MIND's five metrics. Nothing is staged in
`MIND-tests/evoke`.

Standalone use needs no GPU queue:
`python evaluate_evoke.py --video <path> --output-json metrics.json`

Putting Evoke *in* the comparison table means staging it through
`scripts/drive_evoke.sh` and scoring it like any other model -- new work, not a
gap in the current run.

## Attention backends: already settled

Echo's `TROUBLESHOOTING.md` benchmarked this on this GPU: none of xformers,
FlashAttention 1-4, SageAttention or FlashInfer beat plain PyTorch SDPA with
explicit EFFICIENT_ATTENTION/CUDNN_ATTENTION priority. SageAttention runs
correctly but is slower (1.84-1.88s vs 1.74-1.78s per block). The
`[attention] using SDPA ...` line is the code avoiding a trap, not a warning:
FLASH_ATTENTION cannot take a non-causal bias tensor and would fall through to
MATH. Real speed levers: step count, resolution, attention window size, encode
pipelining.

## Still open

- `action` failures should mark the row rather than leaving it indistinguishable
  from a clean skip. Both silent-data incidents trace to this.
- `process_batch_size` is hardcoded at 10 and not exposed as a flag; it is the
  natural per-worker memory knob, finer-grained than worker count.
- 15 drivers still use the exclusive mirror pattern; echo, h3world and zing are
  additive and default-on.
- Leaked VRAM from zombie ViPE processes needs a reboot to reclaim.
