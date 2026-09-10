# MIND Evaluation for Evoke Videos

Evaluate Evoke-generated videos using MIND-like metrics (motion, temporal consistency, sharpness).

## Quick Start

```powershell
cd C:\workspace\world\MIND
setup_mind_evoke.bat
.venv\Scripts\activate.bat
python evaluate_evoke.py --video C:\workspace\world\Evoke\outputs\t2v\geo_pred.mp4
```

## Metrics

### Motion Magnitude
- **What:** Mean optical flow magnitude across frames
- **Range:** 0 (static) → high (fast motion)
- **Use:** Measures dynamism and motion quality

### Temporal Consistency
- **What:** Mean absolute pixel difference between consecutive frames
- **Range:** 0 (perfect consistency) → high (flicker/jitter)
- **Use:** Lower is better (smooth motion, no flicker)

### Sharpness
- **What:** Laplacian variance (edge detection energy)
- **Range:** Low (blurry) → high (sharp)
- **Use:** Higher is better (detail preservation)

## Usage

### Single Video

```bash
python evaluate_evoke.py --video outputs/t2v/geo_pred.mp4 --output-json metrics.json
```

### Batch (Directory)

```bash
python evaluate_evoke.py \
  --video-dir C:\workspace\world\Evoke\outputs \
  --pattern "*.mp4" \
  --output-json evoke_batch_metrics.json
```

### Output

Metrics saved as JSON:
```json
{
  "video": "outputs/t2v/geo_pred.mp4",
  "frames": 54,
  "fps": 24.0,
  "motion_magnitude": 3.45,
  "temporal_consistency": 12.3,
  "sharpness": 1250.5
}
```

## Typical Ranges (Evoke 384×640@24fps)

| Metric | Low | Medium | High |
|--------|-----|--------|------|
| Motion | 0-1 | 1-5 | 5+ |
| Temporal Consistency | 5-15 | 15-30 | 30+ |
| Sharpness | 500-800 | 800-1500 | 1500+ |

**Better videos:** High motion + low consistency + high sharpness.

## Workflow

1. **Generate videos with Evoke:**
   ```powershell
   cd C:\workspace\world\Evoke
   run_examples.bat
   ```

2. **Evaluate with MIND:**
   ```powershell
   cd C:\workspace\world\MIND
   python evaluate_evoke.py --video-dir ..\Evoke\outputs --output-json evoke_results.json
   ```

3. **Compare metrics** across modes (t2v, i2v, v2v, segment)

## Limitations

This is a **lightweight evaluation** using basic computer vision metrics:
- ✓ Motion detection (optical flow)
- ✓ Temporal smoothness (frame diff)
- ✓ Sharpness (Laplacian)
- ✗ No semantic understanding (use external VLM for that)
- ✗ No perceptual quality (LPIPS, DINO, etc. require additional setup)

For full MIND scoring with action metrics, use the official MIND evaluation suite (if available).

## Files

- `setup_mind_evoke.bat` — Install MIND + dependencies
- `evaluate_evoke.py` — Video evaluation script
- `EVOKE_EVALUATION.md` — This guide
