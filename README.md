# mind-metrics

## Environment build-up

First follow vipe's instruction to build conda envrionment.

https://github.com/nv-tlabs/vipe

then run 
```bash
pip install -r requirements.txt 
```
in the same env.

## Multi-GPU Support

The metrics computation now supports multi-GPU parallel processing for faster evaluation.

### Usage

**Single GPU (default)**:
```bash
python process.py --gt_root ../MIND-Data --test_root /path/to/test/videos
```

**Multi-GPU**:
```bash
python process.py --gt_root ../MIND-Data --test_root /path/to/test/videos --num_gpus 4
```

### Command Line Arguments

- `--gt_root`: Ground truth data root directory (default: `../MIND-Data`)
- `--test_root`: Test data root directory (required)
- `--num_gpus`: Number of GPUs to use for parallel processing (default: 1)
- `--video_max_time`: Maximum video frames to process (default: 100)
- `--output`: Output JSON file path (default: `result_YYYY-MM-DD-HH:MM:SS.json`)

### Examples

```bash
# Use 4 GPUs with custom output path
python process.py --num_gpus 4 --output my_results.json

# Process only first 200 frames with 2 GPUs
python process.py --num_gpus 2 --video_max_time 200
```

### How Multi-GPU Works

- Videos are automatically distributed across available GPUs
- Each GPU processes its assigned subset independently
- Progress bars show status for each GPU
- Results are merged into a single output file

## How to order files

### the structure of our ground truth videos (both for training and for testing) looks like:
```bash
MIND-Data
├── 1st_data
│   ├── test
│   │   ├── action_space_test
│   │   │   ├── {gt data name}
│   │   │   │   ├── action.json
│   │   │   │   └── video.mp4
|   |   |   ...
|   |   |    
│   │   └── mem_test
│   │       ├── {gt data name}
│   │       │   ├── action.json
│   │       │   └── video.mp4
|   |       ...
|   └── train
|       ├── 2025-10-18-17-25-BoardcastStudio_B-0m37s
|       │   ├── action.json
|       │   └── video.mp4
|       ...
|
├── 3rd_data
│   ├── test
│   │   ├── action_space_test
│   │   │   ├── {gt data name}
│   │   │   │   ├── action.json
│   │   │   │   └── video.mp4
|   |   |   ...
|   |   |    
│   │   └── mem_test
│   │       ├── {gt data name}
│   │       │   ├── action.json
│   │       │   └── video.mp4
|   |       ...
|   └── train
|       ├── {gt data name}
|       │   ├── action.json
|       │   └── video.mp4
|       ...
```
### suppose the structure your test videos loos like:
```
{model_name}
├── 1st_data
│   ├── action_space_test
│   │   ├── {corresponding data name}
│   │   │   └── video.mp4
|   |   ...
|   |    
│   └── mem_test
│       ├── {corresponding data name}
│       │   └── video.mp4
|       ...
|
├── 3rd_data
│   ├── action_space_test
│   │   ├── {corresponding data name}
│   │   │   └── video.mp4
|   |   ...
|   |    
│   └── mem_test
│       ├── {corresponding data name}
│       │   └── video.mp4
|       ...
```