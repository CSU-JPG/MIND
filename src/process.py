import os as _os

# Suppress noisy libavcodec stderr ("get_buffer() failed", "Error splitting the
# input into NAL units", "no frame!", etc.) emitted by PyAV during H.264 header
# probing. These are recoverable and PyAV retries internally — the warnings
# drown the actual progress bar / per-sample log lines.
_os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
_os.environ.setdefault("AV_LOG_LEVEL", "fatal")  # libav env-level override
try:
    import av as _av  # noqa: E402
    # av.logging.FATAL drops everything except FATAL+; ERROR keeps real errors visible.
    _av.logging.set_level(_av.logging.FATAL)
except Exception:
    pass

import lpips
import torch
from pyiqa.archs.musiq_arch import MUSIQ
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils.utils import load_time_from_json, print_gpu_memory, get_musiq_spaq_path, get_vitl_path, get_aes_path, extract_actions_from_json, ensure_all_models_downloaded, CACHE_DIR, VideoStreamReader, get_video_length
from utils.dino_utils import load_dinov3_model, extract_dinov3_features
from tqdm import tqdm
import threading
import torch.nn.functional as F
import json
import clip
import os
from pathlib import Path
import torch.multiprocessing as mp
import warnings
from metrics.lcm import lcm_metric, merge_lcm_results
from metrics.visual_quality import visual_quality_metric, merge_visual_results
from metrics.dino import dino_mse_metric, merge_dino_results
import time
from collections import namedtuple

Task = namedtuple('Task', ['data', 'retry_count'])

def compute_metrics_single_gpu(task_queue, result_list, gt_root, test_root, dino_path,
                               requested_metrics, video_max_time, process_batch_size, device, gpu_id, stop_event):
    import warnings
    warnings.filterwarnings("ignore", message=".*pretrained.*")
    warnings.filterwarnings("ignore", message=".*Weights.*")
    warnings.filterwarnings("ignore", message=".*video.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*timm\\.models\\.layers.*")
    warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
    """
    在单个GPU上从任务队列获取并处理任务
    """
    # 初始化模型
    def _annotate_gsc(gsc, imgs, source):
        """Tag a gsc result with where it came from and whether the clip moved.

        Two separate problems this makes visible in the result file:

        1. gsc is computed by two different code paths -- a real mirror_test
           go-then-return, and an ordinary action_space/mem clip split and
           time-flipped. On footage that never returns anywhere the second is
           first-half/second-half self-similarity, not spatial memory. Both used
           to land in one 'gsc' key with nothing to tell them apart, so rows
           measuring different things got ranked against each other.

        2. A static clip scores PERFECTLY. If the model barely moves, the two
           halves are near-identical and gsc reports ~0 error -- matrix-game-3's
           mirror clips came back at avg_mse 0.0015 with SSIM 0.991-0.993 and one
           sample at infinite PSNR, 20x "better" than anything else purely by not
           moving. 'motion_mse' is the error between the first frame and the
           midpoint frame: how far the clip actually travelled outbound. When it
           is near zero the gsc score is meaningless and should be dropped, not
           ranked first.
        """
        gsc['source'] = source
        if imgs is not None and len(imgs) >= 2:
            mid = len(imgs) // 2
            gsc['motion_mse'] = float(torch.mean((imgs[0] - imgs[mid]) ** 2))
        return gsc

    if 'lcm' in requested_metrics or 'gsc' in requested_metrics:
        tqdm.write(f"GPU[{gpu_id}]: loading lcm model")
        lpips_metric = lpips.LPIPS(net='alex', spatial=False).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1, 2, 3]).to(device)

    if 'visual' in requested_metrics:
        tqdm.write(f"GPU[{gpu_id}]: loading visual quality model")
        imaging_model = MUSIQ(pretrained_model_path=get_musiq_spaq_path())
        aesthetic_model = torch.nn.Linear(768, 1)
        clip_model, preprocess = clip.load(get_vitl_path(), device=device)

        s = torch.load(get_aes_path(), weights_only=False)
        aesthetic_model.load_state_dict(s)
        aesthetic_model.to(device)
        aesthetic_model.eval()

        imaging_model.to(device)
        imaging_model.training = False

    # 加载DINOv3模型
    if 'dino' in requested_metrics:
        tqdm.write(f"GPU[{gpu_id}]: loading dinov3 model")
        dino_model, dino_processor = load_dinov3_model(dino_path, device)

    tqdm.write(f"GPU[{gpu_id}]: loaded all models")
    processed_count = 0
    try:
        while True:
            # 检查是否需要停止
            if stop_event.is_set():
                tqdm.write(f"GPU{gpu_id}: Stop event received, exiting...")
                break

            try:
                task = task_queue.get(timeout=5)
                data = task.data
                retry_count = task.retry_count
            except:
                continue

            data_path = data['path']
            gt_dir = os.path.join(gt_root, data['perspective'], 'test', data['test_type'])
            test_dir = os.path.join(test_root, data['perspective'], data['test_type'])

            prefix = f"[GPU{gpu_id}] {data_path}"
            try:
                if data['test_type'] == 'mirror_test':
                    if 'gsc' not in requested_metrics:
                        continue
                    result = data.copy()
                    path_videos = [v for v in os.listdir(os.path.join(test_dir, data_path)) if v.endswith('.mp4')]
                    result['video_results'] = []
                    result['videos'] = path_videos
                    result['error'] = None
                    for video in path_videos:
                        vid_result = {'video_name': video, 'error': None}
                        sample_frames = get_video_length(os.path.join(test_dir, data_path, video))
                        vid_result['mark_time'] = sample_frames // 2
                        vid_result['sample_frames'] = sample_frames
                        # Odd frame counts (e.g. dreamx-small's 121-frame default) used to fail
                        # the gsc split. Drop the last frame so it's even and split cleanly —
                        # the gsc metric is symmetry-around-midpoint, dropping the final frame
                        # is harmless.
                        if sample_frames % 2 == 1:
                            tqdm.write(f"{prefix}: odd frame count {sample_frames}, truncating last frame for gsc split")
                            sample_frames -= 1
                            vid_result['sample_frames'] = sample_frames
                            vid_result['mark_time'] = sample_frames // 2

                        tqdm.write(f"{prefix}: [1/2] Reading videos...")
                        sample_reader = VideoStreamReader(os.path.join(test_dir, data_path, video), start_frame=0, total_frames=sample_frames)
                        _, imgs = sample_reader.read_batch(sample_frames)

                        origin_pred = imgs[:sample_frames // 2]
                        mirror_pred = torch.flip(imgs[sample_frames // 2:], dims=[0])

                        tqdm.write(f"{prefix}: part len: {len(origin_pred)}, {len(mirror_pred)}")
                        
                        tqdm.write(f"{prefix}: [2/2] Computing GSC metrics (MSE/PSNR/SSIM/LPIPS)...")
                        gsc = lcm_metric(origin_pred, mirror_pred, lpips_metric, ssim_metric, psnr_metric, process_batch_size, device)
                        vid_result['gsc'] = _annotate_gsc(gsc, imgs, 'mirror')
                        result['video_results'].append(vid_result)

                        del sample_reader                                                                                                                                                                  
                        torch.cuda.empty_cache()

                else:
                    mark_time, total_time = load_time_from_json(os.path.join(gt_dir, data_path, 'action.json'))
                    result = data.copy()
                    result['error'] = None
                    result['mark_time'] = mark_time
                    result['total_time'] = total_time

                    gt_frames = get_video_length(os.path.join(gt_dir, data_path, 'video.mp4'))
                    sample_frames = get_video_length(os.path.join(test_dir, data_path, 'video.mp4'))
                    tqdm.write(f"{prefix}: gt_frames={gt_frames}, sample={sample_frames}, total_time={total_time}, mark_time={mark_time}")
                    result['sample_frames'] = sample_frames

                    if sample_frames != total_time - mark_time:
                        real_time = min(total_time - mark_time, sample_frames)
                    else: real_time = sample_frames

                    if video_max_time is not None:
                        real_time = min(real_time, video_max_time)

                    gt_reader = VideoStreamReader(os.path.join(gt_dir, data_path, 'video.mp4'), start_frame=mark_time, total_frames=mark_time + real_time)
                    sample_reader = VideoStreamReader(os.path.join(test_dir, data_path, 'video.mp4'), start_frame=0, total_frames=real_time)

                    while True:
                        tqdm.write(f"{prefix}: [1/5] Reading videos...")
                        is_ended, gt_imgs = gt_reader.read_batch(process_batch_size * 10)
                        _, sample_imgs = sample_reader.read_batch(process_batch_size * 10)

                        # VideoStreamReader.read_batch returns (is_ended=True, frames_tensor)
                        # when the request consumes the rest of the video — the frames are
                        # still valid. Only skip if we got no frames at all; compute metrics
                        # on whatever the final batch returned, then break.
                        if gt_imgs is None or sample_imgs is None:
                            break

                        if 'lcm' in requested_metrics:
                            tqdm.write(f"{prefix}: [2/5] Computing LCM metrics (MSE/PSNR/SSIM/LPIPS)...")
                            lcm = lcm_metric(sample_imgs, gt_imgs, lpips_metric, ssim_metric, psnr_metric, process_batch_size, device)
                            result['lcm'] = merge_lcm_results(result.get('lcm'), lcm)

                        if 'visual' in requested_metrics:
                            tqdm.write(f"{prefix}: [3/5] Computing visual quality metrics...")
                            vq = visual_quality_metric(sample_imgs, imaging_model, aesthetic_model, clip_model, process_batch_size, device)
                            result['visual_quality'] = merge_visual_results(result.get('visual_quality'), vq)

                        if 'dino' in requested_metrics:
                            tqdm.write(f"{prefix}: [4/5] Computing dino metrics...")
                            dino = dino_mse_metric(sample_imgs, gt_imgs, dino_model, dino_processor, device, process_batch_size)
                            result['dino'] = merge_dino_results(result.get('dino'), dino)

                        del gt_imgs, sample_imgs

                        if is_ended:
                            break

                    del gt_reader, sample_reader
                    torch.cuda.empty_cache()

                    if 'action' in requested_metrics:
                        # Lazy import — pulling in metrics.action drags in vipe_utils, which
                        # spawns ViPE subprocesses that hit _sre.MAGIC mismatch + missing
                        # sm_120 CUDA kernels in DeepVerse venv. Importing only when needed
                        # lets the other 4 metrics (lcm/visual/dino/gsc) score cleanly.
                        from metrics.action import action_accuracy_metric
                        tqdm.write(f"{prefix}: [4/5] Computing action accuracy (ViPE)...")
                        # 计算action accuracy
                        actions = extract_actions_from_json(os.path.join(gt_dir, data_path, 'action.json'), mark_time, 97)
                        action = action_accuracy_metric(
                            os.path.join(test_dir, data_path, 'video.mp4'),
                            os.path.join(gt_dir, data_path, 'video.mp4'),
                            mark_time,
                            actions,
                            max_frames = 97,
                            gt_data_dir = os.path.join(gt_dir, data_path),
                            verbose_prefix=f"  {prefix}",
                            gpu_id=gpu_id
                        )
                        if action is not None:
                            result['action'] = action

                    if 'gsc' in requested_metrics:
                        tqdm.write(f"{prefix}: [5/5] Computing GSC metrics (mirror test)...")
                        try:
                            gsc_frames = real_time
                            if gsc_frames % 2 == 1:
                                tqdm.write(f"{prefix}: odd frame count {gsc_frames}, truncating last frame for gsc split")
                                gsc_frames -= 1

                            sample_reader = VideoStreamReader(os.path.join(test_dir, data_path, 'video.mp4'), start_frame=0, total_frames=gsc_frames)
                            _, imgs = sample_reader.read_batch(gsc_frames)

                            if gsc_frames > 0 and imgs is not None:
                                origin_pred = imgs[:gsc_frames // 2]
                                mirror_pred = torch.flip(imgs[gsc_frames // 2:], dims=[0])
                                gsc = lcm_metric(origin_pred, mirror_pred, lpips_metric, ssim_metric, psnr_metric, process_batch_size, device)
                                # 'self', not 'mirror': this clip is an ordinary
                                # action_space/mem rollout that never returns to
                                # its start, so the midpoint split measures
                                # self-similarity rather than spatial memory.
                                result['gsc'] = _annotate_gsc(gsc, imgs, 'self')
                                tqdm.write(f"{prefix}: GSC computed successfully "
                                           f"(source=self, motion_mse={gsc.get('motion_mse', float('nan')):.4f})")

                            del sample_reader
                            torch.cuda.empty_cache()
                        except Exception as e:
                            tqdm.write(f"{prefix}: GSC calculation failed: {e}")
                            import traceback as _tb
                            tqdm.write(_tb.format_exc())

                    tqdm.write(f"{prefix}: Finish Task!")

            except KeyboardInterrupt:
                tqdm.write(f"{prefix}: Interrupted by user. Exiting...")
                if stop_event is not None:
                    stop_event.set()
                raise
            except Exception as e:
                import traceback as _tb
                # Keep the full traceback in the result JSON so failures aren't
                # cryptic one-liners. tqdm.write only the summary to keep the
                # console readable.
                result['error'] = str(e)
                result['traceback'] = _tb.format_exc()
                if retry_count < 3:
                    tqdm.write(f"{prefix}: Error processing {data_path} on GPU{gpu_id}: {e}, retrying ({retry_count + 1}/3)")
                    task_queue.put(Task(data, retry_count + 1))
                else:
                    tqdm.write(f"{prefix}: Error processing {data_path} on GPU{gpu_id}: {e}, max retries exceeded, giving up")
                    result_list.append(result)
                continue

            result_list.append(result)
            processed_count += 1

    finally:
        tqdm.write(f"GPU{gpu_id}: Cleaning up...")
        if 'lcm' in requested_metrics or 'gsc' in requested_metrics:
            del lpips_metric, ssim_metric, psnr_metric
        if 'visual' in requested_metrics:
            del imaging_model, aesthetic_model, clip_model
        torch.cuda.empty_cache()

def default_worker_count(num_gpus):
    """Scoring workers to use when --num_workers is not given.

    A single video costs 123-659s inside ViPE regardless of worker count -- the
    parallelism overlaps that cost, it does not reduce it. Each concurrent video
    holds ~2 CPU cores and ~21GB of VRAM at steady state (measured: 8 workers =
    171GB on a 256GB GB300). VRAM, not cores, is the binding constraint.

    Capped at 8 because 16 needs ~340GB and OOMs the card. That OOM is worse
    than it sounds: ViPE dying takes `action` with it, and action failures are
    caught per-sample, so the run completes with rows that carry no error and
    no action -- silently unscored. 8 has run for hours without incident.

    NOTE: this is per-process and cannot see other scoring jobs. Two concurrent
    runs both taking the default would ask for 2x the VRAM -- pass an explicit
    --num_workers when deliberately running more than one model at a time.
    """
    cores = os.cpu_count() or 4
    return max(num_gpus, min(cores // 8, 8))

def compute_metrics(gt_root, test_root, dino_path, output_path, requested_metrics=['lcm', 'visual', 'dino', 'action'],
                   video_max_time=100, process_batch_size=10, num_gpus=1, resume_path=None, limit=None,
                   perspectives=None, num_workers=None):
    result_dict = {'data':[], 'video_max_time':video_max_time}

    # Load prior results (if --resume): seed result_dict and build skip-set so
    # samples we already scored aren't recomputed.
    resumed_results = []
    resume_keys = set()
    if resume_path and os.path.exists(resume_path):
        try:
            with open(resume_path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            for entry in prev.get('data', []):
                # An entry is "complete" if it has no top-level error AND at least one
                # of the requested metric keys is present. Re-score anything we never
                # finished cleanly.
                if entry.get('error'):
                    continue
                resumed_results.append(entry)
                resume_keys.add((entry.get('perspective'), entry.get('test_type'), entry.get('path')))
            tqdm.write(f"[resume] loaded {len(resumed_results)} prior result(s) from {resume_path}")
        except Exception as exc:
            tqdm.write(f"[resume] WARNING: failed to load {resume_path}: {exc}; scoring from scratch")
            resumed_results = []
            resume_keys = set()

    mp.set_start_method('spawn', force=True)

    all_perspectives = ['1st_data', '3rd_data']
    active_perspectives = perspectives if perspectives else all_perspectives
    if perspectives:
        tqdm.write(f"[perspectives] restricting to: {','.join(active_perspectives)}")
    all_data = []
    for perspective in active_perspectives:
        for test_type in ['mem_test', 'action_space_test', 'mirror_test']:
            if test_type == 'mirror_test':
                if 'gsc' in requested_metrics:
                    test_dir = os.path.join(test_root, perspective, test_type)
                    # Drivers that don't yet produce mirror_test (e.g. drive_dreamx,
                    # drive_matrix3) leave this subdir absent. Skip instead of crashing
                    # so the other metrics can still score against what IS present.
                    if not os.path.isdir(test_dir):
                        tqdm.write(f"[skip] {perspective}/{test_type}: test_dir missing ({test_dir})")
                        continue
                    all_data += [{'path': d, 'perspective': perspective, 'test_type': test_type }
                        for d in os.listdir(test_dir) if (perspective, test_type, d) not in resume_keys]
            else:
                if 'lcm' in requested_metrics or 'visual' in requested_metrics or 'dino' in requested_metrics or 'action' in requested_metrics or 'gsc' in requested_metrics:
                    gt_dir = os.path.join(gt_root, perspective, 'test', test_type)
                    test_dir = os.path.join(test_root, perspective, test_type)
                    if not os.path.isdir(test_dir):
                        tqdm.write(f"[skip] {perspective}/{test_type}: test_dir missing ({test_dir})")
                        continue

                    all_data += [{'path': d, 'perspective': perspective, 'test_type': test_type}
                        for d in os.listdir(test_dir)
                        if os.path.exists(os.path.join(gt_dir, d))
                        and os.path.exists(os.path.join(test_dir, d, 'video.mp4'))
                        and (perspective, test_type, d) not in resume_keys]

    if resume_keys:
        tqdm.write(f"[resume] skipping {len(resume_keys)} already-scored sample(s); {len(all_data)} to score")

    if limit is not None and limit > 0:
        all_data = all_data[:limit]
        tqdm.write(f"[limit] truncated to first {limit} sample(s)")

    if len(all_data) == 0:
        tqdm.write(f"No data found!")
        return

    total_tasks = len(all_data)
    tqdm.write(f"\n{'='*60}")
    # Worker count is independent of device count: ViPE uses ~2 cores and ~1GB
    # of VRAM per video, so one worker per GPU leaves a large box nearly idle.
    n_workers = num_workers if num_workers else default_worker_count(num_gpus)
    tqdm.write(f"Total data: {total_tasks}, Using {num_gpus} GPU(s), {n_workers} worker(s) with task queue")
    tqdm.write(f"{'='*60}\n")

    ensure_all_models_downloaded()

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_list = manager.list()
    # Seed with already-scored entries so the final json contains both old + new.
    for entry in resumed_results:
        result_list.append(entry)
    stop_event = manager.Event()

    for task in all_data:
        task_queue.put(Task(task, 0))

    # 进度监控线程
    # result_list is pre-seeded with the resumed rows, so completion must be
    # measured against resumed + new, not against the new tasks alone. Comparing
    # len(result_list) to total_tasks meant that whenever resumed >= remaining
    # (e.g. 155 resumed, 95 to score) the monitor declared victory on its first
    # tick and set stop_event before any worker had started -- every worker then
    # logged "Stop event received, exiting..." and the run "succeeded" having
    # scored nothing.
    expected_total = total_tasks + len(resumed_results)

    def monitor_progress():
        pbar = tqdm(total=expected_total, initial=len(resumed_results),
                    desc="Progress", unit="video")
        last_count = len(resumed_results)
        try:
            while not stop_event.is_set() or len(result_list) < expected_total:
                current_count = len(result_list)
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                    with open(output_path, 'w') as f:
                        result_dict['data'] = list(result_list)
                        json.dump(result_dict, f, indent=2)
                if last_count >= expected_total:
                    break
                time.sleep(0.5)
            stop_event.set()
        except (ConnectionResetError, OSError):
            pass  # Ctrl-C时manager连接关闭，忽略
        finally:
            pbar.close()

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    try:
        with mp.Pool(processes=n_workers) as pool:
            worker_args = [
                (task_queue, result_list, gt_root, test_root, dino_path,
                    requested_metrics, video_max_time, process_batch_size,
                    f'cuda:{i % num_gpus}', i % num_gpus, stop_event)
                for i in range(n_workers)
            ]

            pool.starmap(compute_metrics_single_gpu, worker_args)

        monitor_thread.join()
        result_dict['data'] = list(result_list)

    except KeyboardInterrupt:
        tqdm.write("\n\nInterrupted by user! Merging partial results...")
        stop_event.set()
        pool.terminate()
        pool.join()
        monitor_thread.join(timeout=2)
        result_dict['data'] = list(result_list)
        tqdm.write(f"  Merged {len(result_dict['data'])} results from shared list")

    return result_dict

if __name__ == '__main__':
    import argparse
    warnings.filterwarnings("ignore", message=".*pretrained.*")
    warnings.filterwarnings("ignore", message=".*Weights.*")
    warnings.filterwarnings("ignore", message=".*video.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*timm.models.layers.*")
    warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

    parser = argparse.ArgumentParser(description='Compute video metrics with multi-GPU support')
    parser.add_argument('--gt_root', type=str, required=True, help='Ground truth data root directory')
    parser.add_argument('--test_root', type=str, required=True, help='Test data root directory')
    parser.add_argument('--dino_path', type=str, default='./dinov3_vitb16',
                       help='dinov3 weight directory, for example ./dinov3_vitb16')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of scoring worker processes. Default: min(cores//8, 8), at least one '
                            'per GPU. Workers are assigned to devices round-robin, so this may exceed the '
                            'GPU count. Each worker needs ~21GB VRAM and ~2 CPU cores; pass 1 to '
                            'restore serial scoring. Set explicitly when running two models at once -- '
                            'the default is per-process and cannot see the other job.')
    parser.add_argument('--video_max_time', type=int, default=None, help='Maximum video frames (default: None = use all frames)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--metrics', type=str, default='lcm,visual,dino,action,gsc', help='Requested metrics to compute, comma separated (e.g. dino,visual)')
    parser.add_argument('--resume', type=str, default='auto',
                        help='Path to a previous result_*.json to skip already-scored (perspective, test_type, path) entries. '
                             "'auto' (default) picks the most recent result_<basename(test_root)>_*.json next to process.py. "
                             "'none' or '' disables skipping.")
    parser.add_argument('--limit', type=int, default=None,
                        help='Score only the first N samples after dedupe/resume filtering. Use --limit 1 for a smoke test.')
    parser.add_argument('--perspectives', type=str, default=None,
                        help="Comma-separated perspectives to score (e.g. '1st_data' or '1st_data,3rd_data'). "
                             "Default: all perspectives.")

    args = parser.parse_args()

    # 检查GPU数量
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        tqdm.write(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        args.num_gpus = available_gpus

    # 准备输出路径
    if args.output:
        output_path = args.output
    else:
        from datetime import datetime
        output_path = f'result_{os.path.basename(args.test_root)}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'

    tqdm.write(f"Starting computation with {args.num_gpus} GPU(s)...")
    tqdm.write(f"Output will be saved to: {output_path}")

    # Resolve --resume: 'auto' picks the most recent result_<basename>_*.json,
    # 'none'/'' disables, anything else is a literal file path.
    resume_path = None
    if args.resume and args.resume.lower() not in ('none', ''):
        if args.resume == 'auto':
            import glob
            pattern = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f'result_{os.path.basename(args.test_root)}_*.json')
            candidates = sorted(glob.glob(pattern))
            resume_path = candidates[-1] if candidates else None
            if resume_path:
                tqdm.write(f"[resume] auto-picked previous results: {resume_path}")
            else:
                tqdm.write(f"[resume] auto-resume requested but no prior result_{os.path.basename(args.test_root)}_*.json found; scoring from scratch")
        else:
            if os.path.exists(args.resume):
                resume_path = args.resume
                tqdm.write(f"[resume] using explicit prior results: {resume_path}")
            else:
                tqdm.write(f"[resume] WARNING: --resume {args.resume} does not exist; scoring from scratch")

    video_results = None
    try:
        video_results = compute_metrics(
            args.gt_root,
            args.test_root,
            args.dino_path,
            output_path,
            requested_metrics=[m.strip() for m in args.metrics.split(',')],
            video_max_time=args.video_max_time,
            num_gpus=args.num_gpus,
            num_workers=args.num_workers,
            resume_path=resume_path,
            limit=args.limit,
            perspectives=[p.strip() for p in args.perspectives.split(',')] if args.perspectives else None,
        )
    except KeyboardInterrupt:
        tqdm.write("\n\n" + "="*60)
        tqdm.write("INTERRUPTED BY USER (Ctrl+C)")
        tqdm.write("="*60)
        if video_results is None:
            tqdm.write("No results to save (computation was interrupted too early)")

    finally:
        # 保存结果(即使被中断)
        if video_results is not None:
            tqdm.write(f"{len(video_results['data'])} videos are processed")
            tqdm.write(f"checking output path {output_path}:")
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    saved_results = json.load(f)
                    tqdm.write(f"{len(saved_results['data'])} videos are saved")
        else:
            tqdm.write("\nNo results to save.")
    