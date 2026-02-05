import lpips
import torch
from pyiqa.archs.musiq_arch import MUSIQ
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils.utils import load_gt_video, load_sample_video, load_time_from_json, print_gpu_memory, get_musiq_spaq_path, get_vitl_path, get_aes_path, clip_transform_Image, extract_actions_from_json, crop_video_frames, ensure_all_models_downloaded, CACHE_DIR
from utils.dino_utils import load_dinov3_model, extract_dinov3_features
from tqdm import tqdm
import torch.nn.functional as F
import json
import clip
import os
from pathlib import Path
import torch.multiprocessing as mp
import warnings
from metrics.action import action_accuracy_metric
from metrics.lcm import lcm_metric
from metrics.vision_quality import visual_quality_metric
from metrics.dino import dino_mse_metric

def compute_metrics_single_gpu(data_list, gt_root, test_root, dino_path,
                               requested_metrics, video_max_time, process_batch_size, device, gpu_id, gpu_result_file=None):
    import warnings
    # 在子进程中设置warnings过滤
    warnings.filterwarnings("ignore", message=".*pretrained.*")
    warnings.filterwarnings("ignore", message=".*Weights.*")
    warnings.filterwarnings("ignore", message=".*video.*deprecated.*")
    """
    在单个GPU上计算指定数据列表的metrics

    Args:
        data_list: 要处理的数据文件夹名称列表
        gt_root: ground truth数据根目录
        test_root: 测试数据根目录
        perspective: '1st_data' or '3rd_data'
        test_type: 'mem_test' or 'action_space_test'
        requested_metrics: 要计算的指标列表
        video_max_time: 视频最大帧数
        process_batch_size: 批处理大小
        device: GPU设备ID
        gpu_id: GPU编号(用于显示)

    Returns:
        字典，包含所有处理的数据的结果
    """
    # 初始化模型
    tqdm.write(f"GPU[{gpu_id}]: loading lcm model")
    lpips_metric = lpips.LPIPS(net='alex', spatial=False).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1, 2, 3]).to(device)

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
    tqdm.write(f"GPU[{gpu_id}]: loading dinov3 model")
    dino_model, dino_processor = load_dinov3_model(dino_path, device)

    tqdm.write(f"GPU[{gpu_id}]: loaded all models")

    results = []
    try:
        pbar = tqdm(data_list, position=gpu_id, desc=f"GPU{gpu_id}", leave=True)

        for data in pbar:
            data_path = data['path']
            gt_dir = os.path.join(gt_root, data['perspective'], 'test', data['test_type'])
            test_dir = os.path.join(test_root, data['perspective'], data['test_type'])
            if not os.path.exists(os.path.join(test_dir, data_path)):
                continue

            pbar.set_postfix({"file": data_path})

            try:
                mark_time, total_time = load_time_from_json(os.path.join(gt_dir, data_path, 'action.json'))
                result = data
                result['error'] = None
                result['mark_time'] = mark_time
                result['total_time'] = total_time

                prefix = f"[GPU{gpu_id}] {data_path}"
                tqdm.write(f"{prefix}: [1/5] Loading videos...")

                # 读入视频
                gt_imgs = load_gt_video(os.path.join(gt_dir, data_path, 'video.mp4'), mark_time, total_time, video_max_time)
                gt_imgs = gt_imgs.to(device)

                sample_imgs = load_sample_video(os.path.join(test_dir, data_path, 'video.mp4'), mark_time, total_time, video_max_time)
                sample_imgs = sample_imgs.to(device)

                tqdm.write(f"{prefix}: [2/5] Computing LCM metrics (MSE/PSNR/SSIM/LPIPS)...")
                # 计算LCM指标
                lcm = lcm_metric(sample_imgs, gt_imgs, requested_metrics, lpips_metric, ssim_metric, psnr_metric, process_batch_size)
                result['lcm']= lcm

                tqdm.write(f"{prefix}: [3/5] Computing visual quality metrics...")
                # 计算image_quality指标
                vq = visual_quality_metric(sample_imgs, imaging_model, aesthetic_model, clip_model)
                result['visual_quality'] = vq

                tqdm.write(f"{prefix}: [4/5] Computing visual quality metrics...")
                # 计算dino_MSE指标
                dino_mse = dino_mse_metric(sample_imgs, gt_imgs, dino_model, dino_processor, device, process_batch_size)
                result['dino'] = dino_mse

                # 清理内存
                del sample_imgs, gt_imgs
                torch.cuda.empty_cache()

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
                tqdm.write(f"{prefix}: Done")

            except KeyboardInterrupt:
                tqdm.write(f"\nGPU{gpu_id}: Interrupted by user. Returning partial results...")
                raise
            except Exception as e:
                tqdm.write(f"Error processing {data_path} on GPU{gpu_id}: {e}")
                result['error'] = str(e)

            results.append(result)

            # 写入中间结果文件
            if gpu_result_file is not None:
                with open(gpu_result_file, 'w') as f:
                    json.dump(results, f, indent=2)
            #break

    finally:
        # 清理模型和显存
        tqdm.write(f"GPU{gpu_id}: Cleaning up...")
        del lpips_metric, ssim_metric, psnr_metric
        del imaging_model, aesthetic_model, clip_model
        torch.cuda.empty_cache()

    return results

def compute_metrics(gt_root, test_root, dino_path, requested_metrics=['mse','psnr','ssim','lpips'],
                   video_max_time=100, process_batch_size=10, num_gpus=1):
    """
    计算视频metrics，支持多GPU并行（统一架构，单GPU也使用multiprocessing）

    Args:
        gt_root: ground truth数据根目录
        test_root: 测试数据根目录
        requested_metrics: 要计算的指标列表
        video_max_time: 视频最大帧数
        process_batch_size: 批处理大小
        num_gpus: 使用的GPU数量（>=1）

    Returns:
        包含所有结果的字典
    """
    result_dict = {'data':[], 'video_max_time':video_max_time}

    # 统一使用multiprocessing架构
    mp.set_start_method('spawn', force=True)

    # 设置tqdm的全局锁，让多进程共享
    tqdm.set_lock(mp.RLock())

    all_data = []
    for perspective in ['1st_data', '3rd_data']:
        for test_type in ['mem_test', 'action_space_test']:
            gt_dir = os.path.join(gt_root, perspective, 'test', test_type)
            test_dir = os.path.join(test_root, perspective, test_type)

            # 获取所有数据文件夹
            all_data += [{'path': d, 'perspective': perspective, 'test_type': test_type} 
                for d in os.listdir(gt_dir) if os.path.exists(os.path.join(test_dir, d))]

    if len(all_data) == 0:
        tqdm.write(f"No data found!")
        return

    # 将数据分配到各个GPU
    data_per_gpu = len(all_data) // num_gpus
    data_splits = []
    for i in range(num_gpus):
        start_idx = i * data_per_gpu
        end_idx = start_idx + data_per_gpu if i < num_gpus - 1 else len(all_data)
        data_splits.append(all_data[start_idx:end_idx])

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Total data: {len(all_data)}, Using {num_gpus} GPU(s)")
    for i, split in enumerate(data_splits):
        tqdm.write(f"  GPU{i}: {len(split)} videos")
    tqdm.write(f"{'='*60}\n")

    # 在并行前预先下载所有模型文件，避免多进程冲突
    ensure_all_models_downloaded()

    # 为每个GPU创建结果文件路径
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gpu_result_dir = cache_dir / "gpu_results"
    gpu_result_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧的结果文件
    for f in gpu_result_dir.glob("gpu_*.json"):
        f.unlink()

    gpu_result_files = [str(gpu_result_dir / f"gpu_{i}.json") for i in range(num_gpus)]

    # 创建进程池并行处理
    try:
        with mp.Pool(processes=num_gpus) as pool:
            worker_args = [
                (data_splits[i], gt_root, test_root, dino_path,
                    requested_metrics, video_max_time, process_batch_size, f'cuda:{i}', i, gpu_result_files[i])
                for i in range(num_gpus)
            ]

            result_list = pool.starmap(compute_metrics_single_gpu, worker_args)
            for result in result_list:
                result_dict['data'] += result

    except KeyboardInterrupt:
        tqdm.write("\n\nInterrupted by user! Merging partial results from GPU files...")
        pool.terminate()
        pool.join()

        # 从GPU结果文件合并数据
        for gpu_file in gpu_result_files:
            if Path(gpu_file).exists():
                with open(gpu_file, 'r') as f:
                    gpu_results = json.load(f)
                    result_dict['data'] += gpu_results
                tqdm.write(f"  Merged {len(gpu_results)} results from {Path(gpu_file).name}")
        return result_dict

    return result_dict

if __name__ == '__main__':
    import argparse
    warnings.filterwarnings("ignore", message=".*pretrained.*")
    warnings.filterwarnings("ignore", message=".*Weights.*")
    warnings.filterwarnings("ignore", message=".*video.*deprecated.*")

    parser = argparse.ArgumentParser(description='Compute video metrics with multi-GPU support')
    parser.add_argument('--gt_root', type=str, default='../MIND-Data', help='Ground truth data root directory')
    parser.add_argument('--test_root', type=str, default='./i2v',
                       help='Test data root directory')
    parser.add_argument('--dino_path', type=str, default='./dinov3_vitb16',
                       help='dinov3 weight directory, for example ./dinov3_vitb16')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--video_max_time', type=int, default=100, help='Maximum video frames (default: 100)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')

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
        output_path = f'result_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.json'

    tqdm.write(f"Starting computation with {args.num_gpus} GPU(s)...")
    tqdm.write(f"Output will be saved to: {output_path}")

    video_results = None
    try:
        video_results = compute_metrics(
            args.gt_root,
            args.test_root,
            args.dino_path,
            video_max_time=args.video_max_time,
            num_gpus=args.num_gpus
        )
    except KeyboardInterrupt:
        tqdm.write("\n\n" + "="*60)
        tqdm.write("INTERRUPTED BY USER (Ctrl+C)")
        tqdm.write("="*60)
        if video_results is None:
            tqdm.write("No results to save (computation was interrupted too early)")
            exit(1)
    finally:
        # 保存结果(即使被中断)
        if video_results is not None:
            with open(output_path, 'w') as f:
                json.dump(video_results, f, indent=2)
            tqdm.write(f"\nResults saved to: {output_path}")

            # 打印统计信息
            tqdm.write(f"{len(video_results['data'])} videos are processed")
        else:
            tqdm.write("\nNo results to save.")
    