#!/usr/bin/env python
"""
推理脚本 - 使用 visualization.py 展示结果
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from typing import List, Dict, Tuple

# 添加项目根目录到 sys.path
# 无论从哪个目录运行，都能正确找到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.TriBranchCracknet import TriBranchCrackNet
from config.config import Config
from data.dataset import CrackDatasetSplitter
from data.calculateMeanStd import compute_mean_std_from_dir
from utils.visualization import Visualizer, tensor_to_numpy
import albumentations as A


def load_model_weights(checkpoint_path: str, device: torch.device) -> TriBranchCrackNet:
    """加载模型权重"""
    print(f"加载checkpoint: {Path(checkpoint_path).name}")

    # 获取模型配置
    model_config = Config.MODEL_CONFIGS
    stage_count = len(model_config['widths'])
    stage_blocks = Config.expand_stage_blocks(model_config['depths'], stage_count)

    # 创建模型
    model = TriBranchCrackNet(
        in_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        widths=model_config['widths'],
        stage_blocks=stage_blocks,
        k=model_config['k'],
        d_state=model_config['d_state'],
        dropout=model_config['dropout'],
        attn_drop=model_config['attn_drop'],
        proj_drop=model_config['proj_drop'],
        drop_path_rate=model_config['drop_path_rate'],
        aux_stage_indices=Config.AUX_STAGE_INDICES,
        use_aux_outputs=Config.USE_AUX_OUTPUTS,
        first_downsample_stride=Config.ENCODER_FIRST_DOWNSAMPLE_STRIDE,
        use_edge_head=Config.USE_EDGE_HEAD,
        window_size=Config.TRANSFORMER_WINDOW_SIZE,
        global_pool=Config.TRANSFORMER_GLOBAL_POOL
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 优先使用EMA权重（若存在），否则使用常规state_dict
    state_to_load = None
    using_ema = False
    if isinstance(checkpoint, dict):
        ema_state = checkpoint.get('ema_state', None)
        if isinstance(ema_state, dict) and len(ema_state) > 0:
            state_to_load = ema_state
            using_ema = True

    if state_to_load is None and isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 移除Lightning中'model.'前缀
        state_to_load = {
            (k[6:] if k.startswith('model.') else k): v
            for k, v in state_dict.items()
        }

    if state_to_load is None:
        state_to_load = checkpoint

    missing, unexpected = model.load_state_dict(state_to_load, strict=False)

    model.eval()
    model = model.to(device)

    print(f"  ✓ 模型架构: TriBranchCrackNet")
    print(f"  ✓ 参数量: {sum(p.numel() for p in model.parameters()):,}")
    if using_ema:
        print(f"  ✓ 权重来源: EMA (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"  ✓ 权重来源: 常规state_dict (missing={len(missing)}, unexpected={len(unexpected)})")

    return model


def load_image_and_mask(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载图像和mask"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 读取mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取mask: {mask_path}")

    # Resize
    transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
    ])

    transformed = transform(image=image, mask=mask)
    image_resized = transformed['image']
    mask_resized = transformed['mask']

    return image, mask, image_resized, mask_resized


def predict(model: TriBranchCrackNet, image: np.ndarray, device: torch.device,
            mean: np.ndarray = None, std: np.ndarray = None) -> np.ndarray:
    """预测裂缝分割"""
    # 如果未提供 mean/std，则从训练集计算
    if mean is None or std is None:
        train_images_dir = os.path.join(
            Config.DATASET_ROOT, Config.IMAGE_DIR, Config.TRAIN_SPLIT
        )
        mean, std = compute_mean_std_from_dir(train_images_dir)

    image_normalized = (image.astype(np.float32) / 255.0 - mean) / std

    # 转换为tensor [C, H, W]
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, C, H, W]

    # 推理
    with torch.no_grad():
        output = model(image_tensor)  # logits or dict with main_output
        if isinstance(output, dict):
            logits = output['main_output']
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output
        pred_mask = torch.sigmoid(logits).cpu().numpy()[0, 0]  # [H, W]

    return pred_mask


def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """计算评估指标"""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    if getattr(Config, 'INVERT_MASK', False):
        gt_binary = (gt_mask < 128).astype(np.uint8)
    else:
        gt_binary = (gt_mask > 127).astype(np.uint8)

    # IoU
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
    
    # Precision, Recall, F1
    tp = intersection
    fp = (pred_binary & ~gt_binary).sum()
    fn = (gt_binary & ~pred_binary).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def visualize_and_save_results(samples: List[Dict], output_dir: str, model_name: str, 
                                visualizer: Visualizer) -> str:
    """使用 Visualizer 类可视化推理结果"""
    
    # 准备数据
    images = []
    ground_truths = []
    predictions = []
    
    for sample in samples:
        images.append(sample['resized_image'])
        ground_truths.append(sample['resized_mask'])
        predictions.append((sample['pred_mask'] * 255).astype(np.uint8))
    
    # 使用 Visualizer 保存网格
    visualizer.save_prediction_grid(
        images=images,
        ground_truths=ground_truths,
        predictions=predictions,
        title=f'Inference Results - {model_name}',
        save_name='inference_grid',
        show_overlay=True
    )
    
    # 保存单独的样本（前3个）
    for idx, sample in enumerate(samples[:3]):
        visualizer.show_image_with_mask(
            image=sample['resized_image'],
            mask=sample['resized_mask'],
            predicted_mask=(sample['pred_mask'] * 255).astype(np.uint8),
            title=f"Sample {idx+1} - IoU: {sample['metrics']['iou']:.4f} | "
                  f"F1: {sample['metrics']['f1']:.4f}",
            save_name=f'sample_{idx+1}_{Path(sample["name"]).stem}',
            show=False
        )
    
    print(f"\n✓ 可视化结果已保存到: {output_dir}")
    print(f"  - 网格图: inference_grid.png")
    print(f"  - 单独样本: sample_1.png, sample_2.png, sample_3.png")
    
    return os.path.join(output_dir, 'inference_grid.png')


def main():
    print("="*80)
    print("🔍 裂缝检测推理脚本 - TriBranchCrackNet")
    print("="*80)

    # 1. 找到最新的checkpoint
    checkpoints_dir = Path('checkpoints')
    if not checkpoints_dir.exists():
        print("❌ 错误: 没有找到checkpoints目录")
        return

    # 获取最新的训练目录
    training_dirs = sorted(checkpoints_dir.glob('*'),
                          key=lambda x: x.stat().st_mtime, reverse=True)
    if not training_dirs:
        print("❌ 错误: checkpoints目录为空")
        return

    latest_checkpoint_dir = training_dirs[0]
    print(f"\n📁 使用checkpoint目录: {latest_checkpoint_dir.name}")

    # 查找最佳checkpoint
    best_ckpts = list(latest_checkpoint_dir.glob("best-*.ckpt"))
    if not best_ckpts:
        best_ckpts = list(latest_checkpoint_dir.glob("last.ckpt"))

    if not best_ckpts:
        print("❌ 错误: 没有找到checkpoint文件")
        return

    # 兼容两种命名:
    # - best-xx-val_iou=0.xxxx.ckpt -> 选IoU最高
    # - best-xx-val_loss=0.xxxx.ckpt -> 选loss最低
    iou_scored = []
    loss_scored = []
    for ckpt in best_ckpts:
        name = ckpt.stem
        if 'val_iou=' in name:
            try:
                iou_value = float(name.split('val_iou=')[1].split('-')[0])
                iou_scored.append((iou_value, ckpt))
                continue
            except Exception:
                pass
        if 'val_loss=' in name:
            try:
                loss_value = float(name.split('val_loss=')[1].split('-')[0])
                loss_scored.append((loss_value, ckpt))
            except Exception:
                pass

    if iou_scored:
        best_ckpt = max(iou_scored, key=lambda x: x[0])[1]
    elif loss_scored:
        best_ckpt = min(loss_scored, key=lambda x: x[0])[1]
    else:
        best_ckpt = best_ckpts[0]

    # 2. 加载模型
    print("\n🚀 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  📍 设备: {device}")

    model = load_model_weights(str(best_ckpt), device)
    print(f"  ✅ 模型加载成功: {best_ckpt.name}")

    # 3. 准备验证集数据
    print("\n📊 准备验证集数据...")
    splitter = CrackDatasetSplitter()
    train_pairs, val_pairs = splitter.get_train_val_pairs()

    # 选择几个验证样本（均匀分布）
    n_samples = min(6, len(val_pairs))
    sample_indices = np.linspace(0, len(val_pairs) - 1, n_samples, dtype=int)

    print(f"  • 验证集总数: {len(val_pairs)}")
    print(f"  • 选择样本数: {n_samples}")

    # 4. 计算训练集的 mean 和 std
    print("\n📐 计算训练集 mean 和 std...")
    train_images_dir = os.path.join(
        Config.DATASET_ROOT, Config.IMAGE_DIR, Config.TRAIN_SPLIT
    )
    mean, std = compute_mean_std_from_dir(train_images_dir)
    print(f"  ✓ Mean: {mean}")
    print(f"  ✓ Std: {std}")

    # 5. 推理
    print("\n🔮 开始推理...")
    samples = []

    for i, idx in enumerate(sample_indices):
        image_path, mask_path, _ = val_pairs[idx]

        # 加载图像
        orig_image, orig_mask, resized_image, resized_mask = load_image_and_mask(
            image_path, mask_path
        )

        # 预测
        pred_mask = predict(model, resized_image, device, mean, std)

        # 计算指标
        metrics = calculate_metrics(pred_mask, resized_mask, threshold=Config.INFERENCE_THRESHOLD)

        samples.append({
            'original_image': orig_image,
            'gt_mask': orig_mask,
            'resized_image': resized_image,
            'resized_mask': resized_mask,
            'pred_mask': pred_mask,
            'metrics': metrics,
            'name': Path(image_path).name
        })

        print(f"  [{i+1}/{n_samples}] {Path(image_path).name:<30} "
              f"IoU: {metrics['iou']:.4f} | F1: {metrics['f1']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}")

    # 5. 可视化
    print("\n🎨 生成可视化...")
    output_dir = f'results/{latest_checkpoint_dir.name}/inference'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建 Visualizer
    visualizer = Visualizer(save_dir=output_dir, dpi=150)
    
    output_path = visualize_and_save_results(samples, output_dir, best_ckpt.name, visualizer)

    # 6. 统计
    avg_metrics = {
        'iou': np.mean([s['metrics']['iou'] for s in samples]),
        'f1': np.mean([s['metrics']['f1'] for s in samples]),
        'precision': np.mean([s['metrics']['precision'] for s in samples]),
        'recall': np.mean([s['metrics']['recall'] for s in samples])
    }

    print("\n" + "="*80)
    print("✅ 推理完成！")
    print("="*80)
    print(f"📦 模型: {best_ckpt.name}")
    print(f"📊 样本数: {n_samples}")
    print(f"\n📈 平均指标:")
    print(f"  • Threshold: {Config.INFERENCE_THRESHOLD:.3f}")
    print(f"  • IoU:       {avg_metrics['iou']:.4f}")
    print(f"  • F1 Score:  {avg_metrics['f1']:.4f}")
    print(f"  • Precision: {avg_metrics['precision']:.4f}")
    print(f"  • Recall:    {avg_metrics['recall']:.4f}")
    print(f"\n💾 结果保存至: {output_dir}")
    print("="*80)

    return output_path


if __name__ == '__main__':
    try:
        output_path = main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
