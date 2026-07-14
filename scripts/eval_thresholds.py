#!/usr/bin/env python

import os
import sys
import json
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到 sys.path
# 无论从哪个目录运行，都能正确找到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import Config, get_device
from model.TriBranchCracknet import TriBranchCrackNet
from data.dataset import CrackDatasetSplitter, CrackDataLoader
from train.Trainer import LitCrackModel


def evaluate_iou_over_thresholds(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    thresholds: np.ndarray,
    device: torch.device
) -> Dict[float, float]:
    """
    在验证集上评估多个阈值的IoU

    Args:
        model: 加载好的模型
        val_loader: 验证集数据加载器
        thresholds: 阈值数组
        device: 计算设备

    Returns:
        OrderedDict: {threshold: iou} 的有序字典
    """
    num_thresholds = len(thresholds)

    # 初始化累积变量
    total_intersections = np.zeros(num_thresholds, dtype=np.float64)
    total_unions = np.zeros(num_thresholds, dtype=np.float64)

    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating thresholds'):
            # 获取图像和掩膜
            if isinstance(batch, (tuple, list)):
                if len(batch) == 3:
                    images, masks, _ = batch
                elif len(batch) >= 2:
                    images, masks = batch[:2]
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}, len={len(batch)}")
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            images = images.to(device)

            # 前向传播
            output = model(images)

            # 解析模型输出（支持辅助监督输出）
            if isinstance(output, dict):
                logits = output['main_output']
            elif isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output

            # 计算概率
            probs = torch.sigmoid(logits)

            # 转换为numpy
            probs_np = probs.squeeze(1).cpu().numpy()  # [B, H, W]
            gts_np = masks.squeeze(1).cpu().numpy()     # [B, H, W]

            # 对每个阈值计算intersection和union
            for i, thr in enumerate(thresholds):
                # 二值化预测
                preds_bin = (probs_np >= thr).astype(np.uint8)
                # 确保GT是0/1
                gts_bin = (gts_np > 0.5).astype(np.uint8)

                # 计算intersection和union
                inter = np.logical_and(preds_bin == 1, gts_bin == 1).sum()
                union = np.logical_or(preds_bin == 1, gts_bin == 1).sum()

                total_intersections[i] += inter
                total_unions[i] += union

    # 计算IoU
    ious = total_intersections / (total_unions + 1e-6)

    # 构建有序字典
    results = OrderedDict()
    for thr, iou in zip(thresholds, ious):
        results[float(thr)] = float(iou)

    return results


def main():
    """主函数"""
    # 获取配置
    import sys
    if len(sys.argv) < 2:
        print("使用方法: python eval_thresholds.py <checkpoint_path>")
        print("示例: python eval_thresholds.py checkpoints/20231201_120000/best-epoch=50-val_loss=0.1234.ckpt")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    thr_config = Config.THRESHOLD_SEARCH

    # 检查checkpoint是否存在
    if not os.path.exists(ckpt_path):
        print(f"错误: checkpoint文件不存在: {ckpt_path}")
        sys.exit(1)

    # 设备选择
    device = get_device()
    print(f"使用设备: {device}")

    # 输出目录
    output_dir = thr_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 生成阈值列表
    thr_min = thr_config['thr_min']
    thr_max = thr_config['thr_max']
    thr_step = thr_config['thr_step']
    thresholds = np.arange(thr_min, thr_max + 1e-8, thr_step)

    print(f"\n阈值搜索范围: [{thr_min:.2f}, {thr_max:.2f}], 步长: {thr_step:.3f}")
    print(f"共 {len(thresholds)} 个阈值")

    # 1. 准备验证集数据
    print("\n加载验证集数据...")
    splitter = CrackDatasetSplitter()
    train_pairs, val_pairs = splitter.get_train_val_pairs()

    data_loader = CrackDataLoader(
        train_pairs, val_pairs,
        Config.IMAGE_SIZE, Config.BATCH_SIZE
    )
    _, val_loader = data_loader.get_loaders()
    print(f"验证集样本数: {len(val_pairs)}")

    # 2. 创建模型结构
    print("\n创建模型结构...")
    model_config = Config.MODEL_CONFIGS
    stage_count = len(model_config['widths'])
    stage_blocks = Config.expand_stage_blocks(model_config['depths'], stage_count)

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

    # 3. 加载checkpoint
    print(f"\n加载checkpoint: {ckpt_path}")
    lit_model = LitCrackModel.load_from_checkpoint(
        ckpt_path,
        model=model,
        loss_config=Config.LOSS_CONFIG,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        edge_loss_weight=Config.EDGE_LOSS_WEIGHT,
        edge_aux_loss_weight=Config.EDGE_AUX_LOSS_WEIGHT,
        edge_threshold=Config.EDGE_THRESHOLD,
        metric_threshold=Config.METRIC_THRESHOLD,
        map_location=device
    )

    # 若checkpoint包含EMA权重，优先使用EMA权重评估（与训练验证阶段保持一致）
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema_state = raw_ckpt.get('ema_state', None) if isinstance(raw_ckpt, dict) else None
    if isinstance(ema_state, dict) and len(ema_state) > 0:
        missing, unexpected = lit_model.model.load_state_dict(ema_state, strict=False)
        print(f"检测到EMA权重，使用EMA进行评估 (missing={len(missing)}, unexpected={len(unexpected)})")

    lit_model = lit_model.to(device)
    lit_model.eval()

    # 4. 评估所有阈值
    print("\n开始评估阈值...")
    results = evaluate_iou_over_thresholds(
        lit_model.model, val_loader, thresholds, device
    )

    # 5. 打印结果表格
    print("\n" + "=" * 40)
    print("阈值搜索结果:")
    print("=" * 40)

    ious_array = np.array(list(results.values()))
    thresholds_array = np.array(list(results.keys()))

    for thr, iou in results.items():
        print(f"threshold={thr:.3f}, IoU={iou:.4f}")

    # 6. 找到最优阈值
    best_idx = np.argmax(ious_array)
    best_thr = thresholds_array[best_idx]
    best_iou = ious_array[best_idx]

    print("\n" + "=" * 40)
    print(f"最优阈值: {best_thr:.3f}, IoU={best_iou:.4f}")
    print("=" * 40)

    # 7. 保存结果到CSV
    csv_path = os.path.join(output_dir, 'threshold_search.csv')
    with open(csv_path, 'w') as f:
        f.write('threshold,iou\n')
        for thr, iou in results.items():
            f.write(f'{thr:.4f},{iou:.6f}\n')
    print(f"\n结果已保存到: {csv_path}")

    # 8. 保存最优阈值到JSON
    json_path = os.path.join(output_dir, 'best_threshold.json')
    best_result = {
        'best_threshold': float(best_thr),
        'best_iou': float(best_iou),
        'search_range': {
            'min': float(thr_min),
            'max': float(thr_max),
            'step': float(thr_step)
        },
        'checkpoint': ckpt_path
    }
    with open(json_path, 'w') as f:
        json.dump(best_result, f, indent=2)
    print(f"最优阈值已保存到: {json_path}")

    # 9. 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n阈值搜索完成!")

    return best_thr, best_iou


if __name__ == '__main__':
    main()
