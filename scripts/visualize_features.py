#!/usr/bin/env python
"""
CNN vs Mamba 分支特征图可视化 + DAF 方向权重可视化

功能:
  1. CNN vs Mamba feature map: 在 CNNandMambaBlock 中 hook cnn_feat 和 mamba_feat,
     输出 channel-mean activation map, 直观展示两个分支关注什么
  2. DAF 方向权重: 在 SnakeMamba2D 的 DynamicAttentionFusion 中 hook attention weights,
     展示 8 个扫描方向的权重分布

用法:
    python scripts/visualize_features.py --checkpoint checkpoints/XXXXX/best-xxx.ckpt --images_dir path/to/images --output_dir results/vis
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from train.Trainer import LitCrackModel
from data.calculateMeanStd import compute_mean_std_from_dir


# ==================== Hook 工具 ====================

class FeatureHook:
    """通用 forward hook, 保存中间特征"""
    def __init__(self):
        self.features = {}

    def register(self, module, name):
        def hook_fn(mod, inp, out, n=name):
            if isinstance(out, torch.Tensor):
                self.features[n] = out.detach().cpu()
        module.register_forward_hook(hook_fn)

    def clear(self):
        self.features = {}


class CMBlockHook:
    """Hook CNNandMambaBlock 的 CNN 和 Mamba 分支输出"""
    def __init__(self):
        self.cnn_feats = []
        self.mamba_feats = []

    def register(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'CNNandMambaBlock':
                # Hook CNN 分支 (CrackOrientedBlock)
                self._hook_branch(module.cnn, self.cnn_feats, name + '.cnn')
                # Hook Mamba 分支 (MultiScaleSnakeMamba2D)
                self._hook_branch(module.mamba, self.mamba_feats, name + '.mamba')

    def _hook_branch(self, module, storage, tag):
        def hook_fn(mod, inp, out, s=storage):
            if isinstance(out, torch.Tensor):
                s.append(out.detach().cpu())
        module.register_forward_hook(hook_fn)

    def clear(self):
        self.cnn_feats = []
        self.mamba_feats = []


class DAFWeightHook:
    """Hook DynamicAttentionFusion 的 attention weights"""
    def __init__(self):
        self.weights = []
        self.locations = []

    def register(self, model):
        for name, module in model.named_modules():
            if (module.__class__.__name__ == 'DynamicAttentionFusion'
                    and 'dir_attn' in name):
                self._hook_daf(module, name)

    def _hook_daf(self, daf_module, tag):
        original_forward = daf_module.forward

        def hooked_forward(features):
            cat_feat = torch.cat(features, dim=1)
            weights = F.softmax(daf_module.fc(daf_module.gap(cat_feat)), dim=1)
            self.weights.append(weights.detach().cpu())
            self.locations.append(tag)
            stacked = torch.stack(features, dim=1)
            return (stacked * weights.unsqueeze(2)).sum(dim=1)

        daf_module.forward = hooked_forward

    def clear(self):
        self.weights = []
        self.locations = []


# ==================== 可视化函数 ====================

def activation_to_heatmap(feat, size=None):
    """将特征图 (C, H, W) 转为 channel-mean heatmap"""
    heatmap = feat.mean(dim=0).numpy()  # (H, W)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    if size is not None:
        heatmap = cv2.resize(heatmap, (size[1], size[0]))
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap, heatmap_color


def overlay_heatmap(image, heatmap_color, alpha=0.5):
    """将热力图叠加到原图上"""
    if image.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)


def visualize_cnn_vs_mamba(image_bgr, cnn_feats, mamba_feats, save_path):
    """CNN vs Mamba 分支特征图对比可视化"""
    h, w = image_bgr.shape[:2]
    n_stages = min(len(cnn_feats), len(mamba_feats))

    fig, axes = plt.subplots(n_stages, 3, figsize=(12, 4 * n_stages))
    if n_stages == 1:
        axes = axes[np.newaxis, :]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    for i in range(n_stages):
        # CNN heatmap
        cnn_hm, cnn_color = activation_to_heatmap(cnn_feats[i][0], (h, w))
        cnn_overlay = overlay_heatmap(image_bgr, cnn_color)

        # Mamba heatmap
        mamba_hm, mamba_color = activation_to_heatmap(mamba_feats[i][0], (h, w))
        mamba_overlay = overlay_heatmap(image_bgr, mamba_color)

        axes[i, 0].imshow(image_rgb)
        axes[i, 0].set_title(f'Input Image', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cv2.cvtColor(cnn_overlay, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f'CNN Branch (Stage {i + 2})', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(cv2.cvtColor(mamba_overlay, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f'Mamba Branch (Stage {i + 2})', fontsize=10)
        axes[i, 2].axis('off')

    plt.suptitle('CNN vs Mamba Feature Activation Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  CNN vs Mamba 可视化已保存: {save_path}')


def visualize_direction_weights(daf_weights, daf_locations, save_path):
    """DAF 8方向权重可视化"""
    direction_names = [
        '↘ diag', '↙ anti-diag', '↖ frac-45', '↗ frac-135',
        '→ row', '↓ col', '← row-rev', '↑ col-rev',
    ]

    # 收集所有 dir_attn 的权重
    dir_weights = []
    dir_locs = []
    for w, loc in zip(daf_weights, daf_locations):
        if w.shape[1] == 8:  # 只要 8 方向的 (排除 4 分支 COBlock 的)
            dir_weights.append(w)
            dir_locs.append(loc)

    if not dir_weights:
        print('  警告: 未捕获到 8 方向 DAF 权重')
        return

    fig, axes = plt.subplots(1, len(dir_weights), figsize=(6 * len(dir_weights), 5))
    if len(dir_weights) == 1:
        axes = [axes]

    for idx, (w, loc) in enumerate(zip(dir_weights, dir_locs)):
        # w: (B, 8, 1, 1) → (8,) 平均
        avg_w = w[0, :, 0, 0].numpy()
        n = min(len(avg_w), len(direction_names))
        colors = plt.cm.Set2(np.linspace(0, 1, n))

        bars = axes[idx].bar(range(n), avg_w[:n], color=colors)
        axes[idx].set_xticks(range(n))
        axes[idx].set_xticklabels(direction_names[:n], rotation=45, ha='right', fontsize=8)
        axes[idx].set_ylabel('Attention Weight')
        axes[idx].set_title(f'{loc}\n(sum={avg_w[:n].sum():.3f})', fontsize=9)
        axes[idx].set_ylim(0, max(avg_w[:n]) * 1.3)

        for bar, val in zip(bars, avg_w[:n]):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('SnakeMamba2D Direction Attention Weights', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  方向权重可视化已保存: {save_path}')


# ==================== 主流程 ====================

def load_model(checkpoint_path):
    """加载模型"""
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
        global_pool=Config.TRANSFORMER_GLOBAL_POOL,
        scan_directions=model_config.get('scan_directions', 8),
    )

    lit_model = LitCrackModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        loss_config=Config.LOSS_CONFIG,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    lit_model.eval()
    return lit_model


def preprocess_image(image_path, mean, std):
    """预处理单张图片"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取: {image_path}")
    image_bgr = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (Config.IMAGE_SIZE[1], Config.IMAGE_SIZE[0]))

    img_float = image_resized.astype(np.float32) / 255.0
    safe_std = np.maximum(np.array(std), 1e-6)
    img_norm = (img_float - np.array(mean)) / safe_std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, image_resized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--images_dir', type=str, default=None,
                        help='图片目录 (默认使用验证集)')
    parser.add_argument('--output_dir', type=str, default='results/vis')
    parser.add_argument('--num_images', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 计算均值标准差
    train_images_dir = os.path.join(Config.DATASET_ROOT, Config.IMAGE_DIR, Config.TRAIN_SPLIT)
    mean, std = compute_mean_std_from_dir(train_images_dir)
    mean, std = mean.tolist(), std.tolist()

    # 图片来源
    if args.images_dir is None:
        images_dir = os.path.join(Config.DATASET_ROOT, Config.IMAGE_DIR, Config.TEST_SPLIT)
    else:
        images_dir = args.images_dir

    image_files = sorted([
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
    ])[:args.num_images]

    print(f"加载模型: {args.checkpoint}")
    lit_model = load_model(args.checkpoint)
    model = lit_model.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # 注册 hooks
    cm_hook = CMBlockHook()
    cm_hook.register(model)

    daf_hook = DAFWeightHook()
    daf_hook.register(model)

    print(f"\n处理 {len(image_files)} 张图片...\n")
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{basename}]")

        tensor, image_resized = preprocess_image(img_path, mean, std)
        tensor = tensor.to(device)

        cm_hook.clear()
        daf_hook.clear()

        with torch.no_grad():
            _ = model(tensor)

        # 1. CNN vs Mamba 特征图
        if cm_hook.cnn_feats and cm_hook.mamba_feats:
            visualize_cnn_vs_mamba(
                cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR),
                cm_hook.cnn_feats,
                cm_hook.mamba_feats,
                os.path.join(args.output_dir, f'{basename}_cnn_vs_mamba.png'),
            )

        # 2. 方向权重
        if daf_hook.weights:
            visualize_direction_weights(
                daf_hook.weights,
                daf_hook.locations,
                os.path.join(args.output_dir, f'{basename}_direction_weights.png'),
            )

    print(f"\n所有可视化结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
