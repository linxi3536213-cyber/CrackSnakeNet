import torch
import torch.nn as nn
from typing import List
from mamba_ssm import Mamba2
from timm.layers import Mlp
from model.snake_scan import MultiDirectionalSnakeScan
from model.CNNBlock import DynamicAttentionFusion


class SnakeMamba2D(nn.Module):
    """
    8方向蛇形扫描 Mamba 模块

    流程:
      1. 8方向蛇形扫描 (↑↓←→ ↗↘↙↖) → 8条序列
      2. 共享 Mamba2 处理每条序列 (序列内残差)
      3. 各方向 unscan 回 2D → DynamicAttentionFusion 注意力融合
      4. LayerNorm + MLP + 残差

    merge_mode:
      - 'attention': DynamicAttentionFusion (默认, 用于编码器)
      - 'concat': 1×1 Conv 通道压缩 (用于 SkipMamba 等轻量场景)
      - 'add': 直接相加
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        scan_directions: int = 8,
        merge_mode: str = "attention",
        chunk_size: int = 512,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.merge_mode = merge_mode

        # 多方向扫描器 (concat/add 模式由 scanner 内部处理)
        scanner_merge = merge_mode if merge_mode != 'attention' else 'add'
        self.scanner = MultiDirectionalSnakeScan(
            num_directions=scan_directions,
            channels=channels,
            merge_mode=scanner_merge
        )

        head_dim = min(64, channels)
        while head_dim > 1 and channels % head_dim != 0:
            head_dim //= 2
        if channels % head_dim != 0:
            head_dim = 1

        # Mamba 前归一化
        self.norm1 = nn.LayerNorm(channels)

        # 共享 Mamba2 层 (处理所有方向的序列)
        self.mamba = Mamba2(
            d_model=channels,
            d_state=d_state,
            chunk_size=chunk_size,
            headdim=head_dim,
            sequence_parallel=False,
            use_mem_eff_path=False,
        )

        # 注意力融合 (仅 attention 模式)
        if merge_mode == 'attention':
            self.dir_attn = DynamicAttentionFusion(channels, num_branches=scan_directions)

        # MLP
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=int(channels * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  →  (B, C, H, W)
        """
        B, C, H, W = x.shape
        identity = x  # 全局残差: 保留输入中的细裂缝位置信息

        # ===== 8方向扫描 + Mamba =====
        seqs = self.scanner(x)  # List[[B, N, C]]
        processed: List[torch.Tensor] = []
        for seq in seqs:
            y = self.mamba(self.norm1(seq))
            y = seq + y  # 序列内残差
            processed.append(y)

        # ===== 方向融合 =====
        if self.merge_mode == 'attention':
            feat_maps = self.scanner.unscan_all(processed, H, W)  # List[[B, C, H, W]]
            out = self.dir_attn(feat_maps)  # (B, C, H, W)
        else:
            out = self.scanner.merge(processed, H, W)  # (B, C, H, W)

        # ===== MLP =====
        B, C, H, W = out.shape
        x_seq = out.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        out = x_seq.transpose(1, 2).reshape(B, C, H, W)

        return identity + out  # 全局残差: 防止8方向融合稀释细裂缝信号


class MultiScaleSnakeMamba2D(nn.Module):
    """
    多尺度蛇形 Mamba 模块

    在两个分辨率上应用 SnakeMamba2D:
      1. 全分辨率 (H×W): 捕捉细节裂缝
      2. 降采样 (H/2×W/2): 捕捉粗裂缝和全局上下文
      3. 动态注意力融合两个尺度

    适合裂缝检测任务的多尺度特征提取
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        scan_directions: int = 8,
        pool_size: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()
        self.pool_size = pool_size

        # 全分辨率 Mamba (细节)
        self.mamba_full = SnakeMamba2D(
            channels=channels,
            d_state=d_state,
            scan_directions=scan_directions,
            merge_mode='attention',
            drop=drop,
        )

        # 降采样 Mamba (粗粒度)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.mamba_coarse = SnakeMamba2D(
            channels=channels,
            d_state=d_state,
            scan_directions=scan_directions,
            merge_mode='attention',
            drop=drop,
        )
        self.upsample = nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=False)

        # 多尺度融合
        self.scale_fusion = DynamicAttentionFusion(channels, num_branches=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) → (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 全分辨率 Mamba (细节特征)
        out_full = self.mamba_full(x)  # (B, C, H, W)

        # 降采样 → Mamba → 上采样 (粗粒度特征)
        x_coarse = self.pool(x)  # (B, C, H/2, W/2)
        out_coarse = self.mamba_coarse(x_coarse)  # (B, C, H/2, W/2)
        out_coarse = self.upsample(out_coarse)  # (B, C, H, W)

        # 多尺度融合
        out = self.scale_fusion([out_full, out_coarse])  # (B, C, H, W)

        return out
