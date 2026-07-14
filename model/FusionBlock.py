import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import Attention as SelfAttention

from model.CNNBlock import CrackOrientedBlock
from model.MambaBlock import MultiScaleSnakeMamba2D, SnakeMamba2D


def _auto_heads(width: int) -> int:
    candidate = max(1, min(8, width // 32))
    while candidate > 1 and width % candidate != 0:
        candidate -= 1
    return candidate


# ==================== Cross-Attention ====================

class CrossAttention(nn.Module):
    """
    交叉注意力: Q 保持全分辨率, KV 自适应池化

    Q = W_q(CNN features)     [B, N, C]    N = H*W
    K = W_k(pooled features)  [B, M, C]    M = kv_size²  
    V = W_v(pooled features)  [B, M, C]

    当 H=W=kv_size 时退化为标准交叉注意力 (无池化)
    memory: O(N * M)  << O(N²)
    """

    def __init__(
        self,
        dim,
        num_heads,
        kv_size=None,
        kv_pool_ratio=4,
        min_kv_size=4,
        attn_drop=0.0,
        proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv_size = None if kv_size is None else max(1, int(kv_size))
        self.kv_pool_ratio = max(1, int(kv_pool_ratio))
        self.min_kv_size = max(1, int(min_kv_size))

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, H, W):
        """
        q:  (B, N, C) - query at full resolution
        kv: (B, N, C) - key/value (will be pooled to kv_size²)
        H, W: spatial dimensions for pooling
        """
        B, N, C = q.shape

        # Pool KV to controllable spatial size (from config/global_pool)
        kv_2d = kv.transpose(1, 2).reshape(B, C, H, W)
        if self.kv_size is not None:
            kv_h = min(H, self.kv_size)
            kv_w = min(W, self.kv_size)
        else:
            kv_h = max(self.min_kv_size, H // self.kv_pool_ratio)
            kv_w = max(self.min_kv_size, W // self.kv_pool_ratio)
            kv_h = min(H, kv_h)
            kv_w = min(W, kv_w)

        kv_pooled = F.adaptive_max_pool2d(kv_2d, (kv_h, kv_w)).flatten(2).transpose(1, 2)  # (B, M, C)
        M = kv_pooled.shape[1]

        Q = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(kv_pooled).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(kv_pooled).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# ==================== CNN + Mamba Cross-Attention Block ====================

class CNNandMambaBlock(nn.Module):
    """
    CNN-Mamba 双向交叉注意力块 (多尺度版本)

    COBlock(x) ∥ MultiScaleSnakeMamba2D(x)
    →  BiDirectional CrossAttention:
       - Q=CNN, KV=Mamba
       - Q=Mamba, KV=CNN
    →  Concat + Conv Fusion  →  MLP

    双向交叉注意力让 CNN 和 Mamba 特征互相增强
    多尺度 Mamba 同时捕捉细裂缝和粗裂缝
    """

    def __init__(self, channels, k=7, d_state=16,
                 attn_drop=0.0, proj_drop=0.1, drop=0.1,
                 drop_path=0.0, mlp_ratio=2.0, kv_pool_ratio=4,
                 scan_directions=8, use_multiscale_mamba=True, **kwargs):
        super().__init__()
        num_heads = _auto_heads(channels)

        # 并行分支
        self.cnn = CrackOrientedBlock(channels, k=k)
        if use_multiscale_mamba:
            self.mamba = MultiScaleSnakeMamba2D(channels, d_state=d_state, drop=drop,
                                                scan_directions=scan_directions)
        else:
            self.mamba = SnakeMamba2D(channels, d_state=d_state, drop=drop,
                                      scan_directions=scan_directions, merge_mode='attention')

        # 双向交叉注意力
        # Direction 1: Q=CNN, KV=Mamba
        self.norm_q1 = nn.LayerNorm(channels)
        self.norm_kv1 = nn.LayerNorm(channels)
        self.cross_attn1 = CrossAttention(
            channels, num_heads,
            kv_pool_ratio=kv_pool_ratio,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )

        # Direction 2: Q=Mamba, KV=CNN
        self.norm_q2 = nn.LayerNorm(channels)
        self.norm_kv2 = nn.LayerNorm(channels)
        self.cross_attn2 = CrossAttention(
            channels, num_heads,
            kv_pool_ratio=kv_pool_ratio,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )

        # 融合: Concat + 1×1 Conv
        self.fusion = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels),
        )

        # MLP
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio),
                       act_layer=nn.GELU, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, HW, C) 原始输入序列

        # 并行: CNN + Mamba
        cnn_feat = self.cnn(x)       # (B, C, H, W)
        mamba_feat = self.mamba(x)    # (B, C, H, W)

        # 展平为序列
        cnn_seq = cnn_feat.flatten(2).transpose(1, 2)     # (B, HW, C)
        mamba_seq = mamba_feat.flatten(2).transpose(1, 2)   # (B, HW, C)

        # 双向交叉注意力
        out1 = self.cross_attn1(self.norm_q1(cnn_seq), self.norm_kv1(mamba_seq), H, W)     # Q=CNN, KV=Mamba
        out2 = self.cross_attn2(self.norm_q2(mamba_seq), self.norm_kv2(cnn_seq), H, W)     # Q=Mamba, KV=CNN

        # Concat + Fusion
        fused = self.fusion(torch.cat([out1, out2], dim=-1))  # (B, HW, 2C) → (B, HW, C)

        # 残差连接到原始输入: 确保细裂缝位置信息不被双分支平均稀释
        out = x_seq + self.drop_path(fused)

        # MLP + 残差
        out = out + self.drop_path(self.mlp(self.norm_mlp(out)))

        return out.transpose(1, 2).reshape(B, C, H, W)


# ==================== CNN + Transformer Cross-Attention Block ====================

class CNNandTransformerBlock(nn.Module):
    """
    CNN-Transformer 双向交叉注意力块

    COBlock(x) ∥ SelfAttention(x)
    →  BiDirectional CrossAttention:
       - Q=CNN, KV=Transformer
       - Q=Transformer, KV=CNN
    →  Concat + Conv Fusion  →  MLP

    双向交叉注意力让 CNN 和 Transformer 特征互相增强
    仅用于低分辨率阶段 (28×28), 自注意力 O(N²) 可控
    """

    def __init__(self, channels, k=7,
                 attn_drop=0.0, proj_drop=0.1, drop=0.1,
                 drop_path=0.0, mlp_ratio=4.0,
                 window_size=7, kv_pool_ratio=4, **kwargs):
        super().__init__()
        num_heads = _auto_heads(channels)
        self.window_size = max(1, int(window_size))

        # CNN 分支
        self.cnn = CrackOrientedBlock(channels, k=k)

        # Transformer 分支: 自注意力
        self.norm_sa = nn.LayerNorm(channels)
        self.self_attn = SelfAttention(
            channels, num_heads=num_heads, qkv_bias=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )

        # 双向交叉注意力
        # Direction 1: Q=CNN, KV=Transformer
        self.norm_q1 = nn.LayerNorm(channels)
        self.norm_kv1 = nn.LayerNorm(channels)
        self.cross_attn1 = CrossAttention(
            channels, num_heads,
            kv_pool_ratio=kv_pool_ratio,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )

        # Direction 2: Q=Transformer, KV=CNN
        self.norm_q2 = nn.LayerNorm(channels)
        self.norm_kv2 = nn.LayerNorm(channels)
        self.cross_attn2 = CrossAttention(
            channels, num_heads,
            kv_pool_ratio=kv_pool_ratio,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )

        # 融合: Concat + 1×1 Conv
        self.fusion = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels),
        )

        # MLP
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio),
                       act_layer=nn.GELU, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # CNN 分支
        cnn_feat = self.cnn(x)  # (B, C, H, W)

        # Transformer 分支: 自注意力 + 残差
        # window_size 控制 Transformer 分支 token 规模
        if H > self.window_size or W > self.window_size:
            pooled = F.adaptive_avg_pool2d(x, (self.window_size, self.window_size))
            pooled_seq = pooled.flatten(2).transpose(1, 2)  # (B, ws*ws, C)
            trans_pooled = pooled_seq + self.self_attn(self.norm_sa(pooled_seq))
            trans_2d = trans_pooled.transpose(1, 2).reshape(B, C, self.window_size, self.window_size)
            trans_2d = F.interpolate(trans_2d, size=(H, W), mode='bilinear', align_corners=False)
            trans_feat = trans_2d.flatten(2).transpose(1, 2)  # (B, HW, C)
        else:
            x_seq = x.flatten(2).transpose(1, 2)                     # (B, HW, C)
            trans_feat = x_seq + self.self_attn(self.norm_sa(x_seq))  # (B, HW, C)

        # 展平 CNN 特征
        cnn_seq = cnn_feat.flatten(2).transpose(1, 2)   # (B, HW, C)

        # 双向交叉注意力
        out1 = self.cross_attn1(self.norm_q1(cnn_seq), self.norm_kv1(trans_feat), H, W)       # Q=CNN, KV=Trans
        out2 = self.cross_attn2(self.norm_q2(trans_feat), self.norm_kv2(cnn_seq), H, W)       # Q=Trans, KV=CNN

        # Concat + Fusion
        fused = self.fusion(torch.cat([out1, out2], dim=-1))  # (B, HW, 2C) → (B, HW, C)

        # 对称残差连接 (CNN + Mamba 均参与)
        out = (cnn_seq + mamba_seq) * 0.5 + self.drop_path(fused)

        # MLP + 残差
        out = out + self.drop_path(self.mlp(self.norm_mlp(out)))

        return out.transpose(1, 2).reshape(B, C, H, W)
