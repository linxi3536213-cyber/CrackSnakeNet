import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from timm.layers.norm import GroupNorm


def get_group_gn(channels):
    """自适应计算GroupNorm的组数

    Args:
        channels: 输入通道数

    Returns:
        合适的组数（确保channels能被整除）
    """
    if channels >= 32 and channels % 32 == 0:
        return 32
    if channels >= 16 and channels % 16 == 0:
        return 16
    if channels >= 8 and channels % 8 == 0:
        return 8
    if channels >= 4 and channels % 4 == 0:
        return 4
    return 1


def make_group_norm(channels):
    return GroupNorm(num_groups=get_group_gn(channels), num_channels=channels)


class ConvBlock(nn.Module):
    """卷积块（使用GroupNorm）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = make_group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = make_group_norm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.activation(x)
        return x


class ConvGNAct(nn.Module):
    """单层卷积 + GroupNorm + 激活函数"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = make_group_norm(out_channels)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownsampleBlock(nn.Module):
    """下采样模块: 3x3 ConvGNAct"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = ConvGNAct(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )

    def forward(self, x):
        return self.conv(x)



class ConvDownsample(nn.Module):
    """卷积下采样（使用GroupNorm）"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)
        self.norm = make_group_norm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding（使用卷积实现，保持输出为特征图）"""
    def __init__(self, in_channels, embed_dim, patch_size=2, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.norm = make_group_norm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class StemLayer(nn.Module):
    """下采样干层: 将输入图像降采样并提取初始特征。

    Input:  (B, in_channels, H, W)
    Output: (B, out_channels, H//2, W//2)

    其中 out_channels = base // 2 (base = dims[0])
    """

    def __init__(self, in_channels: int = 3, dims: int | list | tuple = 96):
        super().__init__()
        if isinstance(dims, (list, tuple)):
            if not dims:
                raise ValueError("dims must not be empty")
            base = dims[0]
        else:
            base = dims
        self.out_channels = base // 2
        self.layer0 = ConvBlock(in_channels=in_channels, out_channels=base // 4)
        self.downsample = ConvDownsample(base // 4, base // 4, stride=2)
        self.layer1 = ConvBlock(in_channels=base // 4, out_channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)       # (B, base//4, H, W)
        x = self.downsample(x)   # (B, base//4, H//2, W//2)
        x = self.layer1(x)       # (B, out_channels, H//2, W//2)
        return x


class DynamicAttentionFusion(nn.Module):
    """
    动态注意力融合
    Cat 路径: 拼接所有分支输出 → GAP → FC → Softmax → 注意力权重
    Stack 路径: 堆叠所有分支输出
    加权求和: 注意力权重 × 堆叠特征
    """
    def __init__(self, channels: int, num_branches: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels * num_branches, num_branches, kernel_size=1, bias=False)

    def forward(self, features: list) -> torch.Tensor:
        # Cat → GAP → FC → Softmax → 注意力权重 (B, N, 1, 1)
        cat_feat = torch.cat(features, dim=1)
        weights = F.softmax(self.fc(self.gap(cat_feat)), dim=1)

        # Stack → 加权求和
        stacked = torch.stack(features, dim=1)  # (B, N, C, H, W)
        return (stacked * weights.unsqueeze(2)).sum(dim=1)


class DynamicSnakeConv(nn.Module):
    """
    轻量化动态蛇形卷积 (1D 偏移版本)

    学习沿主方向的 1D 偏移量，让卷积核自适应弯曲
    - 水平卷积 (1×k): 学习垂直偏移 (沿裂缝弯曲)
    - 垂直卷积 (k×1): 学习水平偏移 (沿裂缝弯曲)

    相比 2D 偏移，参数更少，更适合小数据集
    """
    def __init__(self, channels, kernel_size=9, direction='horizontal'):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.direction = direction

        # 偏移量预测网络 (轻量级)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, groups=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, kernel_size, 1),  # 每个采样点一个偏移
            nn.Tanh(),  # 限制偏移范围 [-1, 1]
        )

        # 可变形卷积核 (depthwise)
        if direction == 'horizontal':
            self.weight = nn.Parameter(torch.randn(channels, 1, 1, kernel_size))
        else:  # vertical
            self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, 1))

        # 归一化和激活
        self.norm = make_group_norm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 预测偏移量 (B, K, H, W)
        offset = self.offset_conv(x) * (self.kernel_size // 2)  # 缩放到合理范围

        # 2. 生成采样网格
        device = x.device
        if self.direction == 'horizontal':
            # 水平卷积: 采样点沿水平方向，学习垂直偏移
            base_grid_x = torch.linspace(-1, 1, self.kernel_size, device=device)
            base_grid_x = base_grid_x.view(1, self.kernel_size, 1, 1).expand(B, -1, H, W)
            base_grid_y = torch.zeros(B, self.kernel_size, H, W, device=device)

            # 归一化坐标
            grid_x_norm = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, self.kernel_size, H, W).float() / (W - 1) * 2 - 1
            grid_y_norm = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, self.kernel_size, H, W).float() / (H - 1) * 2 - 1

            # 添加偏移 (只在 y 方向偏移)
            grid_x_norm = grid_x_norm + base_grid_x * (self.kernel_size / W)
            grid_y_norm = grid_y_norm + offset / H * 2  # (B, K, H, W)

        else:  # vertical
            # 垂直卷积: 采样点沿垂直方向，学习水平偏移
            base_grid_y = torch.linspace(-1, 1, self.kernel_size, device=device)
            base_grid_y = base_grid_y.view(1, self.kernel_size, 1, 1).expand(B, -1, H, W)
            base_grid_x = torch.zeros(B, self.kernel_size, H, W, device=device)

            # 归一化坐标
            grid_x_norm = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, self.kernel_size, H, W).float() / (W - 1) * 2 - 1
            grid_y_norm = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, self.kernel_size, H, W).float() / (H - 1) * 2 - 1

            # 添加偏移 (只在 x 方向偏移)
            grid_x_norm = grid_x_norm + offset / W * 2  # (B, K, H, W)
            grid_y_norm = grid_y_norm + base_grid_y * (self.kernel_size / H)

        # 3. 采样
        grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1)  # (B, K, H, W, 2)

        # 对每个采样点进行双线性插值
        sampled_features = []
        for i in range(self.kernel_size):
            sampled = F.grid_sample(
                x, grid[:, i, :, :, :],
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # (B, C, H, W)
            sampled_features.append(sampled)

        # 4. 加权求和 (depthwise)
        output = torch.zeros_like(x)
        for i in range(self.kernel_size):
            if self.direction == 'horizontal':
                weight = self.weight[:, :, :, i:i+1].squeeze(3).squeeze(1).view(C, 1, 1)  # (C, 1, 1)
            else:
                weight = self.weight[:, :, i:i+1, :].squeeze(2).squeeze(1).view(C, 1, 1)  # (C, 1, 1)
            output = output + sampled_features[i] * weight

        # 5. 归一化和激活
        output = self.norm(output)
        output = self.act(output)

        return output


# ==================== TopoWaveNet 版 DynamicSnakeConv (零初始化, 更高效) ====================

class DynamicSnakeConvV2(nn.Module):
    """
    Dynamic snake-like depthwise conv along one axis with learned offsets.

    改进自 TopoWaveNet:
    - offset_conv 零初始化: 训练初期等价于标准 strip conv, 避免冷启动不稳定
    - 向量化加权求和: stack → broadcast multiply → sum, 比逐点 for 循环更高效
    - 更简洁的坐标计算
    """

    def __init__(
        self,
        channels: int,
        k: int = 7,
        axis: str = 'h',
        offset_scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        if axis not in ('h', 'v'):
            raise ValueError(f"axis must be 'h' or 'v', got {axis}")
        if k % 2 != 1:
            raise ValueError("k should be odd for symmetric sampling.")
        self.k = k
        self.axis = axis
        self.offset_scale = float(offset_scale)

        # 偏移量预测: 3×3 conv, 零初始化 → 初始行为等价于普通 strip conv
        self.offset_conv = nn.Conv2d(channels, k, kernel_size=3, padding=1, bias=True)

        # Depthwise 权重 (沿蛇形路径的 k 个采样点)
        self.weight = nn.Parameter(torch.empty(channels, k))
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def _base_grid(self, x: torch.Tensor):
        b, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij',
        )
        return xx.unsqueeze(0).expand(b, -1, -1), yy.unsqueeze(0).expand(b, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        offset = torch.tanh(self.offset_conv(x)) * self.offset_scale  # (B, K, H, W)

        xx, yy = self._base_grid(x)
        r = self.k // 2

        samples = []
        for i in range(self.k):
            if self.axis == 'h':
                x_i = xx + (i - r)                   # 沿水平方向均匀采样
                y_i = yy + offset[:, i, :, :]         # 垂直方向学习偏移
            else:
                x_i = xx + offset[:, i, :, :]          # 水平方向学习偏移
                y_i = yy + (i - r)                    # 沿垂直方向均匀采样

            x_norm = (x_i / max(w - 1, 1)) * 2.0 - 1.0
            y_norm = (y_i / max(h - 1, 1)) * 2.0 - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1)

            sample = F.grid_sample(
                x, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
            )
            samples.append(sample)

        # 向量化加权求和
        feats = torch.stack(samples, dim=-1)                    # (B, C, H, W, K)
        weight = self.weight.view(1, c, 1, 1, self.k)           # (1, C, 1, 1, K)
        out = (feats * weight).sum(dim=-1)                       # (B, C, H, W)
        if self.bias is not None:
            out = out + self.bias.view(1, c, 1, 1)
        return out


class DynamicSnakeConvNormAct(nn.Module):
    """DynamicSnakeConvV2 + GroupNorm + GELU"""

    def __init__(self, channels: int, k: int = 7, axis: str = 'h',
                 offset_scale: float = 2.0):
        super().__init__()
        self.conv = DynamicSnakeConvV2(
            channels=channels, k=k, axis=axis,
            offset_scale=offset_scale, bias=False,
        )
        self.norm = make_group_norm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CrackOrientedBlock(nn.Module):
    """
    裂缝导向块 (CO-Block)
    4 个分支 + 动态注意力融合 + 残差
    - 3×3, 5×5 正方形 depthwise conv (各向同性局部特征)
    - 1×k 水平 strip conv (水平方向裂缝特征)
    - k×1 垂直 strip conv (垂直方向裂缝特征)
    - DynamicAttentionFusion 动态加权
    - 1×1 融合 + 残差
    """
    def __init__(self, channels, k=7, **kwargs):
        super().__init__()
        # 深度可分离卷积: depthwise + pointwise (1×1) 跨通道交互
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_group_norm(channels),
            nn.GELU(),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_group_norm(channels),
            nn.GELU(),
        )
        self.branch_h = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, k), padding=(0, k // 2), groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_group_norm(channels),
            nn.GELU(),
        )
        self.branch_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(k, 1), padding=(k // 2, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_group_norm(channels),
            nn.GELU(),
        )

        self.attn = DynamicAttentionFusion(channels, num_branches=4)
        self.fusion = ConvGNAct(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        f1 = self.branch_3x3(x)
        f2 = self.branch_5x5(x)
        f3 = self.branch_h(x)
        f4 = self.branch_v(x)
        out = self.attn([f1, f2, f3, f4])
        out = self.fusion(out)
        return identity + out
