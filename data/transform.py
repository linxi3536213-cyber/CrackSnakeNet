import numpy as np
import cv2
import albumentations as A
import torch
from typing import Tuple, List
from config.config import Config


def get_transforms(mode: str = "train", image_size: Tuple[int, int] = Config.IMAGE_SIZE,
                   augmentation_strength: str = "medium") -> A.Compose:
    """
    获取数据增强变换

    Args:
        mode: 'train', 'val', 'test'
        image_size: 图像大小
        augmentation_strength: 数据增强强度 ['weak', 'medium', 'strong']
    """
    # centerline 与 mask 使用相同的空间变换
    _extra_targets = {'centerline': 'mask'}

    if mode == 'train':
        if augmentation_strength == 'weak':
            # 原始的弱增强配置
            transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.3
                ),
                A.GaussNoise(p=0.1),
            ], additional_targets=_extra_targets)

        elif augmentation_strength == 'medium':
            # 🔧 中等强度增强（推荐用于减少过拟合）
            transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),

                # ========== 几何变换（增加旋转和缩放）==========
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),  # 提高到0.5
                A.RandomRotate90(p=0.5),  # 提高到0.5
                A.ShiftScaleRotate(
                    shift_limit=0.1,      # 平移±10%
                    scale_limit=0.2,      # 缩放±20%
                    rotate_limit=15,      # 旋转±15度
                    border_mode=cv2.BORDER_REFLECT,  # 边界填充
                    p=0.5
                ),

                # ========== 颜色增强（模拟不同光照条件）==========
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,   # 亮度±30%
                    contrast_limit=0.3,     # 对比度±30%
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,     # 色调偏移±10
                    sat_shift_limit=20,     # 饱和度±20
                    val_shift_limit=15,     # 明度±15
                    p=0.3
                ),

                # ========== 模糊和噪声（模拟真实场景）==========
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.3),

                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ], p=0.2),

                # ========== 其他增强 ==========
                A.CLAHE(clip_limit=2.0, p=0.2),  # 对比度受限自适应直方图均衡
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),  # Gamma校正
            ], additional_targets=_extra_targets)

        elif augmentation_strength == 'strong':
            # 🔧 强增强配置（用于严重过拟合情况）
            transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),

                # ========== 更激进的几何变换 ==========
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,     # 平移±15%
                    scale_limit=0.3,      # 缩放±30%
                    rotate_limit=30,      # 旋转±30度
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.7
                ),
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    alpha_affine=5,
                    p=0.3
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),

                # ========== 强颜色增强 ==========
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.4,
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.4
                ),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),

                # ========== 强模糊和噪声 ==========
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], p=0.5),

                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.4),

                # ========== 其他强增强 ==========
                A.CLAHE(clip_limit=3.0, p=0.3),
                A.RandomGamma(gamma_limit=(70, 130), p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3
                ),
            ], additional_targets=_extra_targets)

        else:
            raise ValueError(f"Unknown augmentation_strength: {augmentation_strength}")


    elif mode == 'val':
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
        ], additional_targets=_extra_targets)

    else:  # test
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
        ], additional_targets=_extra_targets)

    return transform


class MixUp:
    """MixUp 数据增强（batch级别）"""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W)
            masks: (B, 1, H, W)
        """
        if self.alpha <= 0:
            return images, masks

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        index = torch.randperm(batch_size, device=images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_masks = lam * masks + (1 - lam) * masks[index]
        return mixed_images, mixed_masks


class CutMix:
    """CutMix 数据增强（batch级别）"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W)
            masks: (B, 1, H, W)
        """
        if self.alpha <= 0:
            return images, masks

        batch_size, _, height, width = images.size()
        lam = np.random.beta(self.alpha, self.alpha)

        index = torch.randperm(batch_size, device=images.device)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = np.random.randint(width)
        cy = np.random.randint(height)

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        mixed_images = images.clone()
        mixed_masks = masks.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        mixed_masks[:, :, bby1:bby2, bbx1:bbx2] = masks[index, :, bby1:bby2, bbx1:bbx2]
        return mixed_images, mixed_masks
