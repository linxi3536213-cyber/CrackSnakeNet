"""
Crack segmentation loss: focal + dice + centerline recall.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CenterlineRecallLoss(nn.Module):
    """
    中心线召回损失：迫使模型在 GT 中心线像素上预测为正。

    recall = sum(centerline * sigmoid(pred)) / sum(centerline)
    loss = 1 - recall

    相比 clDice：无 soft_skel 计算，梯度简单稳定，直接提升细裂缝召回率。
    """

    def __init__(self, from_logits: bool = True):
        super().__init__()
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, gt_centerline: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred

        recall = (gt_centerline * pred_prob).sum() / (gt_centerline.sum() + 1e-6)
        return 1.0 - recall


class BoundaryConstraintLoss(nn.Module):
    """
    边界约束损失：惩罚预测超出 GT mask 的 false positive。

    与 CenterlineRecallLoss 互补：
    - CenterlineRecallLoss: 拉宽预测，覆盖中心线 (提升 Tsens)
    - BoundaryConstraintLoss: 收窄预测，不超出 GT 边界 (提升 Tprec)

    loss = mean(sigmoid(pred) * (1 - gt_mask))
    即：GT 为 0（背景）的区域，预测应尽量为 0。
    """

    def __init__(self, from_logits: bool = True):
        super().__init__()
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred
        false_positive = pred_prob * (1.0 - gt_mask)
        return false_positive.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for crack segmentation.
    Supported loss types:
    - focal
    - dice
    """

    SUPPORTED = {"focal", "dice"}

    def __init__(
        self,
        loss_types: Optional[List[str]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.loss_types = loss_types or ["focal", "dice"]
        unsupported = [name for name in self.loss_types if name not in self.SUPPORTED]
        if unsupported:
            raise ValueError(
                f"Unsupported loss_types: {unsupported}. "
                f"Supported types are: {sorted(self.SUPPORTED)}"
            )

        self.loss_weights = loss_weights or {name: 1.0 for name in self.loss_types}
        self.losses = nn.ModuleDict()

        if "focal" in self.loss_types:
            self.losses["focal"] = smp.losses.FocalLoss(
                mode="binary",
                alpha=focal_alpha,
                gamma=focal_gamma,
            )
        if "dice" in self.loss_types:
            self.losses["dice"] = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        total_loss = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
        loss_dict: Dict[str, torch.Tensor] = {}

        for loss_name in self.loss_types:
            loss_value = self.losses[loss_name](inputs, targets)
            loss_dict[loss_name] = loss_value
            total_loss = total_loss + self.loss_weights.get(loss_name, 1.0) * loss_value

        loss_dict["total_loss"] = total_loss
        return loss_dict


def create_crack_detection_loss(loss_config: Dict, device: Optional[torch.device] = None) -> nn.Module:
    loss_fn = CombinedLoss(
        loss_types=loss_config.get("loss_types", ["focal", "dice"]),
        loss_weights=loss_config.get("loss_weights", {"focal": 1.0, "dice": 1.0}),
        focal_alpha=loss_config.get("focal_alpha", 0.25),
        focal_gamma=loss_config.get("focal_gamma", 2.0),
    )

    if device is not None:
        loss_fn = loss_fn.to(device)

    return loss_fn
