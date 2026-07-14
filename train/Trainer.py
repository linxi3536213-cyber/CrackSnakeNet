import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from typing import Any, Dict
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex
)


class BinaryMeanIoU(Metric):
    """Per-image mIoU matching SCSegamba: mean of per-image (IoU_pos + IoU_neg) / 2."""
    full_state_update = False

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("iou_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pred_bin = (preds >= self.threshold).view(preds.shape[0], -1)
        target_bin = target.view(target.shape[0], -1).bool()

        tp = (pred_bin & target_bin).sum(dim=1).float()
        fp = (pred_bin & ~target_bin).sum(dim=1).float()
        tn = (~pred_bin & ~target_bin).sum(dim=1).float()
        fn = (~pred_bin & target_bin).sum(dim=1).float()

        denom_pos = tp + fp + fn
        denom_neg = tn + fp + fn
        iou_pos = torch.where(denom_pos > 0, tp / denom_pos, torch.zeros_like(tp))
        iou_neg = torch.where(denom_neg > 0, tn / denom_neg, torch.zeros_like(tn))

        per_image_miou = (iou_pos + iou_neg) / 2.0
        self.iou_sum += per_image_miou.sum()
        self.count += preds.shape[0]

    def compute(self):
        return self.iou_sum / self.count.float().clamp(min=1)

from utils.Losses import create_crack_detection_loss, CenterlineRecallLoss, BoundaryConstraintLoss
from utils.schedulers import create_scheduler
from data.transform import CutMix


class LitCrackModel(pl.LightningModule):
    """
    裂缝检测模型的PyTorch Lightning封装

    按照Lightning官方执行顺序组织：
    1. __init__: 初始化模型、损失函数、指标
    2. forward: 前向传播
    3. configure_optimizers: 配置优化器和学习率调度器
    4. training_step/validation_step/test_step: 执行一个batch
    5. on_*_epoch_end: epoch结束时的处理
    """

    def __init__(self, model: nn.Module, loss_config: dict, lr: float, weight_decay: float,
                 max_epochs: int = 100, warmup_epochs: int = 5, min_lr: float = 1e-6,
                 edge_loss_weight: float = 0.0,
                 edge_aux_loss_weight: float = 0.0,
                 edge_threshold: float = 0.1,
                 metric_threshold: float = 0.5,
                 use_cutmix: bool = True,
                 cutmix_alpha: float = 1.0,
                 cutmix_prob: float = 0.3,
                 centerline_loss_weight: float = 0.0,
                 boundary_loss_weight: float = 0.0):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])

        # 缓存关键超参，避免在类型检查时出现属性提示问题
        self.base_lr = lr
        self.base_weight_decay = weight_decay
        self.max_epochs_cfg = max_epochs
        self.warmup_epochs_cfg = warmup_epochs
        self.min_lr_cfg = min_lr
        self.edge_loss_weight = edge_loss_weight
        self.edge_aux_loss_weight = edge_aux_loss_weight
        self.edge_threshold = edge_threshold
        self.metric_threshold = metric_threshold

        # CutMix数据增强（提升边界检测能力）
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.cutmix = CutMix(alpha=cutmix_alpha) if use_cutmix else None

        # 中心线召回损失：迫使模型覆盖细裂缝中心线 (提升 Tsens)
        self.centerline_loss_weight = centerline_loss_weight
        self.cl_recall_loss = CenterlineRecallLoss(from_logits=True)

        # 边界约束损失：惩罚 GT 外的 false positive (提升 Tprec)
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_loss = BoundaryConstraintLoss(from_logits=True)

        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )

        # 损失函数
        self.criterion = create_crack_detection_loss(loss_config)

        # 指标集合（train和val使用独立的metrics）
        # 注意：对于二分类分割，Dice = F1
        metrics = MetricCollection({
            'accuracy': BinaryAccuracy(threshold=self.metric_threshold),
            'precision': BinaryPrecision(threshold=self.metric_threshold),
            'recall': BinaryRecall(threshold=self.metric_threshold),
            'f1': BinaryF1Score(threshold=self.metric_threshold),
            'iou': BinaryJaccardIndex(threshold=self.metric_threshold),
            'miou': BinaryMeanIoU(threshold=self.metric_threshold),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore[override]
        """配置优化器和学习率调度器 - Cosine Annealing with Warmup"""
        if hasattr(self.model, 'get_params_groups') and callable(getattr(self.model, 'get_params_groups', None)):
            # 模型支持参数分组（如差分学习率）
            param_groups = self.model.get_params_groups(  # type: ignore
                lr=self.base_lr,
                weight_decay=self.base_weight_decay
            )
            optimizer = optim.AdamW(
                param_groups,
                lr=self.base_lr,
                weight_decay=0.0  # 已在param_groups中指定
            )
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.base_lr,
                weight_decay=self.base_weight_decay
            )

        # Cosine Annealing 调度器
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='cosine',
            max_epochs=self.max_epochs_cfg,
            warmup_epochs=self.warmup_epochs_cfg,
            min_lr=self.min_lr_cfg
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # ==================== 共享函数（减少重复代码） ====================
    def _parse_model_output(self, output):
        """
        解析模型输出

        Returns:
            main_logits: 主输出logits
            aux_outputs: 辅助输出字典（可为空）
            edge_aux_outputs: 解码器各阶段边缘输出（可为空）
            edge_logits: 边界输出（可为空）
        """
        aux_outputs = {}
        edge_aux_outputs = {}
        edge_logits = None
        if isinstance(output, dict) and 'main_output' in output:
            main_logits = output['main_output']
            aux_outputs = output.get('aux_outputs', {})
            edge_aux_outputs = output.get('edge_aux_outputs', {})
            edge_logits = output.get('edge_output', None)
        elif isinstance(output, (tuple, list)):
            main_logits = output[0]
            if len(output) > 1:
                aux_outputs = {idx: out for idx, out in enumerate(output[1:])}
        else:
            main_logits = output
        return main_logits, aux_outputs, edge_aux_outputs, edge_logits

    def _compute_edge_targets(self, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            targets = targets.float()
            gx = torch.nn.functional.conv2d(targets, self.sobel_x, padding=1)
            gy = torch.nn.functional.conv2d(targets, self.sobel_y, padding=1)
            grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
            return (grad > self.edge_threshold).float()

    def _compute_loss(self, main_logits, targets, stage='train',
                      aux_outputs=None, edge_aux_outputs=None, edge_logits=None,
                      centerlines=None):
        """
        计算损失

        Args:
            main_logits: 主输出logits
            targets: 目标标签
            stage: 'train', 'val', 或 'test'
            aux_outputs: 辅助输出字典（可为空）

        Returns:
            loss: 损失值
        """
        # 计算损失
        loss_dict = self.criterion(main_logits, targets)
        loss = loss_dict.pop('total_loss')

        # 记录各个损失组件（dice, focal等）
        for loss_name, loss_value in loss_dict.items():
            self.log(f'{stage}_{loss_name}_loss', loss_value,
                    on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 辅助输出损失（深监督）
        aux_outputs = aux_outputs or {}
        if aux_outputs:
            aux_losses = []
            for stage_idx, aux_logits in sorted(aux_outputs.items()):
                aux_loss = self.criterion(aux_logits, targets)['total_loss']
                aux_losses.append(aux_loss)
                self.log(f'{stage}_aux{stage_idx}_loss', aux_loss,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            aux_mean = torch.stack(aux_losses).mean()
            # Total = Main + 0.1 * mean(Aux) + edge loss
            loss = loss + 0.1 * aux_mean
            self.log(f'{stage}_aux_mean_loss', aux_mean,
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        edge_targets = None

        # 最终边界损失
        if edge_logits is not None and self.edge_loss_weight > 0:
            if edge_logits.shape[-2:] != targets.shape[-2:]:
                edge_logits = torch.nn.functional.interpolate(
                    edge_logits, size=targets.shape[-2:], mode='bilinear', align_corners=False
                )
            edge_targets = self._compute_edge_targets(targets)
            edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_logits, edge_targets)
            loss = loss + self.edge_loss_weight * edge_loss
            self.log(f'{stage}_edge_loss', edge_loss,
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 解码器多尺度边缘辅助监督
        edge_aux_outputs = edge_aux_outputs or {}
        if edge_aux_outputs and self.edge_aux_loss_weight > 0:
            if edge_targets is None:
                edge_targets = self._compute_edge_targets(targets)
            edge_aux_losses = []
            for stage_idx, edge_aux_logits in sorted(edge_aux_outputs.items()):
                if edge_aux_logits.shape[-2:] != targets.shape[-2:]:
                    edge_aux_logits = torch.nn.functional.interpolate(
                        edge_aux_logits, size=targets.shape[-2:], mode='bilinear', align_corners=False
                    )
                edge_aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_aux_logits, edge_targets)
                edge_aux_losses.append(edge_aux_loss)
                self.log(f'{stage}_edge_aux{stage_idx}_loss', edge_aux_loss,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            edge_aux_mean = torch.stack(edge_aux_losses).mean()
            loss = loss + self.edge_aux_loss_weight * edge_aux_mean
            self.log(f'{stage}_edge_aux_mean_loss', edge_aux_mean,
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 中心线召回损失：迫使模型覆盖 GT 中心线像素
        if centerlines is not None and self.centerline_loss_weight > 0:
            cl_mask = (centerlines > 0.5).float()
            if cl_mask.sum() > 0:
                if cl_mask.shape[-2:] != main_logits.shape[-2:]:
                    cl_mask = torch.nn.functional.interpolate(
                        cl_mask, size=main_logits.shape[-2:], mode='nearest'
                    )
                cl_loss = self.cl_recall_loss(main_logits, cl_mask)
                loss = loss + self.centerline_loss_weight * cl_loss
                self.log(f'{stage}_cl_recall_loss', cl_loss,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 边界约束损失：惩罚预测超出 GT mask 的 false positive (提升 Tprec)
        if self.boundary_loss_weight > 0:
            bd_loss = self.boundary_loss(main_logits, targets)
            loss = loss + self.boundary_loss_weight * bd_loss
            self.log(f'{stage}_boundary_loss', bd_loss,
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 记录总损失（包含辅助损失）
        self.log(f'{stage}_loss', loss,
                 on_step=False, on_epoch=True,
                 prog_bar=(stage == 'train'), sync_dist=True)

        return loss

    def _update_metrics(self, main_logits, targets, metrics):
        """
        更新指标

        Args:
            main_logits: 主输出logits
            targets: 目标标签
            metrics: 指标集合（train_metrics或val_metrics）
        """
        with torch.no_grad():
            preds_prob = torch.sigmoid(main_logits)
            metrics.update(preds_prob, targets.int())

    def _compute_and_log_metrics(self, metrics, stage='train'):
        """
        计算并记录累积指标

        Args:
            metrics: 指标集合（train_metrics或val_metrics）
            stage: 'train' 或 'val'
        """
        # Compute accumulated metrics
        metrics_computed = metrics.compute()

        # 添加dice（等于f1）
        metrics_computed[f'{stage}_dice'] = metrics_computed[f'{stage}_f1']

        # Log all metrics
        for key, value in metrics_computed.items():
            if key in (f'{stage}_iou', f'{stage}_miou'):
                # IoU和mIoU显示在进度条
                self.log(key, value, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
            else:
                self.log(key, value, on_epoch=True, sync_dist=True, prog_bar=False, logger=True)

        # Reset metrics for next epoch
        metrics.reset()

    # ==================== 训练阶段 ====================
    # 注意：lr 由 LearningRateMonitor callback 自动记录，无需手动记录

    def training_step(self, batch, batch_idx):
        """训练步骤（简化版本，使用共享函数）"""
        imgs, targets, centerlines = batch

        # CutMix数据增强（以一定概率应用）
        if self.training and self.use_cutmix and self.cutmix is not None:
            if torch.rand(1).item() < self.cutmix_prob:
                imgs, targets = self.cutmix(imgs, targets)

        output = self(imgs)

        # 解析模型输出
        main_logits, aux_outputs, edge_aux_outputs, edge_logits = self._parse_model_output(output)

        # 计算损失
        loss = self._compute_loss(
            main_logits,
            targets,
            stage='train',
            aux_outputs=aux_outputs,
            edge_aux_outputs=edge_aux_outputs,
            edge_logits=edge_logits,
            centerlines=centerlines,
        )

        # 更新指标
        self._update_metrics(main_logits, targets, self.train_metrics)

        return loss

    def on_train_epoch_end(self):
        """训练epoch结束时compute并记录累积指标"""
        # 计算并记录指标
        self._compute_and_log_metrics(self.train_metrics, stage='train')

        # GPU资源清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ==================== TTA ====================
    def _tta_forward(self, imgs):
        """
        TTA (Test Time Augmentation): 4 次前传取平均

        原图 + 水平翻转 + 垂直翻转 + 水平+垂直翻转
        对 logits 取平均后返回
        """
        # 1. 原图
        out0 = self._parse_model_output(self(imgs))[0]
        logits_sum = out0

        # 2. 水平翻转
        imgs_hflip = torch.flip(imgs, dims=[3])
        out1 = self._parse_model_output(self(imgs_hflip))[0]
        logits_sum = logits_sum + torch.flip(out1, dims=[3])

        # 3. 垂直翻转
        imgs_vflip = torch.flip(imgs, dims=[2])
        out2 = self._parse_model_output(self(imgs_vflip))[0]
        logits_sum = logits_sum + torch.flip(out2, dims=[2])

        # 4. 水平+垂直翻转
        imgs_hvflip = torch.flip(imgs, dims=[2, 3])
        out3 = self._parse_model_output(self(imgs_hvflip))[0]
        logits_sum = logits_sum + torch.flip(out3, dims=[2, 3])

        return logits_sum / 4.0

    # ==================== 验证阶段 ====================
    def validation_step(self, batch, batch_idx):
        """验证步骤: TTA 增强预测 + 原始预测计算损失"""
        imgs, targets, centerlines = batch
        output = self(imgs)

        # 解析模型输出（用于计算损失）
        main_logits, aux_outputs, edge_aux_outputs, edge_logits = self._parse_model_output(output)

        # 计算损失（不用 TTA，保持损失稳定）
        loss = self._compute_loss(
            main_logits,
            targets,
            stage='val',
            aux_outputs=aux_outputs,
            edge_aux_outputs=edge_aux_outputs,
            edge_logits=edge_logits,
            centerlines=centerlines,
        )

        # TTA 增强预测用于指标计算（val_iou 等）
        tta_logits = self._tta_forward(imgs)
        self._update_metrics(tta_logits, targets, self.val_metrics)

        return loss

    def on_validation_epoch_end(self):
        """验证epoch结束时compute并记录累积指标"""
        # 计算并记录指标
        self._compute_and_log_metrics(self.val_metrics, stage='val')

        # GPU资源清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ==================== 测试阶段 ====================
    def test_step(self, batch, batch_idx):
        """测试步骤（复用验证逻辑）"""
        imgs, targets, centerlines = batch
        output = self(imgs)

        # 解析模型输出
        main_logits, aux_outputs, edge_aux_outputs, edge_logits = self._parse_model_output(output)

        # 计算损失
        loss = self._compute_loss(
            main_logits,
            targets,
            stage='test',
            aux_outputs=aux_outputs,
            edge_aux_outputs=edge_aux_outputs,
            edge_logits=edge_logits,
            centerlines=centerlines,
        )

        # 更新指标（测试使用val_metrics）
        self._update_metrics(main_logits, targets, self.val_metrics)

        return loss
