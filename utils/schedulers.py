"""
学习率调度器模块
实现 Warmup + Cosine 学习率策略
"""
import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    """
    Warmup + Cosine Annealing Learning Rate Scheduler

    学习率分两个阶段：
    1. Warmup阶段（前N个epoch）：线性增长从0到base_lr
    2. Cosine阶段（剩余epoch）：余弦退火衰减

    Args:
        optimizer: 优化器
        max_epochs: 总训练轮数
        warmup_epochs: warmup轮数（默认5）
        min_lr: 最小学习率（默认1e-6）
        last_epoch: 上次epoch编号
    """

    def __init__(self, optimizer, max_epochs, warmup_epochs=5,
                 min_lr=1e-6, last_epoch=-1):
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算当前epoch的学习率 - 优化版本，避免重复创建列表"""
        epoch = self.last_epoch

        # 使用缓存列表，避免每次都创建新的列表对象
        if not hasattr(self, '_lr_cache'):
            self._lr_cache = [0.0] * len(self.base_lrs)

        # Warmup阶段：线性增长
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for i, base_lr in enumerate(self.base_lrs):
                self._lr_cache[i] = base_lr * warmup_factor

        # Cosine阶段：余弦退火
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            for i, base_lr in enumerate(self.base_lrs):
                self._lr_cache[i] = self.min_lr + (base_lr - self.min_lr) * cosine_factor

        return self._lr_cache


def create_scheduler(optimizer, scheduler_type, max_epochs, warmup_epochs=5, min_lr=1e-6):
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 (当前仅支持 'cosine')
        max_epochs: 总训练轮数
        warmup_epochs: warmup轮数
        min_lr: 最小学习率

    Returns:
        scheduler: 学习率调度器
    """

    if scheduler_type == 'cosine':
        scheduler = WarmupCosineLR(
            optimizer,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr
        )
        print(f"✓ 使用 Warmup + Cosine LR (warmup={warmup_epochs})")
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}，当前仅支持 'cosine'")

    return scheduler
