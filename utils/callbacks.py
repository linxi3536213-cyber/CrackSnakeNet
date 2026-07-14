"""自定义回调函数"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class VerboseEarlyStopping(EarlyStopping):
    """
    扩展的EarlyStopping回调，显示当前patience计数

    在每个epoch结束时，会打印当前累积的patience值，
    让用户了解距离early stopping还有多远。
    """

    def _run_early_stopping_check(self, trainer: pl.Trainer) -> None:
        """
        重写early stopping检查方法，添加patience计数打印
        """
        # 调用父类的检查方法
        super()._run_early_stopping_check(trainer)

        # 打印当前patience累积值
        # wait_count: 当前累积的未改进epoch数
        # patience: 总的patience值（配置中设定的）
        current_wait = self.wait_count
        total_patience = self.patience

        if current_wait > 0:
            print(f"  ⏳ Early Stopping - 当前patience: {current_wait}/{total_patience} "
                  f"(还需 {total_patience - current_wait} 个epoch无改进才会停止)")
        else:
            print(f"  ✓ 指标改进! patience重置为 0/{total_patience}")


class EMACallback(pl.Callback):
    """
    Exponential Moving Average (EMA) 回调

    在训练过程中维护模型参数的指数移动平均值，
    验证时使用EMA权重以获得更稳定的评估结果。

    EMA通常能带来0.5-2%的免费性能提升。

    Args:
        decay: EMA衰减率，越接近1越平滑（推荐0.999-0.9999）
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_state = None
        self.original_state = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """训练开始时，初始化EMA状态为当前模型参数"""
        self.ema_state = {
            k: v.clone().detach()
            for k, v in pl_module.model.state_dict().items()
        }

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs, batch, batch_idx
    ) -> None:
        """每个训练batch结束后，更新EMA参数"""
        with torch.no_grad():
            model_state = pl_module.model.state_dict()
            for k in self.ema_state:
                if model_state[k].is_floating_point():
                    self.ema_state[k].mul_(self.decay).add_(
                        model_state[k], alpha=1.0 - self.decay
                    )
                else:
                    # 非浮点类型（如int buffer）直接复制
                    self.ema_state[k].copy_(model_state[k])

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """验证开始前，将模型权重替换为EMA权重"""
        if self.ema_state is None:
            return
        # 保存原始权重
        self.original_state = {
            k: v.clone().detach()
            for k, v in pl_module.model.state_dict().items()
        }
        # 加载EMA权重用于验证
        pl_module.model.load_state_dict(self.ema_state)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """验证结束后，恢复原始训练权重"""
        if self.original_state is None:
            return
        pl_module.model.load_state_dict(self.original_state)
        self.original_state = None

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> None:
        """保存checkpoint时，同时保存EMA状态"""
        if self.ema_state is not None:
            checkpoint['ema_state'] = self.ema_state

    def on_load_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> None:
        """加载checkpoint时，恢复EMA状态"""
        if 'ema_state' in checkpoint:
            self.ema_state = checkpoint['ema_state']
