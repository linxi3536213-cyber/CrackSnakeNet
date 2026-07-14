import logging
import os
import sys
import csv
from datetime import datetime
from typing import Optional, Dict, Any
from pytorch_lightning.callbacks import Callback


class Logger:
    """统一的日志管理类"""
    
    def __init__(self, name: str = "CrackMambaNet", log_dir: Optional[str] = None, 
                 log_level: str = "INFO", save_to_file: bool = True):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_dir: 日志文件保存目录
            log_level: 日志级别
            save_to_file: 是否保存到文件
        """
        self.name = name
        self.log_dir = log_dir or "logs"
        self.save_to_file = save_to_file
        
        # 创建日志目录
        if self.save_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        if self.save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, f"{name}_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """记录调试信息"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """记录一般信息"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """记录警告信息"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """记录错误信息"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """格式化日志消息"""
        if kwargs:
            extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {extra_info}"
        return message
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "指标"):
        """记录指标信息"""
        metrics_str = ", ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
            for k, v in metrics.items()
        ])
        self.info(f"{prefix} - {metrics_str}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录训练轮次开始"""
        self.info(f"轮次 [{epoch}/{total_epochs}] 开始")
    
    def log_epoch_end(self, epoch: int, train_metrics: Dict[str, float], 
                     val_metrics: Optional[Dict[str, float]] = None):
        """记录训练轮次结束"""
        self.info(f"轮次 [{epoch}] 完成")
        self.log_metrics(train_metrics, "训练")
        if val_metrics:
            self.log_metrics(val_metrics, "验证")


def create_logger(name: str = "CrackMambaNet", log_dir: Optional[str] = None, 
                 log_level: str = "INFO", save_to_file: bool = True) -> Logger:
    """创建日志器的工厂函数"""
    return Logger(name, log_dir, log_level, save_to_file)


class MetricsCSVLogger(Callback):
    """将训练过程中的所有指标保存为CSV文件

    PyTorch Lightning执行顺序（根据官方文档，注意validation在train之前结束）：
    1. on_train_epoch_start (Callback & LightningModule, epoch=N)
    2. training_step (多次)
    3. validation_step (多次) ← validation在training epoch内部执行
    4. on_validation_epoch_end (LightningModule) ← val指标被compute和log
    5. on_train_epoch_end (LightningModule) ← train指标被compute和log
    6. on_validation_end (Callback) ← 收集所有指标并写入CSV
    7. lr_scheduler.step() ← 学习率更新为epoch N+1

    关键点：
    - on_validation_end时，train和val指标都已经通过log()保存到callback_metrics
    - on_validation_end时，lr还是当前epoch使用的（scheduler.step()在之后）
    - 在on_train_epoch_start记录epoch和lr，此时lr是当前epoch将要使用的lr

    数据收集策略：
    - 在on_train_epoch_start记录epoch和lr（当前epoch使用的lr）
    - 在on_validation_end收集train和val指标并写入CSV
    """

    def __init__(self, save_dir='results', filename='metrics.csv'):
        super().__init__()
        self.save_dir = save_dir
        self.filename = filename
        self.current_row = {}  # 当前epoch的数据行（字典）
        self.metrics_history = []  # 历史记录

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, filename)

    def on_train_epoch_start(self, trainer, pl_module):
        """步骤1：在新 epoch 开始时，写入上一个 epoch 的完整数据

        关键理解（修复后的逻辑）：
        - 在 epoch N 开始时，callback_metrics 中有 epoch N-1 的 train 数据
        - self.current_row 中已经有 epoch N-1 的 val 数据（在上一个 epoch 的 on_validation_end 收集）
        - 现在收集 epoch N-1 的 train 数据，合并后写入 CSV
        - 然后开始准备 epoch N 的数据
        """
        # 如果是 epoch > 0，先写入上一个 epoch 的数据
        if trainer.current_epoch > 0:
            # 从 callback_metrics 收集上一个 epoch 的 train 数据
            train_collected = []
            for key, value in trainer.callback_metrics.items():
                if not key.startswith('train_'):
                    continue
                if key.endswith('_step') or key.endswith('_epoch'):
                    continue

                try:
                    scalar_value = float(value.item()) if hasattr(value, 'item') else float(value)
                    if not (scalar_value != scalar_value or abs(scalar_value) == float('inf')):
                        self.current_row[key] = scalar_value
                        train_collected.append(key)
                except (AttributeError, ValueError, TypeError):
                    continue

            # 检查数据完整性并写入
            has_epoch = 'epoch' in self.current_row
            has_lr = 'lr' in self.current_row
            has_train_data = len(train_collected) > 0
            has_val_data = any(k.startswith('val_') for k in self.current_row.keys())

            if has_epoch and has_lr and has_train_data and has_val_data:
                self.metrics_history.append(self.current_row.copy())
                self._write_csv()

        # 开始准备当前 epoch 的数据
        self.current_row = {}
        self.current_row['epoch'] = trainer.current_epoch

        # 记录当前 epoch 的学习率
        if trainer.optimizers:
            self.current_row['lr'] = trainer.optimizers[0].param_groups[0]['lr']

    def on_validation_end(self, trainer, pl_module):
        """步骤2：验证结束后，只收集 val 指标（不写入CSV）

        关键理解：
        - Callback.on_validation_end 执行时，Module.on_train_epoch_end 还未执行
        - 因此 callback_metrics 中还没有当前 epoch 的 train 数据
        - 我们只收集 val 数据，train 数据要等到下一个 epoch 开始时才能收集

        执行顺序：
        1. Module.on_validation_epoch_end ← val指标已compute
        2. Callback.on_validation_end ← 此时！只有 val 数据可用
        3. Module.on_train_epoch_end ← 还未执行，train 数据还未进入 callback_metrics
        """
        # 只收集 val 指标（不收集 train，因为它们还不在 callback_metrics 中）
        for key, value in trainer.callback_metrics.items():
            # 只收集 val_ 开头的指标
            if not key.startswith('val_'):
                continue
            # 跳过step级别的指标
            if key.endswith('_step') or key.endswith('_epoch'):
                continue

            try:
                scalar_value = float(value.item()) if hasattr(value, 'item') else float(value)
                # 跳过NaN和Inf
                if not (scalar_value != scalar_value or abs(scalar_value) == float('inf')):
                    self.current_row[key] = scalar_value
            except (AttributeError, ValueError, TypeError):
                continue

    def _write_csv(self):
        """将指标写入CSV文件（每次写入一行，立即追加）"""
        if not self.metrics_history:
            return

        # 收集当前所有记录的字段
        current_keys = set()
        for record in self.metrics_history:
            current_keys.update(record.keys())

        # 定义字段排序规则
        def get_sorted_fieldnames(keys_set):
            """
            返回排序后的字段名列表：
            1. epoch, lr
            2. train损失（组件损失 → 总损失）
            3. val损失（组件损失 → 总损失）
            4. train指标（按字母排序）
            5. val指标（按字母排序）
            """
            def sort_loss_keys(loss_keys, prefix):
                """排序损失：组件损失(字母序) → 总损失"""
                component_losses = []
                total_loss = None

                for key in loss_keys:
                    if key == f'{prefix}_loss':
                        total_loss = key
                    else:
                        component_losses.append(key)

                result = sorted(component_losses)
                if total_loss:
                    result.append(total_loss)
                return result

            priority_keys = ['epoch', 'lr']
            train_losses = [k for k in keys_set if k.startswith('train_') and k.endswith('_loss')]
            val_losses = [k for k in keys_set if k.startswith('val_') and k.endswith('_loss')]
            train_metrics = [k for k in keys_set if k.startswith('train_') and not k.endswith('_loss')]
            val_metrics = [k for k in keys_set if k.startswith('val_') and not k.endswith('_loss')]

            # 组合顺序
            fieldnames = [k for k in priority_keys if k in keys_set]
            fieldnames += sort_loss_keys(train_losses, 'train')
            fieldnames += sort_loss_keys(val_losses, 'val')
            fieldnames += sorted(train_metrics)
            fieldnames += sorted(val_metrics)
            return fieldnames

        # 判断是否是首次写入
        file_exists = os.path.exists(self.save_path)

        # 获取字段名并写入
        if file_exists:
            # 文件已存在，读取表头
            try:
                with open(self.save_path, 'r', newline='', encoding='utf-8') as rf:
                    reader = csv.reader(rf)
                    existing_fieldnames = next(reader, [])
                    existing_data = list(reader)

                existing_set = set(existing_fieldnames)
                new_keys = current_keys - existing_set

                if new_keys:
                    # 有新字段，重新排序并重写整个文件
                    all_keys = existing_set | current_keys
                    fieldnames = get_sorted_fieldnames(all_keys)

                    with open(self.save_path, 'w', newline='', encoding='utf-8') as wf:
                        writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction='ignore')
                        writer.writeheader()
                        # 写入旧数据
                        for row_data in existing_data:
                            row_dict = dict(zip(existing_fieldnames, row_data))
                            writer.writerow(row_dict)
                        # 写入新数据
                        writer.writerows(self.metrics_history)
                        wf.flush()
                else:
                    # 没有新字段，直接追加
                    with open(self.save_path, 'a', newline='', encoding='utf-8') as af:
                        writer = csv.DictWriter(af, fieldnames=existing_fieldnames, extrasaction='ignore')
                        writer.writerows(self.metrics_history)
                        af.flush()
            except Exception as e:
                print(f"写入CSV失败: {e}")
        else:
            # 首次写入
            fieldnames = get_sorted_fieldnames(current_keys)
            try:
                with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(self.metrics_history)
                    f.flush()
            except Exception as e:
                print(f"写入CSV失败: {e}")

        # 写入后清空历史记录
        self.metrics_history.clear()

    def on_train_end(self, trainer, pl_module):
        """训练结束时，写入最后一个 epoch 的数据

        因为最后一个 epoch 没有"下一个 epoch"来触发写入，
        所以需要在这里手动收集 train 数据并写入。
        """
        # 检查是否有未写入的数据
        if not self.current_row:
            return

        # 收集最后一个 epoch 的 train 数据
        for key, value in trainer.callback_metrics.items():
            if not key.startswith('train_'):
                continue
            if key.endswith('_step') or key.endswith('_epoch'):
                continue

            try:
                scalar_value = float(value.item()) if hasattr(value, 'item') else float(value)
                if not (scalar_value != scalar_value or abs(scalar_value) == float('inf')):
                    self.current_row[key] = scalar_value
            except (AttributeError, ValueError, TypeError):
                continue

        # 检查数据完整性并写入
        has_epoch = 'epoch' in self.current_row
        has_lr = 'lr' in self.current_row
        has_train_data = any(k.startswith('train_') for k in self.current_row.keys())
        has_val_data = any(k.startswith('val_') for k in self.current_row.keys())

        if has_epoch and has_lr and has_train_data and has_val_data:
            self.metrics_history.append(self.current_row.copy())
            self._write_csv()

        print(f"训练指标已保存到: {self.save_path}")
