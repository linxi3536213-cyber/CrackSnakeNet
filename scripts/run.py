#!/usr/bin/env python
import os
import sys

# 设置环境变量，避免 torchvision 导入时的 inspect 问题
os.environ['PYTORCH_JIT'] = '0'

import torch
import glob
from datetime import datetime

# 导入 pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger

# 添加项目根目录到 sys.path
# 无论从哪个目录运行，都能正确找到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import Config, get_device
from model.TriBranchCracknet import TriBranchCrackNet
from data.dataset import CrackDatasetSplitter, CrackDataLoader
from train.Trainer import LitCrackModel
from utils.logger import MetricsCSVLogger
from utils.callbacks import VerboseEarlyStopping, EMACallback


def main():
    # 设备信息
    device = get_device()
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建带时间戳的目录
    checkpoint_dir = os.path.join('checkpoints', timestamp)
    results_dir = os.path.join('results', timestamp)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"本次训练保存路径:")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Results: {results_dir}")

    # 1. 准备数据
    splitter = CrackDatasetSplitter()
    train_pairs, val_pairs = splitter.get_train_val_pairs()
    
    print(f"\n数据集:")
    print(f"  - 训练集: {len(train_pairs)} 张")
    print(f"  - 验证集: {len(val_pairs)} 张")
    
    data_loader = CrackDataLoader(train_pairs, val_pairs, Config.IMAGE_SIZE, Config.BATCH_SIZE)
    train_loader, val_loader = data_loader.get_loaders()

    # 2. 创建模型
    # 从 Config.MODEL_CONFIGS 读取参数
    model_config = Config.MODEL_CONFIGS
    stage_count = len(model_config['widths'])
    stage_blocks = Config.expand_stage_blocks(model_config['depths'], stage_count)

    # 创建模型
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

    print(f"\n模型配置: TriBranchCrackNet")
    print(f"  - Stage通道数: {model_config['widths']}")
    print(f"  - Stage模块数: {stage_blocks}")
    print(f"  - CO-Block卷积核: {model_config['k']}")
    print(f"  - Mamba状态维度: {model_config['d_state']}")

    # 3. Lightning模块
    lit_model = LitCrackModel(
        model=model,
        loss_config=Config.LOSS_CONFIG,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        max_epochs=Config.EPOCHS,
        warmup_epochs=Config.LR_WARMUP_EPOCHS,
        min_lr=Config.LR_MIN,
        edge_loss_weight=Config.EDGE_LOSS_WEIGHT,
        edge_aux_loss_weight=Config.EDGE_AUX_LOSS_WEIGHT,
        edge_threshold=Config.EDGE_THRESHOLD,
        metric_threshold=Config.METRIC_THRESHOLD,
        use_cutmix=Config.USE_CUTMIX,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        cutmix_prob=Config.CUTMIX_PROB,
        centerline_loss_weight=Config.CENTERLINE_LOSS_WEIGHT,
        boundary_loss_weight=Config.BOUNDARY_LOSS_WEIGHT,
    )

    print(f"\n训练配置:")
    print(f"  - Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")
    print(f"  - Optimizer: AdamW (lr={Config.LEARNING_RATE}, wd={Config.WEIGHT_DECAY})")
    print(f"  - LR Scheduler: Cosine (warmup={Config.LR_WARMUP_EPOCHS}, min_lr={Config.LR_MIN})")
    print(f"  - Loss: {' + '.join(Config.LOSS_CONFIG['loss_types']).upper()}")
    print(f"  - Edge loss: main={Config.EDGE_LOSS_WEIGHT}, aux={Config.EDGE_AUX_LOSS_WEIGHT}")
    print(f"  - clDice loss weight: {Config.CENTERLINE_LOSS_WEIGHT}")
    print(f"  - Metric threshold: {Config.METRIC_THRESHOLD}")
    print(f"  - CutMix: {Config.USE_CUTMIX} (alpha={Config.CUTMIX_ALPHA}, prob={Config.CUTMIX_PROB})")
    early_stop_config = Config.EARLY_STOPPING_CONFIG
    print(f"  - Early Stop: {'disabled' if early_stop_config is None else early_stop_config['monitor']}")

    # 4. 回调函数
    callbacks = [
        EMACallback(decay=0.999),
        # 只保存最优权重，不保存 last
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{val_iou:.4f}',
            monitor='val_iou',
            mode='max',
            save_top_k=1,
            save_last=False,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # 早停（可选）
    if early_stop_config is not None:
        callbacks.append(
            VerboseEarlyStopping(
                monitor=early_stop_config['monitor'],
                patience=early_stop_config['patience'],
                mode=early_stop_config['mode'],
                verbose=early_stop_config['verbose'],
                min_delta=early_stop_config['min_delta']
            )
        )

    # 5. Logger
    csv_logger = CSVLogger(
        save_dir=results_dir,
        name='training_logs'
    )

    # 自定义metrics logger
    metrics_logger = MetricsCSVLogger(save_dir=results_dir)
    callbacks.append(metrics_logger)

    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=Config.EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=Config.LOG_EVERY_N_STEPS,
        val_check_interval=Config.VAL_CHECK_INTERVAL,
        accumulate_grad_batches=Config.ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=Config.GRADIENT_CLIP_VAL,
        precision='16-mixed' if torch.cuda.is_available() else '32',
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
    )

    # 7. 开始训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")
    
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # 8. 训练结束
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    # 输出最佳模型信息
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\n最佳模型保存在: {best_model_path}")
        print(f"最佳 val_iou: {trainer.checkpoint_callback.best_model_score:.4f}")
    
    print(f"\n训练日志保存在: {results_dir}")
    print(f"Checkpoints保存在: {checkpoint_dir}")


if __name__ == '__main__':
    main()
