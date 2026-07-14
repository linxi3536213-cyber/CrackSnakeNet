#!/usr/bin/env python
"""
Unified evaluation script — produces paper-ready metrics identical to SCSegamba's eval.

Usage:
    python scripts/eval_aligned.py --ckpt checkpoints/<timestamp>/best-*.ckpt

Metrics computed (same as SCSegamba evaluate.py):
    mIoU  — threshold-swept, per-image averaged
    ODS   — optimal dataset-scale F1
    OIS   — optimal image-scale F1
    F1 / Precision / Recall — at threshold 0.5
"""
import os
import sys
import argparse
import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import Config, get_device
from model.TriBranchCracknet import TriBranchCrackNet
from data.dataset import CrackDatasetSplitter, CrackDataLoader
from train.Trainer import LitCrackModel


# ==================== SCSegamba-identical eval functions ====================

def get_statistics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, fp, fn


def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01):
    """mIoU: sweep thresholds, per-image (IoU_pos+IoU_neg)/2, average, take max."""
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0) / 2
            iou_list.append(iou)
        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    return np.max(np.array(final_iou))


def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r = 0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
            ODS_list.append(f1)
        final_ODS.append(np.mean(ODS_list))
    return np.max(np.array(final_ODS))


def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r = 0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
            F1_list.append(f1)
        final_F1_list.append(np.max(F1_list))
    return np.mean(final_F1_list)


def prf_at_half(pred_list, gt_list):
    """Precision, Recall, F1 at threshold 0.5 (same as SCSegamba prf_metrics)."""
    statistics = []
    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = ((pred / np.max(pred)) > 0.5).astype('uint8') if np.max(pred) > 0 else np.zeros_like(gt_img)
        tp, fp, fn = get_statistics(pred_img, gt_img)
        statistics.append((tp, fp, fn))
    tp = sum(v[0] for v in statistics)
    fp = sum(v[1] for v in statistics)
    fn = sum(v[2] for v in statistics)
    p = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


# ==================== Model loading & inference ====================

def load_model(ckpt_path, device):
    model_config = Config.MODEL_CONFIGS
    stage_count = len(model_config['widths'])
    stage_blocks = Config.expand_stage_blocks(model_config['depths'], stage_count)

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
    )

    # Direct load with strict=False (ignores optimizer/scheduler keys in checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)

    lit_model = LitCrackModel(
        model=model,
        loss_config=Config.LOSS_CONFIG,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys! Model may not work correctly.")
        print(f"  First 5: {missing[:5]}")
    print(f"  Loaded checkpoint ({len(unexpected)} extra keys skipped)")
    lit_model.eval()
    lit_model.to(device)
    return lit_model


def run_inference(lit_model, val_loader, device, use_tta=True):
    """Run inference and return (pred_list, gt_list) as 0-255 uint8 images."""
    pred_list, gt_list = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            imgs, targets, _ = batch
            imgs, targets = imgs.to(device), targets.to(device)

            if use_tta:
                logits = lit_model._tta_forward(imgs)
            else:
                output = lit_model(imgs)
                logits = lit_model._parse_model_output(output)[0]

            probs = torch.sigmoid(logits)  # [B, 1, H, W]

            for i in range(probs.shape[0]):
                # pred: 0-255 float (SCSegamba eval expects this range)
                pred_np = (probs[i, 0].cpu().numpy() * 255).astype(np.uint8)
                # gt: 0 or 255
                gt_np = (targets[i, 0].cpu().numpy() * 255).astype(np.uint8)
                pred_list.append(pred_np)
                gt_list.append(gt_np)

    return pred_list, gt_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to .ckpt file')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    parser.add_argument('--save-dir', type=str, default='paper/InferenceResult/CrackMambaTransformerNet',
                        help='Save pred images to dir')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load data (only need val set)
    splitter = CrackDatasetSplitter()
    train_pairs, val_pairs = splitter.get_train_val_pairs()
    # Need at least 1 train pair for CrackDataLoader to compute mean/std
    data_loader = CrackDataLoader(train_pairs, val_pairs, Config.IMAGE_SIZE, batch_size=1)
    _, val_loader = data_loader.get_loaders()

    # Get filenames from val_pairs
    name_list = [os.path.splitext(os.path.basename(p[0]))[0] for p in val_pairs]

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    lit_model = load_model(args.ckpt, device)

    # Inference
    use_tta = not args.no_tta
    print(f"Running inference (TTA={'ON' if use_tta else 'OFF'})...")
    pred_list, gt_list = run_inference(lit_model, val_loader, device, use_tta=use_tta)
    print(f"Total images: {len(pred_list)}")

    # Save predictions with original filenames
    save_dir = os.path.join(project_root, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for pred, name in zip(pred_list, name_list):
        cv2.imwrite(os.path.join(save_dir, f"{name}.png"), pred)
    print(f"Saved {len(pred_list)} predictions to {save_dir}")

    # Evaluate — identical to SCSegamba
    print("\nEvaluating (SCSegamba-aligned metrics)...")
    mIoU = cal_mIoU_metrics(pred_list, gt_list)
    ODS = cal_ODS_metrics(pred_list, gt_list)
    OIS = cal_OIS_metrics(pred_list, gt_list)
    P, R, F1 = prf_at_half(pred_list, gt_list)

    print("=" * 50)
    print(f"  mIoU      : {mIoU:.4f}")
    print(f"  ODS       : {ODS:.4f}")
    print(f"  OIS       : {OIS:.4f}")
    print(f"  F1        : {F1:.4f}")
    print(f"  Precision : {P:.4f}")
    print(f"  Recall    : {R:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
