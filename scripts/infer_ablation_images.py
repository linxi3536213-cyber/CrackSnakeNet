#!/usr/bin/env python3
"""Generate ablation inference images for paper figure."""
import os, sys, cv2, torch, types
import numpy as np
import torch.nn.functional as F

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from model.CNNBlock import ConvGNAct
from train.Trainer import LitCrackModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DS_BASE = os.path.join(PROJECT, 'ComparisonModel', 'dataset_8_2')
CKPT_DIR = os.path.join(PROJECT, 'checkpoints', 'ablation_v2')
OUT_DIR = os.path.join(PROJECT, 'paper', 'latex', 'figures', 'fig_ablation')
os.makedirs(OUT_DIR, exist_ok=True)

# Select images that show clear differences
SELECTED_IMAGES = ['CrackTree260_6215_y0000_x0000', 'CrackTree260_6281_y0000_x0000']

def compute_mean_std_from_dir(img_dir):
    means, stds = [], []
    for name in sorted(os.listdir(img_dir)):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        means.append(img.mean(axis=(0,1))); stds.append(img.std(axis=(0,1)))
    return np.mean(means, axis=0), np.mean(stds, axis=0)

def build_model(ckpt_path, scan_directions=8, no_stem_skip=False):
    mc = Config.MODEL_CONFIGS
    widths = mc['widths']
    stage_blocks = Config.expand_stage_blocks(mc['depths'], len(widths))
    model = TriBranchCrackNet(
        in_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES,
        widths=widths, stage_blocks=stage_blocks,
        k=mc['k'], d_state=mc['d_state'],
        dropout=mc['dropout'], attn_drop=mc['attn_drop'],
        proj_drop=mc['proj_drop'], drop_path_rate=mc['drop_path_rate'],
        first_downsample_stride=Config.ENCODER_FIRST_DOWNSAMPLE_STRIDE,
        use_aux_outputs=Config.USE_AUX_OUTPUTS, aux_stage_indices=Config.AUX_STAGE_INDICES,
        use_edge_head=Config.USE_EDGE_HEAD, scan_directions=scan_directions)

    if no_stem_skip:
        model.full_res_refine = torch.nn.Sequential(
            ConvGNAct(widths[0], widths[0]*2, kernel_size=3, stride=1, padding=1),
            ConvGNAct(widths[0]*2, widths[0], kernel_size=3, stride=1, padding=1),
        )
        def _patched_forward(self, x):
            orig_h, orig_w = x.shape[2], x.shape[3]
            stem_out = self.stem(x)
            stage_feats = []
            y = stem_out
            for down, stage in zip(self.stage_downs, self.stages):
                y = stage(down(y))
                stage_feats.append(y)
            y = stage_feats[-1]
            aux_outputs = {}
            for idx in range(self.stage_count - 1):
                skip = stage_feats[-2 - idx]
                y = self.dec_blocks[idx](y, skip)
                if idx in self.aux_stage_indices:
                    aux_outputs[idx] = self.aux_heads[idx](y)
            y = F.interpolate(y, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            y = self.full_res_refine(y)
            logits = self.seg_head(y)
            if logits.shape[-2:] != (orig_h, orig_w):
                logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            if self.use_aux_outputs:
                aux_up = {i: F.interpolate(a, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                          for i, a in aux_outputs.items()}
                return {'main_output': logits, 'aux_outputs': aux_up}
            return logits
        model.forward = types.MethodType(_patched_forward, model)

    lit = LitCrackModel(model=model, loss_config=Config.LOSS_CONFIG,
        lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
        max_epochs=Config.EPOCHS, warmup_epochs=Config.LR_WARMUP_EPOCHS,
        min_lr=Config.LR_MIN, edge_loss_weight=Config.EDGE_LOSS_WEIGHT,
        edge_aux_loss_weight=Config.EDGE_AUX_LOSS_WEIGHT,
        edge_threshold=Config.EDGE_THRESHOLD, metric_threshold=Config.METRIC_THRESHOLD,
        use_cutmix=Config.USE_CUTMIX, cutmix_alpha=Config.CUTMIX_ALPHA,
        cutmix_prob=Config.CUTMIX_PROB, centerline_loss_weight=Config.CENTERLINE_LOSS_WEIGHT,
        boundary_loss_weight=0.0)
    ckpt = torch.load(ckpt_path, map_location=device)
    lit.load_state_dict(ckpt['state_dict'], strict=False)
    lit.to(device).eval()
    return lit

def infer_one(lit, img_path, mean, std):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    for c in range(3): x[c] = (x[c] - mean[c]) / std[c]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        o0 = lit(x); l = o0['main_output'] if isinstance(o0, dict) else o0
        o1 = lit(torch.flip(x,[3])); l = l + torch.flip(o1['main_output'] if isinstance(o1,dict) else o1,[3])
        o2 = lit(torch.flip(x,[2])); l = l + torch.flip(o2['main_output'] if isinstance(o2,dict) else o2,[2])
        o3 = lit(torch.flip(x,[2,3])); l = l + torch.flip(o3['main_output'] if isinstance(o3,dict) else o3,[2,3])
    prob = torch.sigmoid(l / 4.0)
    pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    return pred

ABLATIONS = {
    'A0': {'ckpt': os.path.join(PROJECT, 'checkpoints', 'ours_8_2', 'CrackTree260', 'best-epoch=199-val_iou=0.5890.ckpt'), 'scan_directions': 8, 'no_stem_skip': False},
    'A1': {'ckpt': os.path.join(CKPT_DIR, 'ablation_ckpt_A1_no_centerline_loss.ckpt'), 'scan_directions': 8, 'no_stem_skip': False},
    'A2': {'ckpt': os.path.join(CKPT_DIR, 'ablation_ckpt_A2_4dir_scan.ckpt'), 'scan_directions': 4, 'no_stem_skip': False},
    'A3': {'ckpt': os.path.join(CKPT_DIR, 'ablation_ckpt_A3_no_stem_skip.ckpt'), 'scan_directions': 8, 'no_stem_skip': True},
    'A5': {'ckpt': os.path.join(CKPT_DIR, 'ablation_ckpt_A5_unidir_crossattn.ckpt'), 'scan_directions': 8, 'no_stem_skip': False},
    'A7': {'ckpt': os.path.join(CKPT_DIR, 'ablation_ckpt_A7_no_coblock.ckpt'), 'scan_directions': 8, 'no_stem_skip': False},
}

if __name__ == '__main__':
    ds = 'CrackTree260'
    img_dir = os.path.join(DS_BASE, ds, 'test', 'images')
    mask_dir = os.path.join(DS_BASE, ds, 'test', 'annotations')
    train_img_dir = os.path.join(DS_BASE, ds, 'train', 'images')

    m_, s_ = compute_mean_std_from_dir(train_img_dir)
    mean, std = list(m_), list(s_)

    # Save input and GT for selected images
    for img_name in SELECTED_IMAGES:
        for ext in ['.png', '.jpg']:
            p = os.path.join(img_dir, img_name + ext)
            if os.path.exists(p):
                img = cv2.imread(p)
                img = cv2.resize(img, (512, 512))
                cv2.imwrite(os.path.join(OUT_DIR, f'{img_name}_input.png'), img)
                break

        for ext in ['.png', '.jpg']:
            gp = os.path.join(mask_dir, img_name + ext)
            if os.path.exists(gp):
                gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)
                # Convert to black bg white crack
                gt_vis = np.zeros_like(gt)
                gt_vis[gt < 128] = 255
                cv2.imwrite(os.path.join(OUT_DIR, f'{img_name}_gt.png'), gt_vis)
                break

    # Run inference for each ablation variant
    for ab_name, cfg in ABLATIONS.items():
        print(f'>>> {ab_name}')
        lit = build_model(cfg['ckpt'], scan_directions=cfg['scan_directions'],
                          no_stem_skip=cfg['no_stem_skip'])
        for img_name in SELECTED_IMAGES:
            for ext in ['.png', '.jpg']:
                p = os.path.join(img_dir, img_name + ext)
                if os.path.exists(p):
                    pred = infer_one(lit, p, mean, std)
                    cv2.imwrite(os.path.join(OUT_DIR, f'{img_name}_{ab_name}.png'), pred)
                    print(f'  Saved {img_name}_{ab_name}.png')
                    break
        del lit
        torch.cuda.empty_cache()

    print(f'\nAll ablation images saved to {OUT_DIR}')
