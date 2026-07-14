#!/usr/bin/env python3
"""
Evaluate ablation checkpoints (A1-A7) on CrackTree260 test set.
Metrics: mIoU, ODS, OIS, P, R, F1, clDice, Tprec, Tsens
Uses TTA (4-fold flip augmentation) consistent with paper evaluation.
"""
import os, sys, cv2, torch, json
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from train.Trainer import LitCrackModel

DS_BASE = os.path.join(PROJECT, 'ComparisonModel', 'dataset_8_2')
CKPT_DIR = os.path.join(PROJECT, 'checkpoints', 'ablation_v2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== Metrics =====================

def get_statistics(pred, gt):
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    return tp, fp, fn

def cal_mIoU(pred_list, gt_list, thresh_step=0.01):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = int(np.sum((pred_img == 1) & (gt_img == 1)))
            TN = int(np.sum((pred_img == 0) & (gt_img == 0)))
            FP = int(np.sum((pred_img == 1) & (gt_img == 0)))
            FN = int(np.sum((pred_img == 0) & (gt_img == 1)))
            iou = (TP / (FN + FP + TP) + TN / (FN + FP + TN)) / 2 if (FN + FP + TP) > 0 else 0
            iou_list.append(iou)
        final_iou.append(np.mean(iou_list))
    return float(np.max(final_iou))

def cal_ODS(pred_list, gt_list, thresh_step=0.01):
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        f1_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r = 0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
            f1_list.append(f1)
        final_ODS.append(np.mean(f1_list))
    return float(np.max(final_ODS))

def cal_OIS(pred_list, gt_list, thresh_step=0.01):
    final_F1 = []
    for pred, gt in zip(pred_list, gt_list):
        best_f1 = 0
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r = 0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
            if f1 > best_f1: best_f1 = f1
        final_F1.append(best_f1)
    return float(np.mean(final_F1))

def prf_at_half(pred_list, gt_list):
    all_tp, all_fp, all_fn = 0, 0, 0
    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = ((pred / max(float(np.max(pred)), 1e-8)) > 0.5).astype('uint8')
        tp, fp, fn = get_statistics(pred_img, gt_img)
        all_tp += tp; all_fp += fp; all_fn += fn
    p = 1.0 if all_tp == 0 and all_fp == 0 else all_tp / (all_tp + all_fp)
    r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def compute_cldice(pred_bin, gt_mask, gt_cl):
    cl_sum = gt_cl.sum()
    tsens = (gt_cl * pred_bin).sum() / cl_sum if cl_sum > 0 else 1.0
    pred_skel = skeletonize(pred_bin.astype(bool)).astype(np.float64)
    skel_sum = pred_skel.sum()
    if skel_sum == 0:
        tprec = 0.0 if gt_mask.sum() > 0 else 1.0
    else:
        tprec = (pred_skel * gt_mask).sum() / skel_sum
    if tprec + tsens == 0:
        return 0.0, 0.0, 0.0
    return 2.0 * tprec * tsens / (tprec + tsens), tprec, tsens

def compute_cldice_batch(pred_list, gt_mask_list, gt_cl_list, threshold=0.5):
    cldices, tprecs, tsenses = [], [], []
    for pred, gt_mask, gt_cl in zip(pred_list, gt_mask_list, gt_cl_list):
        pred_bin = (pred / 255.0 > threshold).astype(np.uint8)
        cd, tp, ts = compute_cldice(pred_bin, gt_mask, gt_cl)
        cldices.append(cd); tprecs.append(tp); tsenses.append(ts)
    return float(np.mean(cldices)), float(np.mean(tprecs)), float(np.mean(tsenses))

def evaluate_all(pred_list, gt_pixel_list, gt_mask_list, gt_cl_list):
    print("    mIoU...", end="", flush=True)
    mIoU = cal_mIoU(pred_list, gt_pixel_list)
    print(f" {mIoU:.4f}", flush=True)
    print("    ODS...", end="", flush=True)
    ODS = cal_ODS(pred_list, gt_pixel_list)
    print(f" {ODS:.4f}", flush=True)
    print("    OIS...", end="", flush=True)
    OIS = cal_OIS(pred_list, gt_pixel_list)
    print(f" {OIS:.4f}", flush=True)
    P, R, F1 = prf_at_half(pred_list, gt_pixel_list)
    print(f"    P={P:.4f}  R={R:.4f}  F1={F1:.4f}", flush=True)
    clDice, Tprec, Tsens = compute_cldice_batch(pred_list, gt_mask_list, gt_cl_list, 0.5)
    print(f"    clDice={clDice:.4f}  Tprec={Tprec:.4f}  Tsens={Tsens:.4f}", flush=True)
    return {
        'mIoU': round(mIoU, 4), 'ODS': round(ODS, 4), 'OIS': round(OIS, 4),
        'P': round(P, 4), 'R': round(R, 4), 'F1': round(F1, 4),
        'clDice': round(clDice, 4), 'Tprec': round(Tprec, 4), 'Tsens': round(Tsens, 4),
    }

# ===================== GT loading =====================

def load_gt_all(mask_dir, cl_dir, name, img_size):
    basename = os.path.splitext(name)[0]
    for ext in ['.png', '.jpg', '.bmp']:
        gt_path = os.path.join(mask_dir, basename + ext)
        cl_path = os.path.join(cl_dir, basename + ext)
        if os.path.exists(gt_path) and os.path.exists(cl_path):
            break
    else:
        return None, None, None
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    cl = cv2.imread(cl_path, cv2.IMREAD_GRAYSCALE)
    if gt is None or cl is None:
        return None, None, None
    gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    cl = cv2.resize(cl, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    _, gt_thresh = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    gt_pixel = 255 - gt_thresh
    gt_bin = (gt < 128).astype(np.uint8)
    cl_bin = (cl < 128).astype(np.uint8)
    return gt_pixel, gt_bin, cl_bin

# ===================== Model building + inference =====================

def compute_mean_std_from_dir(img_dir):
    means, stds = [], []
    for name in sorted(os.listdir(img_dir)):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))
    return np.mean(means, axis=0), np.mean(stds, axis=0)

def build_model(ckpt_path, scan_directions=8, no_stem_skip=False, no_coblock=False):
    """Build model and load checkpoint. Handles different ablation configs."""
    model_config = Config.MODEL_CONFIGS.copy()
    model_config['scan_directions'] = scan_directions

    stage_blocks = Config.expand_stage_blocks(model_config['depths'], len(model_config['widths']))
    model = TriBranchCrackNet(
        in_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES,
        widths=model_config['widths'], stage_blocks=stage_blocks,
        k=model_config['k'], d_state=model_config['d_state'],
        dropout=model_config['dropout'], attn_drop=model_config['attn_drop'],
        proj_drop=model_config['proj_drop'], drop_path_rate=model_config['drop_path_rate'],
        first_downsample_stride=Config.ENCODER_FIRST_DOWNSAMPLE_STRIDE,
        use_aux_outputs=Config.USE_AUX_OUTPUTS, aux_stage_indices=Config.AUX_STAGE_INDICES,
        use_edge_head=Config.USE_EDGE_HEAD, scan_directions=scan_directions)

    # A7: replace COBlock with ConvBlock in encoder stages and FusionBlock
    if no_coblock:
        from model.CNNBlock import ConvBlock
        widths = model_config['widths']
        mamba_start = 2
        # Replace COBlock in pure-CNN encoder stages (stages 0, 1)
        for i in range(mamba_start):
            w = widths[i]
            for j in range(len(model.stages[i])):
                model.stages[i][j] = ConvBlock(w, w)
        # Replace COBlock inside CNNandMambaBlock (stages 2, 3)
        for i in range(mamba_start, len(widths)):
            for j in range(len(model.stages[i])):
                block = model.stages[i][j]
                if hasattr(block, 'cnn'):
                    block.cnn = ConvBlock(widths[i], widths[i])

    # A3: no stem skip — patch model to remove stem concat
    if no_stem_skip:
        import torch.nn.functional as F
        from model.CNNBlock import ConvGNAct
        widths = model_config['widths']
        refine_in_ch = widths[0]  # no stem concat
        refine_ch = widths[0] * 2
        model.full_res_refine = torch.nn.Sequential(
            ConvGNAct(refine_in_ch, refine_ch, kernel_size=3, stride=1, padding=1),
            ConvGNAct(refine_ch, widths[0], kernel_size=3, stride=1, padding=1),
        )
        # Monkey-patch forward to skip stem concat
        _orig_forward = model.forward
        def _forward_no_stem(x, _of=_orig_forward):
            # We need to override the forward to skip stem_up concat
            # Run original forward but intercept — simpler to re-implement the key part
            import types
            return _of(x)
        # Instead, patch at source: replace the forward method
        def _patched_forward(self, x):
            orig_h, orig_w = x.shape[2], x.shape[3]
            stem_out = self.stem(x)
            feats = [stem_out]
            y = stem_out
            for stage in self.encoder_stages:
                y = stage(y)
                feats.append(y)
            y = feats[-1]
            aux_outputs = []
            for i, dec in enumerate(self.decoder_blocks):
                skip = feats[-(i+2)]
                y = dec(y, skip)
                if self.use_aux_outputs and i in self._aux_map:
                    aux_idx = self._aux_map[i]
                    aux_outputs.append(self.aux_heads[aux_idx](y))
            # NO stem concat — just upsample and refine
            y = F.interpolate(y, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            y = self.full_res_refine(y)
            main_out = self.seg_head(y)
            if self.use_aux_outputs:
                return {'main_output': main_out, 'aux_outputs': aux_outputs}
            return main_out
        import types
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

def infer_with_tta(lit, img_dir, mask_dir, cl_dir, train_img_dir, desc):
    """Inference with 4-fold TTA, returns predictions + GT lists."""
    m_, s_ = compute_mean_std_from_dir(train_img_dir)
    mean, std = list(m_), list(s_)

    pred_list, gt_pixel_list, gt_mask_list, gt_cl_list = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc=desc):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for c in range(3): x[c] = (x[c] - mean[c]) / std[c]
            x = x.unsqueeze(0).to(device)

            o0 = lit(x); l = o0['main_output'] if isinstance(o0, dict) else o0
            o1 = lit(torch.flip(x, [3])); l = l + torch.flip(o1['main_output'] if isinstance(o1, dict) else o1, [3])
            o2 = lit(torch.flip(x, [2])); l = l + torch.flip(o2['main_output'] if isinstance(o2, dict) else o2, [2])
            o3 = lit(torch.flip(x, [2, 3])); l = l + torch.flip(o3['main_output'] if isinstance(o3, dict) else o3, [2, 3])
            prob = torch.sigmoid(l / 4.0)
            pred_np = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)

            gt_pixel, gt_bin, cl_bin = load_gt_all(mask_dir, cl_dir, name, 512)
            if gt_pixel is None: continue
            pred_list.append(pred_np)
            gt_pixel_list.append(gt_pixel)
            gt_mask_list.append(gt_bin)
            gt_cl_list.append(cl_bin)

    return pred_list, gt_pixel_list, gt_mask_list, gt_cl_list


# ===================== Ablation experiments =====================

ABLATIONS = {
    'A1_no_centerline_loss': {'ckpt': 'ablation_ckpt_A1_no_centerline_loss.ckpt', 'scan_directions': 8},
    'A2_4dir_scan':          {'ckpt': 'ablation_ckpt_A2_4dir_scan.ckpt',          'scan_directions': 4},
    'A3_no_stem_skip':       {'ckpt': 'ablation_ckpt_A3_no_stem_skip.ckpt',       'scan_directions': 8, 'no_stem_skip': True},
    'A4_no_mamba_residual':  {'ckpt': 'ablation_ckpt_A4_no_mamba_residual.ckpt',  'scan_directions': 8},
    'A5_unidir_crossattn':   {'ckpt': 'ablation_ckpt_A5_unidir_crossattn.ckpt',  'scan_directions': 8},
    'A6_no_multiscale':      {'ckpt': 'ablation_ckpt_A6_no_multiscale.ckpt',      'scan_directions': 8},
    'A7_no_coblock':         {'ckpt': 'ablation_ckpt_A7_no_coblock.ckpt',         'scan_directions': 8, 'no_coblock': True},
}


if __name__ == '__main__':
    ds = 'CrackTree260'
    img_dir = os.path.join(DS_BASE, ds, 'test', 'images')
    mask_dir = os.path.join(DS_BASE, ds, 'test', 'annotations')
    cl_dir = os.path.join(DS_BASE, ds, 'test', 'centerlines')
    train_img_dir = os.path.join(DS_BASE, ds, 'train', 'images')

    n_test = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    print(f"{'='*100}")
    print(f"  Ablation evaluation on CrackTree260 — {n_test} test images, 512×512")
    print(f"{'='*100}\n")

    all_results = {}

    for name, cfg in ABLATIONS.items():
        ckpt_path = os.path.join(CKPT_DIR, cfg['ckpt'])
        if not os.path.exists(ckpt_path):
            print(f">>> {name}: SKIPPED (checkpoint not found: {ckpt_path})")
            continue

        print(f">>> {name} (scan_directions={cfg['scan_directions']})")
        try:
            lit = build_model(ckpt_path, scan_directions=cfg['scan_directions'],
                             no_stem_skip=cfg.get('no_stem_skip', False),
                             no_coblock=cfg.get('no_coblock', False))
            preds, gt_pix, gt_mask, gt_cl = infer_with_tta(
                lit, img_dir, mask_dir, cl_dir, train_img_dir, name)
            results = evaluate_all(preds, gt_pix, gt_mask, gt_cl)
            results['n'] = len(preds)
            all_results[name] = results
            print()
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            print()

    # ===================== Summary =====================
    print(f"\n{'='*120}")
    print(f"{'Experiment':<30s} {'mIoU':>7s} {'ODS':>7s} {'OIS':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'clDice':>8s} {'Tprec':>7s} {'Tsens':>7s}")
    print(f"{'-'*120}")

    # A0 baseline from paper
    print(f"{'A0_baseline (paper)':<30s} {'80.35':>7s} {'75.54':>7s} {'75.84':>7s} {'63.01':>7s} {'86.65':>7s} {'72.97':>7s} {'78.21':>8s} {'71.11':>7s} {'87.23':>7s}")

    for name in ABLATIONS:
        if name not in all_results: continue
        r = all_results[name]
        print(f"{name:<30s} {r['mIoU']*100:>7.2f} {r['ODS']*100:>7.2f} {r['OIS']*100:>7.2f} "
              f"{r['P']*100:>7.2f} {r['R']*100:>7.2f} {r['F1']*100:>7.2f} "
              f"{r['clDice']*100:>8.2f} {r['Tprec']*100:>7.2f} {r['Tsens']*100:>7.2f}")
    print(f"{'='*120}")

    out_path = os.path.join(PROJECT, 'paper', 'ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
