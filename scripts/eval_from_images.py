#!/usr/bin/env python3
"""Recompute all metrics from saved inference images for DeepCrack and CrackTree260."""
import os, cv2, json, numpy as np
from skimage.morphology import skeletonize

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = os.path.join(PROJECT, 'ComparisonModel')
DS_BASE = os.path.join(COMP, 'dataset_8_2')
INF_BASE = os.path.join(PROJECT, 'paper', 'InferenceResult')

def get_stat(p, g):
    return int(np.sum((p==1)&(g==1))), int(np.sum((p==1)&(g==0))), int(np.sum((p==0)&(g==1)))

def cal_mIoU(preds, gts, step=0.01):
    best = 0
    for t in np.arange(0.0, 1.0, step):
        ious = []
        for p, g in zip(preds, gts):
            gi = (g/255).astype('uint8'); pi = (p/255 > t).astype('uint8')
            TP = np.sum((pi==1)&(gi==1)); TN = np.sum((pi==0)&(gi==0))
            FP = np.sum((pi==1)&(gi==0)); FN = np.sum((pi==0)&(gi==1))
            iou = (TP/(FN+FP+TP) + TN/(FN+FP+TN))/2 if (FN+FP+TP) > 0 else 0
            ious.append(iou)
        best = max(best, np.mean(ious))
    return best

def cal_ODS(preds, gts, step=0.01):
    best = 0
    for t in np.arange(0.0, 1.0, step):
        f1s = []
        for p, g in zip(preds, gts):
            gi = (g/255).astype('uint8'); pi = (p/255 > t).astype('uint8')
            tp, fp, fn = get_stat(pi, gi)
            pr = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
            rc = tp/(tp+fn) if tp+fn > 0 else 0
            f1s.append(0 if pr+rc==0 else 2*pr*rc/(pr+rc))
        best = max(best, np.mean(f1s))
    return best

def cal_OIS(preds, gts, step=0.01):
    bfs = []
    for p, g in zip(preds, gts):
        bf = 0
        for t in np.arange(0.0, 1.0, step):
            gi = (g/255).astype('uint8'); pi = (p/255 > t).astype('uint8')
            tp, fp, fn = get_stat(pi, gi)
            pr = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
            rc = tp/(tp+fn) if tp+fn > 0 else 0
            bf = max(bf, 0 if pr+rc==0 else 2*pr*rc/(pr+rc))
        bfs.append(bf)
    return float(np.mean(bfs))

def prf_half(preds, gts):
    atp, afp, afn = 0, 0, 0
    for p, g in zip(preds, gts):
        gi = (g/255).astype('uint8')
        pi = ((p / max(float(np.max(p)), 1e-8)) > 0.5).astype('uint8')
        tp, fp, fn = get_stat(pi, gi)
        atp += tp; afp += fp; afn += fn
    pr = 1.0 if atp==0 and afp==0 else atp/(atp+afp)
    rc = atp/(atp+afn) if atp+afn > 0 else 0
    return pr, rc, (2*pr*rc/(pr+rc) if pr+rc > 0 else 0)

def compute_cldice_batch(preds, gt_masks, gt_cls, thr=0.5):
    cds, tps, tss = [], [], []
    for p, gm, gc in zip(preds, gt_masks, gt_cls):
        pb = (p/255.0 > thr).astype(np.uint8)
        cs = gc.sum(); ts = (gc*pb).sum()/cs if cs > 0 else 1.0
        sk = skeletonize(pb.astype(bool)).astype(np.float64); ss = sk.sum()
        tp = (sk*gm).sum()/ss if ss > 0 else (0.0 if gm.sum() > 0 else 1.0)
        cd = 2*tp*ts/(tp+ts) if tp+ts > 0 else 0
        cds.append(cd); tps.append(tp); tss.append(ts)
    return float(np.mean(cds)), float(np.mean(tps)), float(np.mean(tss))

def load_gt(mask_dir, cl_dir, name, sz):
    bn = os.path.splitext(name)[0]
    for ext in ['.png', '.jpg', '.bmp']:
        gp = os.path.join(mask_dir, bn+ext)
        cp = os.path.join(cl_dir, bn+ext)
        if os.path.exists(gp) and os.path.exists(cp):
            break
    else:
        return None, None, None
    g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
    c = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
    if g is None or c is None: return None, None, None
    g = cv2.resize(g, (sz,sz), interpolation=cv2.INTER_NEAREST)
    c = cv2.resize(c, (sz,sz), interpolation=cv2.INTER_NEAREST)
    _, gt = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    return 255-gt, (g<128).astype(np.uint8), (c<128).astype(np.uint8)

models = ['Ours', 'UNet', 'DeepLabV3Plus', 'DeepCrack_Model', 'DBCNet', 'SCSegamba', 'DTrCNet', 'CTCrackSeg']
all_results = {}

for ds in ['DeepCrack', 'CrackTree260']:
    img_dir = os.path.join(DS_BASE, ds, 'test', 'images')
    mask_dir = os.path.join(DS_BASE, ds, 'test', 'annotations')
    cl_dir = os.path.join(DS_BASE, ds, 'test', 'centerlines')

    print(f"\n{'='*80}")
    print(f"  {ds} (from saved inference images)")
    print(f"{'='*80}")

    ds_results = {}
    for model in models:
        pred_dir = os.path.join(INF_BASE, ds, model)
        if not os.path.isdir(pred_dir):
            print(f"  {model}: SKIP (no dir)")
            continue

        preds, gt_pix, gt_m, gt_c = [], [], [], []
        for name in sorted(os.listdir(mask_dir)):
            if not name.endswith('.png'): continue
            pred_path = os.path.join(pred_dir, name)
            if not os.path.exists(pred_path): continue
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if pred is None: continue
            gp, gm, gc = load_gt(mask_dir, cl_dir, name, 512)
            if gp is None: continue
            if pred.shape != gp.shape[:2]:
                pred = cv2.resize(pred, (gp.shape[1], gp.shape[0]))
            preds.append(pred); gt_pix.append(gp); gt_m.append(gm); gt_c.append(gc)

        if len(preds) == 0:
            print(f"  {model}: 0 images"); continue

        print(f"  {model} ({len(preds)} imgs)...", end="", flush=True)
        miou = cal_mIoU(preds, gt_pix)
        ods = cal_ODS(preds, gt_pix)
        ois = cal_OIS(preds, gt_pix)
        p, r, f1 = prf_half(preds, gt_pix)
        cd, tp, ts = compute_cldice_batch(preds, gt_m, gt_c)

        res = {'mIoU': round(miou,4), 'ODS': round(ods,4), 'OIS': round(float(ois),4),
               'P': round(p,4), 'R': round(r,4), 'F1': round(f1,4),
               'clDice': round(cd,4), 'Tprec': round(tp,4), 'Tsens': round(ts,4), 'n': len(preds)}
        ds_results[model] = res
        print(f" mIoU={miou*100:.2f} R={r*100:.2f} Tsens={ts*100:.2f} clDice={cd*100:.2f}")

    all_results[ds] = ds_results

# Summary tables
for ds in ['DeepCrack', 'CrackTree260']:
    print(f"\n{'='*120}")
    print(f"  {ds}")
    print(f"{'Method':<20s} {'mIoU':>7s} {'ODS':>7s} {'OIS':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'clDice':>8s} {'Tprec':>7s} {'Tsens':>7s}")
    print(f"{'-'*120}")
    for model in models:
        if model not in all_results[ds]: continue
        r = all_results[ds][model]
        print(f"{model:<20s} {r['mIoU']*100:>7.2f} {r['ODS']*100:>7.2f} {r['OIS']*100:>7.2f} "
              f"{r['P']*100:>7.2f} {r['R']*100:>7.2f} {r['F1']*100:>7.2f} "
              f"{r['clDice']*100:>8.2f} {r['Tprec']*100:>7.2f} {r['Tsens']*100:>7.2f}")
    print(f"{'='*120}")

out_path = os.path.join(PROJECT, 'paper', 'eval_recomputed.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
