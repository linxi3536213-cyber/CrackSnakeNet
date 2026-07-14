#!/usr/bin/env python3
"""
Unified evaluation: all metrics on 8:2 DeepCrack test set (104 images, 512×512).

Metrics:
  - mIoU, ODS, OIS, Precision, Recall, F1  (pixel-level, threshold-swept)
  - clDice, Tprec, Tsens                    (topology, fixed threshold 0.5)

Models: U-Net(R34), DeepLabV3+(R34), DTrC-Net, DeepCrack, DBCNet, Ours
"""
import os, sys, cv2, torch, json
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from skimage.morphology import skeletonize

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ComparisonModel')
BASE = os.path.abspath(BASE)
PROJECT_ROOT = os.path.dirname(BASE)
DS_BASE = os.path.join(BASE, 'dataset_8_2')

# ===================== Pixel-level metrics (SCSegamba-identical) =====================

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
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou = (TP / (FN + FP + TP) + TN / (FN + FP + TN)) / 2
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

# ===================== clDice =====================

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

# ===================== GT loading =====================

def load_gt_all(mask_dir, cl_dir, name, img_size):
    """Load GT mask (for pixel metrics as 0/255) and centerline (binary 0/1)."""
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

    # For pixel metrics: invert to 255=crack (same as eval_8_2.py load_gt)
    _, gt_thresh = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    gt_pixel = 255 - gt_thresh  # crack=255

    # For clDice: binary 0/1
    gt_bin = (gt < 128).astype(np.uint8)
    cl_bin = (cl < 128).astype(np.uint8)

    return gt_pixel, gt_bin, cl_bin

# ===================== Compute all metrics =====================

def evaluate_all(pred_list, gt_pixel_list, gt_mask_list, gt_cl_list):
    print("    Computing mIoU...", end="", flush=True)
    mIoU = cal_mIoU(pred_list, gt_pixel_list)
    print(f" {mIoU:.4f}")
    print("    Computing ODS...", end="", flush=True)
    ODS = cal_ODS(pred_list, gt_pixel_list)
    print(f" {ODS:.4f}")
    print("    Computing OIS...", end="", flush=True)
    OIS = cal_OIS(pred_list, gt_pixel_list)
    print(f" {OIS:.4f}")
    P, R, F1 = prf_at_half(pred_list, gt_pixel_list)
    print(f"    P={P:.4f}  R={R:.4f}  F1={F1:.4f}")
    clDice, Tprec, Tsens = compute_cldice_batch(pred_list, gt_mask_list, gt_cl_list, 0.5)
    print(f"    clDice={clDice:.4f}  Tprec={Tprec:.4f}  Tsens={Tsens:.4f}")
    return {
        'mIoU': round(mIoU, 4), 'ODS': round(ODS, 4), 'OIS': round(OIS, 4),
        'P': round(P, 4), 'R': round(R, 4), 'F1': round(F1, 4),
        'clDice': round(clDice, 4), 'Tprec': round(Tprec, 4), 'Tsens': round(Tsens, 4),
    }

# ===================== Model inference =====================

def infer_smp(model_name, encoder, ckpt_path, img_dir, mask_dir, cl_dir, img_size=512):
    import segmentation_models_pytorch as smp_lib
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'unet':
        model = smp_lib.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    else:
        model = smp_lib.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    pred_list, gt_pixel_list, gt_mask_list, gt_cl_list = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc=model_name):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            x = transforms.functional.normalize(
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x))
            pred_np = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
            gt_pixel, gt_bin, cl_bin = load_gt_all(mask_dir, cl_dir, name, img_size)
            if gt_pixel is None: continue
            pred_list.append(pred_np)
            gt_pixel_list.append(gt_pixel)
            gt_mask_list.append(gt_bin)
            gt_cl_list.append(cl_bin)
    return pred_list, gt_pixel_list, gt_mask_list, gt_cl_list

def infer_dtrcnet(ckpt_path, img_dir, mask_dir, cl_dir, img_size=256):
    old_cwd = os.getcwd()
    os.chdir(os.path.join(BASE, 'DTrC-Net'))
    sys.path.insert(0, '.')
    from CTCNet import CTCNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CTCNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    pred_list, gt_pixel_list, gt_mask_list, gt_cl_list = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="DTrC-Net"):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            x = transforms.functional.normalize(
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ).unsqueeze(0).to(device)
            logits, _ = model(x)
            pred_np = (torch.sigmoid(logits)[0, 0].cpu().numpy() * 255).astype(np.uint8)
            gt_pixel, gt_bin, cl_bin = load_gt_all(mask_dir, cl_dir, name, img_size)
            if gt_pixel is None: continue
            pred_list.append(pred_np)
            gt_pixel_list.append(gt_pixel)
            gt_mask_list.append(gt_bin)
            gt_cl_list.append(cl_bin)
    os.chdir(old_cwd)
    return pred_list, gt_pixel_list, gt_mask_list, gt_cl_list

def infer_deepcrack_model(ckpt_dir, img_dir, mask_dir, cl_dir, img_size=512):
    sys.path.insert(0, os.path.join(BASE, 'DeepCrack'))
    from models.deepcrack_networks import define_deepcrack
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='batch',
                              init_type='normal', init_gain=0.02, gpu_ids=[])
    ckpt_path = os.path.join(ckpt_dir, 'latest_net_G.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, '200_net_G.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    pred_list, gt_pixel_list, gt_mask_list, gt_cl_list = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="DeepCrack"):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            x = (x - 0.5) / 0.5
            x = x.unsqueeze(0).to(device)
            outputs = model(x)
            prob = torch.sigmoid(outputs[-1])
            pred_np = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
            gt_pixel, gt_bin, cl_bin = load_gt_all(mask_dir, cl_dir, name, img_size)
            if gt_pixel is None: continue
            pred_list.append(pred_np)
            gt_pixel_list.append(gt_pixel)
            gt_mask_list.append(gt_bin)
            gt_cl_list.append(cl_bin)
    return pred_list, gt_pixel_list, gt_mask_list, gt_cl_list

def load_saved_ours(pred_dir, mask_dir, cl_dir, img_size=512):
    pred_list, gt_pixel_list, gt_mask_list, gt_cl_list = [], [], [], []
    for name in tqdm(sorted(os.listdir(mask_dir)), desc="Ours"):
        if not name.endswith('.png'): continue
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(pred_path): continue
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None: continue
        gt_pixel, gt_bin, cl_bin = load_gt_all(mask_dir, cl_dir, name, img_size)
        if gt_pixel is None: continue
        if pred.shape != gt_bin.shape:
            pred = cv2.resize(pred, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred_list.append(pred)
        gt_pixel_list.append(gt_pixel)
        gt_mask_list.append(gt_bin)
        gt_cl_list.append(cl_bin)
    return pred_list, gt_pixel_list, gt_mask_list, gt_cl_list

# ===================== Checkpoint registry =====================

def find_best_dbcnet(log_dir):
    best, best_epoch = None, -1
    if not os.path.isdir(log_dir): return None
    for f in os.listdir(log_dir):
        if f.startswith('epoch-') and f.endswith('.pth') and not os.path.islink(os.path.join(log_dir, f)):
            try:
                epoch = int(f.split('-')[1].split('.')[0])
                if epoch > best_epoch:
                    best_epoch = epoch; best = os.path.join(log_dir, f)
            except ValueError: pass
    return best

# ===================== Main =====================

if __name__ == '__main__':
    ds = 'DeepCrack'
    img_dir = os.path.join(DS_BASE, ds, 'images', 'test')
    mask_dir = os.path.join(DS_BASE, ds, 'annotations', 'test')
    cl_dir = os.path.join(DS_BASE, ds, 'centerlines', 'test')

    n_test = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    print(f"{'='*80}")
    print(f"  DeepCrack 8:2 split — {n_test} test images, 512×512")
    print(f"{'='*80}\n")

    all_results = {}

    # --- U-Net ---
    p = os.path.join(BASE, 'ckpt_8_2', 'unet', ds, 'unet', 'best.pth')
    if os.path.exists(p):
        print(">>> U-Net (R34)")
        preds, gt_pix, gt_mask, gt_cl = infer_smp('unet', 'resnet34', p, img_dir, mask_dir, cl_dir, 512)
        all_results['U-Net'] = evaluate_all(preds, gt_pix, gt_mask, gt_cl)
        all_results['U-Net']['n'] = len(preds)
        print()

    # --- DeepLabV3+ ---
    p = os.path.join(BASE, 'ckpt_8_2', 'deeplabv3plus', ds, 'deeplabv3plus', 'best.pth')
    if os.path.exists(p):
        print(">>> DeepLabV3+ (R34)")
        preds, gt_pix, gt_mask, gt_cl = infer_smp('deeplabv3plus', 'resnet34', p, img_dir, mask_dir, cl_dir, 512)
        all_results['DeepLabV3+'] = evaluate_all(preds, gt_pix, gt_mask, gt_cl)
        all_results['DeepLabV3+']['n'] = len(preds)
        print()

    # --- DTrC-Net ---
    p = os.path.join(BASE, 'ckpt_8_2', 'dtrcnet', ds, 'best.pth')
    if os.path.exists(p):
        print(">>> DTrC-Net")
        preds, gt_pix, gt_mask, gt_cl = infer_dtrcnet(p, img_dir, mask_dir, cl_dir, 256)
        all_results['DTrC-Net'] = evaluate_all(preds, gt_pix, gt_mask, gt_cl)
        all_results['DTrC-Net']['n'] = len(preds)
        print()

    # --- DeepCrack ---
    ckpt_dir = os.path.join(BASE, 'DeepCrack', 'checkpoints', f'deepcrack_{ds}')
    if os.path.isdir(ckpt_dir):
        print(">>> DeepCrack")
        preds, gt_pix, gt_mask, gt_cl = infer_deepcrack_model(ckpt_dir, img_dir, mask_dir, cl_dir, 512)
        all_results['DeepCrack'] = evaluate_all(preds, gt_pix, gt_mask, gt_cl)
        all_results['DeepCrack']['n'] = len(preds)
        print()

    # --- DBCNet (subprocess) ---
    log_dir = os.path.join(BASE, 'DBCNet', 'log_final', 'log_Deepcrack',
                           'log_Deepcrack_my_backbone_7', 'checkpoint')
    dbcnet_ckpt = find_best_dbcnet(log_dir)
    if dbcnet_ckpt:
        print(">>> DBCNet (subprocess)")
        import subprocess
        dbcnet_code = f'''
import os, sys, cv2, torch, json, numpy as np
import torch.nn as nn
from tqdm import tqdm
from skimage.morphology import skeletonize

BASE = "{BASE}"
os.chdir(os.path.join(BASE, "DBCNet"))
sys.path.insert(0, ".")
from configs.config_crack_Deepcrack import config
from models.builder import EncoderDecoder as segmodel
from utils.loss_crack import SoftDiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = segmodel(cfg=config, criterion=SoftDiceLoss(), norm_layer=nn.BatchNorm2d)
ckpt = torch.load("{dbcnet_ckpt}", map_location=device)
sd = {{(k[7:] if k.startswith("module.") else k): v for k,v in ckpt["model"].items()}}
model.load_state_dict(sd, strict=True)
model.to(device).eval()

img_dir = "{img_dir}"
mask_dir = "{mask_dir}"
cl_dir = "{cl_dir}"

def load_all(mask_dir, cl_dir, name, sz):
    bn = os.path.splitext(name)[0]
    for ext in [".png",".jpg",".bmp"]:
        gp = os.path.join(mask_dir, bn+ext)
        cp = os.path.join(cl_dir, bn+ext)
        if os.path.exists(gp) and os.path.exists(cp):
            g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            c = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
            if g is None or c is None: return None,None,None
            g = cv2.resize(g,(sz,sz),interpolation=cv2.INTER_NEAREST)
            c = cv2.resize(c,(sz,sz),interpolation=cv2.INTER_NEAREST)
            _,gt=cv2.threshold(g,127,255,cv2.THRESH_BINARY)
            return 255-gt, (g<128).astype(np.uint8), (c<128).astype(np.uint8)
    return None,None,None

def get_stat(p,g):
    return int(np.sum((p==1)&(g==1))), int(np.sum((p==1)&(g==0))), int(np.sum((p==0)&(g==1)))

preds, gt_pix, gt_m, gt_c = [], [], [], []
with torch.no_grad():
    for name in tqdm(sorted(os.listdir(img_dir)), desc="DBCNet"):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(img).permute(2,0,1).float()/255.0
        x = x.unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x))
        pred_np = (prob[0,0].cpu().numpy()*255).astype(np.uint8)
        gp, gm, gc = load_all(mask_dir, cl_dir, name, 512)
        if gp is None: continue
        preds.append(pred_np); gt_pix.append(gp); gt_m.append(gm); gt_c.append(gc)

# mIoU
best_iou = 0
for t in np.arange(0,1,0.01):
    ious=[]
    for p,g in zip(preds,gt_pix):
        gi=(g/255).astype("uint8"); pi=(p/255>t).astype("uint8")
        TP=np.sum((pi==1)&(gi==1));TN=np.sum((pi==0)&(gi==0))
        FP=np.sum((pi==1)&(gi==0));FN=np.sum((pi==0)&(gi==1))
        iou=(TP/(FN+FP+TP)+TN/(FN+FP+TN))/2 if (FN+FP+TP)>0 else 0
        ious.append(iou)
    best_iou=max(best_iou,np.mean(ious))
# ODS
best_ods=0
for t in np.arange(0,1,0.01):
    f1s=[]
    for p,g in zip(preds,gt_pix):
        gi=(g/255).astype("uint8"); pi=(p/255>t).astype("uint8")
        tp,fp,fn=get_stat(pi,gi)
        pr=1.0 if tp==0 and fp==0 else tp/(tp+fp)
        rc=0 if tp+fn==0 else tp/(tp+fn)
        f1s.append(0 if pr+rc==0 else 2*pr*rc/(pr+rc))
    best_ods=max(best_ods,np.mean(f1s))
# OIS
ois_list=[]
for p,g in zip(preds,gt_pix):
    bf=0
    for t in np.arange(0,1,0.01):
        gi=(g/255).astype("uint8"); pi=(p/255>t).astype("uint8")
        tp,fp,fn=get_stat(pi,gi)
        pr=1.0 if tp==0 and fp==0 else tp/(tp+fp)
        rc=0 if tp+fn==0 else tp/(tp+fn)
        f1=0 if pr+rc==0 else 2*pr*rc/(pr+rc)
        bf=max(bf,f1)
    ois_list.append(bf)
ois_val=np.mean(ois_list)
# PRF
atp,afp,afn=0,0,0
for p,g in zip(preds,gt_pix):
    gi=(g/255).astype("uint8")
    pi=((p/max(float(np.max(p)),1e-8))>0.5).astype("uint8")
    tp,fp,fn=get_stat(pi,gi); atp+=tp;afp+=fp;afn+=fn
pr=1.0 if atp==0 and afp==0 else atp/(atp+afp)
rc=atp/(atp+afn) if atp+afn>0 else 0
f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0
# clDice
cds,tps,tss=[],[],[]
for p,gm,gc in zip(preds,gt_m,gt_c):
    pb=(p/255.0>0.5).astype(np.uint8)
    cs=gc.sum(); ts_v=(gc*pb).sum()/cs if cs>0 else 1.0
    sk=skeletonize(pb.astype(bool)).astype(np.float64)
    ss=sk.sum()
    tp_v=(sk*gm).sum()/ss if ss>0 else (0.0 if gm.sum()>0 else 1.0)
    cd=2*tp_v*ts_v/(tp_v+ts_v) if tp_v+ts_v>0 else 0
    cds.append(cd);tps.append(tp_v);tss.append(ts_v)

res = {{"mIoU":round(best_iou,4),"ODS":round(best_ods,4),"OIS":round(float(ois_val),4),
        "P":round(pr,4),"R":round(rc,4),"F1":round(f1,4),
        "clDice":round(float(np.mean(cds)),4),"Tprec":round(float(np.mean(tps)),4),
        "Tsens":round(float(np.mean(tss)),4),"n":len(preds)}}
print(json.dumps(res))
'''
        try:
            r = subprocess.run([sys.executable, '-c', dbcnet_code],
                               capture_output=True, text=True, timeout=1200)
            if r.returncode == 0:
                for line in r.stdout.strip().split('\n'):
                    if line.startswith('{'):
                        all_results['DBCNet'] = json.loads(line)
                        d = all_results['DBCNet']
                        print(f"    mIoU={d['mIoU']:.4f}  ODS={d['ODS']:.4f}  OIS={d['OIS']:.4f}  P={d['P']:.4f}  R={d['R']:.4f}  F1={d['F1']:.4f}")
                        print(f"    clDice={d['clDice']:.4f}  Tprec={d['Tprec']:.4f}  Tsens={d['Tsens']:.4f}")
                        break
            else:
                print(f"    ERROR: {r.stderr[-500:]}")
        except Exception as e:
            print(f"    ERROR: {e}")
        print()

    # --- Ours ---
    our_pred_dir = os.path.join(PROJECT_ROOT, 'paper', 'InferenceResult', 'CrackMambaTransformerNet')
    if os.path.isdir(our_pred_dir):
        print(">>> Ours")
        preds, gt_pix, gt_mask, gt_cl = load_saved_ours(our_pred_dir, mask_dir, cl_dir, 512)
        all_results['Ours'] = evaluate_all(preds, gt_pix, gt_mask, gt_cl)
        all_results['Ours']['n'] = len(preds)
        print()

    # ===================== Summary =====================
    print("=" * 110)
    print(f"{'Method':<15s} {'mIoU':>7s} {'ODS':>7s} {'OIS':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'clDice':>8s} {'Tprec':>7s} {'Tsens':>7s} {'n':>5s}")
    print("-" * 110)
    order = ['U-Net', 'DBCNet', 'DeepLabV3+', 'Ours', 'DTrC-Net', 'DeepCrack']
    for name in order:
        if name not in all_results: continue
        r = all_results[name]
        print(f"{name:<15s} {r['mIoU']:>7.4f} {r['ODS']:>7.4f} {r['OIS']:>7.4f} "
              f"{r['P']:>7.4f} {r['R']:>7.4f} {r['F1']:>7.4f} "
              f"{r['clDice']:>8.4f} {r['Tprec']:>7.4f} {r['Tsens']:>7.4f} {r['n']:>5d}")
    print("=" * 110)

    out_path = os.path.join(PROJECT_ROOT, 'paper', 'all_metrics_DeepCrack_8_2.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
