#!/usr/bin/env python3
"""
Evaluate clDice on 8:2 split DeepCrack test set (104 images, 512×512).

Re-runs inference from checkpoints for comparison models.
For our model, uses saved predictions.

clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
  Tprec = |skel(pred) ∩ mask_gt| / |skel(pred)|
  Tsens = |centerline_gt ∩ pred| / |centerline_gt|

Reference: Shit et al., CVPR 2021.
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

# ==================== clDice ====================

def compute_cldice(pred_bin, gt_mask, gt_cl):
    """Compute clDice for one image. All inputs are binary 0/1."""
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
    cldice = 2.0 * tprec * tsens / (tprec + tsens)
    return cldice, tprec, tsens


def compute_cldice_optimal(pred_list, gt_mask_list, gt_cl_list, thresh_step=0.01):
    """Sweep thresholds to find optimal clDice (like mIoU sweep)."""
    best_cldice = -1
    best_thresh = 0.5
    best_result = None

    for thresh in np.arange(0.05, 0.95, thresh_step):
        cldices = []
        for pred, gt_mask, gt_cl in zip(pred_list, gt_mask_list, gt_cl_list):
            pred_bin = (pred / 255.0 > thresh).astype(np.uint8)
            cd, _, _ = compute_cldice(pred_bin, gt_mask, gt_cl)
            cldices.append(cd)
        mean_cd = np.mean(cldices)
        if mean_cd > best_cldice:
            best_cldice = mean_cd
            best_thresh = thresh

    # Re-evaluate at best threshold for full stats
    cldices, tprecs, tsenses = [], [], []
    for pred, gt_mask, gt_cl in zip(pred_list, gt_mask_list, gt_cl_list):
        pred_bin = (pred / 255.0 > best_thresh).astype(np.uint8)
        cd, tp, ts = compute_cldice(pred_bin, gt_mask, gt_cl)
        cldices.append(cd)
        tprecs.append(tp)
        tsenses.append(ts)

    return {
        'clDice': round(float(np.mean(cldices)), 4),
        'Tprec': round(float(np.mean(tprecs)), 4),
        'Tsens': round(float(np.mean(tsenses)), 4),
        'threshold': round(float(best_thresh), 2),
        'n_images': len(cldices),
    }


# ==================== GT loading ====================

def load_gt_and_cl(mask_dir, cl_dir, name, img_size):
    """Load GT mask and centerline, invert (raw: 0=crack, 255=bg)."""
    basename = os.path.splitext(name)[0]
    for ext in ['.png', '.jpg', '.bmp']:
        gt_path = os.path.join(mask_dir, basename + ext)
        cl_path = os.path.join(cl_dir, basename + ext)
        if os.path.exists(gt_path) and os.path.exists(cl_path):
            break
    else:
        return None, None

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    cl = cv2.imread(cl_path, cv2.IMREAD_GRAYSCALE)
    if gt is None or cl is None:
        return None, None

    gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    cl = cv2.resize(cl, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    # Invert: raw 0=crack → 1, raw 255=bg → 0
    gt_bin = (gt < 128).astype(np.uint8)
    cl_bin = (cl < 128).astype(np.uint8)
    return gt_bin, cl_bin


# ==================== Model inference (from eval_8_2.py) ====================

def infer_smp(model_name, encoder, ckpt_path, img_dir, mask_dir, cl_dir, img_size=512):
    import segmentation_models_pytorch as smp_lib
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'unet':
        model = smp_lib.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    else:
        model = smp_lib.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    pred_list, gt_mask_list, gt_cl_list = [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc=f"{model_name}"):
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
            gt_bin, cl_bin = load_gt_and_cl(mask_dir, cl_dir, name, img_size)
            if gt_bin is None: continue
            pred_list.append(pred_np)
            gt_mask_list.append(gt_bin)
            gt_cl_list.append(cl_bin)
    return pred_list, gt_mask_list, gt_cl_list


def infer_dtrcnet(ckpt_path, img_dir, mask_dir, cl_dir, img_size=256):
    old_cwd = os.getcwd()
    os.chdir(os.path.join(BASE, 'DTrC-Net'))
    sys.path.insert(0, '.')
    from CTCNet import CTCNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CTCNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    pred_list, gt_mask_list, gt_cl_list = [], [], []
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
            gt_bin, cl_bin = load_gt_and_cl(mask_dir, cl_dir, name, img_size)
            if gt_bin is None: continue
            pred_list.append(pred_np)
            gt_mask_list.append(gt_bin)
            gt_cl_list.append(cl_bin)
    os.chdir(old_cwd)
    return pred_list, gt_mask_list, gt_cl_list


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
    pred_list, gt_mask_list, gt_cl_list = [], [], []
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
            gt_bin, cl_bin = load_gt_and_cl(mask_dir, cl_dir, name, img_size)
            if gt_bin is None: continue
            pred_list.append(pred_np)
            gt_mask_list.append(gt_bin)
            gt_cl_list.append(cl_bin)
    return pred_list, gt_mask_list, gt_cl_list


def infer_dbcnet(ckpt_path, img_dir, mask_dir, cl_dir, img_size=512):
    old_cwd = os.getcwd()
    dbc_dir = os.path.join(BASE, 'DBCNet')
    os.chdir(dbc_dir)
    if dbc_dir not in sys.path:
        sys.path.insert(0, dbc_dir)
    from configs.config_crack_Deepcrack import config
    from models.builder import EncoderDecoder as segmodel
    from utils.loss_crack import SoftDiceLoss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = segmodel(cfg=config, criterion=SoftDiceLoss(), norm_layer=nn.BatchNorm2d)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = {}
    for k, v in ckpt['model'].items():
        state_dict[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    pred_list, gt_mask_list, gt_cl_list = [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="DBCNet"):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x))
            pred_np = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
            gt_bin, cl_bin = load_gt_and_cl(mask_dir, cl_dir, name, img_size)
            if gt_bin is None: continue
            pred_list.append(pred_np)
            gt_mask_list.append(gt_bin)
            gt_cl_list.append(cl_bin)
    os.chdir(old_cwd)
    return pred_list, gt_mask_list, gt_cl_list


def load_saved_predictions(pred_dir, mask_dir, cl_dir, prefix, img_size=512):
    """Load pre-saved predictions (for our model)."""
    pred_list, gt_mask_list, gt_cl_list = [], [], []
    for name in tqdm(sorted(os.listdir(mask_dir)), desc="Ours"):
        if not name.endswith('.png'):
            continue
        # Find matching prediction
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(pred_path):
            continue
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        gt_bin, cl_bin = load_gt_and_cl(mask_dir, cl_dir, name, img_size)
        if gt_bin is None:
            continue
        # Resize prediction if needed
        if pred.shape != gt_bin.shape:
            pred = cv2.resize(pred, (gt_bin.shape[1], gt_bin.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        pred_list.append(pred)
        gt_mask_list.append(gt_bin)
        gt_cl_list.append(cl_bin)
    return pred_list, gt_mask_list, gt_cl_list


# ==================== Checkpoint registry (8:2) ====================

def find_best_dbcnet(log_dir):
    best, best_epoch = None, -1
    if not os.path.isdir(log_dir):
        return None
    for f in os.listdir(log_dir):
        if f.startswith('epoch-') and f.endswith('.pth') and not os.path.islink(os.path.join(log_dir, f)):
            try:
                epoch = int(f.split('-')[1].split('.')[0])
                if epoch > best_epoch:
                    best_epoch = epoch
                    best = os.path.join(log_dir, f)
            except ValueError:
                pass
    return best


def compute_cldice_fixed(pred_list, gt_mask_list, gt_cl_list, threshold=0.5):
    """Compute clDice at a fixed threshold."""
    cldices, tprecs, tsenses = [], [], []
    for pred, gt_mask, gt_cl in zip(pred_list, gt_mask_list, gt_cl_list):
        pred_bin = (pred / 255.0 > threshold).astype(np.uint8)
        cd, tp, ts = compute_cldice(pred_bin, gt_mask, gt_cl)
        cldices.append(cd)
        tprecs.append(tp)
        tsenses.append(ts)
    return {
        'clDice': round(float(np.mean(cldices)), 4),
        'Tprec': round(float(np.mean(tprecs)), 4),
        'Tsens': round(float(np.mean(tsenses)), 4),
        'threshold': round(float(threshold), 2),
        'n_images': len(cldices),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed threshold (default: sweep to find optimal)')
    args = parser.parse_args()

    USE_SWEEP = args.threshold is None
    FIXED_THRESH = args.threshold if args.threshold else 0.5

    ds = 'DeepCrack'
    img_dir = os.path.join(DS_BASE, ds, 'images', 'test')
    mask_dir = os.path.join(DS_BASE, ds, 'annotations', 'test')
    cl_dir = os.path.join(DS_BASE, ds, 'centerlines', 'test')

    n_test = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    print(f"DeepCrack 8:2 split — {n_test} test images")
    print(f"Centerlines: {cl_dir}\n")

    results = {}

    # --- U-Net ---
    p = os.path.join(BASE, 'ckpt_8_2', 'unet', ds, 'unet', 'best.pth')
    if os.path.exists(p):
        print(">>> U-Net")
        preds, gts, cls = infer_smp('unet', 'resnet34', p, img_dir, mask_dir, cl_dir, 512)
        results['U-Net'] = compute_cldice_fixed(preds, gts, cls, FIXED_THRESH) if not USE_SWEEP else compute_cldice_optimal(preds, gts, cls)
        print(f"    clDice={results['U-Net']['clDice']:.4f}  Tprec={results['U-Net']['Tprec']:.4f}  Tsens={results['U-Net']['Tsens']:.4f}  (thresh={results['U-Net']['threshold']:.2f})\n")

    # --- DeepLabV3+ ---
    p = os.path.join(BASE, 'ckpt_8_2', 'deeplabv3plus', ds, 'deeplabv3plus', 'best.pth')
    if os.path.exists(p):
        print(">>> DeepLabV3+")
        preds, gts, cls = infer_smp('deeplabv3plus', 'resnet34', p, img_dir, mask_dir, cl_dir, 512)
        results['DeepLabV3+'] = compute_cldice_fixed(preds, gts, cls, FIXED_THRESH) if not USE_SWEEP else compute_cldice_optimal(preds, gts, cls)
        print(f"    clDice={results['DeepLabV3+']['clDice']:.4f}  Tprec={results['DeepLabV3+']['Tprec']:.4f}  Tsens={results['DeepLabV3+']['Tsens']:.4f}  (thresh={results['DeepLabV3+']['threshold']:.2f})\n")

    # --- DTrC-Net ---
    p = os.path.join(BASE, 'ckpt_8_2', 'dtrcnet', ds, 'best.pth')
    if os.path.exists(p):
        print(">>> DTrC-Net")
        preds, gts, cls = infer_dtrcnet(p, img_dir, mask_dir, cl_dir, 256)
        results['DTrC-Net'] = compute_cldice_fixed(preds, gts, cls, FIXED_THRESH) if not USE_SWEEP else compute_cldice_optimal(preds, gts, cls)
        print(f"    clDice={results['DTrC-Net']['clDice']:.4f}  Tprec={results['DTrC-Net']['Tprec']:.4f}  Tsens={results['DTrC-Net']['Tsens']:.4f}  (thresh={results['DTrC-Net']['threshold']:.2f})\n")

    # --- DeepCrack ---
    ckpt_dir = os.path.join(BASE, 'DeepCrack', 'checkpoints', f'deepcrack_{ds}')
    if os.path.isdir(ckpt_dir):
        print(">>> DeepCrack")
        preds, gts, cls = infer_deepcrack_model(ckpt_dir, img_dir, mask_dir, cl_dir, 512)
        results['DeepCrack'] = compute_cldice_fixed(preds, gts, cls, FIXED_THRESH) if not USE_SWEEP else compute_cldice_optimal(preds, gts, cls)
        print(f"    clDice={results['DeepCrack']['clDice']:.4f}  Tprec={results['DeepCrack']['Tprec']:.4f}  Tsens={results['DeepCrack']['Tsens']:.4f}  (thresh={results['DeepCrack']['threshold']:.2f})\n")

    # --- DBCNet (subprocess isolation to avoid module conflicts) ---
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

def load_gt_cl(mask_dir, cl_dir, name, sz):
    bn = os.path.splitext(name)[0]
    for ext in [".png",".jpg",".bmp"]:
        gp = os.path.join(mask_dir, bn+ext)
        cp = os.path.join(cl_dir, bn+ext)
        if os.path.exists(gp) and os.path.exists(cp):
            g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            c = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
            if g is None or c is None: return None,None
            g = cv2.resize(g,(sz,sz),interpolation=cv2.INTER_NEAREST)
            c = cv2.resize(c,(sz,sz),interpolation=cv2.INTER_NEAREST)
            return (g<128).astype(np.uint8), (c<128).astype(np.uint8)
    return None,None

preds, gts_m, gts_c = [], [], []
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
        gb, cb = load_gt_cl(mask_dir, cl_dir, name, 512)
        if gb is None: continue
        preds.append(pred_np); gts_m.append(gb); gts_c.append(cb)

USE_SWEEP = {USE_SWEEP}
FIXED_T = {FIXED_THRESH}
best_t = FIXED_T
if USE_SWEEP:
    best_cd = -1
    for t in np.arange(0.05, 0.95, 0.01):
        cds = []
        for p, gm, gc in zip(preds, gts_m, gts_c):
            pb = (p/255.0 > t).astype(np.uint8)
            cs = gc.sum()
            ts = (gc*pb).sum()/cs if cs>0 else 1.0
            sk = skeletonize(pb.astype(bool)).astype(np.float64)
            ss = sk.sum()
            tp = (sk*gm).sum()/ss if ss>0 else (0.0 if gm.sum()>0 else 1.0)
            cd = 2*tp*ts/(tp+ts) if tp+ts>0 else 0
            cds.append(cd)
        m = np.mean(cds)
        if m > best_cd: best_cd=m; best_t=t

cds,tps,tss = [],[],[]
for p, gm, gc in zip(preds, gts_m, gts_c):
    pb = (p/255.0 > best_t).astype(np.uint8)
    cs = gc.sum()
    ts = (gc*pb).sum()/cs if cs>0 else 1.0
    sk = skeletonize(pb.astype(bool)).astype(np.float64)
    ss = sk.sum()
    tp = (sk*gm).sum()/ss if ss>0 else (0.0 if gm.sum()>0 else 1.0)
    cd = 2*tp*ts/(tp+ts) if tp+ts>0 else 0
    cds.append(cd); tps.append(tp); tss.append(ts)

print(json.dumps({{"clDice":round(float(np.mean(cds)),4),"Tprec":round(float(np.mean(tps)),4),"Tsens":round(float(np.mean(tss)),4),"threshold":round(float(best_t),2),"n_images":len(cds)}}))
'''
        try:
            import subprocess
            r = subprocess.run([sys.executable, '-c', dbcnet_code],
                               capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                for line in r.stdout.strip().split('\n'):
                    if line.startswith('{'):
                        results['DBCNet'] = json.loads(line)
                        print(f"    clDice={results['DBCNet']['clDice']:.4f}  Tprec={results['DBCNet']['Tprec']:.4f}  Tsens={results['DBCNet']['Tsens']:.4f}  (thresh={results['DBCNet']['threshold']:.2f})\n")
                        break
            else:
                print(f"    ERROR: {r.stderr[-300:]}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # --- Ours (from saved predictions) ---
    our_pred_dir = os.path.join(PROJECT_ROOT, 'paper', 'InferenceResult', 'CrackMambaTransformerNet')
    if os.path.isdir(our_pred_dir):
        print(">>> Ours")
        preds, gts, cls = load_saved_predictions(our_pred_dir, mask_dir, cl_dir, 'DeepCrack', 512)
        results['Ours'] = compute_cldice_fixed(preds, gts, cls, FIXED_THRESH) if not USE_SWEEP else compute_cldice_optimal(preds, gts, cls)
        print(f"    clDice={results['Ours']['clDice']:.4f}  Tprec={results['Ours']['Tprec']:.4f}  Tsens={results['Ours']['Tsens']:.4f}  (thresh={results['Ours']['threshold']:.2f})\n")

    # Summary
    print("=" * 70)
    print(f"{'Model':<15s} {'clDice':>8s} {'Tprec':>8s} {'Tsens':>8s} {'thresh':>8s} {'n':>5s}")
    print("-" * 70)
    order = ['U-Net', 'DBCNet', 'DeepLabV3+', 'Ours', 'DTrC-Net', 'DeepCrack']
    for name in order:
        if name in results:
            r = results[name]
            print(f"{name:<15s} {r['clDice']:>8.4f} {r['Tprec']:>8.4f} "
                  f"{r['Tsens']:>8.4f} {r['threshold']:>8.2f} {r['n_images']:>5d}")
    print("=" * 70)

    # Save
    out_path = os.path.join(PROJECT_ROOT, 'paper', 'cldice_results_DeepCrack_8_2.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
