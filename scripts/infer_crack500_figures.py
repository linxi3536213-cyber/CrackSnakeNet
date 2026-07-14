#!/usr/bin/env python3
"""Generate CRACK500_resize inference images for paper comparison figure."""
import os, sys, cv2, torch, numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
COMP = os.path.join(PROJECT, 'ComparisonModel')
DS_BASE = os.path.join(COMP, 'dataset_8_2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIR = os.path.join(PROJECT, 'paper', 'latex', 'figures', 'fig_crack500')
os.makedirs(OUT_DIR, exist_ok=True)

# Selected images: one with dense cracks, one medium, one thin
SELECTED = [
    'CRACK500_20160222_164141',  # dense cracks 6.95%
    'CRACK500_20160329_094003',  # medium 2.28%
    'CRACK500_20160308_073310',  # thin 2.21%
]

ds = 'CRACK500_resize'
img_dir = os.path.join(DS_BASE, ds, 'test', 'images')
mask_dir = os.path.join(DS_BASE, ds, 'test', 'annotations')
train_img_dir = os.path.join(DS_BASE, ds, 'train', 'images')

def compute_mean_std(d):
    means, stds = [], []
    for name in sorted(os.listdir(d)):
        img = cv2.imread(os.path.join(d, name))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        means.append(img.mean(axis=(0,1))); stds.append(img.std(axis=(0,1)))
    return list(np.mean(means, axis=0)), list(np.mean(stds, axis=0))

mean, std = compute_mean_std(train_img_dir)

# Save input and GT
for img_name in SELECTED:
    for ext in ['.png', '.jpg']:
        p = os.path.join(img_dir, img_name + ext)
        if os.path.exists(p):
            img = cv2.imread(p)
            cv2.imwrite(os.path.join(OUT_DIR, f'{img_name}_input.png'), img)
            break
    for ext in ['.png', '.jpg']:
        gp = os.path.join(mask_dir, img_name + ext)
        if os.path.exists(gp):
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            gt_vis = np.zeros_like(gt); gt_vis[gt < 128] = 255
            cv2.imwrite(os.path.join(OUT_DIR, f'{img_name}_gt.png'), gt_vis)
            break

def save_preds(model, norm_fn, label, sz=512, is_dtrcnet=False):
    model.eval()
    for img_name in SELECTED:
        for ext in ['.png', '.jpg']:
            p = os.path.join(img_dir, img_name + ext)
            if not os.path.exists(p): continue
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if is_dtrcnet:
                img_r = cv2.resize(img, (256, 256))
            else:
                img_r = cv2.resize(img, (sz, sz))
            x = norm_fn(img_r).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_dtrcnet:
                    logits, _ = model(x)
                    prob = torch.sigmoid(logits)
                else:
                    out = model(x)
                    if isinstance(out, (tuple, list)): out = out[-1]
                    if isinstance(out, dict): out = out['main_output']
                    prob = torch.sigmoid(out)
            pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            if is_dtrcnet:
                pred = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(OUT_DIR, f'{img_name}_{label}.png'), pred)
            break
    print(f'  {label} done')

imagenet_norm = lambda img: transforms.functional.normalize(
    torch.from_numpy(img).permute(2,0,1).float()/255.0,
    mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
half_norm = lambda img: (torch.from_numpy(img).permute(2,0,1).float()/255.0 - 0.5) / 0.5
ours_norm = lambda img: (lambda x: torch.stack([(x[c]-mean[c])/std[c] for c in range(3)]))(
    torch.from_numpy(img).permute(2,0,1).float()/255.0)

# 1. CrackSnakeNet (TTA)
print(">>> CrackSnakeNet")
from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from train.Trainer import LitCrackModel
mc = Config.MODEL_CONFIGS
stage_blocks = Config.expand_stage_blocks(mc['depths'], len(mc['widths']))
model = TriBranchCrackNet(in_channels=3, num_classes=1, widths=mc['widths'], stage_blocks=stage_blocks,
    k=mc['k'], d_state=mc['d_state'], dropout=mc['dropout'], attn_drop=mc['attn_drop'],
    proj_drop=mc['proj_drop'], drop_path_rate=mc['drop_path_rate'],
    first_downsample_stride=Config.ENCODER_FIRST_DOWNSAMPLE_STRIDE,
    use_aux_outputs=Config.USE_AUX_OUTPUTS, aux_stage_indices=Config.AUX_STAGE_INDICES,
    use_edge_head=Config.USE_EDGE_HEAD, scan_directions=mc.get('scan_directions', 8))
lit = LitCrackModel(model=model, loss_config=Config.LOSS_CONFIG,
    lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
    max_epochs=Config.EPOCHS, warmup_epochs=Config.LR_WARMUP_EPOCHS,
    min_lr=Config.LR_MIN, edge_loss_weight=Config.EDGE_LOSS_WEIGHT,
    edge_aux_loss_weight=Config.EDGE_AUX_LOSS_WEIGHT,
    edge_threshold=Config.EDGE_THRESHOLD, metric_threshold=Config.METRIC_THRESHOLD,
    use_cutmix=Config.USE_CUTMIX, cutmix_alpha=Config.CUTMIX_ALPHA,
    cutmix_prob=Config.CUTMIX_PROB, centerline_loss_weight=Config.CENTERLINE_LOSS_WEIGHT,
    boundary_loss_weight=0.0)
ckpt = torch.load(os.path.join(PROJECT, 'checkpoints/ours_8_2/CRACK500_resize/best-epoch=153-val_iou=0.5687.ckpt'), map_location=device)
lit.load_state_dict(ckpt['state_dict'], strict=False)
lit.to(device).eval()
for img_name in SELECTED:
    for ext in ['.png', '.jpg']:
        p = os.path.join(img_dir, img_name + ext)
        if not os.path.exists(p): continue
        img = cv2.imread(p); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(img).permute(2,0,1).float()/255.0
        for c in range(3): x[c] = (x[c]-mean[c])/std[c]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            o0=lit(x);l=o0['main_output'] if isinstance(o0,dict) else o0
            o1=lit(torch.flip(x,[3]));l=l+torch.flip(o1['main_output'] if isinstance(o1,dict) else o1,[3])
            o2=lit(torch.flip(x,[2]));l=l+torch.flip(o2['main_output'] if isinstance(o2,dict) else o2,[2])
            o3=lit(torch.flip(x,[2,3]));l=l+torch.flip(o3['main_output'] if isinstance(o3,dict) else o3,[2,3])
        pred = (torch.sigmoid(l/4.0)[0,0].cpu().numpy()>0.5).astype(np.uint8)*255
        cv2.imwrite(os.path.join(OUT_DIR, f'{img_name}_Ours.png'), pred)
        break
print("  Ours done")
del lit; torch.cuda.empty_cache()

# 2. U-Net
print(">>> U-Net")
import segmentation_models_pytorch as smp
m = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
m.load_state_dict(torch.load(os.path.join(COMP, 'ckpt_8_2/unet/CRACK500_resize/unet/unet/best.pth'), map_location=device))
m.to(device)
save_preds(m, imagenet_norm, 'UNet')
del m; torch.cuda.empty_cache()

# 3. DeepLabV3+
print(">>> DeepLabV3+")
m = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
m.load_state_dict(torch.load(os.path.join(COMP, 'ckpt_8_2/deeplabv3plus/CRACK500_resize/deeplabv3plus/deeplabv3plus/best.pth'), map_location=device))
m.to(device)
save_preds(m, imagenet_norm, 'DeepLabV3Plus')
del m; torch.cuda.empty_cache()

# 4. DeepCrack model
print(">>> DeepCrack")
sys.path.insert(0, os.path.join(COMP, 'DeepCrack'))
from models.deepcrack_networks import define_deepcrack
m = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[])
m.load_state_dict(torch.load(os.path.join(COMP, 'DeepCrack/checkpoints/deepcrack_CRACK500_resize/latest_net_G.pth'), map_location=device))
m.to(device)
save_preds(m, half_norm, 'DeepCrack_Model')
del m; torch.cuda.empty_cache()

# 5. DBCNet
print(">>> DBCNet")
old_cwd = os.getcwd()
os.chdir(os.path.join(COMP, 'DBCNet'))
sys.path.insert(0, '.')
from configs.config_crack_CRACK500 import config
from models.builder import EncoderDecoder as segmodel
from utils.loss_crack import SoftDiceLoss
m = segmodel(cfg=config, criterion=SoftDiceLoss(), norm_layer=nn.BatchNorm2d)
ckpt = torch.load('log_final/log_CRACK500/log_CRACK500_resize_my_backbone_2/checkpoint/epoch-200.pth', map_location=device)
sd = {(k[7:] if k.startswith('module.') else k): v for k,v in ckpt['model'].items()}
m.load_state_dict(sd, strict=True)
m.to(device)
os.chdir(old_cwd)
raw_norm = lambda img: torch.from_numpy(img).permute(2,0,1).float()/255.0
save_preds(m, raw_norm, 'DBCNet')
del m; torch.cuda.empty_cache()

# 6. SCSegamba (use saved eval predictions or skip - needs different env)
# We'll generate blank placeholders if needed
print(">>> SCSegamba (needs SCSegamba env, skip inference)")

# 7. DTrC-Net
print(">>> DTrC-Net")
os.chdir(os.path.join(COMP, 'DTrC-Net'))
sys.path.insert(0, '.')
from CTCNet import CTCNet
m = CTCNet(n_channels=3, n_classes=1).to(device)
m.load_state_dict(torch.load(os.path.join(COMP, 'ckpt_8_2/dtrcnet/CRACK500_resize/best.pth'), map_location=device))
save_preds(m, imagenet_norm, 'DTrCNet', is_dtrcnet=True)
os.chdir(old_cwd)
del m; torch.cuda.empty_cache()

# 8. CT-CrackSeg
print(">>> CT-CrackSeg")
sys.path.insert(0, os.path.join(COMP, 'CT-CrackSeg'))
from model.TransMUNet import TransMUNet
m = TransMUNet(n_classes=1).to(device)
ckpt = torch.load(os.path.join(COMP, 'CT-CrackSeg/checkpoints/weights_crack500.model'), map_location=device)
m.load_state_dict(ckpt['model_weights'])
ct_norm = lambda img: torch.from_numpy(img).permute(2,0,1).float()/255.0
for img_name in SELECTED:
    for ext in ['.png', '.jpg']:
        p = os.path.join(img_dir, img_name + ext)
        if not os.path.exists(p): continue
        img = cv2.imread(p); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        x = ct_norm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = m(x)
            if isinstance(out, tuple): out = out[0]
        pred = (torch.sigmoid(out)[0,0].cpu().numpy()>0.5).astype(np.uint8)*255
        pred = cv2.resize(pred, (512,512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT_DIR, f'{img_name}_CTCrackSeg.png'), pred)
        break
print("  CTCrackSeg done")
del m; torch.cuda.empty_cache()

print(f'\nAll images saved to {OUT_DIR}')
print(f'Files: {len(os.listdir(OUT_DIR))}')
