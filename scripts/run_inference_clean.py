#!/usr/bin/env python3
"""Run inference for models that don't have import conflicts: UNet, DeepLabV3+, DeepCrack_Model, Ours."""
import os, sys, cv2, torch, numpy as np
from tqdm import tqdm
from torchvision import transforms
import segmentation_models_pytorch as smp

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from train.Trainer import LitCrackModel
from data.calculateMeanStd import compute_mean_std_from_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DS = f'{PROJECT}/ComparisonModel/dataset_8_2'
CFD = '/home/andy/desktop/CrackDectection/dataset/omnicrack30k_classified/CFD'
OUT = f'{PROJECT}/paper/InferenceResult'
C = f'{PROJECT}/ComparisonModel'

imn = lambda img: transforms.functional.normalize(
    torch.from_numpy(img).permute(2,0,1).float()/255.0,
    mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
hn = lambda img: (torch.from_numpy(img).permute(2,0,1).float()/255.0 - 0.5) / 0.5

def need(ds, md):
    d = f'{OUT}/{ds}/{md}'
    return not os.path.isdir(d) or len([f for f in os.listdir(d) if f.endswith('.png')]) == 0

def inf(model, idir, sdir, nf, desc, sz=512):
    os.makedirs(sdir, exist_ok=True)
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(idir)), desc=desc):
            img = cv2.imread(os.path.join(idir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)
            x = nf(img).unsqueeze(0).to(device)
            out = model(x)
            if isinstance(out, (tuple, list)): out = out[-1]
            if isinstance(out, dict): out = out['main_output']
            pred = (torch.sigmoid(out)[0,0].cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite(os.path.join(sdir, os.path.splitext(name)[0]+'.png'), pred)

def inf_ours(idir, sdir, ckpt, tdir, desc):
    os.makedirs(sdir, exist_ok=True)
    mc = Config.MODEL_CONFIGS
    sb = Config.expand_stage_blocks(mc['depths'], len(mc['widths']))
    model = TriBranchCrackNet(
        in_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES,
        widths=mc['widths'], stage_blocks=sb, k=mc['k'], d_state=mc['d_state'],
        dropout=mc['dropout'], attn_drop=mc['attn_drop'], proj_drop=mc['proj_drop'],
        drop_path_rate=mc['drop_path_rate'],
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
    ck = torch.load(ckpt, map_location=device)
    lit.load_state_dict(ck['state_dict'], strict=False)
    lit.to(device).eval()
    m_, s_ = compute_mean_std_from_dir(tdir)
    mean, std = list(m_), list(s_)
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(idir)), desc=desc):
            img = cv2.imread(os.path.join(idir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(img).permute(2,0,1).float()/255.0
            for c in range(3): x[c] = (x[c]-mean[c])/std[c]
            x = x.unsqueeze(0).to(device)
            o0 = lit(x); l = o0['main_output'] if isinstance(o0,dict) else o0
            for flip in [[3],[2],[2,3]]:
                xf = torch.flip(x, flip); of = lit(xf)
                lf = of['main_output'] if isinstance(of,dict) else of
                l = l + torch.flip(lf, flip)
            pred = (torch.sigmoid(l/4.0)[0,0].cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite(os.path.join(sdir, os.path.splitext(name)[0]+'.png'), pred)

# CrackTree260
for ds_name in ['CrackTree260']:
    idir = f'{DS}/{ds_name}/test/images'
    if need(ds_name, 'UNet'):
        m = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(f'{C}/ckpt_8_2/unet/{ds_name}/unet/best.pth', map_location=device))
        m.to(device).eval(); inf(m, idir, f'{OUT}/{ds_name}/UNet', imn, f'UNet {ds_name}'); del m
    if need(ds_name, 'DeepLabV3Plus'):
        m = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(f'{C}/ckpt_8_2/deeplabv3plus/{ds_name}/deeplabv3plus/best.pth', map_location=device))
        m.to(device).eval(); inf(m, idir, f'{OUT}/{ds_name}/DeepLabV3Plus', imn, f'DLV3+ {ds_name}'); del m
    if need(ds_name, 'DeepCrack_Model'):
        sys.path.insert(0, f'{C}/DeepCrack')
        from models.deepcrack_networks import define_deepcrack
        m = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[])
        m.load_state_dict(torch.load(f'{C}/DeepCrack/checkpoints/deepcrack_{ds_name}/latest_net_G.pth', map_location=device))
        m.to(device).eval(); inf(m, idir, f'{OUT}/{ds_name}/DeepCrack_Model', hn, f'DC {ds_name}'); del m

# CFD
cfd_img = f'{CFD}/images/test'
if need('CFD', 'UNet'):
    m = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(f'{C}/ckpt_8_2/unet/DeepCrack/unet/best.pth', map_location=device))
    m.to(device).eval(); inf(m, cfd_img, f'{OUT}/CFD/UNet', imn, 'UNet CFD'); del m
if need('CFD', 'DeepLabV3Plus'):
    m = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(f'{C}/ckpt_8_2/deeplabv3plus/DeepCrack/deeplabv3plus/best.pth', map_location=device))
    m.to(device).eval(); inf(m, cfd_img, f'{OUT}/CFD/DeepLabV3Plus', imn, 'DLV3+ CFD'); del m
if need('CFD', 'DeepCrack_Model'):
    sys.path.insert(0, f'{C}/DeepCrack')
    from models.deepcrack_networks import define_deepcrack
    m = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[])
    m.load_state_dict(torch.load(f'{C}/DeepCrack/checkpoints/deepcrack_DeepCrack/latest_net_G.pth', map_location=device))
    m.to(device).eval(); inf(m, cfd_img, f'{OUT}/CFD/DeepCrack_Model', hn, 'DC CFD'); del m

# Ours
ours = {
    'DeepCrack': (f'{PROJECT}/checkpoints/ours_8_2/DeepCrack/best-epoch=196-val_iou=0.7923.ckpt', f'{DS}/DeepCrack/train/images'),
    'CrackTree260': (f'{PROJECT}/checkpoints/ours_8_2/CrackTree260/best-epoch=199-val_iou=0.5890.ckpt', f'{DS}/CrackTree260/train/images'),
}
for ds_name, (ckpt, tdir) in ours.items():
    if need(ds_name, 'Ours'):
        inf_ours(f'{DS}/{ds_name}/test/images', f'{OUT}/{ds_name}/Ours', ckpt, tdir, f'Ours {ds_name}')
if need('CFD', 'Ours'):
    inf_ours(cfd_img, f'{OUT}/CFD/Ours',
             f'{PROJECT}/checkpoints/ours_8_2/DeepCrack/best-epoch=196-val_iou=0.7923.ckpt',
             f'{DS}/DeepCrack/train/images', 'Ours CFD')

print('\nDone.')
