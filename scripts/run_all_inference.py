#!/usr/bin/env python3
"""
Generate prediction images for ALL models on DeepCrack, CrackTree260, and CFD (8:2 split).
Saves to paper/InferenceResult/{dataset}/{model}/*.png
SCSegamba predictions are copied from existing results (already 8:2).
"""
import os, sys, cv2, torch, numpy as np
from tqdm import tqdm
import torch.nn as nn

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from train.Trainer import LitCrackModel
from data.calculateMeanStd import compute_mean_std_from_dir
from torchvision import transforms
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DS_BASE = os.path.join(PROJECT, 'ComparisonModel', 'dataset_8_2')
CFD_BASE = '/home/andy/desktop/CrackDectection/dataset/omnicrack30k_classified/CFD'
OUT_BASE = os.path.join(PROJECT, 'paper', 'InferenceResult')
COMP = os.path.join(PROJECT, 'ComparisonModel')

imagenet_norm = lambda img: transforms.functional.normalize(
    torch.from_numpy(img).permute(2,0,1).float()/255.0,
    mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
half_norm = lambda img: (torch.from_numpy(img).permute(2,0,1).float()/255.0 - 0.5) / 0.5
raw_norm = lambda img: torch.from_numpy(img).permute(2,0,1).float()/255.0


def save_pred(pred_np, save_path):
    cv2.imwrite(save_path, pred_np)


def infer_and_save(model, img_dir, save_dir, norm_fn, desc, sz=512, is_dtrcnet=False):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc=desc):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if is_dtrcnet:
                img_r = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
                x = norm_fn(img_r).unsqueeze(0).to(device)
                logits, _ = model(x)
                pred_np = (torch.sigmoid(logits)[0,0].cpu().numpy()*255).astype(np.uint8)
                pred_np = cv2.resize(pred_np, (sz,sz), interpolation=cv2.INTER_LINEAR)
            else:
                img_r = cv2.resize(img, (sz,sz), interpolation=cv2.INTER_CUBIC)
                x = norm_fn(img_r).unsqueeze(0).to(device)
                out = model(x)
                if isinstance(out, (tuple,list)): out = out[-1]
                if isinstance(out, dict): out = out['main_output']
                prob = torch.sigmoid(out)
                pred_np = (prob[0,0].cpu().numpy()*255).astype(np.uint8)
            bn = os.path.splitext(name)[0]
            save_pred(pred_np, os.path.join(save_dir, f'{bn}.png'))


def infer_ours(img_dir, save_dir, ckpt_path, train_img_dir, desc):
    """Ours with TTA"""
    os.makedirs(save_dir, exist_ok=True)
    model_config = Config.MODEL_CONFIGS
    stage_blocks = Config.expand_stage_blocks(model_config['depths'], len(model_config['widths']))
    model = TriBranchCrackNet(
        in_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES,
        widths=model_config['widths'], stage_blocks=stage_blocks,
        k=model_config['k'], d_state=model_config['d_state'],
        dropout=model_config['dropout'], attn_drop=model_config['attn_drop'],
        proj_drop=model_config['proj_drop'], drop_path_rate=model_config['drop_path_rate'],
        first_downsample_stride=Config.ENCODER_FIRST_DOWNSAMPLE_STRIDE,
        use_aux_outputs=Config.USE_AUX_OUTPUTS, aux_stage_indices=Config.AUX_STAGE_INDICES,
        use_edge_head=Config.USE_EDGE_HEAD, scan_directions=model_config.get('scan_directions',8))
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
    m_, s_ = compute_mean_std_from_dir(train_img_dir)
    mean, std = list(m_), list(s_)

    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc=desc):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(img).permute(2,0,1).float()/255.0
            for c in range(3): x[c] = (x[c]-mean[c])/std[c]
            x = x.unsqueeze(0).to(device)
            o0 = lit(x); l = o0['main_output'] if isinstance(o0,dict) else o0
            o1 = lit(torch.flip(x,[3])); l = l + torch.flip(o1['main_output'] if isinstance(o1,dict) else o1,[3])
            o2 = lit(torch.flip(x,[2])); l = l + torch.flip(o2['main_output'] if isinstance(o2,dict) else o2,[2])
            o3 = lit(torch.flip(x,[2,3])); l = l + torch.flip(o3['main_output'] if isinstance(o3,dict) else o3,[2,3])
            prob = torch.sigmoid(l/4.0)
            pred_np = (prob[0,0].cpu().numpy()*255).astype(np.uint8)
            bn = os.path.splitext(name)[0]
            save_pred(pred_np, os.path.join(save_dir, f'{bn}.png'))


# ============================================================
# Dataset configs
# ============================================================
DATASETS = {
    'DeepCrack': {
        'img_dir': f'{DS_BASE}/DeepCrack/test/images',
        'ours_ckpt': f'{PROJECT}/checkpoints/ours_8_2/DeepCrack/best-epoch=196-val_iou=0.7923.ckpt',
        'ours_train': f'{DS_BASE}/DeepCrack/train/images',
    },
    'CrackTree260': {
        'img_dir': f'{DS_BASE}/CrackTree260/test/images',
        'ours_ckpt': f'{PROJECT}/checkpoints/ours_8_2/CrackTree260/best-epoch=199-val_iou=0.5890.ckpt',
        'ours_train': f'{DS_BASE}/CrackTree260/train/images',
    },
}

CFD_CONFIG = {
    'img_dir': f'{CFD_BASE}/images/test',
    'ours_ckpt': f'{PROJECT}/checkpoints/ours_8_2/DeepCrack/best-epoch=196-val_iou=0.7923.ckpt',
    'ours_train': f'{DS_BASE}/DeepCrack/train/images',
}


if __name__ == '__main__':
    # ============================================================
    # DeepCrack + CrackTree260
    # ============================================================
    for ds_name, ds_cfg in DATASETS.items():
        img_dir = ds_cfg['img_dir']
        print(f'\n{"="*60}')
        print(f'  {ds_name} ({len(os.listdir(img_dir))} test images)')
        print(f'{"="*60}')

        # U-Net
        m = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(f'{COMP}/ckpt_8_2/unet/{ds_name}/unet/best.pth', map_location=device))
        m.to(device).eval()
        infer_and_save(m, img_dir, f'{OUT_BASE}/{ds_name}/UNet', imagenet_norm, f'UNet {ds_name}')
        del m

        # DeepLabV3+
        m = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(f'{COMP}/ckpt_8_2/deeplabv3plus/{ds_name}/deeplabv3plus/best.pth', map_location=device))
        m.to(device).eval()
        infer_and_save(m, img_dir, f'{OUT_BASE}/{ds_name}/DeepLabV3Plus', imagenet_norm, f'DeepLabV3+ {ds_name}')
        del m

        # DeepCrack Model
        sys.path.insert(0, f'{COMP}/DeepCrack')
        from models.deepcrack_networks import define_deepcrack
        m = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[])
        m.load_state_dict(torch.load(f'{COMP}/DeepCrack/checkpoints/deepcrack_{ds_name}/latest_net_G.pth', map_location=device))
        m.to(device).eval()
        infer_and_save(m, img_dir, f'{OUT_BASE}/{ds_name}/DeepCrack_Model', half_norm, f'DeepCrack {ds_name}')
        del m

        # DTrC-Net
        old_cwd = os.getcwd()
        os.chdir(f'{COMP}/DTrC-Net')
        sys.path.insert(0, '.')
        from CTCNet import CTCNet
        m = CTCNet(n_channels=3, n_classes=1).to(device)
        m.load_state_dict(torch.load(f'{COMP}/ckpt_8_2/dtrcnet/{ds_name}/best.pth', map_location=device))
        m.eval()
        infer_and_save(m, img_dir, f'{OUT_BASE}/{ds_name}/DTrCNet', imagenet_norm, f'DTrCNet {ds_name}', is_dtrcnet=True)
        del m
        os.chdir(old_cwd)

        # SCSegamba - copy existing 8:2 predictions
        sc_src = f'{COMP}/SCSegamba/results/pred_{ds_name}_82'
        sc_dst = f'{OUT_BASE}/{ds_name}/SCSegamba'
        if os.path.isdir(sc_src):
            print(f'  Copying SCSegamba {ds_name} from existing predictions...')
            for f in sorted(os.listdir(sc_src)):
                if f.endswith('_pre.png'):
                    bn = f.replace('_pre.png', '')
                    src = os.path.join(sc_src, f)
                    dst = os.path.join(sc_dst, f'{bn}.png')
                    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        cv2.imwrite(dst, img)
            print(f'  Copied {len(os.listdir(sc_dst))} SCSegamba predictions')

        # Ours (with TTA)
        infer_ours(img_dir, f'{OUT_BASE}/{ds_name}/Ours',
                   ds_cfg['ours_ckpt'], ds_cfg['ours_train'], f'Ours {ds_name}')

    # ============================================================
    # CFD (cross-domain: DeepCrack-trained models)
    # ============================================================
    img_dir = CFD_CONFIG['img_dir']
    print(f'\n{"="*60}')
    print(f'  CFD ({len(os.listdir(img_dir))} test images) - cross-domain')
    print(f'{"="*60}')

    # U-Net (DeepCrack weights)
    m = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(f'{COMP}/ckpt_8_2/unet/DeepCrack/unet/best.pth', map_location=device))
    m.to(device).eval()
    infer_and_save(m, img_dir, f'{OUT_BASE}/CFD/UNet', imagenet_norm, 'UNet CFD')
    del m

    # DeepLabV3+ (DeepCrack weights)
    m = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(f'{COMP}/ckpt_8_2/deeplabv3plus/DeepCrack/deeplabv3plus/best.pth', map_location=device))
    m.to(device).eval()
    infer_and_save(m, img_dir, f'{OUT_BASE}/CFD/DeepLabV3Plus', imagenet_norm, 'DeepLabV3+ CFD')
    del m

    # DeepCrack Model (DeepCrack weights)
    sys.path.insert(0, f'{COMP}/DeepCrack')
    from models.deepcrack_networks import define_deepcrack
    m = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[])
    m.load_state_dict(torch.load(f'{COMP}/DeepCrack/checkpoints/deepcrack_DeepCrack/latest_net_G.pth', map_location=device))
    m.to(device).eval()
    infer_and_save(m, img_dir, f'{OUT_BASE}/CFD/DeepCrack_Model', half_norm, 'DeepCrack CFD')
    del m

    # DBCNet (DeepCrack weights)
    old_cwd = os.getcwd()
    os.chdir(f'{COMP}/DBCNet')
    sys.path.insert(0, f'{COMP}/DBCNet')
    from configs.config_crack_Deepcrack import config
    from models.builder import EncoderDecoder as segmodel
    from utils.loss_crack import SoftDiceLoss
    m = segmodel(cfg=config, criterion=SoftDiceLoss(), norm_layer=nn.BatchNorm2d)
    ckpt = torch.load('log_final/log_Deepcrack/log_Deepcrack_my_backbone_7/checkpoint/epoch-147.pth', map_location=device)
    sd = {(k[7:] if k.startswith('module.') else k): v for k,v in ckpt['model'].items()}
    m.load_state_dict(sd, strict=True); m.to(device).eval()
    infer_and_save(m, img_dir, f'{OUT_BASE}/CFD/DBCNet', raw_norm, 'DBCNet CFD')
    del m
    os.chdir(old_cwd)

    # Ours (DeepCrack weights, TTA)
    infer_ours(img_dir, f'{OUT_BASE}/CFD/Ours',
               CFD_CONFIG['ours_ckpt'], CFD_CONFIG['ours_train'], 'Ours CFD')

    # ============================================================
    # DBCNet for DeepCrack and CrackTree260 (needs to run from DBCNet dir)
    # ============================================================
    old_cwd = os.getcwd()
    os.chdir(f'{COMP}/DBCNet')
    sys.path.insert(0, f'{COMP}/DBCNet')

    dbcnet_ckpts = {
        'DeepCrack': 'log_final/log_Deepcrack/log_Deepcrack_my_backbone_7/checkpoint/epoch-147.pth',
        'CrackTree260': 'log_final/log_Deepcrack/log_Deepcrack_my_backbone_9/checkpoint/epoch-158.pth',
    }
    for ds_name, ckpt_path in dbcnet_ckpts.items():
        m = segmodel(cfg=config, criterion=SoftDiceLoss(), norm_layer=nn.BatchNorm2d)
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = {(k[7:] if k.startswith('module.') else k): v for k,v in ckpt['model'].items()}
        m.load_state_dict(sd, strict=True); m.to(device).eval()
        infer_and_save(m, f'{DS_BASE}/{ds_name}/test/images',
                       f'{OUT_BASE}/{ds_name}/DBCNet', raw_norm, f'DBCNet {ds_name}')
        del m
    os.chdir(old_cwd)

    print(f'\n{"="*60}')
    print('  ALL INFERENCE COMPLETE!')
    print(f'{"="*60}')
    for ds in ['DeepCrack', 'CrackTree260', 'CFD']:
        ds_dir = f'{OUT_BASE}/{ds}'
        if os.path.isdir(ds_dir):
            for model_dir in sorted(os.listdir(ds_dir)):
                n = len(os.listdir(os.path.join(ds_dir, model_dir)))
                print(f'  {ds}/{model_dir}: {n} images')
