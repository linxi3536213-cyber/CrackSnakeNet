#!/usr/bin/env python3
"""
Evaluate ALL models on CRACK500_resize test set (199 images, 512x512).
Also saves inference images for paper figures.
Metrics: mIoU, ODS, OIS, P, R, F1, clDice, Tprec, Tsens
"""
import os, sys, cv2, torch, json
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from skimage.morphology import skeletonize

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
COMP = os.path.join(PROJECT, 'ComparisonModel')
DS_BASE = os.path.join(COMP, 'dataset_8_2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== Metrics =====================
def get_stat(p,g): return int(np.sum((p==1)&(g==1))),int(np.sum((p==1)&(g==0))),int(np.sum((p==0)&(g==1)))
def cal_mIoU(preds,gts,step=0.01):
    best=0
    for t in np.arange(0,1,step):
        ious=[]
        for p,g in zip(preds,gts):
            gi=(g/255).astype('uint8');pi=(p/255>t).astype('uint8')
            TP=np.sum((pi==1)&(gi==1));TN=np.sum((pi==0)&(gi==0))
            FP=np.sum((pi==1)&(gi==0));FN=np.sum((pi==0)&(gi==1))
            ious.append((TP/(FN+FP+TP)+TN/(FN+FP+TN))/2 if (FN+FP+TP)>0 else 0)
        best=max(best,np.mean(ious))
    return best
def cal_ODS(preds,gts,step=0.01):
    best=0
    for t in np.arange(0,1,step):
        f1s=[]
        for p,g in zip(preds,gts):
            gi=(g/255).astype('uint8');pi=(p/255>t).astype('uint8');tp,fp,fn=get_stat(pi,gi)
            pr=1.0 if tp==0 and fp==0 else tp/(tp+fp);rc=tp/(tp+fn) if tp+fn>0 else 0
            f1s.append(0 if pr+rc==0 else 2*pr*rc/(pr+rc))
        best=max(best,np.mean(f1s))
    return best
def cal_OIS(preds,gts,step=0.01):
    bfs=[]
    for p,g in zip(preds,gts):
        bf=0
        for t in np.arange(0,1,step):
            gi=(g/255).astype('uint8');pi=(p/255>t).astype('uint8');tp,fp,fn=get_stat(pi,gi)
            pr=1.0 if tp==0 and fp==0 else tp/(tp+fp);rc=tp/(tp+fn) if tp+fn>0 else 0
            bf=max(bf,0 if pr+rc==0 else 2*pr*rc/(pr+rc))
        bfs.append(bf)
    return np.mean(bfs)
def prf_half(preds,gts):
    atp,afp,afn=0,0,0
    for p,g in zip(preds,gts):
        gi=(g/255).astype('uint8');pi=((p/max(float(np.max(p)),1e-8))>0.5).astype('uint8')
        tp,fp,fn=get_stat(pi,gi);atp+=tp;afp+=fp;afn+=fn
    pr=1.0 if atp==0 and afp==0 else atp/(atp+afp);rc=atp/(atp+afn) if atp+afn>0 else 0
    return pr,rc,(2*pr*rc/(pr+rc) if pr+rc>0 else 0)
def compute_cldice_batch(preds,gt_m,gt_c,thr=0.5):
    cds,tps,tss=[],[],[]
    for p,gm,gc in zip(preds,gt_m,gt_c):
        pb=(p/255.0>thr).astype(np.uint8);cs=gc.sum();ts=(gc*pb).sum()/cs if cs>0 else 1.0
        sk=skeletonize(pb.astype(bool)).astype(np.float64);ss=sk.sum()
        tp=(sk*gm).sum()/ss if ss>0 else (0.0 if gm.sum()>0 else 1.0)
        cd=2*tp*ts/(tp+ts) if tp+ts>0 else 0;cds.append(cd);tps.append(tp);tss.append(ts)
    return float(np.mean(cds)),float(np.mean(tps)),float(np.mean(tss))
def load_gt_all(mask_dir,cl_dir,name,sz):
    bn=os.path.splitext(name)[0]
    for ext in ['.png','.jpg','.bmp']:
        gp=os.path.join(mask_dir,bn+ext);cp=os.path.join(cl_dir,bn+ext)
        if os.path.exists(gp) and os.path.exists(cp): break
    else: return None,None,None
    g=cv2.imread(gp,cv2.IMREAD_GRAYSCALE);c=cv2.imread(cp,cv2.IMREAD_GRAYSCALE)
    if g is None or c is None: return None,None,None
    g=cv2.resize(g,(sz,sz),interpolation=cv2.INTER_NEAREST);c=cv2.resize(c,(sz,sz),interpolation=cv2.INTER_NEAREST)
    _,gt=cv2.threshold(g,127,255,cv2.THRESH_BINARY)
    return 255-gt,(g<128).astype(np.uint8),(c<128).astype(np.uint8)

def evaluate_all(preds,gt_pix,gt_m,gt_c):
    print("  mIoU...",end="",flush=True);miou=cal_mIoU(preds,gt_pix);print(f" {miou:.4f}")
    print("  ODS...",end="",flush=True);ods=cal_ODS(preds,gt_pix);print(f" {ods:.4f}")
    print("  OIS...",end="",flush=True);ois=float(cal_OIS(preds,gt_pix));print(f" {ois:.4f}")
    p,r,f1=prf_half(preds,gt_pix);print(f"  P={p:.4f} R={r:.4f} F1={f1:.4f}")
    cd,tp,ts=compute_cldice_batch(preds,gt_m,gt_c);print(f"  clDice={cd:.4f} Tprec={tp:.4f} Tsens={ts:.4f}")
    return {'mIoU':round(miou,4),'ODS':round(ods,4),'OIS':round(float(ois),4),
            'P':round(p,4),'R':round(r,4),'F1':round(f1,4),
            'clDice':round(cd,4),'Tprec':round(tp,4),'Tsens':round(ts,4),'n':len(preds)}

def compute_mean_std_from_dir(img_dir):
    means,stds=[],[]
    for name in sorted(os.listdir(img_dir)):
        img=cv2.imread(os.path.join(img_dir,name))
        if img is None: continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        means.append(img.mean(axis=(0,1)));stds.append(img.std(axis=(0,1)))
    return np.mean(means,axis=0),np.mean(stds,axis=0)

# ===================== Dataset =====================
ds = 'CRACK500_resize'
img_dir = os.path.join(DS_BASE, ds, 'test', 'images')
mask_dir = os.path.join(DS_BASE, ds, 'test', 'annotations')
cl_dir = os.path.join(DS_BASE, ds, 'test', 'centerlines')
train_img_dir = os.path.join(DS_BASE, ds, 'train', 'images')

n_test = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
print(f"{'='*80}")
print(f"  CRACK500_resize — {n_test} test images, 512x512")
print(f"{'='*80}\n")

all_results = {}

# ===================== 1. CrackSnakeNet (Ours) with TTA =====================
from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from train.Trainer import LitCrackModel

ours_ckpt = os.path.join(PROJECT, 'checkpoints', 'ours_8_2', 'CRACK500_resize', 'best-epoch=153-val_iou=0.5687.ckpt')
if os.path.exists(ours_ckpt):
    print(">>> CrackSnakeNet (Ours)")
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
    ckpt = torch.load(ours_ckpt, map_location=device)
    lit.load_state_dict(ckpt['state_dict'], strict=False)
    lit.to(device).eval()
    m_, s_ = compute_mean_std_from_dir(train_img_dir)
    mean, std = list(m_), list(s_)
    preds, gt_pix, gt_m, gt_c = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="Ours"):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for c in range(3): x[c] = (x[c] - mean[c]) / std[c]
            x = x.unsqueeze(0).to(device)
            o0=lit(x);l=o0['main_output'] if isinstance(o0,dict) else o0
            o1=lit(torch.flip(x,[3]));l=l+torch.flip(o1['main_output'] if isinstance(o1,dict) else o1,[3])
            o2=lit(torch.flip(x,[2]));l=l+torch.flip(o2['main_output'] if isinstance(o2,dict) else o2,[2])
            o3=lit(torch.flip(x,[2,3]));l=l+torch.flip(o3['main_output'] if isinstance(o3,dict) else o3,[2,3])
            prob = torch.sigmoid(l / 4.0)
            pred_np = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
            gp, gm, gc = load_gt_all(mask_dir, cl_dir, name, 512)
            if gp is None: continue
            preds.append(pred_np); gt_pix.append(gp); gt_m.append(gm); gt_c.append(gc)
    all_results['Ours'] = evaluate_all(preds, gt_pix, gt_m, gt_c)
    del lit; torch.cuda.empty_cache()
    print()
else:
    print(f"CrackSnakeNet checkpoint not found: {ours_ckpt}")

# ===================== 2. U-Net (R34) =====================
import segmentation_models_pytorch as smp
unet_ckpt = os.path.join(COMP, 'ckpt_8_2/unet/CRACK500_resize/unet/unet/best.pth')
if os.path.exists(unet_ckpt):
    print(">>> U-Net (R34)")
    model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(unet_ckpt, map_location=device))
    model.to(device).eval()
    preds, gt_pix, gt_m, gt_c = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="U-Net"):
            img=cv2.imread(os.path.join(img_dir,name))
            if img is None: continue
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);img=cv2.resize(img,(512,512))
            x=transforms.functional.normalize(torch.from_numpy(img).permute(2,0,1).float()/255.0,
                mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]).unsqueeze(0).to(device)
            pred_np=(torch.sigmoid(model(x))[0,0].cpu().numpy()*255).astype(np.uint8)
            gp,gm,gc=load_gt_all(mask_dir,cl_dir,name,512)
            if gp is None: continue
            preds.append(pred_np);gt_pix.append(gp);gt_m.append(gm);gt_c.append(gc)
    all_results['UNet'] = evaluate_all(preds, gt_pix, gt_m, gt_c)
    del model; torch.cuda.empty_cache()
    print()

# ===================== 3. DeepLabV3+ (R34) =====================
dlv3_ckpt = os.path.join(COMP, 'ckpt_8_2/deeplabv3plus/CRACK500_resize/deeplabv3plus/deeplabv3plus/best.pth')
if os.path.exists(dlv3_ckpt):
    print(">>> DeepLabV3+ (R34)")
    model = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(dlv3_ckpt, map_location=device))
    model.to(device).eval()
    preds, gt_pix, gt_m, gt_c = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="DeepLabV3+"):
            img=cv2.imread(os.path.join(img_dir,name))
            if img is None: continue
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);img=cv2.resize(img,(512,512))
            x=transforms.functional.normalize(torch.from_numpy(img).permute(2,0,1).float()/255.0,
                mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]).unsqueeze(0).to(device)
            pred_np=(torch.sigmoid(model(x))[0,0].cpu().numpy()*255).astype(np.uint8)
            gp,gm,gc=load_gt_all(mask_dir,cl_dir,name,512)
            if gp is None: continue
            preds.append(pred_np);gt_pix.append(gp);gt_m.append(gm);gt_c.append(gc)
    all_results['DeepLabV3Plus'] = evaluate_all(preds, gt_pix, gt_m, gt_c)
    del model; torch.cuda.empty_cache()
    print()

# ===================== 4. DeepCrack model =====================
dc_ckpt = os.path.join(COMP, 'DeepCrack/checkpoints/deepcrack_CRACK500_resize/latest_net_G.pth')
if os.path.exists(dc_ckpt):
    print(">>> DeepCrack model")
    sys.path.insert(0, os.path.join(COMP, 'DeepCrack'))
    from models.deepcrack_networks import define_deepcrack
    model = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='instance',
                              init_type='normal', init_gain=0.02, gpu_ids=[])
    model.load_state_dict(torch.load(dc_ckpt, map_location=device))
    model.to(device).eval()
    preds, gt_pix, gt_m, gt_c = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="DeepCrack"):
            img=cv2.imread(os.path.join(img_dir,name))
            if img is None: continue
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);img=cv2.resize(img,(512,512))
            x=torch.from_numpy(img).permute(2,0,1).float()/255.0
            x=(x-0.5)/0.5;x=x.unsqueeze(0).to(device)
            outputs=model(x);prob=torch.sigmoid(outputs[-1])
            pred_np=(prob[0,0].cpu().numpy()*255).astype(np.uint8)
            gp,gm,gc=load_gt_all(mask_dir,cl_dir,name,512)
            if gp is None: continue
            preds.append(pred_np);gt_pix.append(gp);gt_m.append(gm);gt_c.append(gc)
    all_results['DeepCrack_Model'] = evaluate_all(preds, gt_pix, gt_m, gt_c)
    del model; torch.cuda.empty_cache()
    print()

# ===================== 5. DTrC-Net =====================
dtrc_ckpt = os.path.join(COMP, 'ckpt_8_2/dtrcnet/CRACK500_resize/best.pth')
if os.path.exists(dtrc_ckpt):
    print(">>> DTrC-Net")
    old_cwd = os.getcwd()
    os.chdir(os.path.join(COMP, 'DTrC-Net'))
    sys.path.insert(0, '.')
    from CTCNet import CTCNet
    model = CTCNet(n_channels=3, n_classes=1).to(device).eval()
    model.load_state_dict(torch.load(dtrc_ckpt, map_location=device))
    preds, gt_pix, gt_m, gt_c = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="DTrC-Net"):
            img=cv2.imread(os.path.join(img_dir,name))
            if img is None: continue
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);img=cv2.resize(img,(256,256))
            x=transforms.functional.normalize(torch.from_numpy(img).permute(2,0,1).float()/255.0,
                mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]).unsqueeze(0).to(device)
            logits,_=model(x)
            pred_256=(torch.sigmoid(logits)[0,0].cpu().numpy()*255).astype(np.uint8)
            pred_np=cv2.resize(pred_256,(512,512),interpolation=cv2.INTER_LINEAR)
            gp,gm,gc=load_gt_all(mask_dir,cl_dir,name,512)
            if gp is None: continue
            preds.append(pred_np);gt_pix.append(gp);gt_m.append(gm);gt_c.append(gc)
    all_results['DTrCNet'] = evaluate_all(preds, gt_pix, gt_m, gt_c)
    os.chdir(old_cwd)
    del model; torch.cuda.empty_cache()
    print()

# ===================== 6. CT-CrackSeg =====================
ct_ckpt = os.path.join(COMP, 'CT-CrackSeg/checkpoints/weights_crack500.model')
if os.path.exists(ct_ckpt):
    print(">>> CT-CrackSeg")
    sys.path.insert(0, os.path.join(COMP, 'CT-CrackSeg'))
    from model.TransMUNet import TransMUNet
    model = TransMUNet(n_classes=1).to(device).eval()
    ckpt = torch.load(ct_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_weights'])
    preds, gt_pix, gt_m, gt_c = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc="CT-CrackSeg"):
            img=cv2.imread(os.path.join(img_dir,name))
            if img is None: continue
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);img=cv2.resize(img,(256,256))
            x=torch.from_numpy(img).permute(2,0,1).float()/255.0
            x=x.unsqueeze(0).to(device)
            out=model(x)
            if isinstance(out,tuple): out=out[0]
            pred_256=(torch.sigmoid(out)[0,0].cpu().numpy()*255).astype(np.uint8)
            pred_np=cv2.resize(pred_256,(512,512),interpolation=cv2.INTER_LINEAR)
            gp,gm,gc=load_gt_all(mask_dir,cl_dir,name,512)
            if gp is None: continue
            preds.append(pred_np);gt_pix.append(gp);gt_m.append(gm);gt_c.append(gc)
    all_results['CTCrackSeg'] = evaluate_all(preds, gt_pix, gt_m, gt_c)
    del model; torch.cuda.empty_cache()
    print()

# ===================== 7. SCSegamba (load from saved results) =====================
sc_result_path = os.path.join(COMP, 'SCSegamba/eval_crack500_results.json')
if os.path.exists(sc_result_path):
    print(">>> SCSegamba (from saved eval)")
    all_results['SCSegamba'] = json.load(open(sc_result_path))
    r = all_results['SCSegamba']
    print(f"  mIoU={r['mIoU']:.4f} ODS={r['ODS']:.4f} P={r['P']:.4f} R={r['R']:.4f} "
          f"clDice={r['clDice']:.4f} Tprec={r['Tprec']:.4f} Tsens={r['Tsens']:.4f}")
    print()

# ===================== Summary =====================
print(f"\n{'='*120}")
print(f"{'Method':<20s} {'mIoU':>7s} {'ODS':>7s} {'OIS':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'clDice':>8s} {'Tprec':>7s} {'Tsens':>7s}")
print(f"{'-'*120}")
for name in ['Ours', 'UNet', 'DeepLabV3Plus', 'DeepCrack_Model', 'DTrCNet', 'CTCrackSeg', 'SCSegamba']:
    if name not in all_results: continue
    r = all_results[name]
    print(f"{name:<20s} {r['mIoU']:>7.4f} {r['ODS']:>7.4f} {r.get('OIS',0):>7.4f} "
          f"{r['P']:>7.4f} {r['R']:>7.4f} {r['F1']:>7.4f} "
          f"{r['clDice']:>8.4f} {r['Tprec']:>7.4f} {r['Tsens']:>7.4f}")
print(f"{'='*120}")

out_path = os.path.join(PROJECT, 'paper', 'eval_crack500_all.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
