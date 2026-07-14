#!/usr/bin/env python3
"""Re-evaluate A7 (COBlock->ConvBlock) with correct model structure."""
import os, sys, cv2, torch, json, numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from model.CNNBlock import ConvBlock
from train.Trainer import LitCrackModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DS_BASE = os.path.join(PROJECT, 'ComparisonModel', 'dataset_8_2')
CKPT_DIR = os.path.join(PROJECT, 'checkpoints', 'ablation_v2')

# ===================== Metrics =====================
def get_stat(p,g): return int(np.sum((p==1)&(g==1))),int(np.sum((p==1)&(g==0))),int(np.sum((p==0)&(g==1)))
def cal_mIoU(preds,gts,step=0.01):
    best=0
    for t in np.arange(0,1,step):
        ious=[]
        for p,g in zip(preds,gts):
            gi=(g/255).astype('uint8');pi=(p/255>t).astype('uint8')
            TP=np.sum((pi==1)&(gi==1));TN=np.sum((pi==0)&(gi==0));FP=np.sum((pi==1)&(gi==0));FN=np.sum((pi==0)&(gi==1))
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
def compute_mean_std_from_dir(img_dir):
    means,stds=[],[]
    for name in sorted(os.listdir(img_dir)):
        img=cv2.imread(os.path.join(img_dir,name))
        if img is None: continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        means.append(img.mean(axis=(0,1)));stds.append(img.std(axis=(0,1)))
    return np.mean(means,axis=0),np.mean(stds,axis=0)

# ===================== Build A7 model =====================
def build_a7():
    mc = Config.MODEL_CONFIGS
    widths = mc['widths']
    stage_blocks = Config.expand_stage_blocks(mc['depths'], len(widths))
    model = TriBranchCrackNet(
        in_channels=3, num_classes=1,
        widths=widths, stage_blocks=stage_blocks,
        k=mc['k'], d_state=mc['d_state'],
        dropout=mc['dropout'], attn_drop=mc['attn_drop'],
        proj_drop=mc['proj_drop'], drop_path_rate=mc['drop_path_rate'],
        first_downsample_stride=Config.ENCODER_FIRST_DOWNSAMPLE_STRIDE,
        use_aux_outputs=Config.USE_AUX_OUTPUTS, aux_stage_indices=Config.AUX_STAGE_INDICES,
        use_edge_head=Config.USE_EDGE_HEAD, scan_directions=8)

    # Replace COBlock with ConvBlock in encoder stages 0,1
    mamba_start = 2
    for i in range(mamba_start):
        w = widths[i]
        for j in range(len(model.stages[i])):
            model.stages[i][j] = ConvBlock(w, w)
    # Replace COBlock inside CNNandMambaBlock (stages 2,3)
    for i in range(mamba_start, len(widths)):
        for j in range(len(model.stages[i])):
            block = model.stages[i][j]
            if hasattr(block, 'cnn'):
                block.cnn = ConvBlock(widths[i], widths[i])

    ckpt_path = os.path.join(CKPT_DIR, 'ablation_ckpt_A7_no_coblock.ckpt')
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
    missing, unexpected = lit.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing: print(f"  Missing sample: {missing[:3]}")
    if unexpected: print(f"  Unexpected sample: {unexpected[:3]}")
    lit.to(device).eval()
    return lit

if __name__ == '__main__':
    ds = 'CrackTree260'
    img_dir = os.path.join(DS_BASE, ds, 'test', 'images')
    mask_dir = os.path.join(DS_BASE, ds, 'test', 'annotations')
    cl_dir = os.path.join(DS_BASE, ds, 'test', 'centerlines')
    train_img_dir = os.path.join(DS_BASE, ds, 'train', 'images')

    print(">>> A7_no_coblock (correct ConvBlock model)")
    lit = build_a7()

    m_, s_ = compute_mean_std_from_dir(train_img_dir)
    mean, std = list(m_), list(s_)

    preds, gt_pix, gt_m, gt_c = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc='A7'):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for c in range(3): x[c] = (x[c] - mean[c]) / std[c]
            x = x.unsqueeze(0).to(device)
            o0 = lit(x); l = o0['main_output'] if isinstance(o0, dict) else o0
            o1 = lit(torch.flip(x,[3])); l = l + torch.flip(o1['main_output'] if isinstance(o1,dict) else o1,[3])
            o2 = lit(torch.flip(x,[2])); l = l + torch.flip(o2['main_output'] if isinstance(o2,dict) else o2,[2])
            o3 = lit(torch.flip(x,[2,3])); l = l + torch.flip(o3['main_output'] if isinstance(o3,dict) else o3,[2,3])
            prob = torch.sigmoid(l / 4.0)
            pred_np = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
            gp, gm, gc = load_gt_all(mask_dir, cl_dir, name, 512)
            if gp is None: continue
            preds.append(pred_np); gt_pix.append(gp); gt_m.append(gm); gt_c.append(gc)

    print(f"\n{len(preds)} images")
    miou = cal_mIoU(preds, gt_pix); print(f"mIoU={miou:.4f}")
    ods = cal_ODS(preds, gt_pix); print(f"ODS={ods:.4f}")
    ois = float(cal_OIS(preds, gt_pix)); print(f"OIS={ois:.4f}")
    p, r, f1 = prf_half(preds, gt_pix); print(f"P={p:.4f} R={r:.4f} F1={f1:.4f}")
    cd, tp, ts = compute_cldice_batch(preds, gt_m, gt_c); print(f"clDice={cd:.4f} Tprec={tp:.4f} Tsens={ts:.4f}")

    results = {'mIoU':round(miou,4),'ODS':round(ods,4),'OIS':round(float(ois),4),
               'P':round(p,4),'R':round(r,4),'F1':round(f1,4),
               'clDice':round(cd,4),'Tprec':round(tp,4),'Tsens':round(ts,4),'n':len(preds)}

    out_path = os.path.join(PROJECT, 'paper', 'ablation_results.json')
    all_results = json.load(open(out_path))
    all_results['A7_no_coblock'] = results
    json.dump(all_results, open(out_path, 'w'), indent=2)
    print(f"\nA7 corrected: mIoU={miou*100:.2f} R={r*100:.2f} Tsens={ts*100:.2f} clDice={cd*100:.2f}")
    print(f"Saved to {out_path}")
