#!/usr/bin/env python3
"""Evaluate A3 (no stem skip) ablation checkpoint."""
import os, sys, cv2, torch, json, types
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from skimage.morphology import skeletonize

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from config.config import Config
from model.TriBranchCracknet import TriBranchCrackNet
from model.CNNBlock import ConvGNAct
from train.Trainer import LitCrackModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DS_BASE = os.path.join(PROJECT, 'ComparisonModel', 'dataset_8_2')
CKPT_DIR = os.path.join(PROJECT, 'checkpoints', 'ablation_v2')

# ===================== Metrics (same as eval_ablation.py) =====================
def get_statistics(pred, gt):
    return int(np.sum((pred==1)&(gt==1))), int(np.sum((pred==1)&(gt==0))), int(np.sum((pred==0)&(gt==1)))

def cal_mIoU(pred_list, gt_list, thresh_step=0.01):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gi = (gt/255).astype('uint8'); pi = (pred/255>thresh).astype('uint8')
            TP=int(np.sum((pi==1)&(gi==1))); TN=int(np.sum((pi==0)&(gi==0)))
            FP=int(np.sum((pi==1)&(gi==0))); FN=int(np.sum((pi==0)&(gi==1)))
            iou = (TP/(FN+FP+TP)+TN/(FN+FP+TN))/2 if (FN+FP+TP)>0 else 0
            iou_list.append(iou)
        final_iou.append(np.mean(iou_list))
    return float(np.max(final_iou))

def cal_ODS(pred_list, gt_list, thresh_step=0.01):
    final = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        f1s = []
        for pred, gt in zip(pred_list, gt_list):
            gi=(gt/255).astype('uint8'); pi=(pred/255>thresh).astype('uint8')
            tp,fp,fn=get_statistics(pi,gi)
            p=1.0 if tp==0 and fp==0 else tp/(tp+fp)
            r=0 if tp+fn==0 else tp/(tp+fn)
            f1s.append(0 if p+r==0 else 2*p*r/(p+r))
        final.append(np.mean(f1s))
    return float(np.max(final))

def cal_OIS(pred_list, gt_list, thresh_step=0.01):
    final = []
    for pred, gt in zip(pred_list, gt_list):
        bf = 0
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gi=(gt/255).astype('uint8'); pi=(pred/255>thresh).astype('uint8')
            tp,fp,fn=get_statistics(pi,gi)
            p=1.0 if tp==0 and fp==0 else tp/(tp+fp)
            r=0 if tp+fn==0 else tp/(tp+fn)
            f1=0 if p+r==0 else 2*p*r/(p+r)
            bf=max(bf,f1)
        final.append(bf)
    return float(np.mean(final))

def prf_at_half(pred_list, gt_list):
    atp,afp,afn=0,0,0
    for pred, gt in zip(pred_list, gt_list):
        gi=(gt/255).astype('uint8')
        pi=((pred/max(float(np.max(pred)),1e-8))>0.5).astype('uint8')
        tp,fp,fn=get_statistics(pi,gi); atp+=tp;afp+=fp;afn+=fn
    p=1.0 if atp==0 and afp==0 else atp/(atp+afp)
    r=atp/(atp+afn) if atp+afn>0 else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    return p, r, f1

def compute_cldice_batch(pred_list, gt_mask_list, gt_cl_list, threshold=0.5):
    cds, tps, tss = [], [], []
    for pred, gm, gc in zip(pred_list, gt_mask_list, gt_cl_list):
        pb = (pred/255.0>threshold).astype(np.uint8)
        cs = gc.sum(); ts = (gc*pb).sum()/cs if cs>0 else 1.0
        sk = skeletonize(pb.astype(bool)).astype(np.float64)
        ss = sk.sum()
        tp = (sk*gm).sum()/ss if ss>0 else (0.0 if gm.sum()>0 else 1.0)
        cd = 2*tp*ts/(tp+ts) if tp+ts>0 else 0
        cds.append(cd); tps.append(tp); tss.append(ts)
    return float(np.mean(cds)), float(np.mean(tps)), float(np.mean(tss))

def load_gt_all(mask_dir, cl_dir, name, img_size):
    bn = os.path.splitext(name)[0]
    for ext in ['.png','.jpg','.bmp']:
        gp=os.path.join(mask_dir,bn+ext); cp=os.path.join(cl_dir,bn+ext)
        if os.path.exists(gp) and os.path.exists(cp): break
    else: return None,None,None
    gt=cv2.imread(gp,cv2.IMREAD_GRAYSCALE); cl=cv2.imread(cp,cv2.IMREAD_GRAYSCALE)
    if gt is None or cl is None: return None,None,None
    gt=cv2.resize(gt,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    cl=cv2.resize(cl,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    _,gt_t=cv2.threshold(gt,127,255,cv2.THRESH_BINARY)
    return 255-gt_t, (gt<128).astype(np.uint8), (cl<128).astype(np.uint8)

def compute_mean_std_from_dir(img_dir):
    means, stds = [], []
    for name in sorted(os.listdir(img_dir)):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        means.append(img.mean(axis=(0,1))); stds.append(img.std(axis=(0,1)))
    return np.mean(means, axis=0), np.mean(stds, axis=0)

# ===================== Build A3 model =====================
def build_a3_model(ckpt_path):
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
        use_edge_head=Config.USE_EDGE_HEAD, scan_directions=8)

    # Patch: replace full_res_refine with correct input channels (no stem concat)
    model.full_res_refine = torch.nn.Sequential(
        ConvGNAct(widths[0], widths[0]*2, kernel_size=3, stride=1, padding=1),
        ConvGNAct(widths[0]*2, widths[0], kernel_size=3, stride=1, padding=1),
    )

    # Patch forward: skip stem concat
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
        # NO stem concat — just upsample and refine
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

if __name__ == '__main__':
    ds = 'CrackTree260'
    img_dir = os.path.join(DS_BASE, ds, 'test', 'images')
    mask_dir = os.path.join(DS_BASE, ds, 'test', 'annotations')
    cl_dir = os.path.join(DS_BASE, ds, 'test', 'centerlines')
    train_img_dir = os.path.join(DS_BASE, ds, 'train', 'images')

    ckpt_path = os.path.join(CKPT_DIR, 'ablation_ckpt_A3_no_stem_skip.ckpt')
    print(f">>> A3_no_stem_skip evaluation on CrackTree260")
    lit = build_a3_model(ckpt_path)

    m_, s_ = compute_mean_std_from_dir(train_img_dir)
    mean, std = list(m_), list(s_)

    pred_list, gt_pix, gt_mask, gt_cl = [], [], [], []
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(img_dir)), desc='A3'):
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(img).permute(2,0,1).float()/255.0
            for c in range(3): x[c]=(x[c]-mean[c])/std[c]
            x = x.unsqueeze(0).to(device)
            o0=lit(x); l=o0['main_output'] if isinstance(o0,dict) else o0
            o1=lit(torch.flip(x,[3])); l=l+torch.flip(o1['main_output'] if isinstance(o1,dict) else o1,[3])
            o2=lit(torch.flip(x,[2])); l=l+torch.flip(o2['main_output'] if isinstance(o2,dict) else o2,[2])
            o3=lit(torch.flip(x,[2,3])); l=l+torch.flip(o3['main_output'] if isinstance(o3,dict) else o3,[2,3])
            prob = torch.sigmoid(l/4.0)
            pred_np = (prob[0,0].cpu().numpy()*255).astype(np.uint8)
            gp, gm, gc = load_gt_all(mask_dir, cl_dir, name, 512)
            if gp is None: continue
            pred_list.append(pred_np); gt_pix.append(gp); gt_mask.append(gm); gt_cl.append(gc)

    print(f"  {len(pred_list)} images evaluated")
    print("  mIoU...", end="", flush=True)
    mIoU = cal_mIoU(pred_list, gt_pix); print(f" {mIoU:.4f}")
    print("  ODS...", end="", flush=True)
    ODS = cal_ODS(pred_list, gt_pix); print(f" {ODS:.4f}")
    print("  OIS...", end="", flush=True)
    OIS = cal_OIS(pred_list, gt_pix); print(f" {OIS:.4f}")
    P, R, F1 = prf_at_half(pred_list, gt_pix)
    print(f"  P={P:.4f}  R={R:.4f}  F1={F1:.4f}")
    clDice, Tprec, Tsens = compute_cldice_batch(pred_list, gt_mask, gt_cl, 0.5)
    print(f"  clDice={clDice:.4f}  Tprec={Tprec:.4f}  Tsens={Tsens:.4f}")

    results = {'mIoU':round(mIoU,4),'ODS':round(ODS,4),'OIS':round(OIS,4),
               'P':round(P,4),'R':round(R,4),'F1':round(F1,4),
               'clDice':round(clDice,4),'Tprec':round(Tprec,4),'Tsens':round(Tsens,4),'n':len(pred_list)}

    out_path = os.path.join(PROJECT, 'paper', 'ablation_results.json')
    all_results = json.load(open(out_path))
    all_results['A3_no_stem_skip'] = results
    json.dump(all_results, open(out_path, 'w'), indent=2)
    print(f"\nUpdated {out_path}")
    print(f"\nA3: mIoU={mIoU*100:.2f}  Tsens={Tsens*100:.2f}  clDice={clDice*100:.2f}")
