#!/bin/bash
cd /home/andy/desktop/CrackDectection/CrackMambaTransformerNet/ablation/CrackMambaTransformerNetTest3.3TriFFMplus

OUT="paper/InferenceResult"
DS="ComparisonModel/dataset_8_2"
C="ComparisonModel"
CFD="/home/andy/desktop/CrackDectection/dataset/omnicrack30k_classified/CFD"

need() { [ ! -d "$OUT/$1/$2" ] || [ $(ls "$OUT/$1/$2/"*.png 2>/dev/null | wc -l) -eq 0 ]; }

# === DBCNet (must run from DBCNet dir) ===
for DATASET in DeepCrack CrackTree260; do
    if need "$DATASET" DBCNet; then
        echo ">>> DBCNet $DATASET"
        CKPT="log_final/log_Deepcrack/log_Deepcrack_my_backbone_7/checkpoint/epoch-147.pth"
        [ "$DATASET" = "CrackTree260" ] && CKPT="log_final/log_Deepcrack/log_Deepcrack_my_backbone_9/checkpoint/epoch-158.pth"
        cd "$C/DBCNet"
        python -c "
import os,sys,cv2,torch,numpy as np;import torch.nn as nn;from tqdm import tqdm
sys.path.insert(0,'.')
from configs.config_crack_Deepcrack import config
from models.builder import EncoderDecoder as segmodel
from utils.loss_crack import SoftDiceLoss
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m=segmodel(cfg=config,criterion=SoftDiceLoss(),norm_layer=nn.BatchNorm2d)
sd=torch.load('$CKPT',map_location=device);sd={(k[7:] if k.startswith('module.') else k):v for k,v in sd['model'].items()}
m.load_state_dict(sd,strict=True);m.to(device).eval()
idir='../dataset_8_2/$DATASET/test/images';sdir='../../$OUT/$DATASET/DBCNet';os.makedirs(sdir,exist_ok=True)
with torch.no_grad():
    for name in tqdm(sorted(os.listdir(idir)),desc='DBCNet $DATASET'):
        img=cv2.imread(os.path.join(idir,name))
        if img is None:continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);img=cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        x=torch.from_numpy(img).permute(2,0,1).float()/255.0;x=x.unsqueeze(0).to(device)
        pred=(torch.sigmoid(m(x))[0,0].cpu().numpy()*255).astype(np.uint8)
        cv2.imwrite(os.path.join(sdir,os.path.splitext(name)[0]+'.png'),pred)
print(f'DBCNet $DATASET: {len(os.listdir(sdir))}')" 2>&1 | tail -3
        cd /home/andy/desktop/CrackDectection/CrackMambaTransformerNet/ablation/CrackMambaTransformerNetTest3.3TriFFMplus
    fi
done

# CFD DBCNet
if need CFD DBCNet; then
    echo ">>> DBCNet CFD"
    cd "$C/DBCNet"
    python -c "
import os,sys,cv2,torch,numpy as np;import torch.nn as nn;from tqdm import tqdm
sys.path.insert(0,'.')
from configs.config_crack_Deepcrack import config
from models.builder import EncoderDecoder as segmodel
from utils.loss_crack import SoftDiceLoss
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m=segmodel(cfg=config,criterion=SoftDiceLoss(),norm_layer=nn.BatchNorm2d)
sd=torch.load('log_final/log_Deepcrack/log_Deepcrack_my_backbone_7/checkpoint/epoch-147.pth',map_location=device)
sd={(k[7:] if k.startswith('module.') else k):v for k,v in sd['model'].items()}
m.load_state_dict(sd,strict=True);m.to(device).eval()
idir='$CFD/images/test';sdir='../../$OUT/CFD/DBCNet';os.makedirs(sdir,exist_ok=True)
with torch.no_grad():
    for name in tqdm(sorted(os.listdir(idir)),desc='DBCNet CFD'):
        img=cv2.imread(os.path.join(idir,name))
        if img is None:continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);img=cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        x=torch.from_numpy(img).permute(2,0,1).float()/255.0;x=x.unsqueeze(0).to(device)
        pred=(torch.sigmoid(m(x))[0,0].cpu().numpy()*255).astype(np.uint8)
        cv2.imwrite(os.path.join(sdir,os.path.splitext(name)[0]+'.png'),pred)
print(f'DBCNet CFD: {len(os.listdir(sdir))}')" 2>&1 | tail -3
    cd /home/andy/desktop/CrackDectection/CrackMambaTransformerNet/ablation/CrackMambaTransformerNetTest3.3TriFFMplus
fi

# === Rest (UNet, DeepLabV3+, DeepCrack_Model, Ours) — no import conflicts ===
echo ">>> Remaining models"
python -u scripts/run_inference_clean.py 2>&1

echo "=== FINAL ==="
for ds in DeepCrack CrackTree260 CFD; do
    for d in "$OUT/$ds"/*/; do
        n=$(ls "$d"*.png 2>/dev/null | wc -l)
        echo "$ds/$(basename $d): $n"
    done
done
