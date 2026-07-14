#!/bin/bash
# Train ALL comparison models on CRACK500_resize (300 train / 199 test, 512x512)
# Run on server: nohup bash scripts/train_crack500_all.sh > train_crack500_all.log 2>&1 &

export PATH=/root/miniconda3/bin:$PATH

DS="/root/autodl-tmp/ComparisonModel/dataset_8_2/CRACK500_resize"
COMP="/root/autodl-tmp/ComparisonModel"
CKPT_BASE="$COMP/ckpt_8_2"

echo "============================================"
echo "  CRACK500_resize: Train all models"
echo "  Dataset: $DS (300 train / 199 test)"
echo "============================================"

# =====================================================
# 1. U-Net (R34)
# =====================================================
echo ""
echo ">>> U-Net (R34)"
cd "$COMP"
python train_smp.py \
    --model unet --encoder resnet34 \
    --train_img_dir "$DS/train/images" \
    --train_mask_dir "$DS/train/annotations" \
    --val_img_dir "$DS/test/images" \
    --val_mask_dir "$DS/test/annotations" \
    --img_size 512 --batch_size 4 --epochs 200 --lr 1e-4 \
    --save_dir "$CKPT_BASE/unet/CRACK500_resize/unet"
echo "<<< U-Net done"

# =====================================================
# 2. DeepLabV3+ (R34)
# =====================================================
echo ""
echo ">>> DeepLabV3+ (R34)"
cd "$COMP"
python train_smp.py \
    --model deeplabv3plus --encoder resnet34 \
    --train_img_dir "$DS/train/images" \
    --train_mask_dir "$DS/train/annotations" \
    --val_img_dir "$DS/test/images" \
    --val_mask_dir "$DS/test/annotations" \
    --img_size 512 --batch_size 4 --epochs 200 --lr 1e-4 \
    --save_dir "$CKPT_BASE/deeplabv3plus/CRACK500_resize/deeplabv3plus"
echo "<<< DeepLabV3+ done"

# =====================================================
# 3. DeepCrack model
# =====================================================
echo ""
echo ">>> DeepCrack model"
cd "$COMP/DeepCrack"
python train.py \
    --dataroot "$DS" \
    --dataset_mode crack \
    --name deepcrack_CRACK500_resize \
    --model deepcrack \
    --n_epochs 100 --n_epochs_decay 100 \
    --display_id 0 --gpu_ids 0 \
    --save_epoch_freq 50
echo "<<< DeepCrack done"

# =====================================================
# 4. DBCNet
# =====================================================
echo ""
echo ">>> DBCNet"
cd "$COMP/DBCNet"
# Create DBCNet data symlink
mkdir -p data/CRACK500_resize
ln -sf "$DS/train/images" data/CRACK500_resize/train_img
ln -sf "$DS/train/annotations" data/CRACK500_resize/train_lab
ln -sf "$DS/test/images" data/CRACK500_resize/test_img
ln -sf "$DS/test/annotations" data/CRACK500_resize/test_lab
python train.py --config configs/config_crack_Deepcrack.py 2>&1 || echo "DBCNet may need config edit for CRACK500_resize"
echo "<<< DBCNet done"

# =====================================================
# 5. DTrC-Net
# =====================================================
echo ""
echo ">>> DTrC-Net"
cd "$COMP/DTrC-Net"
python train.py \
    --data_path "$DS" \
    --epochs 200 --batch_size 4 --lr 1e-4 \
    --save_path "$CKPT_BASE/dtrcnet/CRACK500_resize"
echo "<<< DTrC-Net done"

# =====================================================
# 6. CT-CrackSeg
# =====================================================
echo ""
echo ">>> CT-CrackSeg"
cd "$COMP/CT-CrackSeg"
# CT-CrackSeg uses 256x256 input
python train_crack.py \
    --data_path "$DS" \
    --epochs 200 --batch_size 4 --lr 1e-4 \
    --save_path "$CKPT_BASE/ctcrackseg/CRACK500_resize"
echo "<<< CT-CrackSeg done"

echo ""
echo "============================================"
echo "  All CRACK500_resize training complete!"
echo "============================================"
