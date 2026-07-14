#!/bin/bash
# Train CrackSnakeNet on CRACK500_resize (300 train, 199 test, 512x512)
# Run on server: nohup bash scripts/train_crack500.sh > train_crack500.log 2>&1 &

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/mymodel

CONFIG="config/config.py"

# Backup config
cp $CONFIG ${CONFIG}.bak

# Set dataset to CRACK500_resize
sed -i 's|DATASET_ROOT = .*|DATASET_ROOT = r"/root/autodl-tmp/ComparisonModel/dataset_8_2/CRACK500_resize"|' $CONFIG

# Ensure correct settings
sed -i 's|BOUNDARY_LOSS_WEIGHT = .*|BOUNDARY_LOSS_WEIGHT = 0.0|' $CONFIG
sed -i "s|'scan_directions': .*|'scan_directions': 8,|" $CONFIG

echo "=== Training CrackSnakeNet on CRACK500_resize ==="
echo "Dataset: /root/autodl-tmp/ComparisonModel/dataset_8_2/CRACK500_resize"
echo "Config:"
grep -E "DATASET_ROOT|BOUNDARY|scan_dir|EPOCHS|BATCH" $CONFIG | head -10
echo ""

python scripts/run.py 2>&1 | tee train_crack500_ours.log

# Find best checkpoint
LATEST_DIR=$(ls -td checkpoints/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_DIR" ]; then
    BEST=$(ls "$LATEST_DIR"best*.ckpt 2>/dev/null | head -1)
    if [ -n "$BEST" ]; then
        mkdir -p checkpoints/ours_8_2/CRACK500_resize
        cp "$BEST" checkpoints/ours_8_2/CRACK500_resize/
        echo "Saved best checkpoint to checkpoints/ours_8_2/CRACK500_resize/"
        echo "Best: $BEST"
    fi
fi

# Restore config
cp ${CONFIG}.bak $CONFIG
rm -f ${CONFIG}.bak

echo ""
echo "=== CrackSnakeNet CRACK500_resize training complete ==="
