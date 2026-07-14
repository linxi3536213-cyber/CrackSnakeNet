#!/bin/bash
# Train CrackMambaTransformerNet on CRACK500 dataset
# 250 train / 50 val / 199 test

cd /home/andy/desktop/CrackDectection/CrackMambaTransformerNet/ablation/CrackMambaTransformerNetTest3.3TriFFMplus

# Backup config
cp config/config.py config/config_backup.py

# Modify config for CRACK500
sed -i 's|DATASET_ROOT = .*|DATASET_ROOT = r"/home/andy/desktop/CrackDectection/dataset/omnicrack30k_classified/CRACK500"|' config/config.py
sed -i 's|TRAIN_SPLIT = .*|TRAIN_SPLIT = "training"|' config/config.py
sed -i 's|TEST_SPLIT = .*|TEST_SPLIT = "validation"|' config/config.py

echo "=== Config updated for CRACK500 ==="
grep -E "DATASET_ROOT|TRAIN_SPLIT|TEST_SPLIT" config/config.py

python3 scripts/run.py

# Restore config
cp config/config_backup.py config/config.py
echo "=== Config restored ==="
