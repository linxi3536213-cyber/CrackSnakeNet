#!/bin/bash
# Ablation study on CrackTree260 (8:2 split)
# A0 baseline uses existing ours_8_2/CrackTree260 result (no need to retrain)
# A1-A7: each removes one component, trains 200 epochs
# BOUNDARY_LOSS_WEIGHT must be 0.0 (verified harmful)

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/mymodel

DS_BASE="/root/autodl-tmp/ComparisonModel/dataset_8_2"
CONFIG="config/config.py"
MODEL="model/TriBranchCracknet.py"
MAMBA="model/MambaBlock.py"
FUSION="model/FusionBlock.py"
LOSS="utils/Losses.py"

# Set dataset to CrackTree260 and ensure BOUNDARY_LOSS_WEIGHT=0.0
sed -i 's|DATASET_ROOT = .*|DATASET_ROOT = r"'"${DS_BASE}/CrackTree260"'"|' $CONFIG
sed -i 's|BOUNDARY_LOSS_WEIGHT = 0.3|BOUNDARY_LOSS_WEIGHT = 0.0|' $CONFIG

# Backup original files (with correct config)
cp $CONFIG ${CONFIG}.bak
cp $MODEL ${MODEL}.bak
cp $MAMBA ${MAMBA}.bak
cp $FUSION ${FUSION}.bak

OUTDIR="ablation_v2"

run_one() {
    local NAME=$1
    echo ""
    echo "============================================"
    echo "  Ablation: $NAME  ($(date))"
    echo "============================================"
    python scripts/run.py 2>&1 | tee "${OUTDIR}/ablation_${NAME}.log"
    # Find best checkpoint
    LATEST_DIR=$(ls -td checkpoints/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        BEST=$(ls "$LATEST_DIR"best*.ckpt 2>/dev/null | head -1)
        if [ -n "$BEST" ]; then
            cp "$BEST" "${OUTDIR}/ablation_ckpt_${NAME}.ckpt"
            echo "  Saved: ${OUTDIR}/ablation_ckpt_${NAME}.ckpt"
        fi
    fi
    echo "<<< $NAME done. ($(date))"
}

restore_all() {
    cp ${CONFIG}.bak $CONFIG
    cp ${MODEL}.bak $MODEL
    cp ${MAMBA}.bak $MAMBA
    cp ${FUSION}.bak $FUSION
    sed -i 's|DATASET_ROOT = .*|DATASET_ROOT = r"'"${DS_BASE}/CrackTree260"'"|' $CONFIG
}

# =====================================================
# A0: SKIP — use existing ours_8_2/CrackTree260 result
# (val_iou=0.5890, mIoU=80.35%, Tsens=87.23%, clDice=78.21%)
# =====================================================
echo ">>> A0: Skipped (using existing baseline)"

# =====================================================
# A1: Remove CenterlineRecallLoss (weight=0)
# =====================================================
echo ">>> A1: No CenterlineRecallLoss"
restore_all
sed -i 's|CENTERLINE_LOSS_WEIGHT = 0.15|CENTERLINE_LOSS_WEIGHT = 0.0|' $CONFIG
run_one "A1_no_centerline_loss"

# =====================================================
# A2: 8-dir → 4-dir scan
# =====================================================
echo ">>> A2: 4-dir scan"
restore_all
sed -i "s|'scan_directions': 8|'scan_directions': 4|" $CONFIG
run_one "A2_4dir_scan"

# =====================================================
# A3: Remove stem skip connection
# =====================================================
echo ">>> A3: No stem skip"
restore_all
python -c "
with open('$MODEL','r') as f: code=f.read()
code=code.replace(
    'refine_in_ch = widths[0] + stem_channels  # decoder_out + stem_skip',
    'refine_in_ch = widths[0]  # no stem skip'
)
code=code.replace(
    '''        # 全分辨率细化: 拼接 stem 浅层特征恢复细裂缝边界
        stem_up = F.interpolate(stem_out, size=(orig_h, orig_w), mode=\'bilinear\', align_corners=False)
        y = F.interpolate(y, size=(orig_h, orig_w), mode=\'bilinear\', align_corners=False)
        y = torch.cat([y, stem_up], dim=1)
        y = self.full_res_refine(y)''',
    '''        # 全分辨率细化 (no stem skip)
        y = F.interpolate(y, size=(orig_h, orig_w), mode=\'bilinear\', align_corners=False)
        y = self.full_res_refine(y)'''
)
with open('$MODEL','w') as f: f.write(code)
print('Stem skip removed')
"
run_one "A3_no_stem_skip"

# =====================================================
# A4: Remove SnakeMamba2D global residual
# =====================================================
echo ">>> A4: No Mamba global residual"
restore_all
sed -i 's|return identity + out  # 全局残差|return out  # no global residual (ablation)|' $MAMBA
run_one "A4_no_mamba_residual"

# =====================================================
# A5: Bidirectional → Unidirectional cross-attention
# =====================================================
echo ">>> A5: Unidirectional cross-attn"
restore_all
python -c "
with open('$FUSION','r') as f: code=f.read()
code=code.replace(
    '''        # 双向交叉注意力
        out1 = self.cross_attn1(self.norm_q1(cnn_seq), self.norm_kv1(mamba_seq), H, W)     # Q=CNN, KV=Mamba
        out2 = self.cross_attn2(self.norm_q2(mamba_seq), self.norm_kv2(cnn_seq), H, W)     # Q=Mamba, KV=CNN

        # Concat + Fusion
        fused = self.fusion(torch.cat([out1, out2], dim=-1))  # (B, HW, 2C) → (B, HW, C)''',
    '''        # 单向交叉注意力 (ablation: only Q=CNN, KV=Mamba)
        out1 = self.cross_attn1(self.norm_q1(cnn_seq), self.norm_kv1(mamba_seq), H, W)     # Q=CNN, KV=Mamba
        out2 = torch.zeros_like(out1)  # disabled

        # Concat + Fusion
        fused = self.fusion(torch.cat([out1, out2], dim=-1))  # (B, HW, 2C) → (B, HW, C)'''
)
with open('$FUSION','w') as f: f.write(code)
print('Unidirectional cross-attn')
"
run_one "A5_unidir_crossattn"

# =====================================================
# A6: Remove multi-scale Mamba (all single-scale)
# =====================================================
echo ">>> A6: No multi-scale Mamba"
restore_all
sed -i 's|use_ms = (idx < len(widths) - 1)|use_ms = False  # ablation: no multiscale|' $MODEL
run_one "A6_no_multiscale"

# =====================================================
# A7: COBlock → standard ConvBlock
# =====================================================
echo ">>> A7: No COBlock (use ConvBlock)"
restore_all
python -c "
with open('$MODEL','r') as f: code=f.read()
code=code.replace(
    'from model.CNNBlock import (',
    'from model.CNNBlock import (\n    ConvBlock,'
)
code=code.replace(
    'block = CrackOrientedBlock(channels=width, k=k)',
    'block = ConvBlock(width, width)  # ablation: plain conv'
)
with open('$MODEL','w') as f: f.write(code)
print('COBlock replaced with ConvBlock')
"
python -c "
with open('$FUSION','r') as f: code=f.read()
code=code.replace(
    'from model.CNNBlock import CrackOrientedBlock',
    'from model.CNNBlock import CrackOrientedBlock, ConvBlock'
)
code=code.replace(
    'self.cnn = CrackOrientedBlock(channels, k=k)',
    'self.cnn = ConvBlock(channels, channels)  # ablation: plain conv'
)
with open('$FUSION','w') as f: f.write(code)
print('COBlock in FusionBlock replaced')
"
run_one "A7_no_coblock"

# =====================================================
# Restore everything
# =====================================================
echo ""
echo "============================================"
echo "  Restoring original files"
echo "============================================"
restore_all
rm -f ${CONFIG}.bak ${MODEL}.bak ${MAMBA}.bak ${FUSION}.bak

echo ""
echo "============================================"
echo "  ALL ABLATION EXPERIMENTS (A1-A7) COMPLETE!"
echo "  Checkpoints: ablation_ckpt_A*.ckpt"
echo "  Logs: ablation_A*.log"
echo "============================================"
