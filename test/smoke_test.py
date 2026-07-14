#!/usr/bin/env python
"""
冒烟测试: 验证 CrossAttention CrackNet 各模块基本功能

测试内容:
1. StemLayer
2. CNNandMambaBlock (Stage 0-2)
3. CNNandTransformerBlock (Stage 3 + Bottleneck)
4. DecoderBlock
5. 完整模型前向传播 (训练模式 + 推理模式)
6. 梯度流检查
7. 不同输入尺寸兼容性
8. 参数分组
"""

import os
import sys

# 添加项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from model.FusionBlock import CNNandMambaBlock, CNNandTransformerBlock
from model.TriBranchCracknet import TriBranchCrackNet, DecoderBlock
from model.CNNBlock import StemLayer
from config.config import Config


def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA 不可用, Mamba2 需要 GPU, 跳过测试")
        sys.exit(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ==================== 单元测试 ====================

def test_stem_layer():
    print("=" * 50)
    print("[1] StemLayer")
    stem = StemLayer(in_channels=3, dims=[192]).cuda()
    x = torch.randn(2, 3, 448, 448, device="cuda")
    out = stem(x)
    assert out.shape == (2, 96, 224, 224), f"形状不匹配: {out.shape}"
    print(f"    输入: {tuple(x.shape)} -> 输出: {tuple(out.shape)}")
    print(f"    参数量: {count_params(stem):,}")
    print("    PASSED")
    print()


def test_cnn_mamba_block():
    print("=" * 50)
    print("[2] CNNandMambaBlock (Stage 0-2)")
    block = CNNandMambaBlock(
        channels=192, k=9, d_state=16,
        attn_drop=0.05, proj_drop=0.1, drop=0.1, drop_path=0.05,
    ).cuda()
    x = torch.randn(2, 192, 56, 56, device="cuda")
    with torch.amp.autocast('cuda'):
        out = block(x)
    assert out.shape == x.shape, f"形状不匹配: {out.shape} != {x.shape}"
    print(f"    输入: {tuple(x.shape)} -> 输出: {tuple(out.shape)}")
    print(f"    参数量: {count_params(block):,}")
    print("    PASSED")
    print()


def test_cnn_transformer_block():
    print("=" * 50)
    print("[3] CNNandTransformerBlock (Stage 3 + Bottleneck)")
    block = CNNandTransformerBlock(
        channels=384, k=9,
        attn_drop=0.05, proj_drop=0.1, drop=0.1, drop_path=0.1,
    ).cuda()
    x = torch.randn(2, 384, 28, 28, device="cuda")
    with torch.amp.autocast('cuda'):
        out = block(x)
    assert out.shape == x.shape, f"形状不匹配: {out.shape} != {x.shape}"
    print(f"    输入: {tuple(x.shape)} -> 输出: {tuple(out.shape)}")
    print(f"    参数量: {count_params(block):,}")
    print("    PASSED")
    print()


def test_decoder_block():
    print("=" * 50)
    print("[4] DecoderBlock")
    dec = DecoderBlock(in_channels=384, skip_channels=192, out_channels=192).cuda()
    x = torch.randn(2, 384, 28, 28, device="cuda")
    skip = torch.randn(2, 192, 56, 56, device="cuda")
    out = dec(x, skip)
    assert out.shape == (2, 192, 56, 56), f"形状不匹配: {out.shape}"
    print(f"    x: {tuple(x.shape)}, skip: {tuple(skip.shape)} -> 输出: {tuple(out.shape)}")
    print(f"    参数量: {count_params(dec):,}")
    print("    PASSED")
    print()


# ==================== 集成测试 ====================

def _create_model():
    model_config = Config.MODEL_CONFIGS
    stage_count = len(model_config['widths'])
    stage_blocks = Config.expand_stage_blocks(model_config['depths'], stage_count)

    return TriBranchCrackNet(
        in_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        widths=model_config['widths'],
        stage_blocks=stage_blocks,
        k=model_config['k'],
        d_state=model_config['d_state'],
        dropout=model_config['dropout'],
        attn_drop=model_config['attn_drop'],
        proj_drop=model_config['proj_drop'],
        drop_path_rate=model_config['drop_path_rate'],
        aux_stage_indices=Config.AUX_STAGE_INDICES,
        use_aux_outputs=Config.USE_AUX_OUTPUTS,
        first_downsample_stride=Config.ENCODER_FIRST_DOWNSAMPLE_STRIDE,
        use_edge_head=Config.USE_EDGE_HEAD,
    ).cuda()


def test_full_model_train_mode():
    print("=" * 50)
    print("[5] 完整模型 - 训练模式")

    model = _create_model()

    total_params = count_params(model)
    trainable_params = count_trainable_params(model)
    print(f"    总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"    可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"    stage_blocks: {model.stage_blocks}")
    print(f"    widths: {model.widths}")

    model.train()
    x = torch.randn(2, 3, 448, 448, device="cuda")

    with torch.amp.autocast('cuda'):
        output = model(x)

    if isinstance(output, dict):
        main = output['main_output']
        assert main.shape == (2, 1, 448, 448), f"main_output 形状不匹配: {main.shape}"
        print(f"    main_output: {tuple(main.shape)}")
    else:
        assert output.shape == (2, 1, 448, 448), f"输出形状不匹配: {output.shape}"
        print(f"    输出: {tuple(output.shape)}")

    print("    PASSED")
    print()
    return model


def test_full_model_eval_mode(model):
    print("=" * 50)
    print("[6] 完整模型 - 推理模式")
    model.eval()
    model.use_aux_outputs = False

    x = torch.randn(1, 3, 448, 448, device="cuda")
    with torch.no_grad(), torch.amp.autocast('cuda'):
        output = model(x)

    assert isinstance(output, torch.Tensor), f"推理模式应返回 Tensor, 但返回了 {type(output)}"
    assert output.shape == (1, 1, 448, 448), f"输出形状不匹配: {output.shape}"
    print(f"    输入: {tuple(x.shape)} -> 输出: {tuple(output.shape)}")
    print("    PASSED")
    print()


def test_gradient_flow(model):
    print("=" * 50)
    print("[7] 梯度流检查")
    model.train()
    model.use_aux_outputs = False
    model.zero_grad()
    x = torch.randn(1, 3, 448, 448, device="cuda")

    with torch.amp.autocast('cuda'):
        output = model(x)
    loss = output.mean()
    loss.backward()

    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            no_grad_params.append(name)

    if no_grad_params:
        print(f"    WARNING: {len(no_grad_params)} 个参数没有梯度:")
        for name in no_grad_params[:10]:
            print(f"      - {name}")
        if len(no_grad_params) > 10:
            print(f"      ... 还有 {len(no_grad_params) - 10} 个")
    else:
        print("    所有可训练参数都有梯度流")

    print("    PASSED")
    print()


def test_different_input_sizes():
    print("=" * 50)
    print("[8] 不同输入尺寸兼容性")

    model_config = Config.MODEL_CONFIGS
    stage_count = len(model_config['widths'])
    stage_blocks = Config.expand_stage_blocks(model_config['depths'], stage_count)

    model = TriBranchCrackNet(
        in_channels=3, num_classes=1,
        widths=model_config['widths'],
        stage_blocks=stage_blocks,
        k=model_config['k'],
        d_state=model_config['d_state'],
        dropout=model_config['dropout'],
        attn_drop=model_config['attn_drop'],
        proj_drop=model_config['proj_drop'],
        drop_path_rate=model_config['drop_path_rate'],
        use_aux_outputs=False,
        use_edge_head=False,
        first_downsample_stride=1,
    ).cuda().eval()

    test_sizes = [(224, 224), (256, 256), (448, 448)]
    for h, w in test_sizes:
        x = torch.randn(1, 3, h, w, device="cuda")
        try:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                out = model(x)
            assert out.shape == (1, 1, h, w), f"尺寸 {h}x{w}: 输出形状不匹配 {out.shape}"
            print(f"    {h}x{w}: PASSED -> {tuple(out.shape)}")
        except Exception as e:
            print(f"    {h}x{w}: FAILED -> {e}")

    print()


def test_get_params_groups():
    print("=" * 50)
    print("[9] get_params_groups 优化器参数分组")

    model_config = Config.MODEL_CONFIGS
    stage_count = len(model_config['widths'])
    stage_blocks = Config.expand_stage_blocks(model_config['depths'], stage_count)

    model = TriBranchCrackNet(
        in_channels=3, num_classes=1,
        widths=model_config['widths'],
        stage_blocks=stage_blocks,
        k=model_config['k'],
        d_state=model_config['d_state'],
        dropout=model_config['dropout'],
        attn_drop=model_config['attn_drop'],
        proj_drop=model_config['proj_drop'],
        drop_path_rate=model_config['drop_path_rate'],
        use_aux_outputs=False,
        use_edge_head=False,
        first_downsample_stride=1,
    ).cuda()

    groups = model.get_params_groups(lr=8e-5, weight_decay=1e-3)
    assert len(groups) == 2, f"期望 2 个参数组, 但得到 {len(groups)}"

    total_grouped = sum(p.numel() for g in groups for p in g['params'])
    total_model = count_trainable_params(model)
    assert total_grouped == total_model, (
        f"参数分组总量 {total_grouped} != 模型可训练参数 {total_model}"
    )

    print(f"    decay 组: {sum(p.numel() for p in groups[0]['params']):,} 参数, wd={groups[0]['weight_decay']}")
    print(f"    no_decay 组: {sum(p.numel() for p in groups[1]['params']):,} 参数, wd={groups[1]['weight_decay']}")
    print("    PASSED")
    print()


# ==================== Main ====================

def main():
    print("\n" + "=" * 50)
    print("CrossAttention CrackNet 冒烟测试")
    print("=" * 50 + "\n")

    check_gpu()

    # 单元测试
    test_stem_layer()
    test_cnn_mamba_block()
    test_cnn_transformer_block()
    test_decoder_block()

    # 集成测试
    model = test_full_model_train_mode()
    test_full_model_eval_mode(model)
    test_gradient_flow(model)
    del model
    torch.cuda.empty_cache()

    test_different_input_sizes()
    test_get_params_groups()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
