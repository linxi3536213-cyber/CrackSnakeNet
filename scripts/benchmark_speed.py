#!/usr/bin/env python3
"""
Benchmark FLOPs and inference speed for all models at 512x512 input.
Reports: Params (M), FLOPs (G), FPS (images/sec), Latency (ms/image).
"""
import os, sys, time, torch
import torch.nn as nn
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
COMP = os.path.join(PROJECT, 'ComparisonModel')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== FLOPs counter =====================
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("WARNING: fvcore not installed, using thop for FLOPs")
    try:
        from thop import profile as thop_profile
        HAS_THOP = True
    except ImportError:
        HAS_THOP = False
        print("WARNING: thop not installed either, FLOPs will be estimated")


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, input_size=(1, 3, 512, 512)):
    x = torch.randn(*input_size).to(device)
    if HAS_FVCORE:
        flops = FlopCountAnalysis(model, x)
        return flops.total() / 1e9
    elif HAS_THOP:
        flops, _ = thop_profile(model, inputs=(x,), verbose=False)
        return flops / 1e9
    return -1


def measure_speed(model, input_size=(1, 3, 512, 512), warmup=20, runs=100):
    x = torch.randn(*input_size).to(device)
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    avg_time = np.mean(times)
    return 1.0 / avg_time, avg_time * 1000  # FPS, latency_ms


# ===================== Build models =====================

def build_ours():
    from config.config import Config
    from model.TriBranchCracknet import TriBranchCrackNet
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
        use_aux_outputs=False, aux_stage_indices=Config.AUX_STAGE_INDICES,
        use_edge_head=False, scan_directions=mc.get('scan_directions', 8))
    return model.to(device).eval()


def build_unet():
    import segmentation_models_pytorch as smp_lib
    model = smp_lib.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    return model.to(device).eval()


def build_deeplabv3plus():
    import segmentation_models_pytorch as smp_lib
    model = smp_lib.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    return model.to(device).eval()


def build_dtrcnet():
    old_cwd = os.getcwd()
    dtrc_path = os.path.join(COMP, 'DTrC-Net')
    os.chdir(dtrc_path)
    sys.path.insert(0, '.')
    from CTCNet import CTCNet
    model = CTCNet(n_channels=3, n_classes=1).to(device).eval()
    os.chdir(old_cwd)
    return model


def build_deepcrack_model():
    sys.path.insert(0, os.path.join(COMP, 'DeepCrack'))
    from models.deepcrack_networks import define_deepcrack
    model = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='batch',
                              init_type='normal', init_gain=0.02, gpu_ids=[])
    return model.to(device).eval()


def build_dbcnet():
    old_cwd = os.getcwd()
    dbcnet_path = os.path.join(COMP, 'DBCNet')
    os.chdir(dbcnet_path)
    sys.path.insert(0, '.')
    from configs.config_crack_Deepcrack import config
    from models.builder import EncoderDecoder as segmodel
    from utils.loss_crack import SoftDiceLoss
    model = segmodel(cfg=config, criterion=SoftDiceLoss(), norm_layer=nn.BatchNorm2d)
    os.chdir(old_cwd)
    return model.to(device).eval()


def build_ctcrackseg():
    sys.path.insert(0, os.path.join(COMP, 'CT-CrackSeg'))
    from model.TransMUNet import TransMUNet
    model = TransMUNet(img_size=256, in_chans=3, out_chans=1, embed_dims=[64, 128, 256, 512])
    return model.to(device).eval()


# ===================== Main =====================

if __name__ == '__main__':
    results = []

    models_config = [
        ('CrackSnakeNet', build_ours, (1, 3, 512, 512)),
        ('U-Net (R34)', build_unet, (1, 3, 512, 512)),
        ('DeepLabV3+ (R34)', build_deeplabv3plus, (1, 3, 512, 512)),
        ('DeepCrack', build_deepcrack_model, (1, 3, 512, 512)),
        ('DBCNet', build_dbcnet, (1, 3, 512, 512)),
        ('DTrC-Net', build_dtrcnet, (1, 3, 256, 256)),
        ('CT-CrackSeg', build_ctcrackseg, (1, 3, 256, 256)),
    ]

    print(f"{'Model':<20s} {'Params(M)':>10s} {'FLOPs(G)':>10s} {'FPS':>8s} {'Latency(ms)':>12s}")
    print('-' * 65)

    for name, build_fn, input_size in models_config:
        try:
            model = build_fn()
            params = count_params(model)

            try:
                flops = count_flops(model, input_size)
            except Exception as e:
                print(f"  {name}: FLOPs error: {e}")
                flops = -1

            try:
                fps, latency = measure_speed(model, input_size)
            except Exception as e:
                print(f"  {name}: Speed error: {e}")
                fps, latency = -1, -1

            flops_str = f"{flops:.2f}" if flops > 0 else "N/A"
            fps_str = f"{fps:.1f}" if fps > 0 else "N/A"
            lat_str = f"{latency:.1f}" if latency > 0 else "N/A"

            print(f"{name:<20s} {params:>10.2f} {flops_str:>10s} {fps_str:>8s} {lat_str:>12s}")
            results.append({
                'model': name, 'params_m': round(params, 2),
                'flops_g': round(flops, 2) if flops > 0 else None,
                'fps': round(fps, 1) if fps > 0 else None,
                'latency_ms': round(latency, 1) if latency > 0 else None,
                'input_size': f"{input_size[2]}x{input_size[3]}"
            })

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{name:<20s}  ERROR: {e}")
            import traceback; traceback.print_exc()

    print('\n' + '=' * 65)

    # Save results
    import json
    out_path = os.path.join(PROJECT, 'paper', 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
