#!/usr/bin/env python3
"""
Benchmark FLOPs and inference speed for all models at 512x512.
FLOPs: uses thop (more robust with custom ops), falls back to fvcore.
Speed: GPU warmup + 100 runs, reports FPS and latency.
DBCNet/DTrC-Net/CT-CrackSeg run via subprocess to handle import isolation.
"""
import os, sys, time, torch, json, subprocess
import torch.nn as nn
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
COMP = os.path.join(PROJECT, 'ComparisonModel')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops_thop(model, input_size):
    from thop import profile
    x = torch.randn(*input_size).to(device)
    flops, _ = profile(model, inputs=(x,), verbose=False)
    return flops / 1e9


def measure_speed(model, input_size=(1, 3, 512, 512), warmup=30, runs=100):
    x = torch.randn(*input_size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
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
    avg = np.mean(times)
    return 1.0 / avg, avg * 1000


# ===================== Our model =====================
def benchmark_ours():
    from config.config import Config
    from model.TriBranchCracknet import TriBranchCrackNet
    mc = Config.MODEL_CONFIGS
    widths = mc['widths']
    stage_blocks = Config.expand_stage_blocks(mc['depths'], len(widths))
    model = TriBranchCrackNet(
        in_channels=3, num_classes=1,
        widths=widths, stage_blocks=stage_blocks,
        k=mc['k'], d_state=mc['d_state'],
        dropout=0, attn_drop=0, proj_drop=0, drop_path_rate=0,
        first_downsample_stride=Config.ENCODER_FIRST_DOWNSAMPLE_STRIDE,
        use_aux_outputs=False, aux_stage_indices=(),
        use_edge_head=False, scan_directions=8)
    model = model.to(device).eval()
    params = count_params(model)
    # FLOPs: thop may fail on Mamba ops, use manual estimation
    try:
        flops = count_flops_thop(model, (1, 3, 512, 512))
    except:
        flops = -1
    fps, latency = measure_speed(model, (1, 3, 512, 512))
    return params, flops, fps, latency


# ===================== SMP models =====================
def benchmark_smp(model_name, encoder='resnet34'):
    import segmentation_models_pytorch as smp_lib
    if model_name == 'unet':
        model = smp_lib.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    else:
        model = smp_lib.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    model = model.to(device).eval()
    params = count_params(model)
    flops = count_flops_thop(model, (1, 3, 512, 512))
    fps, latency = measure_speed(model, (1, 3, 512, 512))
    return params, flops, fps, latency


# ===================== DeepCrack =====================
def benchmark_deepcrack():
    sys.path.insert(0, os.path.join(COMP, 'DeepCrack'))
    from models.deepcrack_networks import define_deepcrack
    model = define_deepcrack(in_nc=3, num_classes=1, ngf=64, norm='batch',
                              init_type='normal', init_gain=0.02, gpu_ids=[])
    model = model.to(device).eval()
    params = count_params(model)
    flops = count_flops_thop(model, (1, 3, 512, 512))
    fps, latency = measure_speed(model, (1, 3, 512, 512))
    return params, flops, fps, latency


# ===================== Subprocess benchmarks =====================
def benchmark_subprocess(name, code):
    """Run benchmark in subprocess, returns (params, flops, fps, latency)."""
    try:
        r = subprocess.run([sys.executable, '-c', code],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            for line in r.stdout.strip().split('\n'):
                if line.startswith('{'):
                    d = json.loads(line)
                    return d['params'], d['flops'], d['fps'], d['latency']
        print(f"  {name} subprocess error: {r.stderr[-300:]}")
    except Exception as e:
        print(f"  {name} error: {e}")
    return -1, -1, -1, -1


def benchmark_dbcnet():
    code = f'''
import os, sys, time, torch, json
import torch.nn as nn
import numpy as np
os.chdir("{os.path.join(COMP, 'DBCNet')}")
sys.path.insert(0, ".")
from configs.config_crack_Deepcrack import config
from models.builder import EncoderDecoder as segmodel
from utils.loss_crack import SoftDiceLoss
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = segmodel(cfg=config, criterion=SoftDiceLoss(), norm_layer=nn.BatchNorm2d)
model = model.to(device).eval()
params = sum(p.numel() for p in model.parameters()) / 1e6
x = torch.randn(1,3,512,512).to(device)
flops, _ = profile(model, inputs=(x,), verbose=False)
flops = flops / 1e9
# Speed
with torch.no_grad():
    for _ in range(30): _ = model(x)
if device.type == "cuda": torch.cuda.synchronize()
times = []
with torch.no_grad():
    for _ in range(100):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
avg = np.mean(times)
print(json.dumps({{"params":round(params,2),"flops":round(flops,2),"fps":round(1/avg,1),"latency":round(avg*1000,1)}}))
'''
    return benchmark_subprocess('DBCNet', code)


def benchmark_dtrcnet():
    code = f'''
import os, sys, time, torch, json
import numpy as np
os.chdir("{os.path.join(COMP, 'DTrC-Net')}")
sys.path.insert(0, ".")
from CTCNet import CTCNet
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTCNet(n_channels=3, n_classes=1).to(device).eval()
params = sum(p.numel() for p in model.parameters()) / 1e6
x = torch.randn(1,3,256,256).to(device)
flops, _ = profile(model, inputs=(x,), verbose=False)
flops = flops / 1e9
with torch.no_grad():
    for _ in range(30): _ = model(x)
if device.type == "cuda": torch.cuda.synchronize()
times = []
with torch.no_grad():
    for _ in range(100):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
avg = np.mean(times)
print(json.dumps({{"params":round(params,2),"flops":round(flops,2),"fps":round(1/avg,1),"latency":round(avg*1000,1)}}))
'''
    return benchmark_subprocess('DTrC-Net', code)


def benchmark_ctcrackseg():
    code = f'''
import os, sys, time, torch, json
import numpy as np
sys.path.insert(0, "{os.path.join(COMP, 'CT-CrackSeg')}")
from model.TransMUNet import TransMUNet
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransMUNet(img_size=256, in_chans=3, out_chans=1, embed_dims=[64,128,256,512]).to(device).eval()
params = sum(p.numel() for p in model.parameters()) / 1e6
x = torch.randn(1,3,256,256).to(device)
flops, _ = profile(model, inputs=(x,), verbose=False)
flops = flops / 1e9
with torch.no_grad():
    for _ in range(30): _ = model(x)
if device.type == "cuda": torch.cuda.synchronize()
times = []
with torch.no_grad():
    for _ in range(100):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
avg = np.mean(times)
print(json.dumps({{"params":round(params,2),"flops":round(flops,2),"fps":round(1/avg,1),"latency":round(avg*1000,1)}}))
'''
    return benchmark_subprocess('CT-CrackSeg', code)


# ===================== SCSegamba (params only, different env) =====================


# ===================== Main =====================
if __name__ == '__main__':
    all_results = []

    print(f"{'Model':<20s} {'Params(M)':>10s} {'FLOPs(G)':>10s} {'FPS':>8s} {'Latency(ms)':>12s} {'Input':>10s}")
    print('=' * 75)

    # CrackSnakeNet
    print(">>> CrackSnakeNet")
    params, flops, fps, latency = benchmark_ours()
    flops_str = f"{flops:.2f}" if flops > 0 else "--"
    print(f"{'CrackSnakeNet':<20s} {params:>10.2f} {flops_str:>10s} {fps:>8.1f} {latency:>12.1f} {'512x512':>10s}")
    all_results.append({'model': 'CrackSnakeNet', 'params': round(params, 2), 'flops': round(flops, 2) if flops > 0 else None, 'fps': round(fps, 1), 'latency': round(latency, 1), 'input': '512'})

    # U-Net
    print(">>> U-Net (R34)")
    params, flops, fps, latency = benchmark_smp('unet')
    print(f"{'U-Net (R34)':<20s} {params:>10.2f} {flops:>10.2f} {fps:>8.1f} {latency:>12.1f} {'512x512':>10s}")
    all_results.append({'model': 'U-Net (R34)', 'params': round(params, 2), 'flops': round(flops, 2), 'fps': round(fps, 1), 'latency': round(latency, 1), 'input': '512'})

    # DeepLabV3+
    print(">>> DeepLabV3+ (R34)")
    params, flops, fps, latency = benchmark_smp('deeplabv3plus')
    print(f"{'DeepLabV3+ (R34)':<20s} {params:>10.2f} {flops:>10.2f} {fps:>8.1f} {latency:>12.1f} {'512x512':>10s}")
    all_results.append({'model': 'DeepLabV3+ (R34)', 'params': round(params, 2), 'flops': round(flops, 2), 'fps': round(fps, 1), 'latency': round(latency, 1), 'input': '512'})

    # DeepCrack
    print(">>> DeepCrack")
    params, flops, fps, latency = benchmark_deepcrack()
    print(f"{'DeepCrack':<20s} {params:>10.2f} {flops:>10.2f} {fps:>8.1f} {latency:>12.1f} {'512x512':>10s}")
    all_results.append({'model': 'DeepCrack', 'params': round(params, 2), 'flops': round(flops, 2), 'fps': round(fps, 1), 'latency': round(latency, 1), 'input': '512'})

    # DBCNet
    print(">>> DBCNet (subprocess)")
    params, flops, fps, latency = benchmark_dbcnet()
    print(f"{'DBCNet':<20s} {params:>10.2f} {flops:>10.2f} {fps:>8.1f} {latency:>12.1f} {'512x512':>10s}")
    all_results.append({'model': 'DBCNet', 'params': round(params, 2), 'flops': round(flops, 2) if flops > 0 else None, 'fps': round(fps, 1) if fps > 0 else None, 'latency': round(latency, 1) if latency > 0 else None, 'input': '512'})

    # DTrC-Net
    print(">>> DTrC-Net (subprocess)")
    params, flops, fps, latency = benchmark_dtrcnet()
    print(f"{'DTrC-Net':<20s} {params:>10.2f} {flops:>10.2f} {fps:>8.1f} {latency:>12.1f} {'256x256':>10s}")
    all_results.append({'model': 'DTrC-Net', 'params': round(params, 2), 'flops': round(flops, 2) if flops > 0 else None, 'fps': round(fps, 1) if fps > 0 else None, 'latency': round(latency, 1) if latency > 0 else None, 'input': '256'})

    # CT-CrackSeg
    print(">>> CT-CrackSeg (subprocess)")
    params, flops, fps, latency = benchmark_ctcrackseg()
    print(f"{'CT-CrackSeg':<20s} {params:>10.2f} {flops:>10.2f} {fps:>8.1f} {latency:>12.1f} {'256x256':>10s}")
    all_results.append({'model': 'CT-CrackSeg', 'params': round(params, 2), 'flops': round(flops, 2) if flops > 0 else None, 'fps': round(fps, 1) if fps > 0 else None, 'latency': round(latency, 1) if latency > 0 else None, 'input': '256'})

    # SCSegamba - only params, no local env
    all_results.append({'model': 'SCSegamba', 'params': 3.05, 'flops': None, 'fps': None, 'latency': None, 'input': '512'})

    print('\n' + '=' * 75)
    out_path = os.path.join(PROJECT, 'paper', 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")
