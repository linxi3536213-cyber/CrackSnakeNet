# CrackSnakeNet

Official implementation of **CrackSnakeNet: Multi-Directional Snake Scanning with Mamba for Topology-Preserving Crack Segmentation**.

CrackSnakeNet is a hybrid CNN–Mamba encoder–decoder for thin-crack segmentation. Its core module, **SnakeMamba2D**, scans feature maps along eight boustrophedon (snake-like) trajectories aligned with the principal crack directions, preserving spatial adjacency along crack trajectories of arbitrary orientation. A **centerline recall loss** enforces prediction coverage along crack medial axes, and **Crack-Oriented (CO) Blocks** with strip-shaped kernels enhance the representation of single-pixel crack structures. The model has 9.14M parameters and is trained from scratch (no ImageNet pretraining).

## Repository structure

```
model/     Network definition
             TriBranchCracknet.py  main network (stem skip, decoder)
             MambaBlock.py         SnakeMamba2D / MultiScaleSnakeMamba2D
             FusionBlock.py        bidirectional CNN–Mamba cross-attention
             CNNBlock.py           Crack-Oriented Block, Dynamic Attention Fusion
             snake_scan.py         eight-direction boustrophedon scan indices
config/    Training / model configuration
train/     Training loop (PyTorch Lightning)
data/      Dataset and transforms
utils/     Utilities (metrics, losses, etc.)
scripts/   Training, inference, evaluation, and analysis scripts
test/      Tests
```

## Requirements

See `requirements.txt`. Key dependencies: PyTorch, PyTorch Lightning, `mamba-ssm`, `causal-conv1d`.

```bash
pip install -r requirements.txt
```

## Training

```bash
bash run_ct260.sh       # CrackTree260
bash run_crack500.sh    # CRACK500 (512-resize)
```

## Note

The code will be made publicly available upon acceptance of the paper. Datasets and pretrained weights are not included in this repository.
