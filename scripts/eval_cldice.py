#!/usr/bin/env python
"""
Evaluate clDice (centerline Dice) for all models on DeepCrack dataset.

clDice measures topological/connectivity correctness:
    Tprec = |S_pred ∩ V_gt| / |S_pred|   (predicted skeleton inside GT mask)
    Tsens = |S_gt ∩ V_pred| / |S_gt|      (GT centerline covered by prediction)
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)

Reference: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
for Tubular Structure Segmentation", CVPR 2021.

Usage:
    python scripts/eval_cldice.py
    python scripts/eval_cldice.py --threshold 0.5
    python scripts/eval_cldice.py --sweep   # find optimal threshold per model
"""
import os
import sys
import argparse
import json

import cv2
import numpy as np
from skimage.morphology import skeletonize

# ---------------------------------------------------------------------------
# clDice computation
# ---------------------------------------------------------------------------

def compute_cldice(pred_bin, gt_mask, gt_centerline):
    """
    Compute clDice for a single image.

    Args:
        pred_bin: binary prediction (H, W), 0/1
        gt_mask: binary GT mask (H, W), 0/1
        gt_centerline: binary GT centerline (H, W), 0/1

    Returns:
        cldice, tprec, tsens
    """
    # Topology sensitivity: GT centerline covered by prediction
    cl_sum = gt_centerline.sum()
    if cl_sum == 0:
        tsens = 1.0
    else:
        tsens = (gt_centerline * pred_bin).sum() / cl_sum

    # Topology precision: predicted skeleton inside GT mask
    pred_skel = skeletonize(pred_bin.astype(bool)).astype(np.float64)
    skel_sum = pred_skel.sum()
    if skel_sum == 0:
        tprec = 0.0 if gt_mask.sum() > 0 else 1.0
    else:
        tprec = (pred_skel * gt_mask).sum() / skel_sum

    # clDice = harmonic mean
    if tprec + tsens == 0:
        cldice = 0.0
    else:
        cldice = 2.0 * tprec * tsens / (tprec + tsens)

    return cldice, tprec, tsens


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_BASE = os.path.join(PROJECT_ROOT, "paper", "InferenceResult")

# Original dataset (79 test, used by comparison models)
ORIG_DATASET = os.path.join(
    PROJECT_ROOT, "ComparisonModel", "dataset", "DeepCrack"
)
ORIG_MASK_DIR = os.path.join(ORIG_DATASET, "annotations", "test")
ORIG_CL_DIR = os.path.join(ORIG_DATASET, "centerlines", "test")

# 8:2 dataset (104 test, used by our model)
DS82_DATASET = os.path.join(
    PROJECT_ROOT, "ComparisonModel", "dataset_8_2", "DeepCrack", "test"
)
DS82_MASK_DIR = os.path.join(DS82_DATASET, "annotations")
DS82_CL_DIR = os.path.join(DS82_DATASET, "centerlines")

# Model registry: name -> (pred_dir, gt_source)
# gt_source: "orig" = original 79-image split, "8_2" = 8:2 104-image split
MODELS = {
    "U-Net":        (os.path.join(PRED_BASE, "DeepCrack", "unet"),          "orig"),
    "DeepLabV3+":   (os.path.join(PRED_BASE, "DeepCrack", "deeplabv3plus"), "orig"),
    "DTrC-Net":     (os.path.join(PRED_BASE, "DeepCrack", "dtrcnet"),       "orig"),
    "DeepCrack":    (os.path.join(PRED_BASE, "DeepCrack", "deepcrack_model"), "orig"),
    "DBCNet":       (os.path.join(PRED_BASE, "DeepCrack", "dbcnet"),        "orig"),
    "SCSegamba":    (os.path.join(PRED_BASE, "DeepCrack", "scsegamba"),      "orig"),
    "Ours":         (os.path.join(PRED_BASE, "CrackMambaTransformerNet"),    "orig"),
}


def load_gt_pairs(gt_source):
    """Load GT mask and centerline pairs.

    Raw GT format: 0=crack, 255=background. We invert to 1=crack, 0=bg.

    Returns: {stem: (mask_uint8, centerline_uint8)}
    """
    if gt_source == "orig":
        mask_dir, cl_dir = ORIG_MASK_DIR, ORIG_CL_DIR
    else:
        mask_dir, cl_dir = DS82_MASK_DIR, DS82_CL_DIR

    gt_files = sorted(os.listdir(mask_dir))
    pairs = {}
    for fname in gt_files:
        if not fname.endswith(".png"):
            continue
        stem = fname.replace(".png", "")
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        cl = cv2.imread(os.path.join(cl_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is None or cl is None:
            continue
        # Invert: 0=crack → 1, 255=bg → 0
        pairs[stem] = ((mask < 128).astype(np.uint8), (cl < 128).astype(np.uint8))
    return pairs


def find_pred_file(pred_dir, gt_stem):
    """Find prediction file matching a GT stem."""
    exact = os.path.join(pred_dir, gt_stem + ".png")
    if os.path.isfile(exact):
        return exact
    return None


def evaluate_model(model_name, pred_dir, gt_pairs, threshold=0.5):
    """Evaluate clDice for a model at a given threshold."""
    cldice_list, tprec_list, tsens_list = [], [], []

    for gt_stem, (gt_mask, gt_cl) in gt_pairs.items():
        pred_path = find_pred_file(pred_dir, gt_stem)
        if pred_path is None:
            continue

        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            continue

        # Resize prediction to match GT if needed
        if pred_img.shape != gt_mask.shape:
            pred_img = cv2.resize(pred_img, (gt_mask.shape[1], gt_mask.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

        # Binarize prediction (high value = crack)
        pred_bin = (pred_img / 255.0 > threshold).astype(np.uint8)

        cldice, tprec, tsens = compute_cldice(pred_bin, gt_mask, gt_cl)
        cldice_list.append(cldice)
        tprec_list.append(tprec)
        tsens_list.append(tsens)

    if len(cldice_list) == 0:
        return None

    return {
        "clDice": float(np.mean(cldice_list)),
        "Tprec": float(np.mean(tprec_list)),
        "Tsens": float(np.mean(tsens_list)),
        "n_images": len(cldice_list),
    }


def sweep_threshold(model_name, pred_dir, gt_pairs, step=0.01):
    """Find optimal threshold for clDice."""
    best_cldice = -1
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, step):
        result = evaluate_model(model_name, pred_dir, gt_pairs, threshold=thresh)
        if result and result["clDice"] > best_cldice:
            best_cldice = result["clDice"]
            best_thresh = thresh
    result = evaluate_model(model_name, pred_dir, gt_pairs, threshold=best_thresh)
    result["best_threshold"] = float(best_thresh)
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate clDice on DeepCrack dataset")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold (default: 0.5)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds to find optimal clDice per model")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to evaluate (default: all)")
    args = parser.parse_args()

    # Pre-load both GT sets
    print("Loading GT (original 79-image split)...")
    gt_orig = load_gt_pairs("orig")
    print(f"  Loaded {len(gt_orig)} pairs")
    print("Loading GT (8:2 104-image split)...")
    gt_82 = load_gt_pairs("8_2")
    print(f"  Loaded {len(gt_82)} pairs\n")

    gt_sets = {"orig": gt_orig, "8_2": gt_82}

    models = MODELS
    if args.models:
        models = {k: v for k, v in MODELS.items() if k in args.models}

    results = {}
    for model_name, (pred_dir, gt_source) in models.items():
        if not os.path.isdir(pred_dir):
            print(f"  [{model_name}] SKIP — directory not found")
            continue

        gt_pairs = gt_sets[gt_source]

        if args.sweep:
            print(f"  [{model_name}] Sweeping thresholds...", end="", flush=True)
            result = sweep_threshold(model_name, pred_dir, gt_pairs)
            if result:
                print(f" best_thresh={result['best_threshold']:.2f}")
        else:
            result = evaluate_model(model_name, pred_dir, gt_pairs,
                                    threshold=args.threshold)

        if result is None:
            print(f"  [{model_name}] SKIP — no matching predictions found")
            continue

        results[model_name] = result
        print(f"  [{model_name:15s}] clDice={result['clDice']:.4f}  "
              f"Tprec={result['Tprec']:.4f}  Tsens={result['Tsens']:.4f}  "
              f"(n={result['n_images']})")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<15s} {'clDice':>8s} {'Tprec':>8s} {'Tsens':>8s} {'n':>5s}")
    print("-" * 70)
    for name in MODELS:
        if name in results:
            r = results[name]
            extra = f"  (thresh={r['best_threshold']:.2f})" if "best_threshold" in r else ""
            print(f"{name:<15s} {r['clDice']:>8.4f} {r['Tprec']:>8.4f} "
                  f"{r['Tsens']:>8.4f} {r['n_images']:>5d}{extra}")
    print("=" * 70)

    # Save
    out_path = os.path.join(PROJECT_ROOT, "paper", "cldice_results_DeepCrack.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
