import json
import os
import glob
import cv2
import numpy as np
from typing import List, Tuple


IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


def list_images(images_dir: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return files


def compute_mean_std(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_sq_rgb = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in image_paths:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[warn] failed to read image: {path}")
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        h, w, _ = img.shape
        pixels = h * w
        pixel_count += pixels
        sum_rgb += img.reshape(-1, 3).sum(axis=0)
        sum_sq_rgb += (img.reshape(-1, 3) ** 2).sum(axis=0)

    if pixel_count == 0:
        raise RuntimeError("No valid images to compute mean/std.")

    mean = sum_rgb / pixel_count
    var = (sum_sq_rgb / pixel_count) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std


def compute_mean_std_from_dir(images_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: list images then compute mean/std."""
    image_paths = list_images(images_dir)
    return compute_mean_std(image_paths)


def save_mean_std(mean: np.ndarray, std: np.ndarray, json_path: str):
    """Save mean/std to a JSON file."""
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Mean/std saved to: {json_path}")


def load_mean_std(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load mean/std from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"mean_std file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return mean, std


def main():
    """Main function to compute and save mean/std for the training dataset."""
    import sys

    # Default paths (can be overridden by command line args)
    if len(sys.argv) > 1:
        images_dir = sys.argv[1]
    else:
        # Try to import Config to get the default training data path
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from config.config import Config
            if hasattr(Config, "DATASET_ROOT") and hasattr(Config, "TRAIN_SPLIT"):
                images_dir = os.path.join(Config.DATASET_ROOT, Config.IMAGE_DIR, Config.TRAIN_SPLIT)
            else:
                images_dir = os.path.join(Config.TRAIN_DATA_PATH, Config.IMAGE_DIR)
        except ImportError:
            print("Error: Cannot import Config. Please provide images directory as argument.")
            print("Usage: python calculateMeanStd.py [images_dir] [output_json]")
            sys.exit(1)

    if len(sys.argv) > 2:
        output_json = sys.argv[2]
    else:
        output_json = os.path.join(os.path.dirname(__file__), "..", "config", "mean_std.json")

    print(f"Computing mean/std from: {images_dir}")
    print(f"Output file: {output_json}")

    # Compute mean and std
    mean, std = compute_mean_std_from_dir(images_dir)

    print(f"\nResults:")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")

    # Save to JSON
    save_mean_std(mean, std, output_json)
    print("\nDone!")


if __name__ == "__main__":
    main()
