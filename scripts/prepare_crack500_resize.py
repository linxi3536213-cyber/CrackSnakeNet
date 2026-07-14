#!/usr/bin/env python3
"""
Prepare CRACK500 dataset: resize original images to 512x512 (no sliding window).
Merge training+validation as train, test as test.

Why resize instead of sliding window:
- Original resolution: 2560x1440 ~ 3264x2448
- Sliding window (512x512, stride 256) produces 10,075 train / 4,104 test patches → too slow to train
- Direct resize to 512x512 reduces crack width from ~26px to ~2.8px,
  creating a challenging thin-crack benchmark that tests fine-structure detection
- 300 train / 199 test images: training completes in ~5h (same as DeepCrack/CrackTree260)
"""
import os, sys, cv2
import numpy as np
from tqdm import tqdm

SRC = "/home/andy/desktop/CrackDectection/dataset/omnicrack30k_classified/CRACK500"
DST = "/home/andy/desktop/CrackDectection/CrackMambaTransformerNet/ablation/CrackMambaTransformerNetTest3.3TriFFMplus/ComparisonModel/dataset_8_2/CRACK500_resize"
SIZE = 512

def resize_and_save(src_img, src_mask, src_cl, dst_img, dst_mask, dst_cl):
    """Resize image, mask, and centerline to SIZExSIZE."""
    img = cv2.imread(src_img)
    if img is None:
        return False
    img_resized = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(dst_img, img_resized)

    mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        mask_resized = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(dst_mask, mask_resized)

    cl = cv2.imread(src_cl, cv2.IMREAD_GRAYSCALE)
    if cl is not None:
        cl_resized = cv2.resize(cl, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(dst_cl, cl_resized)

    return True


def process_split(src_splits, dst_split_name):
    """Process one or more source splits into a destination split."""
    dst_img_dir = os.path.join(DST, dst_split_name, 'images')
    dst_mask_dir = os.path.join(DST, dst_split_name, 'annotations')
    dst_cl_dir = os.path.join(DST, dst_split_name, 'centerlines')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    os.makedirs(dst_cl_dir, exist_ok=True)

    count = 0
    for split in src_splits:
        src_img_dir = os.path.join(SRC, 'images', split)
        src_mask_dir = os.path.join(SRC, 'annotations', split)
        src_cl_dir = os.path.join(SRC, 'centerlines', split)

        if not os.path.isdir(src_img_dir):
            print(f"  SKIP: {src_img_dir} not found")
            continue

        files = sorted(os.listdir(src_img_dir))
        for f in tqdm(files, desc=f"{split} -> {dst_split_name}"):
            bn = os.path.splitext(f)[0]
            src_img = os.path.join(src_img_dir, f)

            # Find matching mask and centerline
            src_mask, src_cl = None, None
            for ext in ['.png', '.jpg', '.bmp', '.tif']:
                mp = os.path.join(src_mask_dir, bn + ext)
                cp = os.path.join(src_cl_dir, bn + ext)
                if os.path.exists(mp):
                    src_mask = mp
                if os.path.exists(cp):
                    src_cl = cp

            if src_mask is None:
                continue

            dst_img = os.path.join(dst_img_dir, bn + '.png')
            dst_mask = os.path.join(dst_mask_dir, bn + '.png')
            dst_cl = os.path.join(dst_cl_dir, bn + '.png')

            if resize_and_save(src_img, src_mask, src_cl or src_mask, dst_img, dst_mask, dst_cl):
                count += 1

    print(f"  {dst_split_name}: {count} images saved")
    return count


if __name__ == '__main__':
    print(f"Source: {SRC}")
    print(f"Destination: {DST}")
    print(f"Resize to: {SIZE}x{SIZE}")
    print()

    # Merge training + validation -> train
    n_train = process_split(['training', 'validation'], 'train')
    print()

    # test -> test
    n_test = process_split(['test'], 'test')
    print()

    print(f"Done: {n_train} train + {n_test} test = {n_train + n_test} total")
