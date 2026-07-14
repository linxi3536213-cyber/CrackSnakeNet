#!/usr/bin/env python
"""
OmniCrack30k 子数据集分类脚本

将 omnicrack30k 按文件名前缀拆分为独立子数据集，
每个子数据集保持与原始目录相同的结构:
  sub_dataset/
    images/
      training/
      validation/
      test/
    annotations/
      training/
      validation/
      test/
    centerlines/
      training/
      validation/
      test/

使用符号链接 (symlink) 避免重复占用磁盘空间。
如需复制文件请将 USE_SYMLINK 改为 False。

Usage:
    python classify.py
    python classify.py --src /path/to/omnicrack30k --dst /path/to/output
    python classify.py --copy   # 复制文件而非符号链接
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


# ==================== 数据集前缀识别规则 ====================

DATASET_PATTERNS = [
    # (regex, dataset_name) — 顺序重要，长前缀优先
    (r'^CrackTree\d+_',    'CrackTree260'),
    (r'^CrackLS\d+_',      'CrackLS315'),
    (r'^CrackSeg\d*k?_',   'CrackSeg9k'),
    (r'^CRKWH\d+_',        'CRKWH100'),
    (r'^CRACK500_',         'CRACK500'),
    (r'^GAPS\d+_',          'GAPS384'),
    (r'^Stone\d+_',         'Stone331'),
    (r'^UAV\d+_',           'UAV75'),
    (r'^Khanh\d+k_',       'Khanh11k'),
    (r'^TopoDS_',           'TopoDS'),
    (r'^Topologically_',    'TopoDS'),
    (r'^DeepCrack_',        'DeepCrack'),
    (r'^CrSpEE_',           'CrSpEE'),
    (r'^Ceramic_',          'Ceramic'),
    (r'^Masonry_',          'Masonry'),
    (r'^BCL_[cfs]',         'BCL'),
    (r'^AEL_',              'AEL'),
    (r'^CFD_',              'CFD'),
    (r'^CSSC_',             'CSSC'),
    (r'^DIC_',              'DIC'),
    (r'^DCD_',              'DCD'),
    (r'^LCW_',              'LCW'),
    (r'^S2DS_',             'S2DS'),
    (r'^SYN_',              'SYN'),
    (r'^GAPs_',             'GAPs'),
]

# 预编译正则
_COMPILED_PATTERNS = [(re.compile(pat), name) for pat, name in DATASET_PATTERNS]


def classify_file(filename: str) -> str:
    """根据文件名识别所属子数据集，返回数据集名或 'unknown'"""
    for regex, name in _COMPILED_PATTERNS:
        if regex.match(filename):
            return name
    return 'unknown'


# ==================== 主逻辑 ====================

def classify_dataset(src_root, dst_root, use_symlink: bool = True):
    """
    扫描 src_root 下的 images/annotations/centerlines，
    按子数据集分类到 dst_root 下。
    """
    src_root = Path(src_root).resolve()
    dst_root = Path(dst_root).resolve()

    subdirs = ['images', 'annotations', 'centerlines']
    splits = ['training', 'validation', 'test']

    # 1. 扫描所有图片文件，建立 {dataset -> {split -> [filenames]}} 映射
    dataset_map = defaultdict(lambda: defaultdict(list))
    unknown_files = defaultdict(list)

    for split in splits:
        img_dir = src_root / 'images' / split
        if not img_dir.exists():
            print(f"[WARN] 目录不存在: {img_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith('.png'):
                continue
            ds_name = classify_file(fname)
            if ds_name == 'unknown':
                unknown_files[split].append(fname)
            else:
                dataset_map[ds_name][split].append(fname)

    # 2. 打印统计
    print("=" * 65)
    print(f"{'子数据集':<16} {'training':>8} {'validation':>10} {'test':>8} {'total':>8}")
    print("=" * 65)

    grand_total = 0
    for ds_name in sorted(dataset_map.keys()):
        splits_data = dataset_map[ds_name]
        tr = len(splits_data.get('training', []))
        va = len(splits_data.get('validation', []))
        te = len(splits_data.get('test', []))
        total = tr + va + te
        grand_total += total
        print(f"  {ds_name:<14} {tr:>8} {va:>10} {te:>8} {total:>8}")

    print("-" * 65)
    unk_total = sum(len(v) for v in unknown_files.values())
    if unk_total > 0:
        print(f"  {'unknown':<14} {len(unknown_files.get('training',[])):>8} "
              f"{len(unknown_files.get('validation',[])):>10} "
              f"{len(unknown_files.get('test',[])):>8} {unk_total:>8}")
    print(f"  {'TOTAL':<14} {'':>8} {'':>10} {'':>8} {grand_total + unk_total:>8}")
    print("=" * 65)
    print()

    if unk_total > 0:
        print(f"[WARN] {unk_total} 个文件无法分类:")
        for split, files in unknown_files.items():
            for f in files[:5]:
                print(f"  {split}/{f}")
            if len(files) > 5:
                print(f"  ... 还有 {len(files) - 5} 个")
        print()

    # 3. 创建目录并链接/复制文件
    action = "符号链接" if use_symlink else "复制"
    print(f"开始分类 ({action})...")
    print(f"  源目录: {src_root}")
    print(f"  目标目录: {dst_root}")
    print()

    file_count = 0
    skip_count = 0

    for ds_name in sorted(dataset_map.keys()):
        splits_data = dataset_map[ds_name]
        ds_total = sum(len(v) for v in splits_data.values())
        print(f"  [{ds_name}] {ds_total} 张图片...", end=" ")

        for split, filenames in splits_data.items():
            for subdir in subdirs:
                # 源文件夹
                src_dir = src_root / subdir / split
                # 目标文件夹
                dst_dir = dst_root / ds_name / subdir / split
                dst_dir.mkdir(parents=True, exist_ok=True)

                for fname in filenames:
                    src_file = src_dir / fname
                    dst_file = dst_dir / fname

                    if not src_file.exists():
                        # annotations/centerlines 可能不是每张都有
                        continue

                    if dst_file.exists() or dst_file.is_symlink():
                        skip_count += 1
                        continue

                    if use_symlink:
                        os.symlink(src_file, dst_file)
                    else:
                        shutil.copy2(src_file, dst_file)
                    file_count += 1

        print("done")

    print()
    print(f"完成! 共处理 {file_count} 个文件" +
          (f", 跳过 {skip_count} 个已存在" if skip_count else ""))
    print(f"输出目录: {dst_root}")

    # 4. 每个子数据集打印最终目录结构
    print()
    print("生成的子数据集目录:")
    for ds_name in sorted(dataset_map.keys()):
        ds_path = dst_root / ds_name
        counts = []
        for subdir in subdirs:
            for split in splits:
                d = ds_path / subdir / split
                if d.exists():
                    n = len(list(d.iterdir()))
                    if n > 0:
                        counts.append(f"{subdir}/{split}: {n}")
        print(f"  {ds_name}/")
        for c in counts:
            print(f"    {c}")


def main():
    parser = argparse.ArgumentParser(description="OmniCrack30k 子数据集分类")
    parser.add_argument(
        '--src', type=str,
        default='/home/andy/desktop/CrackDectection/dataset/omnicrack30k',
        help='OmniCrack30k 源目录'
    )
    parser.add_argument(
        '--dst', type=str,
        default='/home/andy/desktop/CrackDectection/dataset/omnicrack30k_classified',
        help='分类输出目录'
    )
    parser.add_argument(
        '--copy', action='store_true',
        help='复制文件而非创建符号链接'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"[ERROR] 源目录不存在: {args.src}")
        sys.exit(1)

    classify_dataset(args.src, args.dst, use_symlink=not args.copy)


if __name__ == '__main__':
    main()
