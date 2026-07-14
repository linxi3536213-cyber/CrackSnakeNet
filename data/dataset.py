import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from typing import Tuple, List, Optional
import glob
from config.config import Config
from data.transform import get_transforms
from data.calculateMeanStd import compute_mean_std_from_dir

# Reduce OpenCV thread contention inside DataLoader workers.
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class CrackDatasetSplitter:
    """裂缝数据集分割器 (支持 images/annotations/centerlines 目录结构)"""

    def __init__(self):
        root = Config.DATASET_ROOT
        img_dir = Config.IMAGE_DIR
        mask_dir = Config.MASK_DIR
        cl_dir = Config.CENTERLINE_DIR
        train_split = Config.TRAIN_SPLIT
        test_split = Config.TEST_SPLIT

        # 构建路径: root/images/training, root/annotations/training, ...
        self.train_images_dir = os.path.join(root, img_dir, train_split)
        self.test_images_dir = os.path.join(root, img_dir, test_split)
        self.train_masks_dir = os.path.join(root, mask_dir, train_split)
        self.test_masks_dir = os.path.join(root, mask_dir, test_split)
        self.train_centerlines_dir = os.path.join(root, cl_dir, train_split)
        self.test_centerlines_dir = os.path.join(root, cl_dir, test_split)

        # 检查核心目录
        required = [self.train_images_dir, self.train_masks_dir,
                     self.test_images_dir, self.test_masks_dir]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"数据集目录不存在: {missing}")

        # 查找 (image, mask, centerline) 三元组
        self.train_pairs = self._find_triples(
            self.train_images_dir, self.train_masks_dir, self.train_centerlines_dir)
        self.test_pairs = self._find_triples(
            self.test_images_dir, self.test_masks_dir, self.test_centerlines_dir)

    def _find_triples(self, images_dir: str, masks_dir: str,
                       centerlines_dir: str) -> List[Tuple[str, str, Optional[str]]]:
        """查找 (image, mask, centerline) 三元组"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))

        print(f"Found {len(image_files)} images in {images_dir}")

        valid_triples = []
        has_centerlines = os.path.isdir(centerlines_dir)

        for image_path in sorted(image_files):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            if not base_name:
                continue

            # 匹配 mask
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                candidate = os.path.join(masks_dir, base_name + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                continue

            # 匹配 centerline (可选)
            cl_path = None
            if has_centerlines:
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    candidate = os.path.join(centerlines_dir, base_name + ext)
                    if os.path.exists(candidate):
                        cl_path = candidate
                        break

            valid_triples.append((image_path, mask_path, cl_path))

        print(f"Found {len(valid_triples)} valid triples "
              f"({sum(1 for _, _, c in valid_triples if c)} with centerlines)")
        return valid_triples

    def get_train_val_pairs(self):
        """获取训练和验证数据对"""
        print(f"训练集样本数: {len(self.train_pairs)}")
        print(f"验证集样本数: {len(self.test_pairs)}")
        return self.train_pairs, self.test_pairs


class CrackDataset(Dataset):
    """裂缝检测数据集 (支持 centerline + mask 反转)"""

    def __init__(self, valid_pairs: list,
                 mean: Tuple[float, float, float], std: Tuple[float, float, float],
                 image_size: Tuple[int, int] = Config.IMAGE_SIZE, mode: str = 'train',
                 augmentation_strength: str = None,
                 invert_mask: bool = False):
        self.valid_pairs = valid_pairs
        self.image_size = image_size
        self.mode = mode
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.invert_mask = invert_mask

        if augmentation_strength is None:
            augmentation_strength = getattr(Config, 'AUGMENTATION_STRENGTH', 'weak')

        self.transform = get_transforms(mode=mode, image_size=image_size,
                                       augmentation_strength=augmentation_strength)

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path, cl_path = self.valid_pairs[idx]

        # 加载图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 加载掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码: {mask_path}")

        # 二值化 + 反转 (黑色=裂缝 → 1, 白色=背景 → 0)
        if self.invert_mask:
            mask = (mask < 128).astype(np.uint8)
        else:
            mask = (mask > 127).astype(np.uint8)

        # 加载 centerline
        centerline = None
        if cl_path is not None:
            centerline = cv2.imread(cl_path, cv2.IMREAD_GRAYSCALE)
            if centerline is not None:
                if self.invert_mask:
                    centerline = (centerline < 128).astype(np.uint8)
                else:
                    centerline = (centerline > 127).astype(np.uint8)

        # 应用变换 (image + mask + centerline 同步变换)
        if centerline is not None:
            transformed = self.transform(image=image, mask=mask, centerline=centerline)
            centerline = transformed['centerline'].astype(np.float32)
        else:
            transformed = self.transform(image=image, mask=mask)

        image = transformed['image'].astype(np.float32) / 255.0
        mask = transformed['mask'].astype(np.float32)

        # 归一化
        safe_std = np.maximum(self.std, 1e-6)
        image = (image - self.mean) / safe_std

        if np.any(np.isnan(image)):
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        image = np.clip(image, -10.0, 10.0)

        # 转换为 PyTorch 张量
        image = torch.from_numpy(image.transpose(2, 0, 1)).contiguous()   # [3,H,W]
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()           # [1,H,W]

        if centerline is not None:
            centerline = torch.from_numpy(centerline).unsqueeze(0).contiguous()  # [1,H,W]
        else:
            # 无 centerline 时返回全零
            centerline = torch.zeros_like(mask)

        return image, mask, centerline


class CrackDataLoader:
    """裂缝检测数据加载器"""

    def __init__(self, train_pairs: list,
                 valid_pairs: list,
                 image_size: Tuple[int, int] = Config.IMAGE_SIZE,
                 batch_size: int = Config.BATCH_SIZE):

        # 从训练集动态计算均值与标准差
        print("正在计算训练集的 mean 和 std...")
        train_images_dir = os.path.join(
            Config.DATASET_ROOT, Config.IMAGE_DIR, Config.TRAIN_SPLIT)
        mean_np, std_np = compute_mean_std_from_dir(train_images_dir)
        self.mean = tuple(mean_np.tolist())
        self.std = tuple(std_np.tolist())
        print(f"计算完成: mean={self.mean}, std={self.std}")

        invert = getattr(Config, 'INVERT_MASK', False)

        self.train_dataset = CrackDataset(
            valid_pairs=train_pairs,
            mean=self.mean, std=self.std,
            image_size=image_size, mode="train",
            invert_mask=invert,
        )

        self.valid_dataset = CrackDataset(
            valid_pairs=valid_pairs,
            mean=self.mean, std=self.std,
            image_size=image_size, mode="val",
            invert_mask=invert,
        )

        # DataLoader 参数
        num_workers = max(0, int(getattr(Config, "NUM_WORKERS", 4)))
        pin_memory = bool(getattr(Config, "PIN_MEMORY", True))
        prefetch_factor = getattr(Config, "PREFETCH_FACTOR", 2)
        persistent_workers = bool(getattr(Config, "PERSISTENT_WORKERS", num_workers > 0))

        if num_workers == 0:
            persistent_workers = False
            prefetch_factor = None

        loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor

        self.train_loader = DataLoader(
            self.train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
        self.val_loader = DataLoader(
            self.valid_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    def get_loaders(self):
        return self.train_loader, self.val_loader
