import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from model.CNNBlock import make_group_norm


def generate_snake_scan_indices(H: int, W: int, scan_type: str = 'diagonal') -> torch.Tensor:
    """
    生成蛇形扫描索引
    
    参数:
        H: 高度
        W: 宽度
        scan_type: 扫描类型
            - 'diagonal': 主对角线蛇形
            - 'anti_diagonal': 反对角线蛇形
            - 'row': 行扫描（Z字形）
            - 'column': 列扫描（Z字形）
            - 'enhanced_diagonal': 优化主对角线蛇形
            - 'enhanced_anti_diagonal': 优化反对角线蛇形
            - 'fracture_45': 45度裂缝方向（对角线反向扫描）
            - 'fracture_135': 135度裂缝方向（反对角线反向扫描）
            - 'row_enhanced': 增强行扫描
            - 'column_enhanced': 增强列扫描
    
    返回:
        indices: [H*W] 扫描索引序列
    """
    indices = torch.zeros(H * W, dtype=torch.long)

    reverse = False
    if scan_type == 'enhanced_diagonal':
        scan_type = 'diagonal'
    elif scan_type == 'enhanced_anti_diagonal':
        scan_type = 'anti_diagonal'
    elif scan_type == 'fracture_45':
        scan_type = 'diagonal'
        reverse = True
    elif scan_type == 'fracture_135':
        scan_type = 'anti_diagonal'
        reverse = True
    elif scan_type == 'row_enhanced':
        scan_type = 'row'
    elif scan_type == 'column_enhanced':
        scan_type = 'column'
    elif scan_type == 'row_reverse':
        scan_type = 'row'
        reverse = True
    elif scan_type == 'column_reverse':
        scan_type = 'column'
        reverse = True
    
    if scan_type == 'diagonal':
        # 主对角线蛇形扫描（从左上到右下）
        # 沿着和为常数的对角线扫描
        idx = 0
        for diag_sum in range(H + W - 1):
            coords = []
            for i in range(max(0, diag_sum - W + 1), min(H, diag_sum + 1)):
                j = diag_sum - i
                coords.append((i, j))
            
            # 奇数对角线反向，实现蛇形
            if diag_sum % 2 == 1:
                coords = coords[::-1]
            
            for i, j in coords:
                indices[idx] = i * W + j
                idx += 1
                
    elif scan_type == 'anti_diagonal':
        # 反对角线蛇形扫描（从右上到左下）
        # 沿着差为常数的对角线扫描
        idx = 0
        for diag_diff in range(-(W - 1), H):
            coords = []
            for i in range(max(0, diag_diff), min(H, W + diag_diff)):
                j = i - diag_diff
                coords.append((i, j))
            
            # 奇数对角线反向，实现蛇形
            if diag_diff % 2 == 0:
                coords = coords[::-1]
            
            for i, j in coords:
                indices[idx] = i * W + j
                idx += 1
                
    elif scan_type == 'row':
        # 行方向Z字形扫描
        idx = 0
        for i in range(H):
            if i % 2 == 0:  # 偶数行从左到右
                for j in range(W):
                    indices[idx] = i * W + j
                    idx += 1
            else:  # 奇数行从右到左
                for j in range(W - 1, -1, -1):
                    indices[idx] = i * W + j
                    idx += 1
                    
    elif scan_type == 'column':
        # 列方向Z字形扫描
        idx = 0
        for j in range(W):
            if j % 2 == 0:  # 偶数列从上到下
                for i in range(H):
                    indices[idx] = i * W + j
                    idx += 1
            else:  # 奇数列从下到上
                for i in range(H - 1, -1, -1):
                    indices[idx] = i * W + j
                    idx += 1
    else:
        raise ValueError(f"Unknown scan_type: {scan_type}")

    if reverse:
        indices = indices.flip(0)
    
    return indices


def generate_reverse_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    生成反向索引（用于从扫描序列恢复到原始空间）
    
    参数:
        indices: [N] 前向扫描索引
        
    返回:
        reverse_indices: [N] 反向索引，使得 x[reverse_indices][indices] == x
    """
    N = indices.shape[0]
    reverse_indices = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        reverse_indices[indices[i]] = i
    return reverse_indices


class SnakeScanModule(nn.Module):
    """
    蛇形扫描模块
    支持多方向的蛇形扫描：对角线、反对角线、行、列及增强/裂缝方向
    """
    
    def __init__(self):
        super().__init__()
        # 缓存不同尺寸的扫描索引
        self.scan_indices_cache = {}
        self.reverse_indices_cache = {}
        
    def _get_scan_indices(
        self, 
        H: int, 
        W: int, 
        scan_type: str,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取或生成扫描索引（带缓存）"""
        key = (H, W, scan_type, device)
        
        if key not in self.scan_indices_cache:
            scan_indices = generate_snake_scan_indices(H, W, scan_type).to(device)
            reverse_indices = generate_reverse_indices(scan_indices).to(device)
            self.scan_indices_cache[key] = scan_indices
            self.reverse_indices_cache[key] = reverse_indices
            
        return self.scan_indices_cache[key], self.reverse_indices_cache[key]
    
    def scan(
        self, 
        x: torch.Tensor, 
        scan_type: str = 'diagonal'
    ) -> torch.Tensor:
        """
        将2D特征图按照指定路径扫描成1D序列
        
        参数:
            x: [B, C, H, W] 输入特征图
            scan_type: 扫描类型
            
        返回:
            x_scanned: [B, N, C] 扫描后的序列，N=H*W
        """
        B, C, H, W = x.shape
        scan_indices, _ = self._get_scan_indices(H, W, scan_type, x.device)
        
        # 展平为 [B, C, H*W]
        x_flat = x.flatten(2)  # [B, C, H*W]
        
        # 按照扫描索引重排
        x_scanned = x_flat[:, :, scan_indices]  # [B, C, N]
        
        # 转置为 [B, N, C] 以匹配Mamba输入格式
        x_scanned = x_scanned.transpose(1, 2)  # [B, N, C]
        
        return x_scanned
    
    def unscan(
        self, 
        x_seq: torch.Tensor,
        H: int,
        W: int,
        scan_type: str = 'diagonal'
    ) -> torch.Tensor:
        """
        将1D序列恢复为2D特征图
        
        参数:
            x_seq: [B, N, C] 扫描序列
            H: 目标高度
            W: 目标宽度
            scan_type: 扫描类型（需与scan时一致）
            
        返回:
            x: [B, C, H, W] 恢复的特征图
        """
        B, N, C = x_seq.shape
        _, reverse_indices = self._get_scan_indices(H, W, scan_type, x_seq.device)
        
        # 转置为 [B, C, N]
        x_seq = x_seq.transpose(1, 2)  # [B, C, N]
        
        # 按照反向索引恢复原始顺序
        x_flat = x_seq[:, :, reverse_indices]  # [B, C, H*W]
        
        # 重塑为2D
        x = x_flat.view(B, C, H, W)
        
        return x


class MultiDirectionalSnakeScan(nn.Module):
    """
    多方向蛇形扫描
    实现增强后的多方向扫描，并引入方向权重学习
    """
    
    def __init__(self, num_directions: int = 4, channels: Optional[int] = None, merge_mode: str = 'add'):
        super().__init__()
        self.scanner = SnakeScanModule()
        self.merge_mode = merge_mode

        # 定义增强扫描方向（8方向覆盖所有裂缝走向）
        scan_types = [
            'enhanced_diagonal',         # 1. ↘ 主对角线
            'enhanced_anti_diagonal',    # 2. ↙ 反对角线
            'fracture_45',              # 3. ↖ 对角线反向
            'fracture_135',             # 4. ↗ 反对角线反向
            'row_enhanced',             # 5. → 行扫描
            'column_enhanced',          # 6. ↓ 列扫描
            'row_reverse',              # 7. ← 行扫描反向
            'column_reverse',           # 8. ↑ 列扫描反向
        ]
        if num_directions < 1 or num_directions > len(scan_types):
            raise ValueError(
                f"num_directions must be in [1, {len(scan_types)}], got {num_directions}"
            )
        self.scan_types = scan_types[:num_directions]
        self.num_directions = len(self.scan_types)

        # 如果使用concat模式，添加1×1卷积+归一化+激活进行通道压缩
        if merge_mode == 'concat' and channels is not None:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(
                    channels * self.num_directions,
                    channels,
                    kernel_size=1,
                    bias=False
                ),
                make_group_norm(channels),
                nn.GELU()
            )
        else:
            self.channel_reduction = None
        
    def forward(
        self,
        x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        对输入特征图进行多方向扫描

        参数:
            x: [B, C, H, W] 输入特征图

        返回:
            x_scans: List of [B, N, C]，长度为num_directions
        """
        x_scans = []
        for scan_type in self.scan_types:
            x_scan = self.scanner.scan(x, scan_type)
            x_scans.append(x_scan)
        return x_scans

    def unscan_all(
        self,
        x_seqs: List[torch.Tensor],
        H: int,
        W: int,
    ) -> List[torch.Tensor]:
        """
        将多个方向的扫描结果各自恢复为2D特征图

        参数:
            x_seqs: List of [B, N, C]
            H, W: 目标空间尺寸

        返回:
            List of [B, C, H, W]
        """
        feat_maps = []
        for x_seq, scan_type in zip(x_seqs, self.scan_types):
            feat_2d = self.scanner.unscan(x_seq, H, W, scan_type)
            feat_maps.append(feat_2d)
        return feat_maps

    def merge(
        self,
        x_seqs: List[torch.Tensor],
        H: int,
        W: int,
        merge_mode: Optional[str] = None
    ) -> torch.Tensor:
        """
        将多个方向的扫描结果合并回2D特征图

        参数:
            x_seqs: List of [B, N, C]
            H, W: 目标空间尺寸
            merge_mode: 'add' 或 'concat'（默认使用初始化时的模式）

        返回:
            x: [B, C, H, W]
        """
        if merge_mode is None:
            merge_mode = self.merge_mode

        x_2d_list = self.unscan_all(x_seqs, H, W)

        if merge_mode == 'add':
            x_merged = torch.stack(x_2d_list, dim=0).sum(dim=0)
        elif merge_mode == 'concat':
            x_merged = torch.cat(x_2d_list, dim=1)
            if self.channel_reduction is not None:
                x_merged = self.channel_reduction(x_merged)
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")

        return x_merged

