import torch
from typing import Dict, Tuple


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测张量
        target: 目标张量
        threshold: 二值化阈值
        
    Returns:
        IoU值
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    
    return tp / (tp + fp + fn + 1e-8)


def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算Dice系数
    
    Args:
        pred: 预测张量
        target: 目标张量
        threshold: 二值化阈值
        
    Returns:
        Dice系数
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


def calculate_precision_recall(pred: torch.Tensor, target: torch.Tensor, 
                              threshold: float = 0.5) -> Tuple[float, float]:
    """
    计算精确率和召回率
    
    Args:
        pred: 预测张量
        target: 目标张量
        threshold: 二值化阈值
        
    Returns:
        (精确率, 召回率)
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return precision, recall


def calculate_f1_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算F1分数"""
    precision, recall = calculate_precision_recall(pred, target, threshold)
    return 2 * precision * recall / (precision + recall + 1e-8)


def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算准确率"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
    
    return (tp + tn) / (tp + tn + fp + fn + 1e-8)


def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, 
                         threshold: float = 0.5) -> Dict[str, float]:
    """
    计算所有分割指标
    
    Args:
        pred: 预测张量
        target: 目标张量
        threshold: 二值化阈值
        
    Returns:
        包含所有指标的字典
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # 计算基本指标
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
    
    eps = 1e-8
    
    # 计算各项指标
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }