import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os
from typing import List, Dict, Optional, Tuple


class Visualizer:
        
    def __init__(self, save_dir: Optional[str] = None, dpi: int = 300):
        self.save_dir = save_dir or "visualizations"
        self.dpi = dpi
        os.makedirs(self.save_dir, exist_ok=True)
        plt.style.use('default')
    
    def show_image_with_mask(self, image: np.ndarray, mask: np.ndarray, 
                           predicted_mask: Optional[np.ndarray] = None,
                           title: str = "Image and Mask", 
                           save_name: Optional[str] = None,
                           show: bool = True) -> None:
        """Display image with ground truth and predicted masks"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        n_cols = 3 if predicted_mask is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        
        if n_cols == 2:
            axes = [axes] if not hasattr(axes, '__iter__') else list(axes)
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        if predicted_mask is not None:
            axes[2].imshow(predicted_mask, cmap='gray')
            axes[2].set_title("Prediction")
            axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                      color: Tuple[int, int, int] = (255, 0, 0),
                      alpha: float = 0.3) -> np.ndarray:
        """Create overlay of mask on image"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 127] = color
        
        return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    def save_prediction_grid(self, images: List[np.ndarray], 
                           ground_truths: List[np.ndarray],
                           predictions: List[np.ndarray],
                           grid_size: Optional[Tuple[int, int]] = None,
                           title: str = "Prediction Results",
                           save_name: str = "prediction_grid",
                           show_overlay: bool = True) -> None:
        """Save grid of predictions with 4 columns: image, GT, pred, overlay"""
        n_samples = len(images)
        
        if grid_size:
            rows, cols = grid_size
        else:
            cols = 4  # Fixed: image, GT, pred, overlay
            rows = n_samples
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        
        # Handle single row case
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Normalize images
            img = images[i]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            gt = ground_truths[i]
            pred = predictions[i]
            
            # Column 0: Original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')
            
            # Column 1: Ground truth
            axes[i, 1].imshow(gt, cmap='gray')
            axes[i, 1].set_title(f"GT {i+1}")
            axes[i, 1].axis('off')
            
            # Column 2: Prediction
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title(f"Pred {i+1}")
            axes[i, 2].axis('off')
            
            # Column 3: Overlay
            if show_overlay:
                overlay = self.create_overlay(img, pred)
                axes[i, 3].imshow(overlay)
                axes[i, 3].set_title(f"Overlay {i+1}")
                axes[i, 3].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
    
    def plot_training_curves(self, train_losses: List[float], 
                           val_losses: Optional[List[float]] = None,
                           train_metrics: Optional[Dict[str, List[float]]] = None,
                           val_metrics: Optional[Dict[str, List[float]]] = None,
                           title: str = "Training Curves",
                           save_name: Optional[str] = None,
                           show: bool = True) -> None:
        """Plot training curves"""
        n_plots = 1 + (len(train_metrics) if train_metrics else 0)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        
        if val_losses:
            axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if train_metrics:
            for i, (metric_name, values) in enumerate(train_metrics.items()):
                ax_idx = i + 1
                axes[ax_idx].plot(epochs, values, 'b-', label=f'Train {metric_name}', linewidth=2)
                
                if val_metrics and metric_name in val_metrics:
                    axes[ax_idx].plot(epochs, val_metrics[metric_name], 'r-', 
                                    label=f'Val {metric_name}', linewidth=2)
                
                axes[ax_idx].set_xlabel('Epoch')
                axes[ax_idx].set_ylabel(metric_name)
                axes[ax_idx].set_title(f'{metric_name} Curve')
                axes[ax_idx].legend()
                axes[ax_idx].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}_curves.png")
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


def create_visualizer(save_dir: Optional[str] = None, dpi: int = 300) -> Visualizer:
    """Factory function for creating visualizer"""
    return Visualizer(save_dir, dpi)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array"""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range"""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image, 0, 1)