"""Visualization utilities for FLAME AI Challenge."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import numpy as np
import torch
from typing import Optional, Tuple, List
import os

from config.config import Config


class FlowFieldPlotter:
    """Class for plotting flow field data and comparisons."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the plotter with configuration."""
        self.config = config or Config()
        
    def plot_flow_field(self, data: torch.Tensor, title: str = "Flow Field", 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a single flow field with all four channels.
        
        Args:
            data: Tensor of shape (4, H, W) containing [RHO, UX, UY, UZ]
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        data_np = data.detach().cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        channel_names = ['Density (ρ)', 'Velocity X (Ux)', 'Velocity Y (Uy)', 'Velocity Z (Uz)']
        
        for i, (ax, channel_name) in enumerate(zip(axes.flat, channel_names)):
            im = ax.imshow(data_np[i], cmap='viridis', aspect='auto')
            ax.set_title(channel_name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_flow_comparison(self, lr_tensor, hr_tensor, pred_tensor, sample_id, channel_names=None, save_path: Optional[str] = None):
        lr_np = lr_tensor.detach().cpu().numpy()
        hr_np = hr_tensor.detach().cpu().numpy()
        pred_np = pred_tensor.detach().cpu().numpy()
        
        # Debug: Print shapes
        # print(f"LR shape: {lr_np.shape}")
        # print(f"HR shape: {hr_np.shape}")
        # print(f"Pred shape: {pred_np.shape}")
        
        # Handle different tensor formats
        if len(lr_np.shape) == 3:  # (channels, height, width)
            num_channels, lr_h, lr_w = lr_np.shape
            _, hr_h, hr_w = hr_np.shape
            _, pred_h, pred_w = pred_np.shape
            num_samples = 1  # Single sample
            
            # Reshape to add batch dimension: (1, channels, height, width)
            lr_np = lr_np[np.newaxis, ...]
            hr_np = hr_np[np.newaxis, ...]
            pred_np = pred_np[np.newaxis, ...]
            
        elif len(lr_np.shape) == 4:  # (batch_size, channels, height, width)
            num_samples, num_channels, lr_h, lr_w = lr_np.shape
            _, _, hr_h, hr_w = hr_np.shape
            _, _, pred_h, pred_w = pred_np.shape
        else:
            raise ValueError(f"Unexpected tensor shapes. LR: {lr_np.shape}, HR: {hr_np.shape}, Pred: {pred_np.shape}")
        
        if channel_names is None:
            channel_names = ["rho", "ux", "uy", "uz"]
        
        # Ensure we don't exceed available channels
        num_channels = min(num_channels, len(channel_names))
        
        fig, axes = plt.subplots(num_samples, num_channels * 3,
                                figsize=(num_channels * 3 * 3, num_samples * 3))
        
        # Make axes always 2D for consistent indexing
        if num_samples == 1:
            axes = axes[np.newaxis, :]
        if num_channels * 3 == 1:
            axes = axes[:, np.newaxis]
        
        for i in range(num_samples):
            for c in range(num_channels):
                # Extract channel data - now with correct indexing
                lr_channel = lr_np[i, c, :, :]  # Shape: (lr_h, lr_w)
                hr_channel = hr_np[i, c, :, :]  # Shape: (hr_h, hr_w)
                pred_channel = pred_np[i, c, :, :]  # Shape: (pred_h, pred_w)
                
                # Debug: Print individual channel shapes
                # print(f"Sample {i}, Channel {c}:")
                # print(f"  LR channel shape: {lr_channel.shape}")
                # print(f"  HR channel shape: {hr_channel.shape}")
                # print(f"  Pred channel shape: {pred_channel.shape}")
                
                # Verify shapes are 2D
                assert len(lr_channel.shape) == 2, f"LR channel should be 2D, got {lr_channel.shape}"
                assert len(hr_channel.shape) == 2, f"HR channel should be 2D, got {hr_channel.shape}"
                assert len(pred_channel.shape) == 2, f"Pred channel should be 2D, got {pred_channel.shape}"
                
                try:
                    # Calculate vmin/vmax for consistent scaling
                    vmin = min(lr_channel.min(), hr_channel.min(), pred_channel.min())
                    vmax = max(lr_channel.max(), hr_channel.max(), pred_channel.max())
                    
                    # LR - Shape should be (4, 4) based on your output
                    im1 = axes[i, c * 3].imshow(lr_channel, cmap="viridis", vmin=vmin, vmax=vmax)
                    axes[i, c * 3].set_title(f"LR {channel_names[c]} ({lr_channel.shape[0]}x{lr_channel.shape[1]})")
                    axes[i, c * 3].axis('off')
                    
                    # HR - Shape should be (128, 128) based on your output
                    im2 = axes[i, c * 3 + 1].imshow(hr_channel, cmap="viridis", vmin=vmin, vmax=vmax)
                    axes[i, c * 3 + 1].set_title(f"HR {channel_names[c]} ({hr_channel.shape[0]}x{hr_channel.shape[1]})")
                    axes[i, c * 3 + 1].axis('off')
                    
                    # Pred - Shape should be (128, 128) based on your output
                    im3 = axes[i, c * 3 + 2].imshow(pred_channel, cmap="viridis", vmin=vmin, vmax=vmax)
                    axes[i, c * 3 + 2].set_title(f"Pred {channel_names[c]} ({pred_channel.shape[0]}x{pred_channel.shape[1]})")
                    axes[i, c * 3 + 2].axis('off')
                    
                    # Add colorbar to the last plot in each row
                    if c == num_channels - 1:
                        plt.colorbar(im3, ax=axes[i, c * 3:c * 3 + 3], shrink=0.6)
                    
                except Exception as e:
                    print(f"Error processing sample {i}, channel {c}: {e}")
                    # Create empty plots as fallback
                    for offset in range(3):
                        axes[i, c * 3 + offset].text(0.5, 0.5, f"Error:\n{str(e)}", 
                                                ha='center', va='center', transform=axes[i, c * 3 + offset].transAxes)
                        axes[i, c * 3 + offset].set_title(f"Error - {channel_names[c]}")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_velocity_streamlines(self, data: torch.Tensor, title: str = "Velocity Streamlines",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot velocity field as streamlines.
        
        Args:
            data: Tensor of shape (4, H, W)
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        data_np = data.detach().cpu().numpy()
        
        # Extract velocity components
        ux = data_np[1]  # Velocity X
        uy = data_np[2]  # Velocity Y
        
        # Create coordinate grids
        h, w = ux.shape
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot streamlines
        ax.streamplot(X, Y, ux, uy, density=2, color='blue', linewidth=1, arrowsize=1.5)
        
        # Overlay density as background
        density = data_np[0]
        im = ax.imshow(density, extent=[0, w-1, h-1, 0], alpha=0.3, cmap='Reds')
        plt.colorbar(im, ax=ax, label='Density')
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_error_heatmap(self, pred: torch.Tensor, target: torch.Tensor,
                          title: str = "Error Heatmap", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot error heatmaps between prediction and target.
        
        Args:
            pred: Predicted tensor of shape (4, H, W)
            target: Target tensor of shape (4, H, W)
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate absolute error
        error = np.abs(pred_np - target_np)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        channel_names = ['Density Error', 'Velocity X Error', 'Velocity Y Error', 'Velocity Z Error']
        
        for i, (ax, channel_name) in enumerate(zip(axes.flat, channel_names)):
            im = ax.imshow(error[i], cmap='hot', aspect='auto')
            ax.set_title(channel_name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float] = None,
                           metrics_history: Optional[dict] = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training curves.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            metrics_history: Dictionary of metric histories
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        n_metrics = 1 + (1 if val_losses else 0) + (len(metrics_history) if metrics_history else 0)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot training loss
        ax = axes.flat[plot_idx]
        ax.plot(train_losses, label='Train Loss', color='blue')
        if val_losses:
            ax.plot(val_losses, label='Val Loss', color='red')
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        plot_idx += 1
        
        # Plot metrics
        if metrics_history:
            for metric_name, values in metrics_history.items():
                if plot_idx < len(axes.flat):
                    ax = axes.flat[plot_idx]
                    ax.plot(values, label=metric_name, color='green')
                    ax.set_title(f'{metric_name.upper()}')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric_name.upper())
                    ax.grid(True)
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes.flat)):
            axes.flat[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
        return fig