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
    
    def plot_flow_comparison(self, lr_tensor, hr_tensor, pred_tensor, channel_names=None, save_path: Optional[str] = None):
        
        lr_np = lr_tensor.detach().cpu().numpy()
        hr_np = hr_tensor.detach().cpu().numpy()
        pred_np = pred_tensor.detach().cpu().numpy()
        
        num_samples = lr_np.shape[0]
        num_channels = lr_np.shape[1]
        
        if channel_names is None:
            channel_names = ["rho", "ux", "uy", "uz"]
        
        fig, axes = plt.subplots(num_samples, num_channels * 3, 
                                figsize=(num_channels * 3 * 3, num_samples * 3))
        
        # Make axes always 2D for consistent indexing
        if num_samples == 1:
            axes = axes[np.newaxis, :]
        
        for i in range(num_samples):
            for c in range(num_channels):
                vmin = min(lr_np[i, c].min(), hr_np[i, c].min(), pred_np[i, c].min())
                vmax = max(lr_np[i, c].max(), hr_np[i, c].max(), pred_np[i, c].max())
                
                # LR
                axes[i, c].imshow(lr_np[i, c], cmap="viridis", vmin=vmin, vmax=vmax)
                axes[i, c].set_title(f"LR {channel_names[c]}")
                
                # HR
                axes[i, c + num_channels].imshow(hr_np[i, c], cmap="viridis", vmin=vmin, vmax=vmax)
                axes[i, c + num_channels].set_title(f"HR {channel_names[c]}")
                
                # Pred
                axes[i, c + 2 * num_channels].imshow(pred_np[i, c], cmap="viridis", vmin=vmin, vmax=vmax)
                axes[i, c + 2 * num_channels].set_title(f"Pred {channel_names[c]}")
        
        plt.tight_layout()
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