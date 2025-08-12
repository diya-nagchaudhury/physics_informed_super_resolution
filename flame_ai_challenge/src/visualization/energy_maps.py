"""Energy map generation and visualization for flow fields."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple
import cv2

from config.config import Config


class EnergyMapGenerator:
    """Class for generating and visualizing energy maps from flow fields."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the energy map generator."""
        self.config = config or Config()
    
    def generate_kinetic_energy_map(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generate kinetic energy map from flow field data.
        KE = 0.5 * ρ * |v|^2
        
        Args:
            data: Flow field tensor of shape (4, H, W) or (B, 4, H, W)
            
        Returns:
            Kinetic energy map of shape (H, W) or (B, H, W)
        """
        if data.dim() == 3:
            data = data.unsqueeze(0)  # Add batch dimension
        
        rho = data[:, 0]  # Density
        ux = data[:, 1]   # Velocity X
        uy = data[:, 2]   # Velocity Y
        uz = data[:, 3]   # Velocity Z
        
        # Calculate velocity magnitude squared
        velocity_mag_sq = ux**2 + uy**2 + uz**2
        
        # Calculate kinetic energy
        kinetic_energy = 0.5 * rho * velocity_mag_sq
        
        return kinetic_energy.squeeze() if kinetic_energy.shape[0] == 1 else kinetic_energy
    
    def generate_vorticity_map(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generate vorticity map from flow field data.
        Vorticity = ∇ × v (curl of velocity field)
        
        Args:
            data: Flow field tensor of shape (4, H, W) or (B, 4, H, W)
            
        Returns:
            Vorticity magnitude map
        """
        if data.dim() == 3:
            data = data.unsqueeze(0)
        
        ux = data[:, 1]  # Velocity X
        uy = data[:, 2]  # Velocity Y
        
        # Calculate vorticity (ω_z = ∂uy/∂x - ∂ux/∂y)
        duy_dx = torch.gradient(uy, dim=2)[0]
        dux_dy = torch.gradient(ux, dim=1)[0]
        
        vorticity = duy_dx - dux_dy
        
        return vorticity.squeeze() if vorticity.shape[0] == 1 else vorticity
    
    def generate_divergence_map(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generate divergence map from flow field data.
        Divergence = ∇ · v
        
        Args:
            data: Flow field tensor of shape (4, H, W) or (B, 4, H, W)
            
        Returns:
            Divergence map
        """
        if data.dim() == 3:
            data = data.unsqueeze(0)
        
        ux = data[:, 1]  # Velocity X
        uy = data[:, 2]  # Velocity Y
        
        # Calculate divergence (∇ · v = ∂ux/∂x + ∂uy/∂y)
        dux_dx = torch.gradient(ux, dim=2)[0]
        duy_dy = torch.gradient(uy, dim=1)[0]
        
        divergence = dux_dx + duy_dy
        
        return divergence.squeeze() if divergence.shape[0] == 1 else divergence
    
    def generate_strain_rate_map(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generate strain rate magnitude map from flow field data.
        
        Args:
            data: Flow field tensor of shape (4, H, W) or (B, 4, H, W)
            
        Returns:
            Strain rate magnitude map
        """
        if data.dim() == 3:
            data = data.unsqueeze(0)
        
        ux = data[:, 1]  # Velocity X
        uy = data[:, 2]  # Velocity Y
        
        # Calculate velocity gradients
        dux_dx = torch.gradient(ux, dim=2)[0]
        dux_dy = torch.gradient(ux, dim=1)[0]
        duy_dx = torch.gradient(uy, dim=2)[0]
        duy_dy = torch.gradient(uy, dim=1)[0]
        
        # Strain rate tensor components
        s11 = dux_dx
        s22 = duy_dy
        s12 = 0.5 * (dux_dy + duy_dx)
        
        # Strain rate magnitude
        strain_rate_mag = torch.sqrt(s11**2 + s22**2 + 2*s12**2)
        
        return strain_rate_mag.squeeze() if strain_rate_mag.shape[0] == 1 else strain_rate_mag
    
    def generate_turbulent_kinetic_energy_map(self, data: torch.Tensor, mean_flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate turbulent kinetic energy map.
        TKE = 0.5 * (u'^2 + v'^2 + w'^2) where u' = u - <u>
        
        Args:
            data: Flow field tensor of shape (4, H, W) or (B, 4, H, W)
            mean_flow: Mean flow field (if None, uses spatial average)
            
        Returns:
            Turbulent kinetic energy map
        """
        if data.dim() == 3:
            data = data.unsqueeze(0)
        
        ux = data[:, 1]  # Velocity X
        uy = data[:, 2]  # Velocity Y
        uz = data[:, 3]  # Velocity Z
        
        if mean_flow is None:
            # Use spatial mean as reference
            ux_mean = ux.mean(dim=(-2, -1), keepdim=True)
            uy_mean = uy.mean(dim=(-2, -1), keepdim=True)
            uz_mean = uz.mean(dim=(-2, -1), keepdim=True)
        else:
            if mean_flow.dim() == 3:
                mean_flow = mean_flow.unsqueeze(0)
            ux_mean = mean_flow[:, 1:2]
            uy_mean = mean_flow[:, 2:3]
            uz_mean = mean_flow[:, 3:4]
        
        # Calculate fluctuations
        ux_prime = ux - ux_mean
        uy_prime = uy - uy_mean
        uz_prime = uz - uz_mean
        
        # Calculate TKE
        tke = 0.5 * (ux_prime**2 + uy_prime**2 + uz_prime**2)
        
        return tke.squeeze() if tke.shape[0] == 1 else tke
    
    def plot_energy_map(self, energy_map: torch.Tensor, title: str = "Energy Map", 
                       cmap: str = 'hot', save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot an energy map.
        
        Args:
            energy_map: Energy map tensor of shape (H, W)
            title: Plot title
            cmap: Colormap name
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        energy_np = energy_map.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(energy_np, cmap=cmap, aspect='auto', origin='lower')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Energy', rotation=270, labelpad=20)
        
        # Add contour lines for better visualization
        contours = ax.contour(energy_np, colors='white', alpha=0.3, linewidths=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_all_energy_maps(self, data: torch.Tensor, sample_id: str = "",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot all energy maps for a flow field.
        
        Args:
            data: Flow field tensor of shape (4, H, W)
            sample_id: Sample identifier
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Generate all energy maps
        ke_map = self.generate_kinetic_energy_map(data)
        vorticity_map = self.generate_vorticity_map(data)
        divergence_map = self.generate_divergence_map(data)
        strain_rate_map = self.generate_strain_rate_map(data)
        tke_map = self.generate_turbulent_kinetic_energy_map(data)
        
        # Convert to numpy
        maps = {
            'Kinetic Energy': ke_map.detach().cpu().numpy(),
            'Vorticity': vorticity_map.detach().cpu().numpy(),
            'Divergence': divergence_map.detach().cpu().numpy(),
            'Strain Rate': strain_rate_map.detach().cpu().numpy(),
            'Turbulent KE': tke_map.detach().cpu().numpy()
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Energy Maps - Sample {sample_id}', fontsize=20)
        
        cmaps = ['hot', 'RdBu_r', 'RdBu_r', 'plasma', 'viridis']
        
        for i, (title, energy_map) in enumerate(maps.items()):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            
            im = ax.imshow(energy_map, cmap=cmaps[i], aspect='auto', origin='lower')
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Value', rotation=270, labelpad=15)
            
            # Add contour lines
            ax.contour(energy_map, colors='white', alpha=0.3, linewidths=0.5)
        
        # Hide the last subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
        return fig
    
    def create_energy_overlay(self, base_image: torch.Tensor, energy_map: torch.Tensor, 
                            alpha: float = 0.6) -> np.ndarray:
        """
        Create an overlay of energy map on base image.
        
        Args:
            base_image: Base image tensor of shape (H, W)
            energy_map: Energy map tensor of shape (H, W)
            alpha: Transparency of overlay
            
        Returns:
            RGB overlay image
        """
        # Normalize images to [0, 1]
        base_np = base_image.detach().cpu().numpy()
        energy_np = energy_map.detach().cpu().numpy()
        
        base_norm = (base_np - base_np.min()) / (base_np.max() - base_np.min())
        energy_norm = (energy_np - energy_np.min()) / (energy_np.max() - energy_np.min())
        
        # Create RGB overlay
        # Base image in grayscale
        overlay = np.stack([base_norm, base_norm, base_norm], axis=-1)
        
        # Energy map in red channel
        energy_overlay = np.zeros_like(overlay)
        energy_overlay[:, :, 0] = energy_norm  # Red channel for energy
        
        # Blend images
        result = (1 - alpha) * overlay + alpha * energy_overlay
        
        return np.clip(result, 0, 1)
    
    def compute_energy_statistics(self, energy_map: torch.Tensor) -> dict:
        """
        Compute statistics for an energy map.
        
        Args:
            energy_map: Energy map tensor
            
        Returns:
            Dictionary of statistics
        """
        energy_np = energy_map.detach().cpu().numpy()
        
        stats = {
            'mean': float(np.mean(energy_np)),
            'std': float(np.std(energy_np)),
            'min': float(np.min(energy_np)),
            'max': float(np.max(energy_np)),
            'median': float(np.median(energy_np)),
            'total_energy': float(np.sum(energy_np)),
            'energy_concentration': float(np.sum(energy_np > np.percentile(energy_np, 90)) / energy_np.size)
        }
        
        return stats