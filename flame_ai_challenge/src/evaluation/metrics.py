"""Evaluation metrics for FLAME AI Challenge."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted tensor of shape (B, C, H, W)
        target: Target tensor of shape (B, C, H, W)
        max_val: Maximum possible pixel value
        
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted tensor of shape (B, C, H, W)
        target: Target tensor of shape (B, C, H, W)
        
    Returns:
        Average SSIM value across channels
    """
    # Convert to numpy and calculate SSIM for each channel
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    ssim_values = []
    batch_size, channels, height, width = pred_np.shape
    
    for b in range(batch_size):
        for c in range(channels):
            ssim_val = ssim(
                target_np[b, c], 
                pred_np[b, c], 
                data_range=target_np[b, c].max() - target_np[b, c].min()
            )
            ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Squared Error."""
    return F.mse_loss(pred, target).item()


def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Absolute Error."""
    return F.l1_loss(pred, target).item()


def calculate_velocity_magnitude_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate error in velocity magnitude.
    Assumes channels are [RHO, UX, UY, UZ].
    
    Args:
        pred: Predicted tensor of shape (B, 4, H, W)
        target: Target tensor of shape (B, 4, H, W)
        
    Returns:
        Mean absolute error in velocity magnitude
    """
    # Extract velocity components (channels 1, 2, 3)
    pred_vel = pred[:, 1:4]  # UX, UY, UZ
    target_vel = target[:, 1:4]  # UX, UY, UZ
    
    # Calculate velocity magnitudes
    pred_mag = torch.sqrt(torch.sum(pred_vel ** 2, dim=1))
    target_mag = torch.sqrt(torch.sum(target_vel ** 2, dim=1))
    
    # Calculate mean absolute error
    return F.l1_loss(pred_mag, target_mag).item()


def calculate_divergence_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate error in flow divergence.
    
    Args:
        pred: Predicted tensor of shape (B, 4, H, W)
        target: Target tensor of shape (B, 4, H, W)
        
    Returns:
        Mean absolute error in divergence
    """
    def compute_divergence(flow_field):
        """Compute divergence using finite differences."""
        ux, uy, uz = flow_field[:, 1], flow_field[:, 2], flow_field[:, 3]
        
        # Compute gradients using finite differences
        dux_dx = torch.gradient(ux, dim=2)[0]
        duy_dy = torch.gradient(uy, dim=1)[0]
        # For 2D, we ignore duz_dz
        
        divergence = dux_dx + duy_dy
        return divergence
    
    pred_div = compute_divergence(pred)
    target_div = compute_divergence(target)
    
    return F.l1_loss(pred_div, target_div).item()


def calculate_energy_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate energy-based metrics for flow fields.
    
    Args:
        pred: Predicted tensor of shape (B, 4, H, W)
        target: Target tensor of shape (B, 4, H, W)
        
    Returns:
        Dictionary containing energy metrics
    """
    # Kinetic energy (0.5 * rho * |v|^2)
    pred_rho = pred[:, 0:1]  # Density
    target_rho = target[:, 0:1]
    
    pred_vel = pred[:, 1:4]  # Velocity components
    target_vel = target[:, 1:4]
    
    pred_ke = 0.5 * pred_rho * torch.sum(pred_vel ** 2, dim=1, keepdim=True)
    target_ke = 0.5 * target_rho * torch.sum(target_vel ** 2, dim=1, keepdim=True)
    
    # Total energy error
    energy_error = F.l1_loss(pred_ke, target_ke).item()
    
    # Momentum conservation error
    pred_momentum = pred_rho * pred_vel
    target_momentum = target_rho * target_vel
    momentum_error = F.l1_loss(pred_momentum, target_momentum).item()
    
    return {
        'energy_error': energy_error,
        'momentum_error': momentum_error,
        'kinetic_energy_pred': pred_ke.mean().item(),
        'kinetic_energy_target': target_ke.mean().item()
    }


def evaluate_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'psnr': calculate_psnr(pred, target),
        'ssim': calculate_ssim(pred, target),
        'mse': calculate_mse(pred, target),
        'mae': calculate_mae(pred, target),
        'velocity_magnitude_error': calculate_velocity_magnitude_error(pred, target),
        'divergence_error': calculate_divergence_error(pred, target)
    }
    
    # Add energy metrics
    energy_metrics = calculate_energy_metrics(pred, target)
    metrics.update(energy_metrics)
    
    return metrics