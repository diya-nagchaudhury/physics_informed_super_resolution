"""Configuration module for FLAME AI Challenge."""

import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    """Configuration class for the FLAME AI Challenge."""
    model_name: str = 'physics'
    # Data configuration
    input_path: str = '/home/diya/Projects/flame_ai_challenge/dataset'
    output_path: str = '/home/diya/Projects/flame_ai_challenge/outputs/'+ model_name
    
    # Normalization parameters
    mean: List[float] = None
    std: List[float] = None
    
    # Data shapes
    lr_shape: Tuple[int, int] = (16, 16)
    hr_shape: Tuple[int, int] = (128, 128)
    num_channels: int = 4
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    device: str = 'cuda'
    
    # Model parameters
    upsampling_mode: str = 'bilinear'
    
    # Visualization parameters
    save_plots: bool = True
    plot_format: str = 'png'
    dpi: int = 300
    
    # Evaluation parameters
    calculate_psnr: bool = True
    calculate_ssim: bool = True
    calculate_energy_metrics: bool = True

    # checkpoint directory
    checkpoint_dir: str = '/home/diya/Projects/flame_ai_challenge/checkpoints'

    

    def __post_init__(self):
        """Set default values after initialization."""
        if self.mean is None:
            self.mean = [0.24, 28.0, 28.0, 28.0]
        if self.std is None:
            self.std = [0.068, 48.0, 48.0, 48.0]
            
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
    
    def get_data_path(self, resolution: str, split: str) -> str:
        """Get the path for data files."""
        return os.path.join(self.input_path, f"flowfields/{resolution}/{split}")
    
    def get_csv_path(self, split: str) -> str:
        """Get the path for CSV files."""
        return os.path.join(self.input_path, f"{split}.csv")