"""Dataset module for the Stanford Flame dataset."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from typing import Tuple, Union, Optional
import os

from config.config import Config


class FLAMEAIDataset(Dataset):
    """Dataset class for flow field data."""
    
    def __init__(self, split: str, config: Config, apply_norm: bool = True):
        """
        Initialize the dataset.
        
        Args:
            split: One of 'train', 'val', 'test'
            config: Configuration object
            apply_norm: Whether to apply normalization
        """
        self.config = config
        self.split = split
        self.apply_norm = apply_norm
        
        # Load CSV file
        csv_path = config.get_csv_path(split)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Initialize normalization transform
        if self.apply_norm:
            self.norm_transform = Normalize(config.mean, config.std)
    
    def get_files(self, idx: int, resolution: str) -> torch.Tensor:
        """
        Load flow field files for a given index and resolution.
        
        Args:
            idx: Index of the sample
            resolution: 'LR' or 'HR'
            
        Returns:
            Tensor of shape (4, H, W) containing RHO, UX, UY, UZ
        """
        assert resolution in ['LR', 'HR'], f"Invalid resolution: {resolution}"
        
        shape = self.config.lr_shape if resolution == 'LR' else self.config.hr_shape
        data_path = self.config.get_data_path(resolution, self.split)
        
        # Get filenames
        rho_filename = self.df['rho_filename'][idx]
        ux_filename = self.df['ux_filename'][idx]
        uy_filename = self.df['uy_filename'][idx]
        uz_filename = self.df['uz_filename'][idx]
        
        # Load files
        rho = np.fromfile(os.path.join(data_path, rho_filename), dtype="<f4").reshape(shape)
        ux = np.fromfile(os.path.join(data_path, ux_filename), dtype="<f4").reshape(shape)
        uy = np.fromfile(os.path.join(data_path, uy_filename), dtype="<f4").reshape(shape)
        uz = np.fromfile(os.path.join(data_path, uz_filename), dtype="<f4").reshape(shape)
        
        # Stack channels and convert to tensor
        data = torch.from_numpy(np.stack((rho, ux, uy, uz), axis=0))
        
        # Apply normalization if requested
        if self.apply_norm:
            data = self.norm_transform(data)
            
        return data
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df['id'])
    
    def __getitem__(self, idx: int) -> Union[Tuple[int, torch.Tensor], Tuple[int, torch.Tensor, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            For test split: (id, X)
            For train/val split: (id, X, Y)
        """
        sample_id = self.df['id'][idx]
        x = self.get_files(idx, 'LR').float()
        
        if self.split == 'test':
            return sample_id, x
        
        y = self.get_files(idx, 'HR').float()
        return sample_id, x, y
    
    def get_sample_info(self, idx: int) -> dict:
        """Get information about a specific sample."""
        return {
            'id': self.df['id'][idx],
            'rho_filename': self.df['rho_filename'][idx],
            'ux_filename': self.df['ux_filename'][idx],
            'uy_filename': self.df['uy_filename'][idx],
            'uz_filename': self.df['uz_filename'][idx]
        }