"""Upsampling models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from config.config import Config


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD

# class SimpleUpsamplingModel(nn.Module):
#     """Simple learnable upsampling model using transposed convolutions."""
    
#     def __init__(self, config, mode: str = 'bilinear'):
#         """
#         Initialize simple upsampling model.
        
#         Args:
#             config: Configuration object
#             mode: Upsampling mode (kept for compatibility, but we'll use learnable layers)
#         """
#         super().__init__()
#         self.config = config
#         self.mode = mode
#         self.target_size = config.hr_shape
        
#         # Calculate upsampling factor
#         lr_h, lr_w = config.lr_shape
#         hr_h, hr_w = config.hr_shape
#         self.scale_factor_h = hr_h // lr_h
#         self.scale_factor_w = hr_w // lr_w
        
#         # Use learnable transposed convolution for upsampling
#         # This maintains the number of channels (4) while upsampling spatially
#         self.upsample_conv = nn.ConvTranspose2d(
#             in_channels=4, 
#             out_channels=4, 
#             kernel_size=3, 
#             stride=self.scale_factor_h,  # Assuming square upsampling
#             padding=1, 
#             output_padding=self.scale_factor_h-1,
#             bias=True
#         )
        
#         # Optional: Add a refinement layer
#         self.refine = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        
#         # Initialize weights properly
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """Initialize weights to approximate bicubic interpolation."""
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass.
        
#         Args:
#             x: Input tensor of shape (B, 4, m, m)
            
#         Returns:
#             Upsampled tensor of shape (B, 4, c * m, c * m)
#         """
#         # Learnable upsampling
#         upsampled = self.upsample_conv(x)
        
#         # Refinement
#         refined = self.refine(upsampled)
        
#         # Ensure exact target size (handle any size mismatches)
#         if refined.shape[-2:] != self.target_size:
#             refined = F.interpolate(
#                 refined, 
#                 size=self.target_size, 
#                 mode='bilinear', 
#                 align_corners=False
#             )
        
#         return refined
    
class SimpleUpsamplingModel(nn.Module):
    """Minimal learnable model with just a few parameters."""
    
    def __init__(self, config, mode: str = 'bilinear'):
        super().__init__()
        self.config = config
        self.target_size = config.hr_shape
        
        # Just add a simple learnable layer after interpolation
        self.post_process = nn.Conv2d(4, 4, kernel_size=1, bias=True)
        
        # Initialize close to identity
        nn.init.eye_(self.post_process.weight.squeeze())
        nn.init.zeros_(self.post_process.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First do interpolation
        upsampled = F.interpolate(
            x, 
            size=self.target_size, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Then apply learnable post-processing
        output = self.post_process(upsampled)
        return output
    
class ConvolutionalUpsamplingModel(nn.Module):
    """Convolutional upsampling model with learnable parameters."""
    
    def __init__(self, config: Config):
        """Initialize convolutional upsampling model."""
        super().__init__()
        self.config = config
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers using transpose convolution
        self.upsample_layers = nn.Sequential(
            # mxm -> 2m*2m
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 2m*2m -> 4m*4m
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 4m*4m -> 8m*8m
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Final layer to get 4 channels
            nn.Conv2d(32, 4, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_extractor(x)
        output = self.upsample_layers(features)
        return output


class ResidualBlock(nn.Module):
    """Residual block for the residual upsampling model."""
    
    def __init__(self, channels: int):
        """Initialize residual block."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class ResidualUpsamplingModel(nn.Module):
    """Residual upsampling model with skip connections."""
    
    def __init__(self, config: Config, num_residual_blocks: int = 16):
        """Initialize residual upsampling model."""
        super().__init__()
        self.config = config
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(4, 64, kernel_size=9, padding=4)
        
        # Residual blocks
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        # Upsampling path
        self.upsample_conv1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)  # m -> 2m
        
        self.upsample_conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)  # 2m -> 4m
        
        self.upsample_conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixel_shuffle3 = nn.PixelShuffle(2)  # 4m -> 8m
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, 4, kernel_size=9, padding=4)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial features
        features = self.relu(self.initial_conv(x))
        
        # Residual processing
        residual_features = self.residual_layers(features)
        features = features + residual_features  # Skip connection
        
        # Upsampling
        out = self.relu(self.pixel_shuffle1(self.upsample_conv1(features)))
        out = self.relu(self.pixel_shuffle2(self.upsample_conv2(out)))
        out = self.relu(self.pixel_shuffle3(self.upsample_conv3(out)))
        
        # Final output
        output = self.final_conv(out)
        
        return output


class AttentionBlock(nn.Module):
    """Channel attention block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        """Initialize attention block."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionUpsamplingModel(nn.Module):
    """Upsampling model with attention mechanism."""
    
    def __init__(self, config: Config):
        """Initialize attention upsampling model."""
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            AttentionBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            AttentionBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            AttentionBlock(256)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            AttentionBlock(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            AttentionBlock(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 4, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class PhysicsInformedUpsamplingModel(nn.Module):
    """Physics-informed upsampling model that enforces conservation laws."""
    
    def __init__(self, config: Config, lambda_continuity: float = 0.1, lambda_momentum: float = 0.1):
        """
        Initialize physics-informed model.
        
        Args:
            config: Configuration object
            lambda_continuity: Weight for continuity equation constraint
            lambda_momentum: Weight for momentum conservation constraint
        """
        super().__init__()
        self.config = config
        self.lambda_continuity = lambda_continuity
        self.lambda_momentum = lambda_momentum
        
        # Base upsampling network
        self.base_network = ResidualUpsamplingModel(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.base_network(x)
    
    def physics_loss(self, output: torch.Tensor) -> torch.Tensor:
        """
        Calculate physics-based loss terms.
        
        Args:
            output: Model output of shape (B, 4, H, W)
            
        Returns:
            Physics loss
        """
        rho = output[:, 0:1]  # Density
        ux = output[:, 1:2]   # Velocity X
        uy = output[:, 2:3]   # Velocity Y
        
        # Continuity equation: ∂ρ/∂t + ∇·(ρv) = 0
        # For steady state: ∇·(ρv) = 0
        drho_ux_dx = torch.gradient(rho * ux, dim=3)[0]
        drho_uy_dy = torch.gradient(rho * uy, dim=2)[0]
        continuity_loss = torch.mean((drho_ux_dx + drho_uy_dy)**2)
        
        # Momentum conservation
        dux_dx = torch.gradient(ux, dim=3)[0]
        duy_dy = torch.gradient(uy, dim=2)[0]
        momentum_loss = torch.mean((dux_dx + duy_dy)**2)
        
        return self.lambda_continuity * continuity_loss + self.lambda_momentum * momentum_loss


def get_model(model_name: str, config: Config) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model
        config: Configuration object
        
    Returns:
        Model instance
    """
    models = {
        'simple': SimpleUpsamplingModel,
        'convolutional': ConvolutionalUpsamplingModel,
        'residual': ResidualUpsamplingModel,
        'attention': AttentionUpsamplingModel,
        'physics': PhysicsInformedUpsamplingModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](config)