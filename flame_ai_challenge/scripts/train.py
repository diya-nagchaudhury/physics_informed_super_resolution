#!/usr/bin/env python3
"""Training script for the models."""

import argparse
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from src.data.dataset import FLAMEAIDataset
from src.models.upsampling import get_model
from src.training.trainer import Trainer


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train FLAME AI Challenge model')
    
    parser.add_argument('--model', type=str, default='residual',
                       choices=['simple', 'convolutional', 'residual', 'attention', 'physics'],
                       help='Model architecture to use')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    parser.add_argument('--data_path', type=str, default=Config.input_path,
                       help='Path to dataset')
    
    parser.add_argument('--output_path', type=str, default=Config.output_path + 'train/',
                       help='Path to save outputs')
    
    parser.add_argument('--no_validation', action='store_true',
                       help='Skip validation during training')
    
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine', 'plateau'],
                       help='Learning rate scheduler')
    
    parser.add_argument('--loss', type=str, default='mse',
                       choices=['mse', 'l1', 'smooth_l1'],
                       help='Loss function to use')
    
    return parser.parse_args()


def main():
    """Driver function."""
    args = parse_args()
    
    # Create configuration
    config = Config()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.device = args.device
    config.input_path = args.data_path
    config.output_path = args.output_path
    config.optimizer = args.optimizer
    config.scheduler = args.scheduler
    config.loss_function = args.loss
    
    # Ensure output directory exists
    os.makedirs(config.output_path, exist_ok=True)
    
    print("=== Training ===")
    print(f"Model: {args.model}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Scheduler: {config.scheduler}")
    print(f"Loss: {config.loss_function}")
    print("=" * 40)
    
    # Check device availability
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        config.device = 'cpu'
    elif config.device == 'cpu':
        print("Using CPU for training")
    print(f"Using device: {config.device}")
    
    # Create datasets
    print("Loading datasets...")
    try:
        train_dataset = FLAMEAIDataset('train', config, apply_norm=True)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True if config.device == 'cuda' else False
        )
        print(f"Training samples: {len(train_dataset)}")
        
        val_loader = None
        if not args.no_validation:
            try:
                val_dataset = FLAMEAIDataset('val', config, apply_norm=True)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=config.batch_size, 
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True if config.device == 'cuda' else False
                )
                print(f"Validation samples: {len(val_dataset)}")
            except FileNotFoundError:
                print("Validation dataset not found, training without validation")
                val_loader = None
    
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Please check that the dataset path is correct and files exist")
        return
    
    # Create model
    print(f"Creating {args.model} model...")
    try:
        model = get_model(args.model, config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Start training
    try:
        trainer.train(train_loader, val_loader, resume_from_checkpoint=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        current_epoch = len(trainer.train_losses) - 1
        trainer.save_checkpoint(current_epoch, is_best=False)
        print("Checkpoint saved")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()