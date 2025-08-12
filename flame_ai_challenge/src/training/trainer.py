"""Training module for FLAME AI Challenge models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import os
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm

from config.config import Config
from src.evaluation.metrics import evaluate_all_metrics
from src.visualization.plotters import FlowFieldPlotter
from src.models.upsampling import PhysicsInformedUpsamplingModel


class Trainer:
    """Trainer class for FLAME AI Challenge models."""
    
    def __init__(self, model: nn.Module, config: Config, device: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration object
            device: Device to use for training
        """
        self.model = model
        self.config = config
        self.device = device or config.device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._get_scheduler()
        
        # Initialize loss function
        self.criterion = self._get_criterion()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {}
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Visualization
        self.plotter = FlowFieldPlotter(config)
        
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer based on configuration."""
        if hasattr(self.config, 'optimizer'):
            optimizer_name = self.config.optimizer.lower()
        else:
            optimizer_name = 'adam'
            
        if optimizer_name == 'adam':
            return Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif optimizer_name == 'adamw':
            return AdamW(self.model.parameters(), lr=self.config.learning_rate)
        elif optimizer_name == 'sgd':
            return SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        else:
            return Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler."""
        if hasattr(self.config, 'scheduler'):
            scheduler_name = self.config.scheduler.lower()
            
            if scheduler_name == 'step':
                return StepLR(self.optimizer, step_size=30, gamma=0.1)
            elif scheduler_name == 'cosine':
                return CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
            elif scheduler_name == 'plateau':
                return ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
        return None
    
    def _get_criterion(self) -> nn.Module:
        """Get loss criterion."""
        if hasattr(self.config, 'loss_function'):
            loss_name = self.config.loss_function.lower()
            
            if loss_name == 'mse':
                return nn.MSELoss()
            elif loss_name == 'l1':
                return nn.L1Loss()
            elif loss_name == 'smooth_l1':
                return nn.SmoothL1Loss()
        
        return nn.MSELoss()  # Default
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:  # Training data with labels
                sample_ids, inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Add physics loss if using physics-informed model
                if isinstance(self.model, PhysicsInformedUpsamplingModel):
                    physics_loss = self.model.physics_loss(outputs)
                    loss += physics_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            else:
                # Skip test data during training
                continue
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        all_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                if len(batch) == 3:  # Validation data with labels
                    sample_ids, inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # Calculate metrics
                    batch_metrics = evaluate_all_metrics(outputs, targets)
                    
                    # Accumulate metrics
                    for metric_name, value in batch_metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)
                    
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # Average metrics
        avg_metrics = {name: sum(values) / len(values) for name, values in all_metrics.items()}
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.output_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.output_path, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.metrics_history = checkpoint.get('metrics_history', {})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch']
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              resume_from_checkpoint: Optional[str] = None) -> None:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        start_epoch = 0
        
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            start_epoch = self.load_checkpoint(resume_from_checkpoint) + 1
            print(f"Resumed training from epoch {start_epoch}")
        
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # Update metrics history
                for metric_name, value in val_metrics.items():
                    if metric_name not in self.metrics_history:
                        self.metrics_history[metric_name] = []
                    self.metrics_history[metric_name].append(value)
                
                # Check if this is the best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                
                # Print epoch results
                epoch_time = time.time() - start_time
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss: {val_loss:.6f}")
                print(f"PSNR: {val_metrics.get('psnr', 0):.2f} dB")
                print(f"SSIM: {val_metrics.get('ssim', 0):.4f}")
                print(f"Time: {epoch_time:.2f}s")
                
                if is_best:
                    print("*** New best model! ***")
                
            else:
                val_loss = train_loss
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot training curves periodically
            if (epoch + 1) % 20 == 0 and self.config.save_plots:
                self.plot_training_progress(epoch + 1)
        
        print(f"\nTraining completed!")
        print(f"Best model at epoch {self.best_epoch + 1} with validation loss: {self.best_val_loss:.6f}")
    
    def plot_training_progress(self, current_epoch: int) -> None:
        """
        Plot and save training progress.
        
        Args:
            current_epoch: Current epoch number
        """
        save_path = os.path.join(self.config.output_path, f'training_curves_epoch_{current_epoch}.png')
        
        self.plotter.plot_training_curves(
            train_losses=self.train_losses,
            val_losses=self.val_losses if self.val_losses else None,
            metrics_history=self.metrics_history,
            save_path=save_path
        )
    
    def evaluate_sample(self, sample_data: Tuple, save_visualizations: bool = True) -> Dict[str, float]:
        """
        Evaluate model on a single sample and optionally save visualizations.
        
        Args:
            sample_data: Tuple containing (sample_id, inputs, targets)
            save_visualizations: Whether to save visualization plots
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        sample_id, inputs, targets = sample_data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # Calculate metrics
            metrics = evaluate_all_metrics(outputs, targets)
            
            if save_visualizations:
                # Create comparison plot
                comparison_path = os.path.join(
                    self.config.output_path, 
                    f'comparison_sample_{sample_id}.png'
                )
                
                # Remove batch dimension for plotting
                lr_data = inputs.squeeze(0)
                hr_data = targets.squeeze(0)
                pred_data = outputs.squeeze(0)
                
                self.plotter.plot_flow_comparison(
                    lr_data, hr_data, pred_data, 
                    sample_id=str(sample_id), 
                    save_path=comparison_path
                )
                
                # Create error heatmap
                error_path = os.path.join(
                    self.config.output_path, 
                    f'error_heatmap_sample_{sample_id}.png'
                )
                
                self.plotter.plot_error_heatmap(
                    pred_data, hr_data, 
                    title=f"Error Heatmap - Sample {sample_id}",
                    save_path=error_path
                )
        
        return metrics

# """Training module for FLAME AI Challenge models."""

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.optim import Adam, SGD, AdamW
# from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
# import os
# from typing import Dict, List, Optional, Tuple
# import time
# from tqdm import tqdm

# from config.config import Config
# from src.evaluation.metrics import evaluate_all_metrics
# from src.visualization.plotters import FlowFieldPlotter
# from src.models.upsampling import PhysicsInformedUpsamplingModel


# class Trainer:
#     """Trainer class for FLAME AI Challenge models."""
    
#     def __init__(self, model: nn.Module, config: Config, device: Optional[str] = None, 
#                  skip_training_if_checkpoint: bool = False):
#         """
#         Initialize trainer.
        
#         Args:
#             model: Model to train
#             config: Configuration object
#             device: Device to use for training
#             skip_training_if_checkpoint: If True, skip training if checkpoint exists
#         """
#         self.model = model
#         self.config = config
#         self.device = device or config.device
#         self.skip_training_if_checkpoint = skip_training_if_checkpoint
        
#         # Move model to device
#         self.model.to(self.device)
        
#         # Initialize optimizer
#         self.optimizer = self._get_optimizer()
        
#         # Initialize scheduler
#         self.scheduler = self._get_scheduler()
        
#         # Initialize loss function
#         self.criterion = self._get_criterion()
        
#         # Training history
#         self.train_losses = []
#         self.val_losses = []
#         self.metrics_history = {}
        
#         # Best model tracking
#         self.best_val_loss = float('inf')
#         self.best_epoch = 0
#         self.checkpoint_loaded = False
        
#         # Visualization
#         self.plotter = FlowFieldPlotter(config)

#         # Load checkpoint if available
#         self._load_checkpoint_if_exists()

#     def _load_checkpoint_if_exists(self):
#         """Load checkpoint if it exists."""
#         checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
#         if os.path.exists(checkpoint_path):
#             try:
#                 checkpoint = torch.load(checkpoint_path, map_location=self.device)
#                 self.model.load_state_dict(checkpoint['model_state_dict'])
#                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
#                 if self.scheduler and checkpoint.get('scheduler_state_dict') is not None:
#                     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
#                 self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
#                 self.best_epoch = checkpoint.get('epoch', 0)
#                 self.checkpoint_loaded = True
                
#                 print(f"✅ Loaded checkpoint from {checkpoint_path} (epoch {self.best_epoch+1})")
#                 print(f"   Best validation loss: {self.best_val_loss:.6f}")
                
#                 if self.skip_training_if_checkpoint:
#                     print("🛑 Training will be skipped due to existing checkpoint and skip_training_if_checkpoint=True")
                
#             except Exception as e:
#                 print(f"⚠️ Failed to load checkpoint: {e}")
#                 print("   Starting training from scratch.")
#                 self.checkpoint_loaded = False
#         else:
#             print("ℹ️ No checkpoint found, starting training from scratch.")
#             self.checkpoint_loaded = False

        
#     def _get_optimizer(self) -> torch.optim.Optimizer:
#         """Get optimizer based on configuration."""
#         if hasattr(self.config, 'optimizer'):
#             optimizer_name = self.config.optimizer.lower()
#         else:
#             optimizer_name = 'adam'
            
#         if optimizer_name == 'adam':
#             return Adam(self.model.parameters(), lr=self.config.learning_rate)
#         elif optimizer_name == 'adamw':
#             return AdamW(self.model.parameters(), lr=self.config.learning_rate)
#         elif optimizer_name == 'sgd':
#             return SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
#         else:
#             return Adam(self.model.parameters(), lr=self.config.learning_rate)
    
#     def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
#         """Get learning rate scheduler."""
#         if hasattr(self.config, 'scheduler'):
#             scheduler_name = self.config.scheduler.lower()
            
#             if scheduler_name == 'step':
#                 return StepLR(self.optimizer, step_size=30, gamma=0.1)
#             elif scheduler_name == 'cosine':
#                 return CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
#             elif scheduler_name == 'plateau':
#                 return ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
#         return None
    
#     def _get_criterion(self) -> nn.Module:
#         """Get loss criterion."""
#         if hasattr(self.config, 'loss_function'):
#             loss_name = self.config.loss_function.lower()
            
#             if loss_name == 'mse':
#                 return nn.MSELoss()
#             elif loss_name == 'l1':
#                 return nn.L1Loss()
#             elif loss_name == 'smooth_l1':
#                 return nn.SmoothL1Loss()
        
#         return nn.MSELoss()  # Default
    
#     def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
#               force_train: bool = False):
#         """
#         Full training loop.

#         Args:
#             train_loader: DataLoader for training set
#             val_loader: Optional DataLoader for validation set
#             force_train: If True, train even if checkpoint exists and skip_training_if_checkpoint=True
#         """
#         # Check if we should skip training
#         if self.skip_training_if_checkpoint and self.checkpoint_loaded and not force_train:
#             print("🛑 Skipping training - checkpoint loaded and skip_training_if_checkpoint=True")
#             print("   Use force_train=True to override this behavior")
#             return
        
#         # Determine starting epoch
#         start_epoch = self.best_epoch + 1 if self.checkpoint_loaded else 0
        
#         if start_epoch >= self.config.num_epochs:
#             print(f"🏁 Training already completed ({start_epoch}/{self.config.num_epochs} epochs)")
#             return
        
#         print(f"🚀 Starting training from epoch {start_epoch + 1}")
        
#         for epoch in range(start_epoch, self.config.num_epochs):
#             print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")

#             train_loss = self.train_epoch(train_loader)
#             self.train_losses.append(train_loss)

#             val_loss = None
#             if val_loader is not None:
#                 val_loss = self.validate_epoch(val_loader)
#                 self.val_losses.append(val_loss)

#                 # Save best model checkpoint
#                 if val_loss < self.best_val_loss:
#                     self.best_val_loss = val_loss
#                     self.best_epoch = epoch
#                     self._save_checkpoint(epoch)
#                     print(f"🏆 Best model saved at epoch {epoch+1}")

#             # Step scheduler
#             if self.scheduler:
#                 if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                     self.scheduler.step(val_loss if val_loss is not None else train_loss)
#                 else:
#                     self.scheduler.step()

#             # Epoch summary
#             if val_loss is not None:
#                 print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
#             else:
#                 print(f"Train Loss: {train_loss:.6f}")

#     def _save_checkpoint(self, epoch: int):
#         """Save model checkpoint."""
#         os.makedirs(self.config.checkpoint_dir, exist_ok=True)
#         checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
#             'best_val_loss': self.best_val_loss,
#             'train_losses': self.train_losses,
#             'val_losses': self.val_losses
#         }
        
#         torch.save(checkpoint, checkpoint_path)

#     def train_epoch(self, loader: DataLoader) -> float:
#         """
#         Train for one epoch.
        
#         Args:
#             loader: Training data loader
            
#         Returns:
#             Average training loss
#         """
#         self.model.train()
#         total_loss = 0.0
#         num_batches = len(loader)
        
#         pbar = tqdm(loader, desc="Training", leave=False)
#         for batch_idx, batch in enumerate(pbar):
#             self.optimizer.zero_grad()
            
#             # Handle different batch formats
#             if len(batch) == 3:
#                 _, inputs, targets = batch
#             else:
#                 inputs, targets = batch
                
#             inputs = inputs.to(self.device)
#             targets = targets.to(self.device)
            
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, targets)
            
#             loss.backward()
#             self.optimizer.step()
            
#             total_loss += loss.item()
#             pbar.set_postfix({"train_loss": loss.item()})
        
#         avg_loss = total_loss / num_batches
#         return avg_loss

#     def validate_epoch(self, val_loader: DataLoader) -> float:
#         """
#         Validate for one epoch.

#         Args:
#             val_loader: Validation data loader

#         Returns:
#             Average validation loss
#         """
#         self.model.eval()
#         total_loss = 0.0
#         num_batches = len(val_loader)

#         with torch.no_grad():
#             pbar = tqdm(val_loader, desc="Validation", leave=False)
#             for batch_idx, batch in enumerate(pbar):
#                 if len(batch) == 3:
#                     _, inputs, targets = batch
#                 else:
#                     inputs, targets = batch

#                 inputs = inputs.to(self.device)
#                 targets = targets.to(self.device)

#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, targets)

#                 total_loss += loss.item()
#                 pbar.set_postfix({"val_loss": loss.item()})

#         avg_loss = total_loss / num_batches
#         return avg_loss

#     def evaluate(self, test_loader: DataLoader) -> Dict:
#         """
#         Evaluate the model on test data.
        
#         Args:
#             test_loader: Test data loader
            
#         Returns:
#             Dictionary of evaluation metrics
#         """
#         self.model.eval()
#         all_predictions = []
#         all_targets = []
        
#         with torch.no_grad():
#             pbar = tqdm(test_loader, desc="Evaluating")
#             for batch in pbar:
#                 if len(batch) == 3:
#                     _, inputs, targets = batch
#                 else:
#                     inputs, targets = batch
                    
#                 inputs = inputs.to(self.device)
#                 targets = targets.to(self.device)
                
#                 outputs = self.model(inputs)
                
#                 all_predictions.append(outputs.cpu())
#                 all_targets.append(targets.cpu())
        
#         predictions = torch.cat(all_predictions, dim=0)
#         targets = torch.cat(all_targets, dim=0)
        
#         # Calculate metrics using your evaluate_all_metrics function
#         metrics = evaluate_all_metrics(predictions.numpy(), targets.numpy())
        
#         return metrics

#     def load_best_model(self):
#         """Load the best saved model."""
#         checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
#         if os.path.exists(checkpoint_path):
#             checkpoint = torch.load(checkpoint_path, map_location=self.device)
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             print(f"✅ Loaded best model from epoch {checkpoint.get('epoch', 0) + 1}")
#         else:
#             print("⚠️ No saved model found!")

#     def get_training_summary(self) -> Dict:
#         """Get a summary of training progress."""
#         return {
#             'total_epochs_trained': len(self.train_losses),
#             'best_epoch': self.best_epoch + 1,
#             'best_val_loss': self.best_val_loss,
#             'final_train_loss': self.train_losses[-1] if self.train_losses else None,
#             'final_val_loss': self.val_losses[-1] if self.val_losses else None,
#             'checkpoint_loaded': self.checkpoint_loaded
#         }