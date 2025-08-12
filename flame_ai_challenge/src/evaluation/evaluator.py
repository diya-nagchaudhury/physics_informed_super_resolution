"""Model evaluation module."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm

from config.config import Config
from src.evaluation.metrics import evaluate_all_metrics
from src.visualization.plotters import FlowFieldPlotter
from src.visualization.energy_maps import EnergyMapGenerator


class ModelEvaluator:
    """Class for comprehensive model evaluation."""
    
    def __init__(self, model: nn.Module, config: Config, device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            config: Configuration object
            device: Device to use for evaluation
        """
        self.model = model
        self.config = config
        self.device = device or config.device
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize visualization tools
        self.plotter = FlowFieldPlotter(config)
        self.energy_generator = EnergyMapGenerator(config)
    
    def evaluate_dataset(self, data_loader: DataLoader, save_individual_results: bool = False) -> Dict[str, float]:
        """
        Evaluate model on entire dataset.
        
        Args:
            data_loader: Data loader for evaluation
            save_individual_results: Whether to save individual sample results
            
        Returns:
            Dictionary containing averaged metrics
        """
        print("Evaluating model on dataset...")
        
        all_metrics = {}
        individual_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                if len(batch) == 3:  # Has ground truth
                    sample_ids, inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Calculate metrics for each sample in batch
                    for i in range(len(sample_ids)):
                        sample_id = sample_ids[i].item() if hasattr(sample_ids[i], 'item') else sample_ids[i]
                        
                        # Extract single sample
                        sample_output = outputs[i:i+1]
                        sample_target = targets[i:i+1]
                        
                        # Calculate metrics
                        sample_metrics = evaluate_all_metrics(sample_output, sample_target)
                        sample_metrics['sample_id'] = sample_id
                        
                        individual_results.append(sample_metrics)
                        
                        # Accumulate metrics
                        for metric_name, value in sample_metrics.items():
                            if metric_name != 'sample_id':
                                if metric_name not in all_metrics:
                                    all_metrics[metric_name] = []
                                all_metrics[metric_name].append(value)
                        
                        # Save individual visualizations if requested
                        if save_individual_results and batch_idx < 10:  # Save first 10 batches only
                            self._save_individual_results(
                                sample_id, 
                                inputs[i:i+1], 
                                targets[i:i+1], 
                                outputs[i:i+1]
                            )
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        # Save individual results to CSV
        if save_individual_results:
            results_df = pd.DataFrame(individual_results)
            results_path = os.path.join(self.config.output_path, 'individual_evaluation_results.csv')
            results_df.to_csv(results_path, index=False)
            print(f"Individual results saved to {results_path}")
        
        return avg_metrics
    
    def _save_individual_results(self, sample_id: int, inputs: torch.Tensor, 
                               targets: torch.Tensor, outputs: torch.Tensor) -> None:
        """Save visualization results for individual sample."""
        # Create output directory
        sample_dir = os.path.join(self.config.output_path, 'individual_samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Remove batch dimension
        lr_data = inputs.squeeze(0)
        hr_data = targets.squeeze(0)
        pred_data = outputs.squeeze(0)
        
        # Flow field comparison
        comparison_path = os.path.join(sample_dir, f'comparison_sample_{sample_id}.png')
        self.plotter.plot_flow_comparison(
            lr_data, hr_data, pred_data,
            sample_id=str(sample_id),
            save_path=comparison_path
        )
        
        # Error heatmap
        error_path = os.path.join(sample_dir, f'error_sample_{sample_id}.png')
        self.plotter.plot_error_heatmap(
            pred_data, hr_data,
            title=f"Error Analysis - Sample {sample_id}",
            save_path=error_path
        )
        
        # Energy maps
        energy_path = os.path.join(sample_dir, f'energy_maps_sample_{sample_id}.png')
        self.energy_generator.plot_all_energy_maps(
            pred_data,
            sample_id=str(sample_id),
            save_path=energy_path
        )
    
    def compare_models(self, other_models: List[Tuple[str, nn.Module]], 
                      data_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            other_models: List of (name, model) tuples
            data_loader: Data loader for comparison
            
        Returns:
            Dictionary with model names as keys and metrics as values
        """
        all_models = [("Current Model", self.model)] + other_models
        comparison_results = {}
        
        for model_name, model in all_models:
            print(f"\nEvaluating {model_name}...")
            
            # Temporarily set model for evaluation
            original_model = self.model
            self.model = model.to(self.device)
            self.model.eval()
            
            # Evaluate model
            results = self.evaluate_dataset(data_loader, save_individual_results=False)
            comparison_results[model_name] = {
                metric_name: stats['mean'] for metric_name, stats in results.items()
            }
            
            # Restore original model
            self.model = original_model
        
        # Save comparison results
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_path = os.path.join(self.config.output_path, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path)
        
        print(f"Model comparison saved to {comparison_path}")
        return comparison_results
    
    def analyze_failure_cases(self, data_loader: DataLoader, threshold_psnr: float = 20.0) -> List[Dict]:
        """
        Analyze samples where the model performs poorly.
        
        Args:
            data_loader: Data loader for analysis
            threshold_psnr: PSNR threshold below which samples are considered failures
            
        Returns:
            List of failure case information
        """
        print(f"Analyzing failure cases (PSNR < {threshold_psnr} dB)...")
        
        failure_cases = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Analyzing")):
                if len(batch) == 3:
                    sample_ids, inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    for i in range(len(sample_ids)):
                        sample_id = sample_ids[i].item() if hasattr(sample_ids[i], 'item') else sample_ids[i]
                        
                        sample_output = outputs[i:i+1]
                        sample_target = targets[i:i+1]
                        
                        metrics = evaluate_all_metrics(sample_output, sample_target)
                        
                        if metrics['psnr'] < threshold_psnr:
                            failure_info = {
                                'sample_id': sample_id,
                                'psnr': metrics['psnr'],
                                'ssim': metrics['ssim'],
                                'mse': metrics['mse'],
                                'batch_idx': batch_idx,
                                'sample_idx': i
                            }
                            failure_cases.append(failure_info)
                            
                            # Save detailed analysis for worst cases
                            if len(failure_cases) <= 20:  # Save first 20 failure cases
                                self._analyze_failure_case(
                                    sample_id,
                                    inputs[i:i+1],
                                    targets[i:i+1],
                                    outputs[i:i+1],
                                    metrics
                                )
        
        # Save failure cases summary
        if failure_cases:
            failure_df = pd.DataFrame(failure_cases)
            failure_path = os.path.join(self.config.output_path, 'failure_cases.csv')
            failure_df.to_csv(failure_path, index=False)
            print(f"Found {len(failure_cases)} failure cases. Summary saved to {failure_path}")
        else:
            print("No failure cases found!")
        
        return failure_cases
    
    def _analyze_failure_case(self, sample_id: int, inputs: torch.Tensor,
                            targets: torch.Tensor, outputs: torch.Tensor,
                            metrics: Dict[str, float]) -> None:
        """Perform detailed analysis of a failure case."""
        failure_dir = os.path.join(self.config.output_path, 'failure_analysis')
        os.makedirs(failure_dir, exist_ok=True)
        
        lr_data = inputs.squeeze(0)
        hr_data = targets.squeeze(0)
        pred_data = outputs.squeeze(0)
        
        # Detailed comparison plot
        comparison_path = os.path.join(failure_dir, f'failure_comparison_{sample_id}.png')
        self.plotter.plot_flow_comparison(
            lr_data, hr_data, pred_data,
            sample_id=f"{sample_id} (PSNR: {metrics['psnr']:.2f})",
            save_path=comparison_path
        )
        
        # Streamline plot for velocity analysis
        streamline_path = os.path.join(failure_dir, f'failure_streamlines_{sample_id}.png')
        self.plotter.plot_velocity_streamlines(
            pred_data,
            title=f"Predicted Velocity Field - Sample {sample_id}",
            save_path=streamline_path
        )
        
        # Energy map analysis
        energy_path = os.path.join(failure_dir, f'failure_energy_{sample_id}.png')
        self.energy_generator.plot_all_energy_maps(
            pred_data,
            sample_id=str(sample_id),
            save_path=energy_path
        )
    
    def generate_evaluation_report(self, data_loader: DataLoader) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Path to generated report
        """
        print("Generating comprehensive evaluation report...")
        
        # Evaluate dataset
        metrics = self.evaluate_dataset(data_loader, save_individual_results=True)
        
        # Analyze failure cases
        failure_cases = self.analyze_failure_cases(data_loader)
        
        # Generate report
        report_lines = [
            "# FLAME AI Challenge Model Evaluation Report\n",
            f"## Model Configuration\n",
            f"- Model Type: {type(self.model).__name__}\n",
            f"- Device: {self.device}\n",
            f"- Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n\n",
            
            "## Overall Performance Metrics\n"
        ]
        
        for metric_name, stats in metrics.items():
            report_lines.extend([
                f"### {metric_name.upper()}\n",
                f"- Mean: {stats['mean']:.6f}\n",
                f"- Std: {stats['std']:.6f}\n",
                f"- Min: {stats['min']:.6f}\n",
                f"- Max: {stats['max']:.6f}\n",
                f"- Median: {stats['median']:.6f}\n\n"
            ])
        
        report_lines.extend([
            "## Failure Analysis\n",
            f"- Total failure cases: {len(failure_cases)}\n",
            f"- Failure rate: {len(failure_cases) / len(data_loader.dataset) * 100:.2f}%\n\n"
        ])
        
        if failure_cases:
            worst_case = min(failure_cases, key=lambda x: x['psnr'])
            report_lines.extend([
                f"### Worst Case\n",
                f"- Sample ID: {worst_case['sample_id']}\n",
                f"- PSNR: {worst_case['psnr']:.2f} dB\n",
                f"- SSIM: {worst_case['ssim']:.4f}\n\n"
            ])
        
        # Save report
        report_path = os.path.join(self.config.output_path, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        
        print(f"Evaluation report saved to {report_path}")
        return report_path


# import torch
# from typing import Dict, Optional
# from src.evaluation.metrics import calculate_psnr, calculate_ssim

# class ModelEvaluator:
#     """
#     Evaluates a trained model on a dataset.
#     """

#     def __init__(self, model: torch.nn.Module, config, device: Optional[str] = None):
#         self.model = model
#         self.config = config
#         self.device = device or config.device
#         self.model.to(self.device)

#     def evaluate(self, data_loader) -> Dict[str, float]:
#         """
#         Run evaluation on a given dataloader.

#         Returns:
#             dict with aggregated metrics {"PSNR": float, "SSIM": float}
#         """
#         self.model.eval()
#         total_psnr, total_ssim = 0.0, 0.0
#         num_batches = 0

#         with torch.no_grad():
#             for batch in data_loader:
#                 if len(batch) == 2:  # test split: (id, inputs)
#                     ids, inputs = batch
#                     targets = None
#                 elif len(batch) == 3:  # val/train split: (id, inputs, targets)
#                     ids, inputs, targets = batch
#                 else:
#                     raise ValueError(f"Unexpected batch structure: {len(batch)} elements.")

#                 inputs = inputs.float().to(self.device)

#                 outputs = self.model(inputs)

#                 if targets is not None:
#                     targets = targets.float().to(self.device)
#                     psnr_val = calculate_psnr(outputs, targets).item()
#                     ssim_val = calculate_ssim(outputs, targets).item()

#                     total_psnr += psnr_val
#                     total_ssim += ssim_val
#                     num_batches += 1

#         if num_batches > 0:
#             avg_psnr = total_psnr / num_batches
#             avg_ssim = total_ssim / num_batches
#             return {"PSNR": avg_psnr, "SSIM": avg_ssim}
#         else:
#             return {}
