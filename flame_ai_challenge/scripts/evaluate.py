#!/usr/bin/env python3
"""Evaluation script for the models."""

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
from src.evaluation.evaluator import ModelEvaluator


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model')
    
    parser.add_argument('--model', type=str, default='residual',
                       choices=['simple', 'convolutional', 'residual', 'attention', 'physics'],
                       help='Model architecture to use')
    
    parser.add_argument('--checkpoint', type=str, default='/home/diya/Projects/physics_informed_super_resolution/flame_ai_challenge/outputs/simple_model/train/best_model.pth',
                       help='Path to model checkpoint')
    
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for evaluation')
    
    parser.add_argument('--data_path', type=str, default=Config.input_path,
                       help='Path to dataset')
    
    parser.add_argument('--output_path', type=str, default=Config.output_path+'evaluation/',
                       help='Path to save evaluation results')
    
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual sample results and visualizations')
    
    parser.add_argument('--analyze_failures', action='store_true',
                       help='Analyze failure cases')
    
    parser.add_argument('--failure_threshold', type=float, default=20.0,
                       help='PSNR threshold for failure analysis (dB)')
    
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate comprehensive evaluation report')
    
    return parser.parse_args()


def main():
    """Driver function."""
    args = parse_args()
    
    # Create configuration
    config = Config()
    config.batch_size = args.batch_size
    config.device = args.device
    config.input_path = args.data_path
    config.output_path = args.output_path
    
    # Ensure output directory exists
    os.makedirs(config.output_path, exist_ok=True)
    
    print("=== Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")
    print("=" * 40)
    
    # Check device availability
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        config.device = 'cpu'
    elif config.device == 'cpu':
        print("Using CPU for evaluation")
    print(f"Using device: {config.device}")
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    try:
        dataset = FLAMEAIDataset(args.split, config, apply_norm=True)
        data_loader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True if config.device == 'cuda' else False
        )
        print(f"Dataset samples: {len(dataset)}")
        
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please check that the dataset path is correct and files exist")
        return
    
    # Create model
    print(f"Creating {args.model} model...")
    try:
        model = get_model(args.model, config)
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    evaluator = ModelEvaluator(model, config)
    print("\nStarting evaluation...")
    
    if args.generate_report:
        report_path = evaluator.generate_evaluation_report(data_loader)
        print(f"Comprehensive report generated: {report_path}")
        
    else:
        metrics = evaluator.evaluate_dataset(
            data_loader, 
            save_individual_results=args.save_individual
        )
        
        # Print results
        print("\n=== Results ===")
        for metric_name, stats in metrics.items():
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std:  {stats['std']:.6f}")
            print(f"  Min:  {stats['min']:.6f}")
            print(f"  Max:  {stats['max']:.6f}")
        
        # Analyze failure cases if requested
        if args.analyze_failures:
            print(f"\nAnalyzing failure cases (PSNR < {args.failure_threshold} dB)...")
            failure_cases = evaluator.analyze_failure_cases(
                data_loader, 
                threshold_psnr=args.failure_threshold
            )
            
            if failure_cases:
                print(f"Found {len(failure_cases)} failure cases")
                print(f"Failure rate: {len(failure_cases) / len(dataset) * 100:.2f}%")
            else:
                print("No failure cases found!")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()