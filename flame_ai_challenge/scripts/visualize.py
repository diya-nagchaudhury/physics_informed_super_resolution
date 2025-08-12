#!/usr/bin/env python3
"""Visualization script for results."""

import argparse
import torch
from torch.utils.data import DataLoader
import sys
import os
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from src.data.dataset import FLAMEAIDataset
from src.models.upsampling import get_model
from src.visualization.plotters import FlowFieldPlotter
from src.visualization.energy_maps import EnergyMapGenerator


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize FLAME AI Challenge data and results')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['data', 'predictions', 'energy_maps', 'comparison'],
                       help='Visualization mode')
    
    parser.add_argument('--data_path', type=str, default=Config.input_path,
                       help='Path to dataset')
    
    parser.add_argument('--output_path', type=str, default=Config.output_path + 'visualizations/',
                       help='Path to save visualizations')
    
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to visualize')
    
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    parser.add_argument('--sample_ids', type=str, default=None,
                       help='Specific sample IDs to visualize (comma-separated)')
    
    # Model-related arguments
    parser.add_argument('--model', type=str, default='residual',
                       choices=['simple', 'convolutional', 'residual', 'attention', 'physics'],
                       help='Model architecture to use')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (required for predictions)')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    # Visualization options
    parser.add_argument('--show_streamlines', action='store_true',
                       help='Show velocity streamlines')
    
    parser.add_argument('--show_energy_stats', action='store_true',
                       help='Show energy statistics')
    
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved images')
    
    return parser.parse_args()


def visualize_data(args, config, dataset, plotter, energy_generator):
    """Visualize raw dataset samples."""
    print("Visualizing dataset samples...")
    
    # Select samples to visualize
    if args.sample_ids:
        sample_indices = [int(x.strip()) for x in args.sample_ids.split(',')]
    else:
        sample_indices = random.sample(range(len(dataset)), min(args.samples, len(dataset)))
    
    for i, idx in enumerate(sample_indices):
        print(f"Visualizing sample {i+1}/{len(sample_indices)} (ID: {idx})")
        
        sample = dataset[idx]
        if len(sample) == 3:  # Has ground truth
            sample_id, lr_data, hr_data = sample
            
            # Plot LR data
            lr_path = os.path.join(config.output_path, f'lr_sample_{sample_id}.png')
            plotter.plot_flow_field(lr_data, f"Low Resolution - Sample {sample_id}", lr_path)
            
            # Plot HR data
            hr_path = os.path.join(config.output_path, f'hr_sample_{sample_id}.png')
            plotter.plot_flow_field(hr_data, f"High Resolution - Sample {sample_id}", hr_path)
            
            # Plot velocity streamlines
            if args.show_streamlines:
                streamline_path = os.path.join(config.output_path, f'streamlines_sample_{sample_id}.png')
                plotter.plot_velocity_streamlines(hr_data, f"Velocity Streamlines - Sample {sample_id}", streamline_path)
            
        else:  # Test data
            sample_id, lr_data = sample
            lr_path = os.path.join(config.output_path, f'test_lr_sample_{sample_id}.png')
            plotter.plot_flow_field(lr_data, f"Test Low Resolution - Sample {sample_id}", lr_path)


def visualize_predictions(args, config, dataset, model, plotter, energy_generator):
    """Visualize model predictions."""
    if args.checkpoint is None:
        print("Error: Checkpoint required for prediction visualization")
        return
    
    print("Loading model and generating predictions...")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    # Select samples
    if args.sample_ids:
        sample_indices = [int(x.strip()) for x in args.sample_ids.split(',')]
    else:
        sample_indices = random.sample(range(len(dataset)), min(args.samples, len(dataset)))
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            print(f"Processing sample {i+1}/{len(sample_indices)} (ID: {idx})")
            
            sample = dataset[idx]
            if len(sample) == 3:  # Has ground truth
                sample_id, lr_data, hr_data = sample
                
                # Generate prediction
                lr_input = lr_data.unsqueeze(0).to(config.device)
                prediction = model(lr_input).squeeze(0).cpu()
                
                # Plot comparison
                comparison_path = os.path.join(config.output_path, f'prediction_comparison_{sample_id}.png')
                plotter.plot_flow_comparison(lr_data, hr_data, prediction, str(sample_id), comparison_path)
                
                # Plot error heatmap
                error_path = os.path.join(config.output_path, f'prediction_error_{sample_id}.png')
                plotter.plot_error_heatmap(prediction, hr_data, f"Prediction Error - Sample {sample_id}", error_path)
                
                # Plot streamlines if requested
                if args.show_streamlines:
                    streamline_path = os.path.join(config.output_path, f'prediction_streamlines_{sample_id}.png')
                    plotter.plot_velocity_streamlines(prediction, f"Predicted Velocity - Sample {sample_id}", streamline_path)
                
            else:  # Test data
                sample_id, lr_data = sample
                
                lr_input = lr_data.unsqueeze(0).to(config.device)
                prediction = model(lr_input).squeeze(0).cpu()
                
                # Plot prediction
                pred_path = os.path.join(config.output_path, f'test_prediction_{sample_id}.png')
                plotter.plot_flow_field(prediction, f"Test Prediction - Sample {sample_id}", pred_path)


def visualize_energy_maps(args, config, dataset, model, plotter, energy_generator):
    """Visualize energy maps."""
    print("Generating energy maps...")
    
    if args.checkpoint and model:
        # Load checkpoint for model predictions
        checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)
        model.eval()
    
    # Select samples
    if args.sample_ids:
        sample_indices = [int(x.strip()) for x in args.sample_ids.split(',')]
    else:
        sample_indices = random.sample(range(len(dataset)), min(args.samples, len(dataset)))
    
    for i, idx in enumerate(sample_indices):
        print(f"Processing sample {i+1}/{len(sample_indices)} (ID: {idx})")
        
        sample = dataset[idx]
        if len(sample) == 3:  # Has ground truth
            sample_id, lr_data, hr_data = sample
            
            # Energy maps for ground truth
            gt_energy_path = os.path.join(config.output_path, f'energy_maps_gt_{sample_id}.png')
            energy_generator.plot_all_energy_maps(hr_data, f"{sample_id} (Ground Truth)", gt_energy_path)
            
            # Energy maps for prediction if model available
            if args.checkpoint and model:
                with torch.no_grad():
                    lr_input = lr_data.unsqueeze(0).to(config.device)
                    prediction = model(lr_input).squeeze(0).cpu()
                    
                    pred_energy_path = os.path.join(config.output_path, f'energy_maps_pred_{sample_id}.png')
                    energy_generator.plot_all_energy_maps(prediction, f"{sample_id} (Prediction)", pred_energy_path)
            
            # Print energy statistics if requested
            if args.show_energy_stats:
                ke_map = energy_generator.generate_kinetic_energy_map(hr_data)
                stats = energy_generator.compute_energy_statistics(ke_map)
                print(f"Sample {sample_id} energy statistics:")
                for stat_name, value in stats.items():
                    print(f"  {stat_name}: {value:.6f}")
        
        else:  # Test data - only show predictions if model available
            if args.checkpoint and model:
                sample_id, lr_data = sample
                
                with torch.no_grad():
                    lr_input = lr_data.unsqueeze(0).to(config.device)
                    prediction = model(lr_input).squeeze(0).cpu()
                    
                    pred_energy_path = os.path.join(config.output_path, f'test_energy_maps_{sample_id}.png')
                    energy_generator.plot_all_energy_maps(prediction, f"{sample_id} (Test Prediction)", pred_energy_path)


def main():
    """Driver function."""
    args = parse_args()
    
    # Create configuration
    config = Config()
    config.device = args.device
    config.input_path = args.data_path
    config.output_path = args.output_path
    config.dpi = args.dpi
    
    # Ensure output directory exists
    os.makedirs(config.output_path, exist_ok=True)
    
    print("=== Visualization ===")
    print(f"Mode: {args.mode}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.samples}")
    print(f"Output path: {config.output_path}")
    print("=" * 40)
    
    # Check device availability
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        config.device = 'cpu'
    print(f"Using device: {config.device}")
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    try:
        dataset = FLAMEAIDataset(args.split, config, apply_norm=True)
        print(f"Dataset samples: {len(dataset)}")
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize visualization tools
    plotter = FlowFieldPlotter(config)
    energy_generator = EnergyMapGenerator(config)
    
    # Initialize model if needed
    model = None
    if args.mode in ['predictions', 'energy_maps'] and args.checkpoint:
        try:
            model = get_model(args.model, config)
        except Exception as e:
            print(f"Error creating model: {e}")
            return
    
    # Run visualization based on mode
    if args.mode == 'data':
        visualize_data(args, config, dataset, plotter, energy_generator)
    
    elif args.mode == 'predictions':
        visualize_predictions(args, config, dataset, model, plotter, energy_generator)
    
    elif args.mode == 'energy_maps':
        visualize_energy_maps(args, config, dataset, model, plotter, energy_generator)
    
    elif args.mode == 'comparison':
        # Combine multiple visualization modes
        visualize_data(args, config, dataset, plotter, energy_generator)
        if args.checkpoint:
            visualize_predictions(args, config, dataset, model, plotter, energy_generator)
            visualize_energy_maps(args, config, dataset, model, plotter, energy_generator)
    
    print(f"\nVisualization completed! Results saved to {config.output_path}")


if __name__ == "__main__":
    main()