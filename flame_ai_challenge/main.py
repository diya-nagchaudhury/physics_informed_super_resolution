import torch
from torch.utils.data import DataLoader
from src.training.trainer import Trainer
from config.config import Config
from src.data.dataset import FLAMEAIDataset  # Assuming your dataset is here
from src.models.upsampling import get_model
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import calculate_psnr, calculate_ssim
from src.visualization.plotters import FlowFieldPlotter
from src.visualization.energy_maps import EnergyMapGenerator
import os

def prepare_for_model(x: torch.Tensor, device: str) -> torch.Tensor:
    """
    Ensures x is in shape (B, C, H, W) for Conv2d models.
    Moves tensor to the correct device.
    """
    if x.ndim == 3:  # (C, H, W) → add batch dim
        x = x.unsqueeze(0)
    elif x.ndim == 1:  # completely flat vector
        raise ValueError(f"Invalid input shape {x.shape} for Conv2d. Expected at least (C, H, W).")
    return x.to(device)

# 1. Config
config = Config()

# 2. Dataset & DataLoaders
train_dataset = FLAMEAIDataset("train", config)
val_dataset   = FLAMEAIDataset("val", config)
test_dataset  = FLAMEAIDataset("test", config)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# 3. Model
model = get_model(config.model_name, config)

# 4. Trainer
trainer = Trainer(model, config)
trainer.train(train_loader, val_loader)

# 5. Evaluation
evaluator = ModelEvaluator(model, config)
results = evaluator.evaluate(test_loader)
print("Evaluation results:", results)

# 6. Visualization (example usage)
plotter = FlowFieldPlotter(config)

# get first batch from val_loader
_, lr_data, hr_data = next(iter(val_loader))

# Convert to float and move to device
lr_data = lr_data.float().to(config.device)
hr_data = hr_data.float().to(config.device)

print("Before prepare:", lr_data.shape)  # Should now be (B, 4, 16, 16)
lr_data = prepare_for_model(lr_data, config.device)

predicted_data = model(lr_data).detach().cpu()
# create output path if it doesnt exist
output_path = config.output_path
if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)
plotter.plot_flow_comparison(lr_data.cpu(), hr_data.cpu(), predicted_data, save_path=output_path)

# 7. Energy map
energy_gen = EnergyMapGenerator()
flow_data = predicted_data[0]  # example: first sample's flow field
energy_map = energy_gen.generate_divergence_map(flow_data)
