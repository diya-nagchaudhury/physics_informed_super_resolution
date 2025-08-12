import torch
from typing import Dict, Optional
from src.evaluation.metrics import calculate_psnr, calculate_ssim

class ModelEvaluator:
    """
    Evaluates a trained model on a dataset.
    """

    def __init__(self, model: torch.nn.Module, config, device: Optional[str] = None):
        self.model = model
        self.config = config
        self.device = device or config.device
        self.model.to(self.device)

    def evaluate(self, data_loader) -> Dict[str, float]:
        """
        Run evaluation on a given dataloader.

        Returns:
            dict with aggregated metrics {"PSNR": float, "SSIM": float}
        """
        self.model.eval()
        total_psnr, total_ssim = 0.0, 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:  # test split: (id, inputs)
                    ids, inputs = batch
                    targets = None
                elif len(batch) == 3:  # val/train split: (id, inputs, targets)
                    ids, inputs, targets = batch
                else:
                    raise ValueError(f"Unexpected batch structure: {len(batch)} elements.")

                inputs = inputs.float().to(self.device)

                outputs = self.model(inputs)

                if targets is not None:
                    targets = targets.float().to(self.device)
                    psnr_val = calculate_psnr(outputs, targets).item()
                    ssim_val = calculate_ssim(outputs, targets).item()

                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    num_batches += 1

        if num_batches > 0:
            avg_psnr = total_psnr / num_batches
            avg_ssim = total_ssim / num_batches
            return {"PSNR": avg_psnr, "SSIM": avg_ssim}
        else:
            return {}
