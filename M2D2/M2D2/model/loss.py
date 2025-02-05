import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
import logging
import numpy as np
from sklearn.metrics import r2_score

@dataclass
class LossConfig:
    mse_weight: float = 1.0
    energy_weight: float = 0.1
    entropy_weight: float = 0.1
    free_energy_weight: float = 0.1
    temporal_weight: float = 0.1
    alpha: float = 0.5  # Weighing factor for combined losses
    beta: float = 0.5   # Temperature scaling factor
    gamma: float = 0.1  # Additional regularization parameter
    delta: float = 0.05 # Fine-tuning parameter

class ThermodynamicLoss(nn.Module):
    """
    Custom loss function incorporating statistical mechanics principles.
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets, energies, temperature, degeneracies, access_history=None):
        # Use raw outputs without softmax or scaling
        mse_loss = self.mse(outputs, targets)
        
        # Log raw values for debugging
        logging.info("\nLoss Calculation Debug:")
        logging.info(f"Raw outputs range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
        logging.info(f"Raw targets range: [{targets.min().item():.6f}, {targets.max().item():.6f}]")
        logging.info(f"MSE Loss: {mse_loss.item():.6f}")
        
        return {'total_loss': mse_loss}

def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Calculate raw MSE metrics."""
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Calculate raw MSE without any scaling
    raw_mse = float(np.mean((outputs_np - targets_np) ** 2))
    
    # Log raw values for debugging
    logging.debug(f"Output values: min={outputs_np.min():.6f}, max={outputs_np.max():.6f}, mean={outputs_np.mean():.6f}")
    logging.debug(f"Target values: min={targets_np.min():.6f}, max={targets_np.max():.6f}, mean={targets_np.mean():.6f}")
    
    return {
        'raw_mse': raw_mse,  # Unscaled MSE
        'mse': raw_mse       # Keep this for compatibility
    }
