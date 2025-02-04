import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class LossConfig:
    alpha: float = 0.2  # Weight for energy conservation
    beta: float = 0.1   # Weight for entropy maximization
    gamma: float = 0.1  # Weight for free energy minimization
    delta: float = 0.05 # Weight for temporal consistency

class ThermodynamicLoss(nn.Module):
    """
    Custom loss function incorporating statistical mechanics principles.
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        energies: torch.Tensor,
        temperature: torch.Tensor,
        degeneracies: torch.Tensor,
        access_history: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss with multiple components.
        
        Args:
            outputs: Predicted access frequencies [batch_size, n_nodes]
            labels: True access frequencies [batch_size, n_nodes]
            energies: Energy levels [batch_size, n_nodes]
            temperature: System temperature [batch_size]
            degeneracies: State degeneracies [batch_size, n_nodes]
            access_history: Optional access history [batch_size, seq_len, n_nodes]
            
        Returns:
            Dictionary containing total loss and individual components
        """
        # Base prediction loss (MSE)
        mse_loss = torch.nn.functional.mse_loss(outputs, labels)
        
        # Energy conservation constraint
        energy_conservation_loss = self._compute_energy_conservation(
            outputs, labels, energies
        )
        
        # Entropy term (maximization)
        entropy_loss = self._compute_entropy(outputs, degeneracies)
        
        # Free energy minimization
        free_energy_loss = self._compute_free_energy(
            outputs, energies, temperature, degeneracies
        )
        
        # Temporal consistency (if access_history provided)
        temporal_loss = torch.tensor(0.0, device=outputs.device)
        if access_history is not None:
            temporal_loss = self._compute_temporal_consistency(
                outputs, access_history
            )
        
        # Combine all loss terms
        total_loss = (
            mse_loss +
            self.config.alpha * energy_conservation_loss -
            self.config.beta * entropy_loss +
            self.config.gamma * free_energy_loss +
            self.config.delta * temporal_loss
        )
          # Debug: Print loss components
        print(f"mse_loss: {mse_loss.item()}")
        print(f"energy_conservation_loss: {energy_conservation_loss.item()}")
        print(f"entropy_loss: {entropy_loss.item()}")
        print(f"free_energy_loss: {free_energy_loss.item()}")
        print(f"temporal_loss: {temporal_loss.item()}")
        print(f"total_loss: {total_loss.item()}")
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'energy_conservation_loss': energy_conservation_loss,
            'entropy_loss': entropy_loss,
            'free_energy_loss': free_energy_loss,
            'temporal_loss': temporal_loss
        }

    def _compute_energy_conservation(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        energies: torch.Tensor
    ) -> torch.Tensor:
        """Compute energy conservation loss term."""
        pred_energy = torch.sum(outputs * energies, dim=1)
        true_energy = torch.sum(labels * energies, dim=1)
        return torch.mean(torch.abs(pred_energy - true_energy))

    def _compute_entropy(
        self,
        outputs: torch.Tensor,
        degeneracies: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy term using Gibbs entropy formula."""
        eps = 1e-10  # Small constant for numerical stability
        entropy = -torch.sum(
            outputs * torch.log(outputs + eps) * degeneracies,
            dim=1
        )
        return torch.mean(entropy)

    def _compute_free_energy(
        self,
        outputs: torch.Tensor,
        energies: torch.Tensor,
        temperature: torch.Tensor,
        degeneracies: torch.Tensor
    ) -> torch.Tensor:
        """Compute free energy term."""
        eps = 1e-10
        energy_term = torch.sum(outputs * energies, dim=1)
        entropy_term = torch.sum(
            degeneracies * outputs * torch.log(outputs + eps),
            dim=1
        )
        free_energy = energy_term - temperature * entropy_term
        return torch.mean(free_energy)

    def _compute_temporal_consistency(
        self,
        outputs: torch.Tensor,
        access_history: torch.Tensor
    ) -> torch.Tensor:
        """Compute temporal consistency loss."""
        # Get recent access patterns (last few timesteps)
        recent_access = access_history[:, -5:, :]  # Last 5 timesteps
        
        # Calculate average recent access pattern
        avg_recent_access = torch.mean(recent_access, dim=1)
        
        # Penalize large deviations from recent access patterns
        temporal_consistency = torch.mean(
            torch.abs(outputs - avg_recent_access)
        )
        
        return temporal_consistency
