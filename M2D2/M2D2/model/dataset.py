#dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DataConfig:
    n_samples: int
    sequence_length: int
    n_nodes: int
    node_features: int
    train_size: float
    num_workers: int
    #pin_memory: bool


class EnhancedStorageDataGenerator:
    """Generates synthetic data for storage system simulation."""
    
    def __init__(self, config: DataConfig):
        self.config = config

    def generate_dataset(self) -> List[Dict[str, np.ndarray]]:
        """
        Generate synthetic dataset with thermodynamic properties.
        
        Returns:
            List[Dict[str, np.ndarray]]: List of samples with features and labels
        """
        dataset = []
        for _ in range(self.config.n_samples):
            # Generate node features
            node_features = np.random.normal(
                loc=0.0, 
                scale=0.1, 
                size=(self.config.n_nodes, self.config.node_features)
            )
            
            # Generate access history with temporal patterns
            access_history = self._generate_access_history()
            
            # Generate thermodynamic properties
            energies = np.abs(np.random.normal(1.0, 0.2, size=(self.config.n_nodes,)))
            temperature = np.abs(np.random.normal(1.0, 0.1))
            degeneracies = np.abs(np.random.normal(1.0, 0.1, size=(self.config.n_nodes,)))
            
            # Generate access frequencies based on Boltzmann distribution
            access_frequency = self._compute_boltzmann_distribution(energies, temperature)
            
            # Generate time series data
            time_series = self._generate_time_series()
            
            # Generate adjacency matrix
            adj_matrix = self._generate_adj_matrix()
            
            sample = {
                'node_features': node_features,
                'access_history': access_history,
                'energies': energies,
                'temperature': temperature,
                'degeneracies': degeneracies,
                'access_frequency': access_frequency,
                'time_series': time_series,
                'adj_matrix': adj_matrix
            }
            dataset.append(sample)
        
        return dataset

    def _generate_access_history(self) -> np.ndarray:
        """Generate synthetic access history with temporal patterns."""
        history = np.zeros((self.config.sequence_length, self.config.n_nodes))
        
        # Add temporal patterns
        for t in range(self.config.sequence_length):
            base_prob = np.random.dirichlet(np.ones(self.config.n_nodes) * 0.1)
            time_factor = np.sin(2 * np.pi * t / self.config.sequence_length)
            pattern = np.abs(np.sin(np.arange(self.config.n_nodes) + time_factor))
            combined_prob = 0.7 * base_prob + 0.3 * pattern
            combined_prob /= combined_prob.sum()
            history[t] = np.random.multinomial(1, combined_prob)
        
        return history

    def _compute_boltzmann_distribution(self, energies: np.ndarray, temperature: float) -> np.ndarray:
        """Compute access frequencies using Boltzmann distribution."""
        beta = 1.0 / (temperature + 1e-10)
        probabilities = np.exp(-beta * energies)
        return probabilities / (np.sum(probabilities) + 1e-10)

    def _generate_time_series(self) -> np.ndarray:
        """Generate synthetic time series data."""
        time_series = np.random.normal(
            size=(self.config.sequence_length, 1)
        ).astype(np.float32)
        return time_series

    def _generate_adj_matrix(self) -> np.ndarray:
        """Generate random adjacency matrix for graph structure."""
        n = self.config.n_nodes
        adj = np.random.rand(n, n) < 0.1  # 10% connection probability
        np.fill_diagonal(adj, 0)  # No self-loops
        adj = np.maximum(adj, adj.T)  # Undirected graph
        return adj.astype(np.float32)

class StorageDataset(Dataset):
    """PyTorch Dataset wrapper for storage system data."""
    
    def __init__(self, data: List[Dict[str, np.ndarray]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        return {
            'features': torch.tensor(sample['node_features'], dtype=torch.float32),
            'access_history': torch.tensor(sample['access_history'], dtype=torch.float32),
            'energies': torch.tensor(sample['energies'], dtype=torch.float32),
            'temperature': torch.tensor(sample['temperature'], dtype=torch.float32),
            'degeneracies': torch.tensor(sample['degeneracies'], dtype=torch.float32),
            'labels': torch.tensor(sample['access_frequency'], dtype=torch.float32),
            'time_series': torch.tensor(sample['time_series'], dtype=torch.float32),
            'adj_matrix': torch.tensor(sample['adj_matrix'], dtype=torch.float32)
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        'features': torch.stack([item['features'] for item in batch]),
        'access_history': torch.stack([item['access_history'] for item in batch]),
        'energies': torch.stack([item['energies'] for item in batch]),
        'temperature': torch.stack([item['temperature'] for item in batch]),
        'degeneracies': torch.stack([item['degeneracies'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'time_series': torch.stack([item['time_series'] for item in batch]),
        'adj_matrix': torch.stack([item['adj_matrix'] for item in batch])
    }