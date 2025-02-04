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
    """Generates synthetic data for hospital medical imaging storage system simulation."""
    
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
        """Generate synthetic access history with hospital-specific temporal patterns."""
        history = np.zeros((self.config.sequence_length, self.config.n_nodes))
        
        # Add random schedule variations
        shift_variations = np.clip(np.random.normal(0, 1, self.config.sequence_length), -3, 3)
        workload_variations = np.clip(np.random.gamma(2, 2, self.config.sequence_length), 0.1, 10.0)
        
        for t in range(self.config.sequence_length):
            hour = t % 24
            day = (t // 24) % 7
            
            # Base probability with more variation but ensure positive values
            base_prob = np.abs(np.random.dirichlet(np.ones(self.config.n_nodes) * 0.05))
            
            # Time-of-day factor with bounded noise
            shifted_hour = (hour + shift_variations[t]) % 24
            if 8 <= shifted_hour <= 17:
                time_factor = np.clip(np.random.normal(1.5, 0.3), 0.5, 3.0)
            elif 22 <= shifted_hour or shifted_hour <= 5:
                time_factor = np.clip(np.random.normal(0.3, 0.15), 0.1, 1.0)
            else:
                time_factor = np.clip(np.random.normal(1.0, 0.2), 0.3, 2.0)
            
            # Day-of-week factor with bounded variations
            if day < 5:  # Weekdays
                day_factor = np.clip(np.random.normal(1.0, 0.2), 0.5, 2.0)
                if np.random.random() < 0.1:
                    day_factor *= np.clip(np.random.uniform(1.5, 2.5), 1.5, 2.5)
            else:  # Weekends
                day_factor = np.clip(np.random.normal(0.6, 0.15), 0.3, 1.5)
                if np.random.random() < 0.15:
                    day_factor *= np.clip(np.random.uniform(1.8, 3.0), 1.8, 3.0)
            
            # Recency factor with bounded decay
            decay_rate = np.clip(np.random.normal(0.1, 0.02), 0.05, 0.2)
            recency = np.exp(-np.arange(self.config.n_nodes) * decay_rate)
            
            # Seasonal factor
            seasonal_factor = np.clip(1.0 + 0.2 * np.sin(2 * np.pi * t / (self.config.sequence_length * 4)), 0.5, 1.5)
            
            # Combine all factors safely
            combined_prob = base_prob * time_factor * day_factor * recency * seasonal_factor * workload_variations[t]
            
            # Handle urgent events
            if np.random.random() < 0.08:
                n_urgent = np.random.randint(1, 4)
                for _ in range(n_urgent):
                    urgent_node = np.random.randint(0, self.config.n_nodes)
                    urgency_factor = np.clip(np.random.gamma(3, 2), 1.0, 10.0)
                    combined_prob[urgent_node] *= urgency_factor
            
            # Handle correlations
            for i in range(self.config.n_nodes):
                if np.random.random() < 0.15:
                    neighbors = np.random.choice([-1, 1]) + i
                    if 0 <= neighbors < self.config.n_nodes:
                        combined_prob[neighbors] = combined_prob[i] * np.clip(np.random.normal(0.8, 0.2), 0.4, 1.2)
            
            # Add bounded noise
            noise = np.clip(np.random.normal(1.0, 0.2, size=self.config.n_nodes), 0.5, 2.0)
            outlier_mask = np.random.random(self.config.n_nodes) < 0.02
            noise[outlier_mask] = np.clip(np.random.normal(1.0, 1.0, size=outlier_mask.sum()), 0.1, 5.0)
            
            combined_prob *= noise
            
            # Ensure valid probabilities
            combined_prob = np.maximum(1e-10, combined_prob)  # Avoid zeros
            combined_prob /= combined_prob.sum()  # Normalize
            
            # Generate events
            n_events = max(1, np.random.poisson(1.2))  # Ensure at least 1 event
            history[t] = np.random.multinomial(n_events, combined_prob)
        
        return history

    def _compute_boltzmann_distribution(self, energies: np.ndarray, temperature: float) -> np.ndarray:
        """Compute access frequencies using Boltzmann distribution."""
        beta = 1.0 / (temperature + 1e-10)
        probabilities = np.exp(-beta * energies)
        return probabilities / (np.sum(probabilities) + 1e-10)

    def _generate_time_series(self) -> np.ndarray:
        """Generate synthetic time series data with more realistic variations."""
        time_series = np.zeros((self.config.sequence_length, 1))
        
        # Generate random daily pattern variations
        pattern_noise = np.random.normal(0, 0.3, 24)  # Daily pattern variation
        workload_trend = np.random.normal(0, 0.1)  # Overall workload trend
        
        for t in range(self.config.sequence_length):
            hour = t % 24
            day = (t // 24) % 7
            
            # Base load with trend
            base_load = 1.0 + (t / self.config.sequence_length) * workload_trend
            
            # Daily pattern with variations
            daily_pattern = np.sin(2 * np.pi * (hour - 8) / 24) + 1 + pattern_noise[hour]
            
            # Weekly pattern with random variations
            if day < 5:
                weekly_pattern = np.random.normal(1.0, 0.15)
            else:
                weekly_pattern = np.random.normal(0.7, 0.2)
            
            # Add random events
            event_factor = 1.0
            if np.random.random() < 0.1:  # 10% chance of special event
                event_factor = np.random.gamma(2, 1)
            
            # Combine patterns with multiple noise sources
            value = base_load * daily_pattern * weekly_pattern * event_factor
            
            # Add different types of noise
            short_term_noise = np.random.normal(0, 0.1)
            spike_noise = np.random.exponential(0.1) if np.random.random() < 0.05 else 0
            
            time_series[t] = max(0, value + short_term_noise + spike_noise)
        
        return time_series.astype(np.float32)

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