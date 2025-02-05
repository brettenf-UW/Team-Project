#dataset.py
import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from typing import List, Dict, Any
from dataclasses import dataclass
from scipy.stats import zipf, pareto

@dataclass
class DataConfig:
    n_samples: int
    sequence_length: int
    n_nodes: int
    node_features: int
    train_size: float
    num_workers: int
    anomaly_rate: float = 0.05
    lifecycle_states: int = 4  # hot, warm, cool, cold

def normalize_probs(probs: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Safely normalize probability distribution."""
    # Ensure non-negative
    probs = np.maximum(probs, epsilon)
    # Handle zero sum
    total = probs.sum()
    if total <= epsilon:
        # If sum is too small, return uniform distribution
        return np.ones_like(probs) / len(probs)
    return probs / total

class EnhancedStorageDataGenerator:
    """Generates synthetic data for hospital medical imaging storage system simulation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        # Initialize lifecycles
        self.lifecycle_transitions = self._initialize_lifecycle_matrix()
        # Initialize feature correlations
        self.feature_correlations = self._initialize_feature_correlations()

    def _initialize_lifecycle_matrix(self) -> np.ndarray:
        """Initialize Markov transition matrix for data lifecycles."""
        # Transition probabilities between states (hot -> warm -> cool -> cold)
        transitions = np.array([
            [0.80, 0.15, 0.04, 0.01],  # hot state transitions
            [0.10, 0.70, 0.15, 0.05],  # warm state transitions
            [0.05, 0.10, 0.75, 0.10],  # cool state transitions
            [0.01, 0.04, 0.15, 0.80],  # cold state transitions
        ])
        return transitions

    def _initialize_feature_correlations(self) -> np.ndarray:
        """Initialize correlation matrix for features with guaranteed positive definiteness."""
        n_features = self.config.node_features
        
        # Start with a random matrix
        A = np.random.randn(n_features, n_features)
        # Make it symmetric
        A = (A + A.T) / 2
        
        # Add diagonal dominance to ensure positive definiteness
        A = A + (n_features * np.eye(n_features))
        
        # Convert to correlation matrix
        D = np.sqrt(np.diag(1.0 / np.diag(A)))
        corr = D @ A @ D
        
        # Add meaningful patterns while preserving positive definiteness
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if (i + j) % 3 == 0:
                    factor = 0.3  # Strong positive correlation
                elif (i + j) % 2 == 0:
                    factor = -0.2  # Moderate negative correlation
                else:
                    factor = 0.1  # Weak positive correlation
                    
                # Apply correlation while maintaining symmetry
                corr[i, j] = factor
                corr[j, i] = factor
        
        # Ensure diagonal is 1
        np.fill_diagonal(corr, 1.0)
        
        # Make it positive definite by adding a small constant to eigenvalues
        eigenvals, eigenvecs = np.linalg.eigh(corr)
        min_eig = eigenvals.min()
        if (min_eig < 0):
            corr = corr + (-min_eig + 1e-6) * np.eye(n_features)
        
        # Final normalization to ensure valid correlations
        D = np.sqrt(np.diag(1.0 / np.diag(corr)))
        corr = D @ corr @ D
        
        return corr

    def generate_dataset(self) -> List[Dict[str, np.ndarray]]:
        """
        Generate synthetic dataset with thermodynamic properties.
        
        Returns:
            List[Dict[str, np.ndarray]]: List of samples with features and labels
        """
        dataset = []
        for _ in range(self.config.n_samples):
            # Generate correlated features
            node_features = self._generate_correlated_features()
            
            # Generate access history with heavy-tailed distributions
            access_history = self._generate_access_history()
            
            # Add lifecycle states
            lifecycle_states = self._generate_lifecycle_states()
            
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
                'adj_matrix': adj_matrix,
                'lifecycle_states': lifecycle_states
            }
            dataset.append(sample)
        
        return dataset

    def _generate_correlated_features(self) -> np.ndarray:
        """Generate correlated feature vectors using safe decomposition."""
        try:
            # Try Cholesky decomposition first
            L = np.linalg.cholesky(self.feature_correlations)
            method = "cholesky"
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition if Cholesky fails
            eigenvals, eigenvecs = np.linalg.eigh(self.feature_correlations)
            L = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))
            method = "eigen"
        
        # Generate uncorrelated samples
        uncorrelated = np.random.normal(0, 1, (self.config.n_nodes, self.config.node_features))
        
        # Apply correlation structure
        correlated = uncorrelated @ L.T
        
        # Log which method was used (for debugging)
        logging.debug(f"Used {method} decomposition for feature correlation")
        
        return correlated

    def _generate_lifecycle_states(self) -> np.ndarray:
        """Generate lifecycle states for each node."""
        states = np.zeros((self.config.sequence_length, self.config.n_nodes), dtype=int)
        # Initialize states (mostly hot and warm initially)
        current_states = np.random.choice(
            4, 
            size=self.config.n_nodes, 
            p=[0.4, 0.3, 0.2, 0.1]
        )
        
        states[0] = current_states
        # Evolve states using Markov transitions
        for t in range(1, self.config.sequence_length):
            for n in range(self.config.n_nodes):
                current_states[n] = np.random.choice(
                    4,
                    p=self.lifecycle_transitions[current_states[n]]
                )
            states[t] = current_states
        return states

    def _generate_access_history(self) -> np.ndarray:
        """Generate synthetic access history with realistic hospital temporal patterns."""
        history = np.zeros((self.config.sequence_length, self.config.n_nodes))
        
        # Initialize patterns and schedules
        department_schedules = {
            'emergency': {'active': 24, 'peak_hours': [8, 16, 23], 'variation': 0.4},
            'radiology': {'active': [8, 20], 'peak_hours': [10, 14], 'variation': 0.3},
            'outpatient': {'active': [8, 17], 'peak_hours': [9, 15], 'variation': 0.2},
            'inpatient': {'active': 24, 'peak_hours': [6, 14, 22], 'variation': 0.25}
        }
        
        node_departments = np.random.choice(list(department_schedules.keys()), self.config.n_nodes)
        special_events = self._generate_special_events_calendar()
        shift_changes = [7, 15, 23]
        
        # Generate variations
        shift_variations = np.clip(np.random.normal(0, 1, self.config.sequence_length), -3, 3)
        workload_variations = np.clip(np.random.gamma(2, 2, self.config.sequence_length), 0.1, 10.0)
        
        # Base popularity (Zipf distribution)
        popularity = zipf.rvs(1.5, size=self.config.n_nodes)
        popularity = normalize_probs(popularity)
        
        for t in range(self.config.sequence_length):
            hour = t % 24
            day = (t // 24) % 7
            week = (t // (24 * 7))
            
            # Generate base probabilities
            base_prob = np.zeros(self.config.n_nodes)
            for i, dept in enumerate(node_departments):
                schedule = department_schedules[dept]
                if isinstance(schedule['active'], int) and schedule['active'] == 24:
                    is_active = True
                else:
                    active_start, active_end = schedule['active']
                    is_active = active_start <= hour < active_end
                
                if is_active:
                    peak_distances = [min((hour - p) % 24, (p - hour) % 24) for p in schedule['peak_hours']]
                    min_distance = min(peak_distances)
                    peak_factor = np.exp(-min_distance * schedule['variation'])
                    base_prob[i] = peak_factor
            
            # Normalize after each major modification
            base_prob = normalize_probs(base_prob)
            
            # Apply time-based effects
            time_factor = 1.0
            
            # Shift changes
            for shift_hour in shift_changes:
                time_since_shift = min((hour - shift_hour) % 24, 1)
                if time_since_shift < 1:
                    time_factor *= (0.5 + 0.5 * time_since_shift)
            
            # Lunch and rounds
            if (hour == 12 or hour == 13) and day < 5:
                time_factor *= 0.7
            if 8 <= hour <= 10 and day < 5:
                time_factor *= 1.5
            
            base_prob *= time_factor
            base_prob = normalize_probs(base_prob)
            
            # Apply special events
            if t in special_events:
                event_type = special_events[t]
                if event_type == 'maintenance':
                    affected_nodes = np.random.choice(
                        self.config.n_nodes,
                        size=self.config.n_nodes//4,
                        replace=False
                    )
                    base_prob[affected_nodes] *= 0.1
                elif event_type == 'meeting':
                    base_prob *= 0.8
                base_prob = normalize_probs(base_prob)
            
            # Apply weekly and monthly patterns
            base_prob *= (1.0 - 0.3 * (day >= 5))  # Weekend reduction
            base_prob *= (1.0 + 0.1 * np.sin(2 * np.pi * (week % 4) / 4))  # Monthly cycle
            base_prob = normalize_probs(base_prob)
            
            # Combine with popularity and workload
            combined_prob = base_prob * popularity * workload_variations[t]
            combined_prob = normalize_probs(combined_prob)
            
            # Add noise and bursts
            noise = np.clip(np.random.normal(1.0, 0.2, size=self.config.n_nodes), 0.5, 2.0)
            combined_prob *= noise
            combined_prob = normalize_probs(combined_prob)
            
            # Add bursts
            if np.random.random() < 0.1:
                burst_size = np.clip(pareto.rvs(2, size=1)[0], 1.0, 10.0)
                burst_node = np.random.choice(self.config.n_nodes)
                combined_prob[burst_node] *= burst_size
                combined_prob = normalize_probs(combined_prob)
            
            # Add anomalies
            if np.random.random() < self.config.anomaly_rate:
                combined_prob = normalize_probs(self._inject_anomaly(combined_prob, t))
            
            # Final normalization and validation
            combined_prob = normalize_probs(combined_prob)
            assert np.isclose(combined_prob.sum(), 1.0), f"Probabilities sum to {combined_prob.sum()}"
            assert np.all(combined_prob >= 0), "Negative probabilities detected"
            
            # Generate events
            n_events = max(1, np.random.poisson(1.2))
            history[t] = np.random.multinomial(n_events, combined_prob)
        
        return history

    def _generate_special_events_calendar(self) -> Dict[int, str]:
        """Generate calendar of special events."""
        events = {}
        sequence_days = self.config.sequence_length // 24
        
        # Schedule maintenance events (usually early morning hours)
        for day in range(sequence_days):
            if np.random.random() < 0.1:  # 10% chance of maintenance per day
                maintenance_hour = np.random.randint(2, 6)  # Between 2 AM and 6 AM
                event_time = day * 24 + maintenance_hour
                events[event_time] = 'maintenance'
        
        # Schedule regular meetings (weekdays during working hours)
        for day in range(sequence_days):
            if day % 7 < 5:  # Weekdays only
                if np.random.random() < 0.3:  # 30% chance of meetings on weekdays
                    meeting_hour = np.random.randint(9, 16)  # Between 9 AM and 4 PM
                    event_time = day * 24 + meeting_hour
                    events[event_time] = 'meeting'
        
        return events

    def _compute_boltzmann_distribution(self, energies: np.ndarray, temperature: float) -> np.ndarray:
        """Compute access frequencies using Boltzmann distribution."""
        beta = 1.0 / (temperature + 1e-10)
        probabilities = np.exp(-beta * energies)
        return probabilities / (np.sum(probabilities) + 1e-10)

    def _generate_time_series(self) -> np.ndarray:
        """Generate synthetic time series data with realistic variations."""
        time_series = np.zeros((self.config.sequence_length, 1))
        
        # Generate more realistic pattern variations
        hourly_pattern = np.array([
            0.2, 0.15, 0.1, 0.1, 0.15, 0.3,  # 12am-6am
            0.6, 1.0, 1.4, 1.3, 1.2, 1.1,    # 6am-12pm
            1.0, 1.2, 1.4, 1.3, 1.2, 1.0,    # 12pm-6pm
            0.8, 0.6, 0.4, 0.3, 0.25, 0.2    # 6pm-12am
        ])
        
        # Add random variations to the basic pattern
        pattern_noise = np.random.normal(0, 0.1, 24)
        hourly_pattern += pattern_noise
        
        # Rest of the existing time series generation code
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

    def _inject_anomaly(self, prob: np.ndarray, t: int) -> np.ndarray:
        """Inject different types of anomalies."""
        anomaly_type = np.random.choice([
            'sudden_spike',
            'gradual_drift',
            'pattern_break',
            'system_wide'
        ])
        
        if anomaly_type == 'sudden_spike':
            # Sudden spike in specific nodes
            affected_nodes = np.random.choice(
                self.config.n_nodes,
                size=max(1, self.config.n_nodes // 10),
                replace=False
            )
            prob[affected_nodes] *= np.random.uniform(5, 10)
            
        elif anomaly_type == 'gradual_drift':
            # Gradual increase/decrease over time
            drift_factor = 1 + 0.1 * (t / self.config.sequence_length)
            prob *= drift_factor
            
        elif anomaly_type == 'pattern_break':
            # Complete break from normal patterns
            prob = np.random.dirichlet(np.ones(self.config.n_nodes) * 0.1)
            
        elif anomaly_type == 'system_wide':
            # System-wide anomaly
            prob *= np.random.uniform(0.1, 0.5)
        
        return prob

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