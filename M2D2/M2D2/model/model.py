# model.py

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    node_features: int
    sequence_length: int
    hidden_size: int = 128
    num_gru_layers: int = 2
    dropout: float = 0.2
    train_size: float = 0.8

class StatMechThermodynamicModule(nn.Module):
    """
    Neural network model incorporating statistical mechanics principles.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Feature processing
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.node_features, config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Access history processing
        self.history_encoder = nn.GRU(
            input_size=1,
            hidden_size=config.hidden_size // 2,
            num_layers=config.num_gru_layers,
            batch_first=True,
            dropout=config.dropout if config.num_gru_layers > 1 else 0
        )
        
        # Thermodynamic variables processing
        self.thermo_encoder = nn.Sequential(
            nn.Linear(3, config.hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size // 2)
        )
        
        # Attention mechanism for temporal dependencies
        self.attention = TemporalAttention(config.hidden_size // 2)
        
        # Final processing layers
        combined_size = config.hidden_size * 2  # features + history + thermo
        self.output_layers = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size // 2),
            nn.Linear(config.hidden_size // 2, 1)
        )

    def forward(self, features, access_history, energies, temperature, degeneracies):
        print(f"features shape: {features.shape}")  # Debug print
        print(f"access_history shape: {access_history.shape}")  # Debug print
        print(f"energies shape: {energies.shape}")  # Debug print
        print(f"temperature shape: {temperature.shape}")  # Debug print
        print(f"degeneracies shape: {degeneracies.shape}")  # Debug print

        batch_size, n_nodes, _ = features.shape
        
        # Ensure features tensor is contiguous
        features = features.contiguous()
        
        # Process node features
        features_flat = features.reshape(-1, features.size(-1))  # Use .reshape()
        features_encoded = self.feature_encoder(features_flat)
        features_encoded = features_encoded.reshape(batch_size, n_nodes, -1)  # Use .reshape()
        print(f"features_encoded shape: {features_encoded.shape}")  # Debug print
        
        # Process access history with attention
        history_encoded = self._process_history(access_history)
        print(f"history_encoded shape: {history_encoded.shape}")  # Debug print
        
        # Process thermodynamic variables
        thermo_encoded = self._process_thermo_vars(energies, temperature, degeneracies)
        print(f"thermo_encoded shape: {thermo_encoded.shape}")  # Debug print
        
        # Combine all encodings
        combined = torch.cat([
            features_encoded,  # [batch, nodes, hidden]
            history_encoded.squeeze(2),  # Remove the extra dimension: [batch, nodes, hidden]
            thermo_encoded  # [batch, nodes, hidden]
        ], dim=-1)
        print(f"combined shape: {combined.shape}")  # Debug print
        
        # Ensure combined tensor is contiguous
        combined = combined.contiguous()
        
        # Final processing
        combined_flat = combined.reshape(-1, combined.size(-1))  # Use .reshape()
        output = self.output_layers(combined_flat)
        output = output.reshape(batch_size, n_nodes)  # Use .reshape()
        print(f"output shape: {output.shape}")  # Debug print
        
        # Return raw logits instead of softmax
        return output  # Remove softmax here
    
    def _process_history(self, access_history: torch.Tensor) -> torch.Tensor:
        """Process access history with attention mechanism."""
        batch_size, seq_len, n_nodes = access_history.shape
        
        # Ensure access_history is contiguous
        access_history = access_history.contiguous()
        
        # Reshape history for GRU
        history_reshaped = access_history.transpose(1, 2)  # [batch, nodes, seq]
        history_reshaped = history_reshaped.reshape(-1, seq_len, 1)  # Use .reshape()
        
        # Process each node's history
        history_encoded, _ = self.history_encoder(history_reshaped)
        
        # Reshape back to [batch, nodes, seq_len, hidden_size]
        history_encoded = history_encoded.reshape(batch_size, n_nodes, seq_len, -1)
        
        # Apply attention
        history_encoded = self.attention(history_encoded)
        
        return history_encoded

    def _process_thermo_vars(
        self,
        energies: torch.Tensor,
        temperature: torch.Tensor,
        degeneracies: torch.Tensor
    ) -> torch.Tensor:
        """Process thermodynamic variables."""
        batch_size, n_nodes = energies.shape
        
        # Ensure tensors are contiguous
        energies = energies.contiguous()
        temperature = temperature.contiguous()
        degeneracies = degeneracies.contiguous()
        
        # Combine thermodynamic variables
        thermo_vars = torch.stack([
            energies,
            temperature.unsqueeze(1).expand(-1, n_nodes),
            degeneracies
        ], dim=-1)
        
        # Ensure thermo_vars is contiguous
        thermo_vars = thermo_vars.contiguous()
        
        # Reshape and encode
        thermo_flat = thermo_vars.reshape(-1, 3)  # Use .reshape()
        thermo_encoded = self.thermo_encoder(thermo_flat)
        
        return thermo_encoded.reshape(batch_size, n_nodes, -1)  # Use .reshape()

class TemporalAttention(nn.Module):
    """Attention mechanism for temporal dependencies."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism.
        
        Args:
            x: Input tensor [batch_size, n_nodes, seq_len, hidden_size]
            
        Returns:
            Attention-weighted tensor [batch_size, n_nodes, hidden_size]
        """
        # Calculate attention weights
        weights = self.attention(x)  # [batch, nodes, seq, 1]
        weights = torch.softmax(weights, dim=2)
        
        # Apply attention weights
        attended = torch.sum(weights * x, dim=2)  # [batch, nodes, hidden]
        return attended

class TimeSeriesModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim  # Store input_dim
        # Change LSTM to expect single feature per timestep
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 32)

    def forward(self, x):
        # Reshape input to (batch, sequence_length, 1)
        x = x.view(x.size(0), -1, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class SimpleGraphModule(nn.Module):
    def __init__(self, node_features, hidden_dim=64):
        super().__init__()
        self.node_transform = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)
        )
        self.edge_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, adj_matrix):
        # Ensure correct dimensions
        if len(x.shape) == 4:  # [batch, 1, nodes, features]
            x = x.squeeze(1)   # -> [batch, nodes, features]

        if len(adj_matrix.shape) == 3:  # [1, nodes, nodes]
            adj_matrix = adj_matrix.squeeze(0)  # -> [nodes, nodes]

        batch_size = x.size(0)

        # Transform node features
        node_embeddings = self.node_transform(x)  # [batch, nodes, hidden]

        # Create pairs of node embeddings
        node_pairs_1 = node_embeddings.unsqueeze(2)  # [batch, nodes, 1, hidden]
        node_pairs_2 = node_embeddings.unsqueeze(1)  # [batch, 1, nodes, hidden]

        # Expand to create all pairs
        node_pairs = torch.cat([
            node_pairs_1.expand(-1, -1, node_embeddings.size(1), -1),  # [batch, nodes, nodes, hidden]
            node_pairs_2.expand(-1, node_embeddings.size(1), -1, -1)   # [batch, nodes, nodes, hidden]
        ], dim=-1)

        # Calculate attention scores
        attention = self.edge_attention(node_pairs).squeeze(-1)  # [batch, nodes, nodes]

        # Apply adjacency matrix mask
        adj_matrix = adj_matrix.expand(batch_size, -1, -1)  # [batch, nodes, nodes]
        attention = attention * adj_matrix

        # Apply attention to node embeddings
        attended = torch.bmm(attention, node_embeddings)  # [batch, nodes, hidden]

        # Pool nodes
        graph_embedding = torch.mean(attended, dim=1)  # [batch, hidden]

        return graph_embedding
    
class MultiModalEnsemble(nn.Module):
    def __init__(self, feature_dim, sequence_length, node_features):
        super().__init__()
        self.thermo = StatMechThermodynamicModule(ModelConfig(node_features, sequence_length))
        self.time_series = TimeSeriesModule(feature_dim)
        self.graph = SimpleGraphModule(node_features)

        # Ensemble layer
        ensemble_input = 20 + 32 + 32  # Adjusted combined dimensions: thermo_out is [batch_size, 20]
        self.ensemble = nn.Sequential(
            nn.Linear(ensemble_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 outputs: transition prob, timing, resource
        )

    def forward(self, thermo_features, access_history, energies, temperature, degeneracies, time_features, graph_features, adj_matrix):
        # Process each modality
        thermo_out = self.thermo(thermo_features, access_history, energies, temperature, degeneracies)
        time_out = self.time_series(time_features)
        graph_out = self.graph(graph_features, adj_matrix)

        # Combine outputs
        combined = torch.cat([thermo_out, time_out, graph_out], dim=1)
        return self.ensemble(combined)