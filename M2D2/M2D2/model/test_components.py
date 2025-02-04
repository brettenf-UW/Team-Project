# test_components.py
import torch
from model import (
    StatMechThermodynamicModule,
    SimpleGraphModule,
    TimeSeriesModule,
    MultiModalEnsemble,
    ModelConfig
)
from dataset import EnhancedStorageDataGenerator, DataConfig

def test_stat_mech_thermodynamic_module():
    """Test the StatMechThermodynamicModule."""
    print("\nTesting StatMechThermodynamicModule...")
    
    # Create config
    config = ModelConfig(node_features=10, sequence_length=100)
    
    # Initialize model
    model = StatMechThermodynamicModule(config)
    
    # Create dummy input
    batch_size = 2
    n_nodes = 20
    features = torch.randn(batch_size, n_nodes, config.node_features)
    access_history = torch.randn(batch_size, config.sequence_length, n_nodes)
    energies = torch.randn(batch_size, n_nodes)
    temperature = torch.randn(batch_size)
    degeneracies = torch.randn(batch_size, n_nodes)
    
    # Forward pass
    outputs = model(features, access_history, energies, temperature, degeneracies)
    
    # Check output shape
    assert outputs.shape == (batch_size, n_nodes), f"Unexpected output shape: {outputs.shape}"
    print("StatMechThermodynamicModule test passed! Output shape:", outputs.shape)

def test_simple_graph_module():
    """Test the SimpleGraphModule."""
    print("\nTesting SimpleGraphModule...")
    
    # Initialize model
    node_features = 10
    model = SimpleGraphModule(node_features)
    
    # Create dummy input
    batch_size = 2
    n_nodes = 20
    x = torch.randn(batch_size, n_nodes, node_features)
    adj_matrix = torch.randn(n_nodes, n_nodes)
    
    # Forward pass
    outputs = model(x, adj_matrix)
    
    # Check output shape
    assert outputs.shape == (batch_size, 32), f"Unexpected output shape: {outputs.shape}"
    print("SimpleGraphModule test passed! Output shape:", outputs.shape)

def test_time_series_module():
    """Test the TimeSeriesModule."""
    print("\nTesting TimeSeriesModule...")
    
    # Initialize model
    input_dim = 100
    model = TimeSeriesModule(input_dim)
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    outputs = model(x)
    
    # Check output shape
    assert outputs.shape == (batch_size, 32), f"Unexpected output shape: {outputs.shape}"
    print("TimeSeriesModule test passed! Output shape:", outputs.shape)

def test_multi_modal_ensemble():
    """Test the MultiModalEnsemble."""
    print("\nTesting MultiModalEnsemble...")
    
    # Initialize model
    feature_dim = 100
    sequence_length = 100
    node_features = 10
    model = MultiModalEnsemble(feature_dim, sequence_length, node_features)
    
    # Create dummy input
    batch_size = 2
    n_nodes = 20
    thermo_features = torch.randn(batch_size, n_nodes, node_features)
    access_history = torch.randn(batch_size, sequence_length, n_nodes)
    energies = torch.randn(batch_size, n_nodes)
    temperature = torch.randn(batch_size)
    degeneracies = torch.randn(batch_size, n_nodes)
    time_features = torch.randn(batch_size, feature_dim)
    graph_features = torch.randn(batch_size, n_nodes, node_features)
    adj_matrix = torch.randn(n_nodes, n_nodes)
    
    # Forward pass
    outputs = model(thermo_features, access_history, energies, temperature, degeneracies, time_features, graph_features, adj_matrix)
    
    # Check output shape
    assert outputs.shape == (batch_size, 3), f"Unexpected output shape: {outputs.shape}"
    print("MultiModalEnsemble test passed! Output shape:", outputs.shape)
def test_data_generator():
    """Test the EnhancedStorageDataGenerator."""
    print("\nTesting EnhancedStorageDataGenerator...")
    
    # Create config
    config = DataConfig(
        n_samples=10,
        sequence_length=100,
        n_nodes=20,
        node_features=10
    )
    
    # Initialize generator
    generator = EnhancedStorageDataGenerator(config)
    
    # Generate dataset
    dataset = generator.generate_dataset()
    
    # Check dataset size
    assert len(dataset) == config.n_samples, f"Unexpected dataset size: {len(dataset)}"
    print("EnhancedStorageDataGenerator test passed! Dataset size:", len(dataset))

if __name__ == "__main__":
    # Run all tests
    test_stat_mech_thermodynamic_module()
    test_simple_graph_module()
    test_time_series_module()
    test_multi_modal_ensemble()
    test_data_generator()
    
    print("\nAll tests passed!")