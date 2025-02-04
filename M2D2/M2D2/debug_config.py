debug_config = {
    'data': {
        'n_samples': 1000,
        'sequence_length': 50,
        'n_nodes': 10,
        'node_features': 5,
        'train_size': 0.8,  # Ensure this is present
        'num_workers': 4    # Ensure this is present
    },
    'model': {
        'node_features': 5,
        'sequence_length': 50,
        'hidden_size': 128,
        'num_gru_layers': 2,
        'dropout': 0.2
    },
    'loss': {
        'alpha': 0.2,
        'beta': 0.1,
        'gamma': 0.1,
        'delta': 0.05
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001
    }
}