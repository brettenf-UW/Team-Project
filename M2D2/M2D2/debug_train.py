# debug_train.py
import torch
from debug_config import debug_config
from train import train_epoch, validate
import pdb

from model.dataset import DataConfig, EnhancedStorageDataGenerator, StorageDataset
from model.model import ModelConfig, StatMechThermodynamicModule
from model.loss import LossConfig, ThermodynamicLoss
from torch.utils.data import DataLoader
from model.dataset import collate_fn

# Early Stopping Class
class EarlyStopping:
    """
    Early stopping to stop training when the validation loss does not improve after a given patience.
    """
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after the last time the validation loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call this method every epoch to check if training should stop.
        Args:
            val_loss (float): The validation loss from the current epoch.
        """
        if self.best_loss is None:
            # First epoch, set the best loss
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # Validation loss did not improve
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Validation loss improved
            if self.verbose and val_loss < self.best_loss:
                print(f"Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0

def debug_training():
    try:
        # Initialize with debug config
        #data_config = DataConfig(**debug_config['data'])
        data_config = DataConfig(
            n_samples=debug_config['data']['n_samples'],
            sequence_length=debug_config['data']['sequence_length'],
            n_nodes=debug_config['data']['n_nodes'],
            node_features=debug_config['data']['node_features'],
            train_size=debug_config['data']['train_size'],  # Ensure this is passed
            num_workers=debug_config['data']['num_workers']  # Ensure this is passed
        )
        model_config = ModelConfig(**debug_config['model'])
        loss_config = LossConfig(**debug_config['loss'])
        
        # Setup components
        generator = EnhancedStorageDataGenerator(data_config)
        dataset = generator.generate_dataset()
        train_dataset = StorageDataset(dataset)
        val_dataset = StorageDataset(dataset)  # Use the same dataset for debugging

        # Modify DataLoader creation
        train_loader = DataLoader(
            train_dataset,
            batch_size=debug_config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn  # <-- Critical fix
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=debug_config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Initialize model, loss, and optimizer
        model = StatMechThermodynamicModule(model_config)
        criterion = ThermodynamicLoss(loss_config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=debug_config['training']['learning_rate']
        )
        
        # Single batch test
        batch = next(iter(train_loader))
        print("Batch keys:", batch.keys())  # Debug: Check batch keys
        outputs = model(
            batch['features'],
            batch['access_history'],
            batch['energies'],
            batch['temperature'],
            batch['degeneracies']
        )
        print("Single batch successful!")
        print("Output shape:", outputs.shape)
        
        # Test early stopping with a simple training loop
        early_stopping = EarlyStopping(patience=3, min_delta=0.001, verbose=True)
        for epoch in range(10):  # Run for 10 epochs for testing
            print(f"\nEpoch {epoch + 1}")
            
            # Train for one epoch
            train_loss, loss_components = train_epoch(
                model, train_loader, criterion, optimizer, torch.device('cpu'), epoch, None
            )
            print(f"Train Loss: {train_loss:.4f}")
            print("Loss Components:", loss_components)  # Debug: Check loss components
            
            # Validate
            val_loss = validate(model, val_loader, criterion, torch.device('cpu'))
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Check early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        pdb.set_trace()

if __name__ == "__main__":
    debug_training()