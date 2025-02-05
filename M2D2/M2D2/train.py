import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np

from model.model import StatMechThermodynamicModule, ModelConfig
from model.dataset import EnhancedStorageDataGenerator, StorageDataset, DataConfig, collate_fn
from model.loss import ThermodynamicLoss, LossConfig

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

def setup_logging(save_dir: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Calculate MSE, R², correlation, and MAPE."""
    # Convert to numpy for calculations
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate existing metrics
    mse = torch.mean((outputs - targets) ** 2).item()
    r2 = r2_score(targets_np, outputs_np)
    
    # Add correlation coefficient
    corr, _ = pearsonr(targets_np.flatten(), outputs_np.flatten())
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((targets_np - outputs_np) / (targets_np + 1e-8))) * 100
    
    return {
        'mse': mse, 
        'r2': r2,
        'correlation': corr,
        'mape': mape
    }

def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: ThermodynamicLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> dict:
    """Train for one epoch."""
    model.train()
    metrics_tracker = {
        # Core performance metrics
        'loss/total': 0,  # Overall training loss
        'performance/mse': 0,  # Mean squared error
        'performance/r2': 0,  # R² score for prediction accuracy
        'performance/mape': 0,  # Mean Absolute Percentage Error
        
        # Important components of the loss
        'loss_components/energy': 0,  # Energy conservation violation
        'loss_components/entropy': 0,  # Entropy maximization term
        'loss_components/prediction': 0,  # Direct prediction error
        
        # Data characteristics
        'data/access_frequency': 0,  # Average access frequency
        'data/temperature': 0,  # System temperature
    }
    
    n_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            batch['features'],
            batch['access_history'],
            batch['energies'],
            batch['temperature'],
            batch['degeneracies']
        )
        
        # Compute loss
        loss_dict = criterion(
            outputs,
            batch['labels'],
            batch['energies'],
            batch['temperature'],
            batch['degeneracies'],
            batch['access_history']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        metrics = calculate_metrics(outputs, batch['labels'])
        for key, value in metrics.items():
            if key in metrics_tracker:
                metrics_tracker[key] += value
        
        # Track important data characteristics
        metrics_tracker['data/access_frequency'] += torch.mean(batch['access_history']).item()
        metrics_tracker['data/temperature'] += torch.mean(batch['temperature']).item()
        
        # Update progress bar with key metrics
        pbar.set_postfix({
            'loss': metrics_tracker['loss/total'] / (batch_idx + 1),
            'mse': metrics_tracker['performance/mse'] / (batch_idx + 1),
            'r2': metrics_tracker['performance/r2'] / (batch_idx + 1)
        })
    
    # Average metrics
    metrics_avg = {k: v / n_batches for k, v in metrics_tracker.items()}
    
    # Log to TensorBoard without description parameter
    for key, value in metrics_avg.items():
        writer.add_scalar(f'Train/{key}', value, epoch)
        # Add description as text tag instead (once per run)
        if epoch == 0:
            writer.add_text(
                f'Descriptions/{key}',
                get_metric_description(key),
                epoch
            )
    
    return metrics_avg

def get_metric_description(metric_key: str) -> str:
    """Return description for each metric for TensorBoard tooltips."""
    descriptions = {
        'loss/total': 'Overall training loss combining all components',
        'performance/mse': 'Mean Squared Error - Lower is better',
        'performance/r2': 'R² Score (0-1) - Higher is better',
        'performance/mape': 'Mean Absolute Percentage Error - Lower is better',
        'loss_components/energy': 'Energy conservation violation term',
        'loss_components/entropy': 'Entropy maximization component',
        'loss_components/prediction': 'Direct prediction error component',
        'data/access_frequency': 'Average frequency of data access',
        'data/temperature': 'System temperature affecting access patterns'
    }
    return descriptions.get(metric_key, "No description available")

def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: ThermodynamicLoss,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int
) -> float:
    """Validate the model."""
    model.eval()
    metrics_tracker = {
        'loss/total': 0,
        'performance/mse': 0,
        'performance/r2': 0,
        'performance/mape': 0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                batch['features'],
                batch['access_history'],
                batch['energies'],
                batch['temperature'],
                batch['degeneracies']
            )
            loss_dict = criterion(
                outputs,
                batch['labels'],
                batch['energies'],
                batch['temperature'],
                batch['degeneracies'],
                batch['access_history']
            )
            metrics_tracker['loss/total'] += loss_dict['total_loss'].item()
            
            metrics = calculate_metrics(outputs, batch['labels'])
            for key, value in metrics.items():
                if key in metrics_tracker:
                    metrics_tracker[key] += value
    
    # Average metrics
    metrics_avg = {k: v / len(val_loader) for k, v in metrics_tracker.items()}
    
    # Log validation metrics without description parameter
    for key, value in metrics_avg.items():
        writer.add_scalar(f'Val/{key}', value, epoch)
        # Add description as text tag instead (once per run)
        if epoch == 0:
            writer.add_text(
                f'Descriptions/{key}',
                get_metric_description(key),
                epoch
            )
    
    return metrics_avg

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Setup save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('checkpoints') / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging and TensorBoard
    setup_logging(save_dir)
    writer = SummaryWriter(save_dir / 'tensorboard')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create datasets
    data_config_params = config['data'].copy()
    # Only remove pin_memory from data_config params
    dataloader_params = {
        'pin_memory': data_config_params.pop('pin_memory', False)
    }
    
    data_config = DataConfig(**data_config_params)
    generator = EnhancedStorageDataGenerator(data_config)
    dataset = generator.generate_dataset()

    # Split dataset
    train_size = int(data_config.train_size * len(dataset))  # Use train_size from DataConfig
    train_dataset = StorageDataset(dataset[:train_size])
    val_dataset = StorageDataset(dataset[train_size:])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        **dataloader_params,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        **dataloader_params,
        collate_fn=collate_fn
    )
    
    # Create model
    model_config = ModelConfig(**config['model'])
    model = StatMechThermodynamicModule(model_config).to(device)
    
    # Setup loss and optimizer
    loss_config = LossConfig(**config['loss'])
    criterion = ThermodynamicLoss(loss_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['optimization']['weight_decay']
    )
    
    # Training loop
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)
    best_val_loss = float('inf')
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        logging.info(f'Epoch {epoch}: Train Loss = {train_metrics["loss/total"]:.4f}, MSE = {train_metrics["performance/mse"]:.4f}, R² = {train_metrics["performance/r2"]:.4f}')
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, writer, epoch)
        logging.info(f'Epoch {epoch}: Val Loss = {val_metrics["loss/total"]:.4f}, MSE = {val_metrics["performance/mse"]:.4f}, R² = {val_metrics["performance/r2"]:.4f}')
        
        # Check early stopping
        early_stopping(val_metrics['loss/total'])
        if early_stopping.early_stop:
            logging.info("Early stopping triggered!")
            break
        
        # Save checkpoint
        if val_metrics['loss/total'] < best_val_loss:
            best_val_loss = val_metrics['loss/total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss/total'],
                'config': config
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            logging.info(f'Saved new best model with validation loss: {val_metrics["loss/total"]:.4f}')

if __name__ == '__main__':
    main()