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
    total_loss = 0
    loss_components = {
        'mse_loss': 0,
        'energy_conservation_loss': 0,
        'entropy_loss': 0,
        'free_energy_loss': 0,
        'temporal_loss': 0
    }
    
    total_mse = 0
    total_r2 = 0
    
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
        total_loss += loss_dict['total_loss'].item()
        for k in loss_components.keys():
            loss_components[k] += loss_dict[k].item()
        
        # Calculate additional metrics
        metrics = calculate_metrics(outputs, batch['labels'])
        total_mse += metrics['mse']
        total_r2 += metrics['r2']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss_dict['total_loss'].item(),
            'mse': metrics['mse'],
            'corr': f"{metrics['correlation']:.3f}",
            'mape': f"{metrics['mape']:.1f}%"
        })
    
    # Average losses
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    
    # Average metrics
    avg_mse = total_mse / len(train_loader)
    avg_r2 = total_r2 / len(train_loader)
    
    # Log to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    for k, v in avg_components.items():
        writer.add_scalar(f'LossComponents/{k}', v, epoch)
    
    # Log metrics
    writer.add_scalar('Metrics/train_mse', avg_mse, epoch)
    writer.add_scalar('Metrics/train_r2', avg_r2, epoch)
    writer.add_scalar('Metrics/train_correlation', metrics['correlation'], epoch)
    writer.add_scalar('Metrics/train_mape', metrics['mape'], epoch)
    logging.info(f'Epoch {epoch}: MSE = {avg_mse:.4f}, R² = {avg_r2:.4f}, '
                f'Correlation = {metrics["correlation"]:.4f}, MAPE = {metrics["mape"]:.1f}%')
    
    return avg_loss, avg_components, {'mse': avg_mse, 'r2': avg_r2}

def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: ThermodynamicLoss,
    device: torch.device
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_mse = 0
    total_r2 = 0
    
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
            total_loss += loss_dict['total_loss'].item()
            
            metrics = calculate_metrics(outputs, batch['labels'])
            total_mse += metrics['mse']
            total_r2 += metrics['r2']
    
    avg_mse = total_mse / len(val_loader)
    avg_r2 = total_r2 / len(val_loader)
    return total_loss / len(val_loader), {'mse': avg_mse, 'r2': avg_r2}

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
        train_loss, loss_components, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        logging.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, MSE = {train_metrics["mse"]:.4f}, R² = {train_metrics["r2"]:.4f}')
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        logging.info(f'Epoch {epoch}: Val Loss = {val_loss:.4f}, MSE = {val_metrics["mse"]:.4f}, R² = {val_metrics["r2"]:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/val_mse', val_metrics['mse'], epoch)
        writer.add_scalar('Metrics/val_r2', val_metrics['r2'], epoch)
        
        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered!")
            break
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            logging.info(f'Saved new best model with validation loss: {val_loss:.4f}')

if __name__ == '__main__':
    main()