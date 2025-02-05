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
    def __init__(self, patience=5, min_delta=0.001, verbose=False, min_epochs=10):
        """
        Args:
            patience (int): How many epochs to wait after the last time the validation loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            min_epochs (int): Minimum number of epochs to train before considering early stopping.
                            Default: 10
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_epochs_reached = False

    def __call__(self, val_loss, epoch):
        """
        Call this method every epoch to check if training should stop.
        Args:
            val_loss (float): The validation loss from the current epoch.
            epoch (int): The current epoch number.
        """
        # Don't stop before minimum epochs
        if epoch < self.min_epochs:
            return
        
        self.min_epochs_reached = True
        
        if self.best_loss is None:
            # First epoch, set the best loss
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # Validation loss did not improve
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience and self.min_epochs_reached:
                self.early_stop = True
        else:
            # Validation loss improved
            if self.verbose and val_loss < self.best_loss:
                print(f"Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0

def setup_logging(save_dir: Path):
    """Setup logging configuration."""
    # Create debug directory
    debug_dir = save_dir.parent / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped debug file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_file = debug_dir / f'debug_{timestamp}.txt'
    
    # Setup logging with multiple handlers
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'train.log'),
            logging.FileHandler(debug_file),  # Additional debug file
            logging.StreamHandler()
        ]
    )
    
    # Log system info
    logging.debug("\n=== System Information ===")
    logging.debug(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logging.debug(f"PyTorch Version: {torch.__version__}")
    logging.debug(f"Debug file location: {debug_file}")
    logging.debug("="*30 + "\n")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Calculate raw MSE metrics."""
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Calculate raw MSE directly
    mse = float(np.mean((outputs_np - targets_np) ** 2))
    
    # Calculate r2 score on raw values
    target_var = np.var(targets_np)
    if target_var > 1e-10:
        r2 = r2_score(targets_np.flatten(), outputs_np.flatten())
    else:
        r2 = 0.0
    
    return {
        'mse': mse,
        'r2': r2
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
    
    # Add more detailed logging
    logging.debug("\n" + "="*50)
    logging.debug(f"Starting epoch {epoch}")
    logging.debug("="*50)
    
    metrics_tracker = {
        'loss': 0.0,  # Simplified metrics - just track what we need
        'mse': 0.0
    }
    
    n_batches = len(train_loader)
    
    # Add batch counter for proper averaging
    batch_counts = {k: 0 for k in metrics_tracker.keys()}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Add detailed batch logging
        if batch_idx % 10 == 0:  # Reduced logging frequency
            logging.info(f"\nBatch {batch_idx}/{len(train_loader)}")
            logging.info(f"Current MSE: {metrics_tracker['mse']/(batch_idx+1):.6f}")
        
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
        
        # Update loss components
        metrics_tracker['loss'] += loss_dict['total_loss'].item()
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate metrics
        metrics = calculate_metrics(outputs, batch['labels'])
        
        # Update metrics with proper accumulation
        metrics_tracker['mse'] += metrics['mse']
        
        # Update batch counts
        for k in batch_counts:
            batch_counts[k] += 1
        
        # Cleaner progress bar without scaling
        pbar.set_postfix({
            'MSE': f'{metrics["mse"]:.6f}',
            'Loss': f'{loss_dict["total_loss"].item():.6f}'
        })
        
        # Reduced logging without scaling
        if batch_idx == 0 or batch_idx == n_batches - 1:
            logging.info(f"Batch {batch_idx} metrics:")
            logging.info(f"MSE: {metrics['mse']:.6f}")
            logging.info(f"Loss: {loss_dict['total_loss'].item():.6f}")
    
    # Properly average metrics
    metrics_avg = {
        k: v / max(batch_counts[k], 1) for k, v in metrics_tracker.items()
    }
    
    # TensorBoard logging with raw MSE
    writer.add_scalar('Training/MSE', metrics_avg['mse'], epoch)
    writer.add_scalar('Training/Loss', metrics_avg['loss'], epoch)
    
    logging.info(f'Epoch {epoch}: Loss = {metrics_avg["loss"]:.4f}, '
                f'MSE = {metrics_avg["mse"]:.4f}')
    
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

def validate(model, val_loader, criterion, device, writer, epoch, config, last_val_metrics=None):
    """Validate the model."""
    model.eval()
    metrics_tracker = {
        'loss': 0.0,
        'mse': 0.0
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
            metrics_tracker['loss'] += loss_dict['total_loss'].item()
            
            # Calculate raw MSE without scaling
            metrics = calculate_metrics(outputs, batch['labels'])
            metrics_tracker['mse'] += metrics['mse']
    
    # Average metrics
    metrics_avg = {k: v / len(val_loader) for k, v in metrics_tracker.items()}
    
    # Log validation metrics
    writer.add_scalar('Validation/MSE', metrics_avg['mse'], epoch)
    writer.add_scalar('Validation/Loss', metrics_avg['loss'], epoch)
    
    # Log relative improvement if we have previous metrics
    if last_val_metrics is not None:
        mse_improvement = ((last_val_metrics['mse'] - metrics_avg['mse']) / last_val_metrics['mse']) * 100
        logging.info(f"MSE improvement: {mse_improvement:.2f}%")
    
    return metrics_avg, metrics_avg['mse'] <= config['training']['target_mse']

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
    
    # Training loop with maximum duration check
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        verbose=True,
        min_epochs=config['training']['early_stopping']['min_epochs']
    )
    
    best_val_loss = float('inf')
    max_epochs = config['training']['num_epochs']
    max_no_improve = config['training']['early_stopping']['max_epochs_without_improvement']
    epochs_no_improve = 0
    
    start_time = time.time()
    max_training_time = 3600 * 2  # 2 hours maximum
    
    last_val_metrics = None
    
    for epoch in range(max_epochs):
        # Check training time
        if time.time() - start_time > max_training_time:
            logging.info("Maximum training time reached. Stopping training.")
            break
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        logging.info(f'Epoch {epoch}: Loss = {train_metrics["loss"]:.4f}, '
                    f'MSE = {train_metrics["mse"]:.4f}')
        
        # Validate with last metrics
        val_metrics, target_reached = validate(
            model, val_loader, criterion, device, writer, epoch, config, last_val_metrics
        )
        logging.info(f'Val: Loss = {val_metrics["loss"]:.4f}, '
                    f'MSE = {val_metrics["mse"]:.4f}')
        
        # Update last_val_metrics for next epoch
        last_val_metrics = val_metrics
        
        if target_reached:
            logging.info("Target MSE reached. Stopping training.")
            break
        
        # Check early stopping
        early_stopping(val_metrics['loss'], epoch)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered!")
            break
        
        # Check for improvement
        if val_metrics['loss'] < best_val_loss - config['training']['early_stopping']['min_delta']:
            best_val_loss = val_metrics['loss']
            epochs_no_improve = 0
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            logging.info(f'Saved new best model with validation loss: {val_metrics["loss"]:.4f}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= max_no_improve:
                logging.info(f"No improvement for {max_no_improve} epochs. Stopping training.")
                break

if __name__ == '__main__':
    main()