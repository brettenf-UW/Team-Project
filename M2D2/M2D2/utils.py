# utils.py
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