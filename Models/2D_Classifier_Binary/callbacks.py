"""
Callback configuration for training a binary classifier using PyTorch Lightning.

This file provides two essential callbacks:
- ModelCheckpoint: Saves the best model based on validation loss (or any monitored metric).
- EarlyStopping: Stops training early if the monitored metric does not improve.

Usage:
    callbacks = get_callbacks(monitor_v='val_loss', mode_v='min')
    trainer = pl.Trainer(callbacks=callbacks)

Note:
- All checkpoints are saved to: 'checkpoints/binary_classifier/'
- Only the model weights are saved (not full LightningModule).
"""


# Import necessary libraries
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Callback functions for training a binary classifier using PyTorch Lightning
def get_callbacks(monitor_v='val_loss', mode_v='min', patience_v=3):
    """
    Returns a list of callbacks for training:
    - ModelCheckpoint: Saves the best model weights based on monitored metric.
    - EarlyStopping: Stops training if no improvement is seen in given patience.

    Args:
        monitor_v (str): Metric to monitor, e.g., 'val_loss' or 'val_acc'.
        mode_v (str): 'min' or 'max' depending on whether lower or higher is better.

    Returns:
        list: [ModelCheckpoint, EarlyStopping]
    """
    
    # Save the best model weights (based on val_loss by default)
    checkpoint_cb = ModelCheckpoint(
        # Folder to save checkpoints
        dirpath = 'checkpoints/binary_classifier',
        # Custom filename format
        filename = 'best-val-loss-{epoch:02d}-{val_loss:.2f}',
        # Metric to track
        monitor = monitor_v,
        # Mode: 'min' for loss, 'max' for accuracy
        mode = mode_v,
        # Only save the best model weights
        save_top_k = 1,
        # Save only the model weights, not the full LightningModule
        save_weight_only = True,
        # Verbose logging
        verbose = True
    )
    
    # Early stopping callback
    early_stopping_cb = EarlyStopping(
        monitor = monitor_v,
        mode = mode_v,
        patience = patience_v
    )
    
    return [checkpoint_cb, early_stopping_cb]
    