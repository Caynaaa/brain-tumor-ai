"""
Callback configuration for training a binary classifier using PyTorch Lightning.

This file provides two essential callbacks:
- ModelCheckpoint: Saves the best model based on validation loss (or any monitored metric).
- EarlyStopping: Stops training early if the monitored metric does not improve.

Usage:
    callbacks = get_callbacks(monitor='val_loss', mode='min')
    trainer = pl.Trainer(callbacks=callbacks)

Note:
- All checkpoints are saved to: 'checkpoints/binary_classifier/'
- Only the model weights are saved (not full LightningModule).
"""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def get_callbacks(dirpath='checkpoints/binary_classifier',
                  filename='best-val-loss-{epoch:02d}-{val_loss:.2f}',
                  monitor='val_loss',
                  mode='min',
                  patience=3):
    """
    Returns a list of callbacks for training:
    - ModelCheckpoint: Saves the best model weights based on monitored metric.
    - EarlyStopping: Stops training if no improvement is seen within the given patience.

    Args:
        dirpath (str): Directory to save checkpoints.
        filename (str): Filename pattern for the saved model.
        monitor (str): Metric to monitor, e.g., 'val_loss' or 'val_acc'.
        mode (str): 'min' or 'max' depending on whether lower or higher is better.
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        list: [ModelCheckpoint, EarlyStopping]
    """
    
    # Save the best model weights
    checkpoint_cb = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_weights_only=True,
        verbose=True
    )
    
    # Early stopping
    early_stopping_cb = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience
    )
    
    return [checkpoint_cb, early_stopping_cb]
