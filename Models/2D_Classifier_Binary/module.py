"""
This file defines the PyTorch Lightning Module for binary brain tumor classification.

It handles:
- Loading a pre-trained DenseNet121 model
- Replacing the classifier head with a binary output layer
- Optional unfreezing of specific layers for fine-tuning
- Defining training, validation, and test steps
- Logging metrics (Binary Accuracy and AUROC)
- Integrating optimizer and learning rate scheduler

This design separates model logic from data and training pipeline,
enabling more maintainable and modular experimentation.

Components:
------------
1. DenseNetClassifierBinary:
   - Inherits from pl.LightningModule
   - Takes training hyperparameters (learning rate, weight decay, layers to unfreeze)
   - Defines forward pass and step-wise training/validation/test procedures
   - Logs relevant metrics via `self.log`
   - Returns a scheduler config compatible with PyTorch Lightning's Trainer
"""


# Import necessary libraries
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchmetrics.classification import BinaryAccuracy, AUROC
from imbalanced_data_utils import compute_poss_weight

# Define the "Lightning Module" for binary classification
class DenseNetClassifierBinary(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-3,
                 weight_decay=1e-5,
                 unfreeze_layers=None
    ):
        # Initialize the parent class
        super().__init__()
        # Save hyperparameters for logging
        self.save_hyperparameters()
        self.criterion = None
        
        # Load a pre-trained DenseNet121 model
        backbone = models.densenet121(pretrained=True)
        
        # freeze all layers
        for param in backbone.parameters():
            param.requires_grad = False 
        # Unfreeze the classifier head
        # This allows the model to learn a new binary classification head
        for name, param in backbone.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
        # Selectively unfreeze specified layers
        if unfreeze_layers is not None:
            # Unfreeze specific layers if provided
            for name, param in backbone.named_parameters():
                # Check if the layer name matches any in the unfreeze list
                # and set requires_grad to True
                if any(layer_name in name for layer_name in unfreeze_layers):
                    param.requires_grad = True
        
        # Replace classifier head
        num_features = backbone.classifier.in_features
        # Replace classifier head for binary output
        backbone.classifier = nn.Linear(num_features, 1)
        # Store the modified model
        self.model = backbone
        
        # Initialize metrics
        # Binary Accuracy and AUROC for binary classification
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auroc = AUROC(task="binary")
        self.test_acc = BinaryAccuracy()
        self.test_auroc = AUROC(task="binary")
    
    # Setup method to compute pos_Weight for BCEWithLogitsLoss
    # This is called by PyTorch Lightning Trainer before training starts
    def setup(self, stage=None):
        train_labels = self.trainer.datamodule.train_labels
        pos_weight_v = compute_poss_weight(torch.tensor(train_labels))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_v)

    # Define the forward pass
    # Defines how input x flows through the model — called during training, validation, and testing
    def forward(self, x):
        return self.model(x)
    
    # Define the training and validation steps
    # These are called by the Trainer during training/validation
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass through the model
        # Squeeze to remove the extra dimension for binary output
        # Output (logits) is raw prediction before sigmoid activation
        logits = self(x).squeeze(1)
        
        # Compute loss and accuracy
        # Convert labels to float for BCE loss
        loss = self.criterion(logits, y.float())
        acc = self.train_acc(logits, y)
        
        # Log the loss and accuracy
        # This logs metrics for visualization (TensorBoard, WandB, etc.) and internal tracking
        self.log("train_loss", loss, 
                 on_step=True, 
                 on_epoch=True
        )
        self.log("train_acc", acc, 
                 on_step=False, 
                 on_epoch=True,
                 prog_bar=True
        )
        # Return the loss for optimization
        # This is used by the optimizer to update model weights
        return loss
    
    # Define the validation step
    # This is called by the Trainer during validation
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        acc = self.val_acc(logits, y)
        probs = torch.sigmoid(logits)
        auroc = self.val_auroc(probs, y)
        self.log("val_loss", loss, 
                 on_step=False, 
                 on_epoch=True
        )
        self.log("val_acc", acc,
                 on_step=False, 
                 on_epoch=True,
                 prog_bar=True
        )
        self.log("val_auroc", auroc,
                 on_step=False, 
                 on_epoch=True,
                 prog_bar=True
        )
        return loss
    
    # Define the test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        acc = self.test_acc(logits, y)
        probs = torch.sigmoid(logits)
        auroc = self.test_auroc(probs, y)
        self.log("test_loss", loss, 
                 on_step=False, 
                 on_epoch=True
        )
        self.log("test_acc", acc,
                 on_step=False, 
                 on_epoch=True,
                 prog_bar=True
        )
        self.log("test_auroc", auroc,
                 on_step=False, 
                 on_epoch=True,
                 prog_bar=True
        )
    
    # Define prediction step
    # This is called when making predictions (e.g., during inference)
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = batch
        else:
            x = batch
        logits = self(x).squeeze(1)
        return torch.sigmoid(logits)
    
   # Define Optimizer and learning rate scheduler
    def configure_optimizers(self):
        # Filter parameters to optimize
        params_to_optimize = filter(lambda p:p.requires_grad, self.model.parameters())
        
        # Use AdamW optimizer with specified learning rate and weight decay
        optimizer = optim.AdamW(
            params_to_optimize,
            lr = self.hparams.learning_rate,
            weight_decay = self.hparams.weight_decay
        )
        # Use ReduceLROnPlateau scheduler to reduce learning rate on validation loss plateau
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2
        )
        # Return the optimizer and scheduler configuration
        # This is compatible with PyTorch Lightning's Trainer
        # Lightning will automatically call scheduler.step() using 'monitor' key ('val_loss')
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }