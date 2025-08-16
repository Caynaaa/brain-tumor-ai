"""
This file defines the PyTorch Lightning Module for multi-class image classification.

It handles:
- Loading a pre-trained backbone model (e.g., DenseNet121, ResNet50, EfficientNet)
- Replacing the classifier head with an output layer matching the number of target classes
- Optional unfreezing of specific layers for fine-tuning
- Defining training, validation, and test steps
- Logging metrics (Accuracy, Top-K Accuracy, AUROC per class, Loss, etc.)
- Integrating optimizer and learning rate scheduler

This design separates model logic from data and training pipeline,
enabling more maintainable and modular experimentation.

Components:
------------
1. ImageClassifierMultiClass:
   - Inherits from pl.LightningModule
   - Accepts training hyperparameters (learning rate, weight decay, layers to unfreeze, number of classes)
   - Defines forward pass and step-wise training/validation/test procedures
   - Supports metric logging for both overall and per-class performance
   - Returns a scheduler configuration compatible with PyTorch Lightning's Trainer
"""


# import necessary libraries
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

# Define the "Lightning Module" for multiclass classification
class DenseNetClassifierMulticlass(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
        unfreeze_layers=None,
        backbone_model=models.densenet121,
        class_weight = None
        ):
               
        # Initialize the parent class
        super().__init__()
        # Save hyperparameters for logging
        self.save_hyperparameters()
        # Store hyperparameters
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.backbone_model = backbone_model
        
        # Load a pre-trained DenseNet121 model
        backbone = self.backbone_model(pretrained=True)
        
        # freeze all layers
        for param in backbone.parameters():
            param.requires_grad = False
        # Unfreeze the classifier head
        for name, param in backbone.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
        # Selectively unfreeze specified layers
        if unfreeze_layers is not None:
            # Unfreeze specific layers if provided
            for name, param in backbone.named_parameters():
                if any(layer_name in name for layer_name in unfreeze_layers):
                    param.requires_grad = True
        
        # Replace classifier head
        num_features = backbone.classifier.in_features
        # Replace classifier head for multiclass output
        backbone.classifier = nn.Linear(num_features, 4)
        
        # Store the modified model
        self.model = backbone
        
        # Define the loss function
        # If class weights are provided, use weighted CrossEntropyLoss
        if class_weight is not None:
            class_weight = torch.tensor(class_weight, dtype=torch.float32)    
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics
        # Multiclass Accuracy and AUROC for multiclass classification
        self.train_acc = MulticlassAccuracy(num_classes=4)
        self.val_acc = MulticlassAccuracy(num_classes=4)
        self.val_auroc = MulticlassAUROC(task="multiclass", num_classes=4)
        self.test_acc = MulticlassAccuracy(num_classes=4) 
        self.test_auroc = MulticlassAUROC(task="multiclass", num_classes=4)
        
    # Define the forward pass
    def forward(self, x):
        return self.model(x)
     
    # Define the training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)   
        loss = self.criterion(outputs, y)
        acc = self.train_acc(outputs, y)
        # log the metrics
        self.log("train_loss", 
                 loss,
                 on_step=True,
                 on_epoch=True,
                )        
        self.log("train_acc",
                 acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                )
        return loss
    
    # Define the validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        acc = self.val_acc(outputs, y)
        probs = torch.softmax(outputs, dim=1)
        auroc = self.val_auroc(probs, y)
        # log the metrics
        self.log("val_loss",
                 loss,
                 on_step=False,
                 on_epoch=True
                )
        self.log("val_acc",
                 acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True
                )
        self.log("val_auroc",
                 auroc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True
                )
        return loss
    
    # Define the test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        acc = self.test_acc(outputs, y)
        probs = torch.softmax(outputs, dim=1)
        auroc = self.test_auroc(probs, y)
        # log the metrics
        self.log("test_loss",
                 loss,
                 on_step=False,
                 on_epoch=True
                )
        self.log("test_acc",
                 acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True
                )
        self.log("test_auroc",
                 auroc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True
                )
    
    # Define prediction step
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = batch
        else:
            x = batch
        outputs = self(x)
        return torch.softmax(outputs, dim=1) 
    
    # Define the optimizer
    def configure_optimizers(self):
        # Filter parameters that require gradients
        # This ensures only trainable parameters are optimized
        param_to_optimize = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = optim.AdamW(
            param_to_optimize,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # Use ReduceLROnPlateau scheduler to reduce learning rate on plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=2,
        )
        # Return optimizer and scheduler configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }
        