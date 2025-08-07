import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchmetrics.classification import BinaryAccuracy, AUROC



class DenseNetClassifierBinary(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-3,
                 weight_decay=1e-5,
                 unfreeze_layers=None
    ):
        super(). __init__()
        self.save_hyperparameters()
        
        backbone = models.densenet121(pretrained=True)
        
        # freeze all layers
        for param in backbone.parameters():
            param.requires_grad = False
            
        # Selectify unfreeze specified Layers
        if unfreeze_layers is not None:
            for name, param in backbone.named_parameters():
                if any(layer_name in name for layer_name in unfreeze_layers):
                    param.requires_grad = True
        
        # Replace classifier head
        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(num_features, 1)
        
        self.model = backbone
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auroc = AUROC(task="binary")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, 
                 on_step=True, 
                 on_epoch=True
        )
        self.log("train_acc", acc, 
                 on_step=False, 
                 on_epoch=True,
                 prog_bar=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        acc = self.val_acc(logits, y)
        auroc = self.val_auroc(logits, y)
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
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr = self.hparams.learning_rate,
            weight_decay = self.hparams.weight_decay
        )
        scheduler = optim.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2,
            monitor='val_loss'
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }