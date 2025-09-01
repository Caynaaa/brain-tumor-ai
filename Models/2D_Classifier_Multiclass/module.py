# --- import necessary library ---
import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy

# --- LightningModule ---
class NN_PLmodule(pl.LightningModule):
    def __init__(self, backbone=None, manual_device=None, num_classes=None):
        super().__init__()
        self.backbone = backbone
        self.manual_device = manual_device
        self.num_classes = num_classes
        
        # --- Metrics ---
        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes).to(self.device)
        self.test_acc = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes).to(self.device)
        
        # --- Inject Placeholder ---
        self.optimizer = None
        self.scheduler = None
        self.criterion = None 
        
    # --- Inject Methods ---
    def set_optimizer(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def set_criterion(self, criterion):
        self.criterion = criterion
    
    # --- return optimizer to lightning ---
    def configure_optimizers(self):
        if self.scheduler:
            return {"optimizer": self.optimizer,
                    "lr_scheduler": {
                        "scheduler": self.scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                        "frequency": 1
                    }
                }
        return self.optimizer
    
    # --- data movement ---
    def batch_to_device(self, batch, device):
        x, y = batch
        target_device = self.manual_device if self.manual_device else device
        return x.to(target_device, non_blocking=True), y.to(target_device, non_blocking=True)
    
    # --- share step ---
    def shared_step(self, batch, stage):
        x, y = self.batch_to_device(batch, self.device)
        outputs = self(x)
        y_loss = y.long()
        y_metrics = y_loss
        loss = self.criterion(outputs, y_loss)
        probs = torch.softmax(outputs, dim=1)
        
        if stage == "train":
            self.train_acc.update(probs, y_metrics)
            self.log("train_loss",
                    loss, 
                    on_step=True,
                    on_epoch=True)
    
        elif stage == "val":
            self.val_acc.update(probs, y_metrics)
            self.val_auroc.update(probs, y_metrics)
            self.log("val_loss",
                loss, 
                on_epoch=True,
                prog_bar=True)
            
        elif stage == "test":
            self.test_acc.update(probs, y_metrics)
            self.test_auroc.update(probs, y_metrics)
            self.log("test_loss",
                    loss, 
                    on_epoch=True)
        return loss
    
    # --- forward pass ---
    def forward(self, x):
        return self.backbone(x)
    
    # --- training step ---
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")
    
    # --- validation step ---
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
    
    # --- test step ---
    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, "test")
        return None
    
    # --- train epoch end ---
    def on_train_epoch_end(self):
        self.log("train_acc",
                 self.train_acc.compute(),
                 prog_bar=True)
        self.train_acc.reset()
    
    # --- val epoch end ---
    def on_validation_epoch_end(self):
        self.log("val_acc",
                self.val_acc.compute(), 
                prog_bar=True)
        self.val_acc.reset()
        self.log("val_auroc",
                self.val_auroc.compute(), 
                prog_bar=True)
        self.val_auroc.reset()
    
    # --- test epoch end ---
    def on_test_epoch_end(self):
        self.log("test_acc",
                self.test_acc.compute(), 
                prog_bar=True)
        self.test_acc.reset()
        self.log("test_auroc",
                self.test_auroc.compute(), 
                prog_bar=True)
        self.test_auroc.reset()
        
    # --- predict step ---
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = self.batch_to_device(batch, self.device)
        else:
            x = batch.to(self.manual_device if self.manual_device else self.device, non_blocking=True)
        outputs = self(x)
        return torch.softmax(outputs, dim=1)
                