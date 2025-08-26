# Import necessary library
import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy, MulticlassAUROC

class Hybrid_PLmodule(pl.LightningModule):
    def __init__(self, backbone=None, num_classes=None, manual_device=None):
        super().__init__()
        self.backbone = backbone
        self.manual_device = manual_device
        self.num_classes = num_classes
        
        # Inject placeholder
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # metrics
        self.metrics_initialized = False
    
    # --- Injection metodhs ---
    def set_optimizer(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def set_criterion(self, criterion):
        self.criterion = criterion
    
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
    
    # --- metrics ---
    def _init_metrics(self, outputs):
        if self.num_classes is None:
            self.num_classes = outputs.shape[1] if outputs.ndim > 1 else 1
        # --- binary ---
        if self.num_classes <= 2:
            self.train_acc = BinaryAccuracy().to(self.device)
            self.val_acc = BinaryAccuracy().to(self.device)
            self.test_acc = BinaryAccuracy().to(self.device)
            self.val_auroc = BinaryAUROC().to(self.device)
            self.test_auroc = BinaryAUROC().to(self.device)
        # --- multiclass ---
        else:
            self.train_acc = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
            self.val_acc = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
            self.test_acc = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
            self.val_auroc = MulticlassAUROC(num_classes=self.num_classes).to(self.device)
            self.test_auroc = MulticlassAUROC(num_classes=self.num_classes).to(self.device)
        
        self.metrics_initialized = True
    
    # --- automactly get probs in cases binary outputs ---
    def get_probs(self, outputs):
        if outputs.shape[1] == 1:
            # BCEW style-(sigmoid for 1 neuron output)
            return torch.sigmoid(outputs).view(-1)
        else:
            # CE style-(softmax for multiclass, get probability positif[1] class)
            return torch.softmax(outputs, dim=1)[:,1]
        
    # --- shared step so i not need to handwork ---
    def shared_step(self, batch, stage):
        x, y = self.batch_to_device(batch, self.device)
        outputs = self(x)
        
        if not self.metrics_initialized:
            self._init_metrics(outputs)
            
        # casting y for loss and metrics
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            y_loss = y.long()
            y_metrics = y_loss
        elif isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            y_loss = y.float().unsqueeze(1)
            y_metrics = y.int().view(-1)
        
        loss = self.criterion(outputs, y_loss)
        probs = self.get_probs(outputs)
            
        if stage == 'train':
            acc = self.train_acc(probs, y_metrics)
            self.log("train_loss", 
                 loss, on_step=True, 
                 on_epoch=True)
            self.log("train_acc",
                 acc, on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        
        elif stage == 'val':
            acc = self.val_acc(probs, y_metrics)
            self.val_auroc.update(probs, y_metrics)
            self.log("val_loss", 
                     loss, on_epoch=True, 
                     prog_bar=True)
            self.log("val_acc", 
                     acc, on_epoch=True, 
                     prog_bar=True)
        
        elif stage == 'test':
            acc = self.test_acc(probs, y_metrics)
            self.test_auroc.update(probs, y_metrics)
            self.log("test_loss", 
                     loss, on_epoch=True)
            self.log("test_acc", 
                     acc, on_epoch=True, 
                     prog_bar=True)
        return loss
    
    # --- forward pass ---
    def forward(self, x):
        return self.backbone(x)
    
    # --- training loop ---        
    def training_step(self, batch, batch_idx):
       return self.shared_step(batch, 'train')
   
    # --- validation loop ---
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')
    
    # --- test loop ---
    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, 'test')
        return None
        
    def on_validation_epoch_end(self):
        if self.metrics_initialized:
            val_auroc = self.val_auroc.compute()
            self.log("val_auroc", 
                val_auroc, on_epoch=True,
                prog_bar=True)
            self.val_auroc.reset()
            
    def on_test_epoch_end(self):
        if self.metrics_initialized:
            test_auroc = self.test_auroc.compute()
            self.log("test_auroc", 
                test_auroc, on_epoch=True, 
                prog_bar=True)  
            self.test_auroc.reset()      

    # --- predict step ---
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = self.batch_to_device(batch, self.device)
        else:
            x = batch.to(self.manual_device if self.manual_device else self.device, non_blocking=True)
        outputs = self(x)
        return self.get_probs(outputs)