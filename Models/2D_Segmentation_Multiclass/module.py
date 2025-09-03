# --- import necessary library ---
import torch
import pytorch_lightning as pl
from torchmetrics import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassDice

# --- Lightning Module class ---
class SegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning Module for multi-class semantic segmentation.

    This module wraps a given backbone model and provides:
    - Training, validation, and test loops
    - Metric tracking (Accuracy, IoU/Jaccard Index, Dice Score) for each stage
    - Flexible optimizer, scheduler, and criterion injection
    - Support for custom device placement

    Metrics:
        - Accuracy (macro): Measures how often predictions match labels across all classes equally.
        - IoU / Jaccard Index (macro): Measures the overlap ratio between predicted and true masks.
        - Dice Score (macro): Similar to IoU but more sensitive to small regions.

    Important:
        - Input logits shape: (B, C, H, W)
        - Target masks shape: (B, H, W) or (B, 1, H, W)
        - Metrics expect logits or probabilities; CrossEntropyLoss expects integer labels (long type).

    Args:
        backbone (nn.Module): The segmentation backbone model.
        num_classes (int): Number of segmentation classes (including background).
        manual_device (torch.device or str, optional): If provided, manually moves data to this device.

    Example:
        >>> model = SegmentationModule(backbone=MyUNet(), num_classes=4)
        >>> model.set_criterion(ComboLoss(num_classes=4))
        >>> model.set_optimizer(torch.optim.Adam(model.parameters(), lr=1e-3), None)
    """

    def __init__(self, backbone=None, num_classes=4, manual_device=None):
        self.backbone = backbone
        self.num_classes = num_classes
        self.manual_device = manual_device
        
        # --- metrics ---
        # MulticlassAcc -> calculates how often the model's predictions are correct for each pixel or mask.
        # IoU (MulticlassJaccardIndex) -> calculates the overlap ratio between predictions and the ground truth per class.
        # MulticlassDice -> similar to IoU, but more sensitive to small areas.
        # average='macro' -> the metric is calculated per class and then averaged, so each class has equal weight.
        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes, average='macro')
        self.train_iou = MulticlassJaccardIndex(num_classes=self.num_classes, average='macro')
        self.train_dice = MulticlassDice(num_classes=self.num_classes, average='macro', task='multiclass')
        
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes, average='macro')
        self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes, average='macro')
        self.val_dice = MulticlassDice(num_classes=self.num_classes, average='macro', task='multiclass')
        
        self.test_acc = MulticlassAccuracy(num_classes=self.num_classes, average='macro')
        self.test_iou = MulticlassJaccardIndex(num_classes=self.num_classes, average='macro')
        self.test_dice = MulticlassDice(num_classes=self.num_classes, average='macro', task='multiclass')
        
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
    
    # --- return optimizer to LightningModule ---
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
        else:
            return self.optimizer
        
    # --- data movement ---
    def batch_to_device(self, batch, device):
        x, y = batch
        target_device = self.manual_device if self.manual_device else device
        return x.to(target_device, non_blocking=True), y.to(target_device, non_blocking=True)
    
    # --- shared step ---
    def shared_step(self, batch, stage):
        x, y = self.batch_to_device(batch, self.device)
        logits = self(x)
        y_loss = y.long()
        y_metrics = y_loss
        loss = self.criterion(logits, y_loss)
        
        if stage == "train":
            self.train_acc.update(logits, y_metrics)
            self.train_iou.update(logits, y_metrics)
            self.train_dice.update(logits, y_metrics)
            self.log("train_loss", loss,
                     on_step=True, 
                     on_epoch=True)
        
        elif stage == "val":
            self.val_acc.update(logits, y_metrics)
            self.val_iou.update(logits, y_metrics)
            self.val_dice.update(logits, y_metrics)
            self.log("val_loss", loss,
                     on_epoch=True,
                     prog_bar=True)
        
        elif stage == "test":
            self.test_acc.update(logits, y_metrics)
            self.test_iou.update(logits, y_metrics)
            self.test_dice.update(logits, y_metrics)
            self.log("test_loss", loss, 
                     on_epoch=True)
        return loss

    # --- on_x_epoch_end ---
    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.log("train_iou", self.train_iou.compute(), prog_bar=True)
        self.log("train_dice", self.train_dice.compute(), prog_bar=True)
        self.train_acc.reset()
        self.train_iou.reset()
        self.train_dice.reset()
    
    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_iou", self.val_iou.compute(), prog_bar=True)
        self.log("val_dice", self.val_dice.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_iou.reset()
        self.val_dice.reset()
    
    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.log("test_iou", self.test_iou.compute(), prog_bar=True)
        self.log("test_dice", self.test_dice.compute(), prog_bar=True)
        self.test_acc.reset()
        self.test_iou.reset()
        self.test_dice.reset()
        
    # --- forward pass ---
    def forward(self, x):
        return self.backbone(x)
    
    # --- training step ---
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")
    
    # --- val step ---
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
    
    # --- test step ---
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
    
    # --- predict step ---
    def predict_step(self, batch, batch_idx, datalaoder_idx=0):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = self.batch_to_device(batch, self.device)
        else:
            x = batch.to(self.manual_device if self.manual_device else self.device, non_blocking=True)
        outputs = self(x)
        return torch.softmax(outputs, dim=1)
                
    
        