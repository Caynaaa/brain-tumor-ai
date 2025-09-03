# --- import necessary library ---
import torch 
import torch.nn as nn
from monai.losses import DiceLoss

# --- ComboLoss (CELoss + DiceLoss) ---
class ComboLoss(nn.Module):
    """
    Combines CrossEntropyLoss and DiceLoss for multi-class segmentation.

    Formula:
        total_loss = ce_scale * CrossEntropyLoss + dice_weight * DiceLoss

    Args:
        num_classes (int): Number of classes including background.
        ce_weight (Tensor, optional): Class weights for CE loss.
        dice_weight (float): Weight for Dice loss. Default: 1.0.
        ce_scale (float): Weight for CE loss. Default: 1.0.
    """
    
    
    def __init__(self, num_classes=None, ce_weight=None, dice_weight=1.0, ce_scale=1.0):
        # call parents class
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_scale = ce_scale
        # --- Celoss ---
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        # --- Diceloss ---
        # to_onehot_y -> auto onehot target (B, 1, H, W) or (b, H, W) to (B, C, H, W). 
        # softmax - > auto apply softmax to prediction. 
        # include_background -> background class is also calculated.
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
    
    # --- forward pass ---
    def forward(self, preds: torch.tensor, targets: torch.tensor):
        if self.num_classes is None:
            self.num_classes = preds.shape[1]
        
        # safty check
        assert preds.ndim == 4, f"preds must be (B, C, H, W), can be {preds.shape}"
        assert preds.shape[1] == self.num_classes, f"preds channel={preds.shape[1]} not compatibel with num_classes={self.num_classes}"
        assert targets.ndim in (3, 4), f"targets shape must be (B, H, W) or (B, 1, H, W), can be {targets.shape}"
        
        # ensure targets shape for CE
        targets_ce = targets.squeeze(1) if targets.ndim == 4 else targets
        targets_ce = targets_ce.long()
        
        # ensure targets shape for Dice
        targets_dice = targets.unsqueeze(1) if targets.ndim == 3 else targets
        
        # compute individual losses
        loss_ce = self.ce(preds, targets_ce)
        loss_dice = self.dice(preds, targets_dice)
        
        # combined weighted losses
        total_loss = self.ce_scale * loss_ce + self.dice_weight * loss_dice
        return total_loss 
        