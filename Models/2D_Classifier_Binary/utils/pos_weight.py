import torch

def compute_poss_weight(labels: torch.tensor) -> torch.tensor:
    """
    Compute the positive class weight for BCEWithLogitsLoss.

    This weight helps balance the contribution of positive and negative samples
    when training on imbalanced datasets.

    Args:
        labels (torch.Tensor): 1D tensor of shape (N,) containing binary labels (0 or 1).

    Returns:
        torch.Tensor: Scalar tensor representing the positive class weight.
    """
    
    # Count positive and negative samples
    # pos_count is the number of positive samples (label == 1)
    pos_count = labels.sum().item()
    neg_count = len(labels) - pos_count
    
    if pos_count == 0 or neg_count == 0:
        # Edge case: dataset contains only one class
        return torch.tensor(1.0) 
    return torch.tensor(neg_count / pos_count, dtype=torch.float)
