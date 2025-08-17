"""
This file provides utility functions for handling class imbalance in binary classification tasks.
It includes:
- Computation of `pos_weight` for BCEWithLogitsLoss.
- Creation of a WeightedRandomSampler for DataLoader.

These utilities can be used in PyTorch or PyTorch Lightning projects to
balance datasets either by adjusting the loss function or by sampling strategy.
"""


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

def make_weights_sampler(labels: torch.tensor) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle imbalanced datasets.

    This sampler assigns higher weights to underrepresented classes so that
    the DataLoader samples them more frequently during training.

    Args:
        labels (torch.Tensor): 1D tensor of shape (N,) containing binary labels (0 or 1).

    Returns:
        torch.utils.data.WeightedRandomSampler: Sampler object for DataLoader.
    """
    
    # Count samples for each class
    class_sample_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels)])
    # Invert counts to weights(less frequent classes get higher weights)
    weight = 1. / class_sample_count.float()
    # Assign weights to each sample based on its class
    samples_weight = torch.tensor([weight[t] for t in labels])
    
    return torch.utils.data.WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),
        replacement=True  # Sample with replacement
    )