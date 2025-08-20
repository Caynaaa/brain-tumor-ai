from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

def compute_class_weights(labels: torch.tensor, num_classes: int) -> torch.tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        labels (torch.Tensor): Tensor of shape (N,) containing class labels.
        num_classes (int): Total number of classes.
    
    Returns:
        torch.Tensor: Weights for each class (size = num_classes).
    """
    labels = labels.cpu().numpy().astype(int)
    class_weights = compute_class_weight(
        class_weight = "balanced",
        classes = np.arange(num_classes),
        y = labels
    )
    return torch.tensor(class_weights, dtype=torch.float)