import torch

def make_random_sampler(labels: torch.tensor) -> torch.utils.data.WeightedRandomSampler:
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
        replacement=False
    )