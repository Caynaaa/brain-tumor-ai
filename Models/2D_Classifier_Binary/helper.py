"""
This file contains utility/helper functions used throughout the project.

Includes:
- set_seed: to ensure reproducible results across runs
"""


# Import necessary libraries
import random
import numpy as np
import torch

# Function to set random seed for reproducibility
# This ensures that results are consistent across runs
def set_seed(seed: int=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to use (default is 42)
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior for CUDA operations
    # This can slow down training but ensures consistent results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    