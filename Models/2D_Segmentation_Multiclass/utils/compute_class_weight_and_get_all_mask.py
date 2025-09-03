# --- library ---
import numpy as np
import torch
import h5py
import os
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# --- compute class weight -> get all mask from train path and used it for calculate class_weight for CELoss ---
def compute_class_weight_and_get_mask(file_path, remap=None, save_path=None, load_if_available=True):
    """
    Compute class weights for multi-class segmentation from HDF5 mask files.

    This function scans through all given HDF5 files, extracts the segmentation 
    masks, optionally remaps label values to a continuous integer range, and 
    calculates class weights using sklearn's `compute_class_weight` for use 
    with losses like CrossEntropyLoss.

    Args:
        file_path (list[str]): List of paths to HDF5 files containing masks under the key 'y'.
        remap (bool, optional): If True, remap mask pixel values from {0,50,100,150} to {0,1,2,3}.
                                Useful when masks have non-contiguous label values. Default: False.
        save_path (str, optional): Path to save computed class weights as a .pt file. Default: None.
        load_if_available (bool, optional): If True and `save_path` exists, load cached weights instead 
                                            of recomputing. Default: False.

    Returns:
        torch.Tensor: Class weights tensor of shape (num_classes,) with dtype=torch.float32.
    
    Example:
        >>> paths = ["train_1.h5", "train_2.h5"]
        >>> weights = compute_class_weight(paths, remap=True, save_path="class_weights.pt")
        >>> loss_fn = nn.CrossEntropyLoss(weight=weights)
    """

    # --- laod file index if have in path_save ---
    if load_if_available and save_path and os.path.exists(save_path):
        print(f"[INFO] Index found, loading from {save_path}...")
        return torch.load(save_path)  
    
    all_labels = []
    
    # loop for get all mask
    for path in tqdm(file_path, desc="collecting mask..."):
        with h5py.File(path, 'r') as f:
            # get all mask
            masks = f['y'][:]
            if remap is not None:
                masks = np.vectorize(remap.get)(masks)
            # add to all_labels and convert to 1D with "flatten"
            all_labels.append(masks.flatten())
            
    # compute class weight
    # use "np.concatenate" for combine all values ​​in "all_labels"
    all_labels = np.concatenate(all_labels)
    weights = compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(all_labels),
        y = all_labels
    ) 
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # --- for save the file index ---
    if save_path:
        torch.save(weights, save_path)
        print(f"[INFO] class weight saved to {save_path}")
    
    return weights

        
                 