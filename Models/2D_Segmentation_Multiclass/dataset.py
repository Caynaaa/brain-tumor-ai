# --- import necessary library ---
import os
import h5py
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

# --- CustomDataset with Hybrid Pathc ---
class CustomDataset(Dataset):
    """
    Custom Dataset for medical segmentation with hybrid patch and caching index support.

    This dataset reads data from an HDF5 file (.h5) containing two main datasets:
    - 'x': image (slice)
    - 'y': mask (label) in grayscale (0, 50, 100, 150)

    Key Features:
    - Hybrid patch: a mix of full slice resize and random patch crop.
    - Caching index: saves the mapping (file_path, slice_index) to a .pt file
    to speed up subsequent dataset loading.
    - Optional augmentation transformation using Albumentations.

    Args:
    file_path (list[str]): List of HDF5 file paths.
    img_size (tuple[int, int], optional): Output image and mask size (H, W). Default: (240, 240).
    patch_size (int, optional): Patch size before resizing to `img_size`. Default: 160.
    hybrid_prob (float, optional): Probability of using a full slice versus a random patch. Default: 0.3.
    transform (callable, optional): Augmentation transform (e.g., from Albumentations). Default: None.
    debug (bool, optional): If True, displays the shape and dtype info of the first image/mask. Default: False.
    index_cache_path (str, optional): Path to store/load the index cache. Default: None.
    load_index_if_available (bool, optional): If True, loads the index cache if available. Default: True.

    Notes:
    - Mask labels will be automatically remapped:
    0 -> 0
    50 -> 1
    100 -> 2
    150 -> 3
    - For patch cropping, only patches with >10 pixels of class other than the background are selected.
    """

    # --- define params in init (built-in Dataset) ---
    def __init__(self, file_path, img_size=(240, 240), patch_size=160, hybrid_prob=0.3, transform=None, 
                 debug=False, index_cache_path=None, load_index_if_available=True):
        self.file_path = file_path
        self.img_size = img_size
        self.patch_size = patch_size
        self.hybrid_prob = hybrid_prob
        self.transform = transform
        self.debug = debug
        self.index_cache_path = index_cache_path
        self.load_index_if_available = load_index_if_available
        self.full_index = []
        
        # --- if condition for load and save index file_path ---
        if self.load_index_if_available and self.index_cache_path and os.path.exists(self.index_cache_path):
            print(f"[INFO] Loading index from {self.index_cache_path}...")
            self.full_index = torch.load(self.index_cache_path)
        else:
            # --- build mapping to file_path (file, slice_idx) ---
            for fpath in tqdm(self.file_path, desc="mapping file path..."):
                with h5py.File(fpath, 'r') as f:
                    n_slices = len(f['x']) 
                for i in range(n_slices):
                    self.full_index.append((fpath, i))
            if self.index_cache_path:
                torch.save(self.full_index, self.index_cache_path)
                print(f"[INFO] Saved index to {self.index_cache_path}...")
        
        # flag: for debug just once
        self._debug_done = False
    
    # --- return len dataset (built-in Dataset) ---    
    def __len__(self):
        return len(self.full_index)
    
    # --- load single slice ---
    def load_slice(self, file_path, idx):
        with h5py.File(file_path, 'r') as f:
            image = f['x'][idx]
            mask = f['y'][idx]
        # remap labels 
        remap = {0:0, 50:1, 100:2, 150:3}
        mask = np.vectorize(remap.get)(mask).astype(np.int64)
        return image, mask
    
    # --- random patch extraction ---
    def random_patch(self, image, mask):
        # get Height & Width and ignore Channel
        H, W = image.shape[:2]
        patch_size = self.patch_size
        # if slice < pathc size -> return full slice
        if H < patch_size or W < patch_size:
            return image, mask
        
        # try loop 10 time for get random patch
        for _ in range(10):
            x = np.random.randint(0, W-patch_size) # coordinat x top-left patch
            y = np.random.randint(0, H-patch_size) # coordinat y top-left patch
            img_patch = image[y:y+patch_size, x:x+patch_size] # crop image
            mask_patch = mask[y:y+patch_size, x:x+patch_size] # crop mask
            # if patch have enough tumor pixel -> used
            if np.sum(mask_patch > 0) > 10:
                # interpolation -> method for filling in new pixels "when zooming in or out" of an image
                # cv2.INTER_LINEAR -> calculates the new pixel value from the linear average of surrounding pixels, output: smooth, not broken - suitable for images (Grayscale/RGB)
                # cv2.INTER_NEAREST -> selects the nearest pixel without averaging, output: not smooth, pixels remain sharp, - suitable for masks/labels because the class value does not change 
                img_patch = cv2.resize(img_patch, self.img_size, interpolation=cv2.INTER_LINEAR) 
                mask_patch = cv2.resize(mask_patch, self.img_size, interpolation=cv2.INTER_NEAREST)
                return img_patch, mask_patch
        
        # fallback: if 10 times random patch fails to find tumor
        # get top-left (0,0) patch
        img_patch = cv2.resize(image[0:patch_size, 0:patch_size], self.img_size, interpolation=cv2.INTER_LINEAR)
        mask_patch = cv2.resize(mask[0:patch_size, 0:patch_size], self.img_size, interpolation=cv2.INTER_NEAREST)
        return img_patch, mask_patch

    # --- __getitem__ (built-in Dataset) ---
    def __getitem__(self, idx):
        fpath, slice_idx = self.full_index[idx]
        image, mask = self.load_slice(fpath, slice_idx)
        
        # expand channel -> so that cv2 can perform modifications, used in func.random_patch
        # np.repeat -> changes the channel to 3 (RGB) to match the pretrained model
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
    
        # hybrid patch logic
        if np.random.rand() < self.hybrid_prob:
            # resize full slice 
            img_patch = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
            mask_patch = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            is_patch = False
        
        else:
            # random patch
            img_patch, mask_patch = self.random_patch(image, mask)
            is_patch = True
            
        # debug: before transform
        if self.debug and not self._debug_done:
            print(f"[DEBUG] BEFORE transform: img {img_patch.shape}, type {img_patch.dtype}," 
                  f"mask {mask_patch.shape}, type {mask_patch.dtype}")
            
        # transform metods
        if self.transform:
            aug = self.transform(image=img_patch, mask=mask_patch)
            img_patch = aug['image']
            mask_patch = aug['mask']
        
        # check if mask_patch have ndim > 2 will be remove, because loss want 2d shape (H, W)
        if mask_patch.ndim == 3 and mask_patch.shape[-1] == 1:
            mask_patch = mask_patch.squeeze(-1)
        mask_patch = mask_patch.astype(np.int64)
        
        # debug: after tarnsform
        if self.debug and not self._debug_done:
           print(f"[DEBUG AFTER] patch={is_patch} img {img_patch.shape}, type {img_patch.dtype}, "
                 f"mask {mask_patch.shape}, type {mask_patch.dtype}, "
                 f"unique mask: {np.unique(mask_patch)}")
        # set debug_done to True, so that debug print does not appear again 
        self._debug_done = True
        return img_patch, mask_patch