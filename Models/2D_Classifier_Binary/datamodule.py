"""
This file defines the LightningDataModule for binary classification using PyTorch Lightning.

It handles:
- CustomDataset creation for image-label pairs
- Applying transformations (augmentations & normalization)
- Creating DataLoader for training and validation
- Clean integration with PyTorch Lightning Trainer

This improves modularity and separates data concerns from model/training logic.

Components:
------------
1. CustomDataset: 
   - Wraps image paths and corresponding labels
   - Loads images using PIL and applies Albumentations transforms
   - Returns image tensor and label tensor

2. BrainTumorDataModule:
   - Inherits from pl.LightningDataModule
   - Accepts training/validation data (already split), image size, batch size, etc.
   - Prepares datasets and DataLoaders
   - Can be reused across training, validation, and testing
"""


# Import necessary libraries
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transform import get_transform
from imbalanced_data_utils import make_weights_sampler 

# Define the "Custom Dataset"
class CustomDataset(Dataset):
    def __init__(self, img_input, labels, transform=None):
        # Initialize the dataset with image paths, labels, and transformations
        self.img_input= img_input
        self.labels = labels
        self.transform = transform
    
    # Override the __len__ and __getitem__ methods
    # to make it compatible with PyTorch DataLoader
    def __len__(self):
        return len(self.img_input)
    
    def __getitem__(self, idx):
        img_idx = self.img_input[idx]
        
        if isinstance(img_idx, str):
            img = Image.open(img_idx)
            img = img.convert('RGB')
            img = np.array(img)
        
        elif isinstance(img_idx, Image.Image):
            img = img_idx.convert('RGB')
            img = np.array(img)
        
        elif isinstance(img_idx, np.ndarray):
            img = img_idx
        
        else:
            raise ValueError("Unsupported image input type. Must be str, PIL Image, or numpy array.")
        
        # Get the corresponding label
        labels = self.labels[idx]
        # Convert label to tensor and ensure it's float for BCE loss
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Apply transformations if provided
        if self.transform:
            img = self.transform(image=img)['image']
        
        # Return the image tensor and label tensor
        return img, labels
    
# Define the "LightningDataModule"
class BrainTumorDataModule(pl.LightningDataModule):
    def __init__(self, train_data=None, val_data=None, test_data=None, batch_size=64, img_size=(224, 224), num_workers=4, use_sampler=False):
        """
        PyTorch Lightning DataModule for binary brain tumor classification.

        Args:
            train_data (tuple): Tuple containing training image paths and labels.
            val_data (tuple): Tuple containing validation image paths and labels.
            batch_size (int, optional): Number of samples per batch. Defaults to 64.
            img_size (tuple, optional): Target image size (height, width) for resizing. Defaults to (224, 224).
            num_workers (int, optional): Number of subprocesses used for data loading. Defaults to 4.
        """

        
        # Initialize the parent class
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.use_sampler = use_sampler
    
    # Setup method to create datasets and transformations
    # This is called by PyTorch Lightning Trainer
    def setup(self, stage=None):
        # Create transform:
        train_T, val_T = get_transform(self.img_size)
        
        # Create datasets using the CustomDataset class
        # Assuming train_data and val_data are tuples of (image_paths, labels)
        if stage == 'fit' or stage is None:
            if self.train_data is not None and self.val_data is not None:
                train_images, train_labels = self.train_data
                val_images, val_labels = self.val_data
                self.train_dataset = CustomDataset(train_images, train_labels, transform=train_T)
                self.val_dataset = CustomDataset(val_images, val_labels, transform=val_T)
            
        if stage == 'test' or stage is None:
            if self.test_data is not None:
                test_images, test_labels = self.test_data
            else:
                test_images, test_labels = self.val_data
            self.test_dataset = CustomDataset(test_images, test_labels, transform=val_T)
    
    # Define DataLoaders for training and validation
    # These will be used by the Trainer during training/validation
    def train_dataloader(self):
        # make sampler
        if self.use_sampler and self.train_data is not None:
            self.train_images, self.train_labels = self.train_data
            sampler = make_weights_sampler(torch.tensor(self.train_labels))
            return DataLoader(self.train_dataset,
                              batch_size = self.batch_size,
                              sampler = sampler,
                              num_workers = self.num_workers
                        )
        else:
            return DataLoader(self.train_dataset,
                            batch_size = self.batch_size,
                            shuffle = True,
                            num_workers = self.num_workers
                            )
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size = self.batch_size,
                          shuffle = False,
                          num_workers = self.num_workers
                        )
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size = self.batch_size,
                          shuffle = False,
                          num_workers = self.num_workers
                        )
    
        