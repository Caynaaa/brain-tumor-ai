"""
This file defines the LightningDataModule for multiclass classification using PyTorch Lightning.

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


# import necessary libraries
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transform import get_transform

# Define the "Custom Dataset"
class CustomDataset(Dataset):
    def __init__(self, img_input, labels, transform=None):
        self.img_input = img_input
        self.labels = labels
        self.transform = transform
    
    # Override the __len__ 
    def __len__(self):
        return len(self.img_input)
    
    # Override the __getitem__ method
    def __getitem__(self, idx):
        img_idx = self.img_input[idx]
        
        # Check the type of img_idx and load the image accordingly
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
        labels = torch.tensor(labels, dtype=torch.long)
        # Apply transformations if provided
        if self.transform:
            img = self.transform(image=img)['image']
        return img, labels

# Define the "LightningDataModule"
class BrainTumorDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_data, val_data, 
        test_data=None, 
        batch_size=64, 
        img_size=(224, 224), 
        num_workers=4
    ):
    
        # Initialize the parent class 
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
    
    # Setup method to initialize datasets and transformations
    def setup(self, stage=None):
        train_T, val_T = get_transform(self.img_size)

        # Create datasets for training and validation
        if stage == 'fit' or stage is None:
            train_imgs, train_labels = self.train_data
            val_imgs, val_labels = self.val_data
            self.train_dataset = CustomDataset(train_imgs, train_labels, transform=train_T)
            self.val_dataset = CustomDataset(val_imgs, val_labels, transform=val_T)
        
        if stage == 'test' or stage is None:
            if self.test_data is not None:
                test_imgs, test_labels = self.test_data
                self.test_dataset = CustomDataset(test_imgs, test_labels, transform=val_T)
            else:
                test_imgs, test_labels = self.val_dataset
            self.test_dataset = CustomDataset(test_imgs, test_labels, transform=val_T) 
    
    # Define the data loaders for training, validation, and testing
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
            