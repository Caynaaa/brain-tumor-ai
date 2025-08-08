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

# Define the "Custom Dataset"
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom Dataset for loading brain tumor images and corresponding labels.

        Args:
            image_paths (list): A list of file paths to the input images.
            labels (list): A list of binary labels corresponding to each image.
            transform (callable, optional): Albumentations transformation to apply to each image.
                                            Defaults to None.
        """
        
        # Initialize the dataset with image paths, labels, and transformations
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    # Override the __len__ and __getitem__ methods
    # to make it compatible with PyTorch DataLoader
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image using PIL and convert to RGB
        img = Image.open(self.image_paths[idx]).convert('RGB')
        # Convert to numpy array for Albumentations
        img = np.array(img)
        
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
    def __init__(self, train_data, val_data, batch_size=64, img_size=(224, 224), num_workers=4):
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
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
    
    # Setup method to create datasets and transformations
    # This is called by PyTorch Lightning Trainer
    def setup(self, stage=None):
        # Create transform:
        train_T, val_T = get_transform(self.img_size)
        
        # Create datasets using the CustomDataset class
        # Assuming train_data and val_data are tuples of (image_paths, labels)
        train_images, train_labels = self.train_data
        val_images, val_labels = self.val_data
        
        # Initialize the datasets
        self.train_dataset = CustomDataset(train_images, train_labels, transform=train_T)
        self.val_dataset = CustomDataset(val_images, val_labels, transform=val_T)
        
    # Define DataLoaders for training and validation
    # These will be used by the Trainer during training/validation
    def train_dataloader(self):
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
        
    
        