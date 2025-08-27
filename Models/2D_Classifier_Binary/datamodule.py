"""
DataModule — PyTorch Lightning Data Management for Brain Tumor Classification

This module implements a `LightningDataModule` that organizes dataset loading, transformations,
and batching for training, validation, and testing in the binary brain tumor classification project.

Key Features:
- Unified Data Handling: Encapsulates train, validation, and test dataset preparation in one place.
- Flexible Input: Accepts pre-split datasets as `(images, labels)` tuples for each phase.
- Custom Transforms: Integrates with Albumentations-based transforms from `transform.py` for preprocessing.
- Sampler Support: Optional random sampler for balanced training when dealing with class imbalance.
- Reproducibility: Automatically sets a fixed random seed to ensure consistent results across runs.

Usage:
    data_module = DataModule(
        train_data=(train_imgs, train_labels),
        val_data=(val_imgs, val_labels),
        test_data=(test_imgs, test_labels),
        batch_size=64,
        num_workers=4,
        use_sampler=True,
        seed=42
    )

    trainer.fit(model, datamodule=data_module)

Notes:
- The `train_dataloader`, `val_dataloader`, and `test_dataloader` methods return ready-to-use
  PyTorch DataLoader objects.
- Sampler usage is recommended for datasets with significant class imbalance.
"""


import torch
import pytorch_lightning as pl
from utils import make_random_sampler, set_seed
from transform import get_transform
from torch.utils.data import DataLoader
from dataset import CustomDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_data=None, 
                 val_data=None, 
                 test_data=None, 
                 batch_size=64, 
                 num_workers=2,
                 use_sampler=False,
                 seed = 42):
        
        # Initialize the parent class
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_sampler = use_sampler
        self.class_weights = None
        self.seed = seed
        set_seed(self.seed)
    
    # Setup method to create datasets and transformations
    # This is called by PyTorch Lightning Trainer
    def setup(self, stage=None):
        # set seed   
        set_seed(self.seed)
        # get transform
        train_T, val_T = get_transform()
        
        # Create datasets using the CustomDataset class
        # Assuming train_data and val_data are tuples of (image_paths, labels)
        if stage in ('fit', None):
            if self.train_data is not None and self.val_data is not None:
                train_images, train_labels = self.train_data
                val_images, val_labels = self.val_data
                
                self.train_dataset = CustomDataset(train_images, train_labels, transform=train_T)
                self.val_dataset = CustomDataset(val_images, val_labels, transform=val_T)
            
                self.train_labels = train_labels
                self.val_labels = val_labels
                
                # Sampler
                if self.use_sampler and self.train_labels is not None:  
                    self.train_sampler = make_random_sampler(torch.tensor(self.train_labels))
                else:
                    self.train_sampler = None
                
        if stage in ('test', None):
            if self.test_data is not None:
                test_images, test_labels = self.test_data
            else:
                test_images, test_labels = self.val_data
            self.test_dataset = CustomDataset(test_images, test_labels, transform=val_T)
        
    # Define DataLoaders for training and validation
    # These will be used by the Trainer during training/validation
    def train_dataloader(self):
        if self.use_sampler and hasattr(self, 'train_sampler') and self.train_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                sampler = self.train_sampler,
                num_workers = self.num_workers,
                pin_memory = True,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True,
                persistent_workers=True
            )
            
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers=True
            )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers=True
            )
    
        