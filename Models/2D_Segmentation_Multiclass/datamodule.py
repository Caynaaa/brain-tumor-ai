# --- import necessary library ---
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import set_seed
from transform import get_transform
from dataset import CustomDataset

# --- lightning datamodule class ---
class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for segmentation tasks.

    This module wraps training, validation, and test datasets into DataLoader objects.
    It uses the `CustomDataset` class for dataset creation and supports configurable 
    batch size, number of workers, and random seed.

    Args:
        train_data (list[str] | None): List of file paths for training data.
        val_data (list[str] | None): List of file paths for validation data.
        test_data (list[str] | None): List of file paths for test data.
        batch_size (int, optional): Batch size for DataLoader. Default is 64.
        num_workers (int, optional): Number of worker threads for DataLoader. Default is 2.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Methods:
        setup(stage=None):
            Initializes datasets based on the provided stage ('fit', 'test', or None).
        train_dataloader():
            Returns DataLoader for the training dataset.
        val_dataloader():
            Returns DataLoader for the validation dataset.
        test_dataloader():
            Returns DataLoader for the test dataset.
    """
    
    
    def __init__(self, train_data=None, val_data=None, test_data=None, batch_size=64, num_workers=2, seed=42, 
                 index_train_path=None, index_val_path=None, index_test_path=None, load_index=True):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.index_train_path = index_train_path
        self.index_val_path = index_val_path
        self.index_test_path = index_test_path
        self.load_index = load_index
        
    def setup(self, stage=None):
        # set_seed so can be deterministic
        set_seed(seed=self.seed)
        # get transform from transform.py
        train_T, val_T = get_transform()
    
        if stage in ('fit', None):
            # --- used CustomDataset for train and val ---
            self.train_dataset = CustomDataset(file_path=self.train_data, transform=train_T, debug=True, index_cache_path=self.index_train_path)
            self.val_dataset = CustomDataset(file_path=self.val_data, transform=val_T, debug=True, index_cache_path=self.index_val_path)
        
        if stage in ('test', None):
            # --- used CustomDataset for test ---
            self.test_dataset = CustomDataset(file_path=self.test_data, transform=val_T, debug=False)
    
    # --- DataLoader ---
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
            pin_memory = True,
            persistent_workers = True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory = True,
            persistent_workers = True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
            pin_memory = True,
            persistent_workers = True
        )
            