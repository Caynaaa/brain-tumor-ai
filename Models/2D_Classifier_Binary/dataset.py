import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_input, labels, transform=None):
        # Initialize the dataset with image paths, labels, and transformations
        self.img_input= img_input
        self.labels = labels
        self.transform = transform
    
    # Override the __len__ and __getitem__ methods
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
            if img_idx.ndim == 2:
                img = np.stack([img_idx]*3, axis=-1)
            else:
                img = img_idx
        else:
            raise ValueError("Unsupported image input type. Must be str, PIL Image, or numpy array.")
        
        # Get the corresponding label
        labels = self.labels[idx]
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Apply transformations if provided
        if self.transform:
            img = self.transform(image=img)['image']
        
        # Return the image tensor and label tensor
        return img, labels
