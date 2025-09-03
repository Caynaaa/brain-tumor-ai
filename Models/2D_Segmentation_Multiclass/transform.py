"""
This module defines the transformation pipelines used for brain tumor 
image classification. It provides separate augmentations for training 
and validation phases using the Albumentations library.

These transformations include resizing, normalization (using ImageNet stats),
and various data augmentation techniques (e.g., flipping, noise, distortion).
"""


# Import necessary libraries
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(img_size=(224, 224)):
    # Define the training transformations
    train_transform = A.Compose([
        # Resize the image to the target size
        A.Resize(*img_size),
        # Random horizontal and vertical flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Slight brightness/contrast variation
        A.RandomBrightnessContrast(p=0.2),  
        # Grid-based distortion
        A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.2),  
        # Add Gaussian noise
        A.GaussNoise(p=0.1), 
        # Normalize using ImageNet stats
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
        # Convert to PyTorch tensor
        ToTensorV2() 
    ])
    
    # Define the validation transformations
    val_transform = A.Compose([
        A.Resize(*img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
        A.ToTensorV2() 
    ])
    
    # Return the training and validation transformations
    return train_transform, val_transform