# src/biomass2pred/data/transforms.py

import torchvision.transforms.v2 as T

def get_train_transform(image_size):
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.ToTensor(),
    ])

def get_valid_transform(image_size):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor()
    ])