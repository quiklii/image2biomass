# src/biomass2pred/data/transforms.py

from torch.utils.data import DataLoader
from .dataset import BiomassRegressionDataset
from .transforms import get_train_transform, get_valid_transform

class BiomassDataModule:
    '''
        Datamodule-like class:
            - creates dataset
            - creates dataloaders
            - centralizes all data-related logic
    '''

    def __init__(
            self,
            train_csv,
            valid_csv,
            images_root,
            target_names,
            image_size,
            batch_size,
            num_workers
    ):
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.images_root = images_root
        self.target_names = target_names
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset=None
        self.valid_dataset=None

    def setup(self):
        '''
        Creates datasets and dataloaders. Call ONCE before training.
        '''
        self.train_dataset = BiomassRegressionDataset(
            csv_path=self.train_csv,
            images_root=self.images_root,
            target_names=self.target_names,
            transform=get_train_transform(self.image_size))

        self.valid_dataset = BiomassRegressionDataset(
            csv_path=self.valid_csv,
            images_root=self.images_root,
            target_names=self.target_names,
            transform=get_valid_transform(self.image_size))

    def train_dataloader(self):
        assert self.train_dataset is not None, 'You need to call setup() before'

        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def valid_dataloader(self):
        assert self.valid_dataset is not None, 'You need to call setup() before'

        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

