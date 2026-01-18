# src/biomass2pred/data/dataset.py

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class BiomassRegressionDataset(Dataset):
    """
    Dataset for biomass regression tasks.

    The data is ungrouped and each image appears multiple times once per each target.
    The targets for this task are: [Dry_Clover_g, Dry_Green_h, Dry_Dead_g, Dry_Total_g, GSM_g],
    since they are not independent we need only three of them.
    """

    def __init__(
            self,
            csv_path: str | Path,
            images_root: str | Path,
            target_names: list[str],
            transform=None
    ):
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.target_names = target_names
        self.transform = transform

        self.image_paths = list(self.df['image_path'].unique())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        rows = self.df[self.df['image_path'] == image_path]

        image = Image.open(self.images_root / image_path)
        if self.transform:
            image = self.transform(image)

        target_map = {
            row['target_name'] : row['target']
            for _, row in rows.iterrows()
        }

        try:
            targets = torch.tensor(
                [target_map[name] for name in self.target_names],
                dtype=torch.float32
            )
        except KeyError as e:
            raise KeyError(
                f'Missing target {e} for image {image_path}'
            )

        return image, targets

