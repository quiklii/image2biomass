# src/biomass2pred/train/trainer.py

import timm
from pathlib import Path

class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            criterion,
            device,
            output_dir: str | Path
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
