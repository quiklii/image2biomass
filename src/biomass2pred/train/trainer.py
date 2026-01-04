# src/biomass2pred/train/trainer.py

import torch
from pathlib import Path

from biomass2pred.utils.metrics import weighted_r2

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
            'val_metric': []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader, metric_fn=weighted_r2):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)

                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()

                all_preds.append(y_pred.cpu())
                all_targets.append(y_batch.cpu())

            val_loss = total_loss / len(val_loader)

            metric_value = None
            if metric_fn is not None:
                y_pred = torch.cat(all_preds, dim=0)
                y_true = torch.cat(all_targets, dim=0)
                metric_value = metric_fn(y_pred, y_true)
            return val_loss, metric_value

    def save_checkpoint(self, epoch, model_name):
        checkpoint_path = self.output_dir / f'{model_name}_epoch_{epoch:03d}.pt'
        torch.save(
            {
                'epoch' : epoch,
                'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict' : self.optimizer.state_dict(),
                'history' : self.history
            },
            checkpoint_path
        )

    def fit(self, train_loader, val_loader, epochs, metric_fn=weighted_r2):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, metric_value = self.validate(val_loader, metric_fn)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metric'].append(metric_value)

            msg = (
                f'Epoch: [{epoch}/{epochs}] '
                f'Train loss: {train_loss:.4f} '
                f'Validation loss: {val_loss:.4f} '
            )

            if metric_fn is not None:
                msg += f'Validation R^2: {metric_value:.4f}'
            print(msg)



