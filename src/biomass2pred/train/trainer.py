# src/biomass2pred/train/trainer.py

import torch
from pathlib import Path

from anyio import sleep_until

from biomass2pred.utils.metrics import weighted_r2

class Trainer:
    def __init__(
            self,
            run_name: str,
            model,
            optimizer,
            criterion,
            device,
            output_dir: str | Path,
            unfreeze_at_epoch: int | None = None,
            lr_multipliers: dict | None = None
    ):
        self.run_name = run_name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.lr_multipliers = lr_multipliers or {}
        self.best_epoch = None
        self.current_epoch = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': []
        }

    def _freeze_all(self):
        """
        Freezes all parameters of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        print('Freezing all parameters.')

    def _unfreeze_linear(self):
        """
        Unfreezes the linear layers of the model.
        """
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            print(f'Unfreezing linear layer (classifier) for training.')
            self._rebuild_optimizer()

    def _unfreeze_conv_head(self):
        """
        Unfreezes the convolutional head of the model.
        """
        if hasattr(self.model, 'conv_head') and hasattr(self.model, 'bn2'):
            for param in self.model.conv_head.parameters():
                param.requires_grad = True
            for param in self.model.bn2.parameters():
                param.requires_grad = True
            print(f'Unfreezing convolutional head for training.')
            self._rebuild_optimizer()

    def _rebuild_optimizer(self):
        """
        Rebuilds the optimizer with current trainable parameters.
        """
        old_state = self.optimizer.state_dict()
        old_config = old_state['param_groups'][0].copy()
        base_lr = old_config.pop('lr')
        old_config.pop('params')

        # Remove internal PyTorch parameters that aren't constructor arguments
        # Keep only valid optimizer kwargs
        valid_keys = {'weight_decay', 'betas', 'eps', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused'}
        old_config = {k: v for k, v in old_config.items() if k in valid_keys}

        param_groups = []

        # HEAD
        mods = []
        for name in ('conv_head', 'bn2'):
            m = getattr(self.model, name, None)
            if m is not None:
                mods.append(m)
        params = [p for m in mods for p in m.parameters() if p.requires_grad]
        if params:
            lr_multiplier = self.lr_multipliers.get('head', 0.1)
            param_groups.append({
                'params': params,
                'lr': base_lr * lr_multiplier,
                'name': 'head'
            })

        # CLASSIFIER
        if hasattr(self.model, 'classifier'):
            classifier_params = [
                p for p in self.model.classifier.parameters() if p.requires_grad
            ]
            param_groups.append({
                'params' : classifier_params,
                'lr' : base_lr,
                'name': 'classifier'
            })

        optimizer_class = type(self.optimizer)
        self.optimizer = optimizer_class(param_groups, **old_config)

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
                'history' : self.history,
                'unfreeze_at_epoch' : self.unfreeze_at_epoch
            },
            checkpoint_path
        )

    def fit(self, train_loader, val_loader, epochs, metric_fn=weighted_r2):
        self._freeze_all()
        self._unfreeze_linear()

        print('Training on device:', self.device, '\n')
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            if self.unfreeze_at_epoch and epoch == self.unfreeze_at_epoch:
                self._unfreeze_conv_head()
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

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch, self.run_name + '_best')
                msg += ' (best)'
            print(msg)

        if self.best_epoch == self.current_epoch:
            self.save_checkpoint(self.current_epoch, self.run_name + '_best_last')
        else:
            self.save_checkpoint(self.best_epoch, self.run_name + '_last')





