# src/biomass2pred/train/run.py

import argparse
import torch
from pathlib import Path

from biomass2pred.config import load_config
from biomass2pred.utils.seed import seed_everything
from biomass2pred.data.datamodule import BiomassDataModule
from biomass2pred.models.factory import create_model
from biomass2pred.train.losses import create_loss
from biomass2pred.train.optimizers import create_optimizer
from biomass2pred.train.trainer import Trainer

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

def main(config_path: str):
    # 1. Load config
    cfg = load_config(PROJECT_DIR / config_path)

    # 2. Seed
    seed_everything(cfg['project']['seed'])

    # 3. Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Datamodule
    datamodule = BiomassDataModule(
        train_csv=PROJECT_DIR / cfg['data']['train_csv'],
        valid_csv=PROJECT_DIR / cfg['data']['valid_csv'],
        images_root=PROJECT_DIR / cfg['data']['images_root'],
        target_names=cfg['data']['target_names'],
        image_size=tuple(cfg['data']['image_size']),
        batch_size=cfg['dataloader']['batch_size'],
        num_workers=cfg['dataloader']['num_workers']
    )
    datamodule.setup()

    # 5. Model
    model = create_model(
        cfg['model']['backbone'],
        pretrained=cfg['model']['pretrained'],
        num_outputs=len(cfg['data']['target_names'])
    )
    model.to(device)

    # 6. Loss & optimizer
    criterion = create_loss(cfg['loss'])
    optimizer = create_optimizer(
        cfg['optimizer'],
        params=model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay']
    )

    # 7. Trainer
    output_dir = Path(cfg['train']['output_dir'])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir= PROJECT_DIR / output_dir,
        unfreeze_at_epoch=cfg['train']['unfreeze_at_epoch'],
        lr_multipliers=cfg['train']['lr_multipliers']
    )

    #8. Train
    trainer.fit(
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.valid_dataloader(),
        epochs=cfg['train']['epochs']
    )

    print('Training complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    args = parser.parse_args()
    main(args.config)
