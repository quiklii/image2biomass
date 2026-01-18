# src/biomass2pred/train/losses.py

import torch.nn as nn

LOSS_REGISTRY = {
    'mse' : nn.MSELoss,
    'mae' : nn.L1Loss
}

def create_loss(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f'Unknown loss function: {name}. Avaliable: {list(LOSS_REGISTRY.keys())}')
    return LOSS_REGISTRY[name](**kwargs)