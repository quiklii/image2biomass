# src/biomass2pred/train/optimizers.py

import torch.optim as optim

OPTIMIZER_REGISTRY = {
    'adamw' : optim.AdamW
}

def create_optimizer(name: str, **kwargs) -> optim.Optimizer:
    if name not in OPTIMIZER_REGISTRY:
        ValueError(f'Unknown optimizer: {name}. Avaliable: {list(OPTIMIZER_REGISTRY.keys())}')
    return OPTIMIZER_REGISTRY[name](**kwargs)