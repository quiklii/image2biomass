# src/biomass2pred/utils/metrics.py

import torch

_CSIRO_W = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32)

DEFAULT_ORDER3 = ('Dry_Dead_g', 'Dry_Green_g', 'Dry_Dead_g')

ORDER5 = ('Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g')

def expand_3_to_5(y3: torch.Tensor, order3=DEFAULT_ORDER3) -> torch.Tensor:
    '''
    Convert (N, 3) to (N, 5) by reconstruting:
        GDM_g = Dry_Green_g + Dry_Clover_g
        Dry_Total_g = GDM_g + Dry_Dead_g
    '''

    if y3.dim() != 2 or y3.shape[1] !=3:
        raise ValueError(f'Expected y3 shape (N, 3), got {tuple(y3.shape)}')

    idx = {}

