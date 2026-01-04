# src/biomass2pred/utils/metrics.py

import torch

_CSIRO_W = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32)

DEFAULT_ORDER3 = ('Dry_Clover_g', 'Dry_Green_g', 'Dry_Dead_g')

ORDER5 = ('Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g')

def expand_3_to_5(y3: torch.Tensor, order3=DEFAULT_ORDER3) -> torch.Tensor:
    """
    Convert (N, 3) to (N, 5) by reconstructing:
        GDM_g = Dry_Green_g + Dry_Clover_g
        Dry_Total_g = GDM_g + Dry_Dead_g
    """

    if y3.dim() != 2 or y3.shape[1] !=3:
        raise ValueError(f'Expected y3 shape (N, 3), got {tuple(y3.shape)}')

    clover = y3[:, 0]
    green = y3[:, 1]
    dead = y3[:, 2]

    gdm = green + clover
    total = gdm + dead

    y5 = torch.stack(([green, dead, clover, gdm, total]), dim=1)
    return y5

def weighted_r2(
        y_true3: torch.Tensor,
        y_pred3: torch.Tensor,
        order3=DEFAULT_ORDER3
):
    """
    Competition metric: weighted R^2 is computed using 5 targets (reconstructed from 3)
    """
    y_true = expand_3_to_5(y_true3, order3)
    y_pred = expand_3_to_5(y_pred3, order3)

    w = _CSIRO_W.to(device=y_true.device, dtype=y_true.dtype).unsqueeze(0).expand_as(y_true)

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    w = w.reshape(-1)

    yw = torch.sum(w * y_true) / torch.sum(w)
    ss_res = torch.sum(w * (y_true - y_pred)**2)
    ss_tot = torch.sum(w * (y_true - yw)**2)

    return 1 - ss_res / ss_tot



