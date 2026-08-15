from __future__ import annotations

import torch
import torch.nn.functional as F


STAGE1_LOSS_WEIGHTS = {"point": 0.10, "conditional": 0.50, "regional": 0.20, "low_frequency": 0.15, "total_variation": 0.03}


def stage1_composite_loss(pred: torch.Tensor, target: torch.Tensor, lead: torch.Tensor,
                          huber_beta: float = 0.30, regional_block: int = 16,
                          lowpass_kernel: int = 9, conditional_min_group: int = 4):
    residual = pred - target
    point = F.smooth_l1_loss(pred, target, reduction="none", beta=huber_beta).mean()
    groups = []
    for value in torch.unique(lead):
        selected = residual[lead == value]
        if selected.shape[0] >= conditional_min_group:
            groups.append(selected.mean(dim=0).abs().mean())
    conditional = torch.stack(groups).mean() if groups else residual.new_zeros(())
    regional = F.avg_pool2d(residual, regional_block, regional_block).abs().mean()
    pad = lowpass_kernel // 2
    low = F.avg_pool2d(F.pad(residual, (pad, pad, pad, pad), mode="reflect"), lowpass_kernel, 1).abs().mean()
    dx = pred[..., 1:] - pred[..., :-1]
    dy = pred[..., 1:, :] - pred[..., :-1, :]
    tv = dx.abs().mean() + dy.abs().mean()
    parts = {"point": point, "conditional": conditional, "regional": regional,
             "low_frequency": low, "total_variation": tv}
    total = sum(STAGE1_LOSS_WEIGHTS[name] * value for name, value in parts.items())
    return total, parts
