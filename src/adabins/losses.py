"""Losses for AdaBins training."""

from __future__ import annotations

import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    """Main dense loss used in the AdaBins paper."""

    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if mask is not None and mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]

        if pred.shape[-2:] != target.shape[-2:]:
            pred = nn.functional.interpolate(
                pred.unsqueeze(1),
                target.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        if pred.numel() <= 1 or target.numel() <= 1:
            # torch.var with Bessel correction returns NaN for a single element.
            return pred.new_zeros(())
        pred = pred.clamp_min(self.eps)
        target = target.clamp_min(self.eps)
        g = torch.log(pred) - torch.log(target)
        dg = torch.var(g, correction=0) + 0.15 * torch.pow(torch.mean(g), 2)
        dg = dg.clamp_min(0.0)
        return 10.0 * torch.sqrt(dg)
