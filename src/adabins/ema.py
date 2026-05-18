"""Exponential Moving Average of model weights for stable evaluation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains an EMA shadow copy of a model used exclusively for evaluation.

    The shadow copy is kept in eval mode with gradients disabled.
    Call ``update()`` after every optimizer step.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, device: torch.device | None = None) -> None:
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.is_floating_point():
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return self.ema.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.ema.load_state_dict(state)
