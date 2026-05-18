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
        self._fix_activation_hook_refs()

    def _fix_activation_hook_refs(self) -> None:
        # deepcopy breaks two activation-dict patterns used by MiDaS:
        #   1. utils.py  – `activations` is a module-level global; hooks write to
        #      the global but pretrained.activations was copied to a new empty dict.
        #      Fix: re-point via hook.__globals__['activations'].
        #   2. midas.py  – `core_out` is an instance dict passed as closure arg
        #      `bank`; hooks still reference the original dict after deepcopy.
        #      Fix: re-point via the dict found in hook.__closure__.
        for attr in ('activations', 'core_out'):
            for module in self.ema.modules():
                current = getattr(module, attr, None)
                if not isinstance(current, dict):
                    continue
                for submod in module.modules():
                    for hook_fn in submod._forward_hooks.values():
                        # Pattern 1: closure variable (core_out / bank)
                        for cell in getattr(hook_fn, '__closure__', None) or ():
                            try:
                                v = cell.cell_contents
                            except ValueError:
                                continue
                            if isinstance(v, dict) and v is not current:
                                setattr(module, attr, v)
                                break
                        else:
                            # Pattern 2: module-level global (activations)
                            v = getattr(hook_fn, '__globals__', {}).get(attr)
                            if isinstance(v, dict) and v is not current:
                                setattr(module, attr, v)
                        if getattr(module, attr) is not current:
                            break
                    if getattr(module, attr) is not current:
                        break
                break  # one module per attr is enough

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
