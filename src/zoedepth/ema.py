"""Exponential Moving Average of model weights for stable evaluation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _first_str_in_closure(fn) -> str | None:
    for cell in getattr(fn, '__closure__', None) or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if isinstance(v, str):
            return v
    return None


def _first_dict_in_closure(fn, exclude: dict) -> dict | None:
    for cell in getattr(fn, '__closure__', None) or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if isinstance(v, dict) and v is not exclude:
            return v
    return None


def _make_hook(name: str, target: dict):
    def hook(mod, inp, out):
        target[name] = out
    return hook


class ModelEMA:
    """The shadow copy is kept in eval mode with gradients disabled.
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
        # deepcopy severs two activation-dict patterns used by MiDaS:
        # A) utils.py global `activations` — hooks write to the global but
        #    pretrained.activations was copied to a fresh dict.  Fix: re-register
        #    hooks so they write to the module's own instance dict.
        # B) midas.py closure `bank` (core_out) — re-point the stale attribute to
        #    the dict still held in the hooks' closures.
        # core_out must be fixed first; new activations hooks also have a dict in
        # their closure and would otherwise be mistaken for a stale core_out.
        self._fix_closure_attr('core_out')
        self._fix_global_activations_attr('activations')

    def _fix_global_activations_attr(self, attr: str) -> None:
        for module in self.ema.modules():
            instance_dict = getattr(module, attr, None)
            if not isinstance(instance_dict, dict):
                continue
            for submod in module.modules():
                for hook_id, hook_fn in list(submod._forward_hooks.items()):
                    global_dict = getattr(hook_fn, '__globals__', {}).get(attr)
                    if not isinstance(global_dict, dict) or global_dict is instance_dict:
                        continue
                    name = _first_str_in_closure(hook_fn)
                    if name is None:
                        continue
                    submod._forward_hooks.pop(hook_id)
                    submod.register_forward_hook(_make_hook(name, instance_dict))
            break  # only one module owns this attr

    def _fix_closure_attr(self, attr: str) -> None:
        for module in self.ema.modules():
            current = getattr(module, attr, None)
            if not isinstance(current, dict):
                continue
            for submod in module.modules():
                for hook_fn in submod._forward_hooks.values():
                    stale = _first_dict_in_closure(hook_fn, exclude=current)
                    if stale is not None:
                        setattr(module, attr, stale)
                        break
                if getattr(module, attr) is not current:
                    break
            break  # only one module owns this attr

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.is_floating_point():
                v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return self.ema.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.ema.load_state_dict(state)
