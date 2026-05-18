"""Checkpoint and pretrained weight loading for Depth Anything V2 (metric)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def unwrap_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_state(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    blob = torch.load(path, map_location=map_location)
    if isinstance(blob, dict) and "model" in blob:
        return unwrap_state_dict(blob["model"])
    return unwrap_state_dict(blob)


def load_pretrained_encoder(model: nn.Module, pretrained_path: str) -> None:
    """Load only DINOv2 encoder weights (keys containing ``pretrained``), as in upstream ``train.py``."""
    state = load_checkpoint_state(pretrained_path)
    enc = {k: v for k, v in state.items() if "pretrained" in k}
    missing, unexpected = model.load_state_dict(enc, strict=False)
    if not enc:
        raise RuntimeError(
            f"No 'pretrained' keys found in {pretrained_path}. "
            "Use the official Depth Anything V2 checkpoint (e.g. depth_anything_v2_vitl.pth)."
        )
    # Missing head weights are expected when starting from relative-depth ckpt.
    _ = missing, unexpected


def maybe_sync_batchnorm(model: nn.Module, distributed: bool) -> nn.Module:
    if distributed and torch.cuda.is_available():
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model
