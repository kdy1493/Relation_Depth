"""Checkpoint loaders for depth backends (model-agnostic visualization entry point)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKEND_CHOICES = ("depth_anything_v2", "zoedepth", "adabins")


@dataclass
class VisualizeDataSettings:
    val_list: str
    list_root: str | None
    depth_scale: float
    img_size: int
    min_depth: float
    max_depth: float


class DepthPredictor(ABC):
    def __init__(self, model: nn.Module, settings: VisualizeDataSettings, device: torch.device) -> None:
        self.model = model
        self.settings = settings
        self.device = device

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
        *,
        encoder: str | None = None,
    ) -> DepthPredictor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, image_bchw: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        image_bchw = image_bchw.to(self.device).float()
        pred = self._forward(image_bchw)
        if pred.ndim == 4:
            pred = pred[:, 0]
        elif pred.ndim != 3:
            raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")
        pred = F.interpolate(pred[:, None], target_hw, mode="bilinear", align_corners=True)[:, 0, 0]
        return pred

    @abstractmethod
    def _forward(self, image_bchw: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DepthAnythingV2Predictor(DepthPredictor):
    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
        *,
        encoder: str | None = None,
    ) -> DepthAnythingV2Predictor:
        from depthanything.checkpoints import unwrap_state_dict
        from depthanything.da2_vendor.depth_anything_v2.dpt import DepthAnythingV2
        from depthanything.model_configs import MODEL_CONFIGS
        from depthanything.yaml_config import load_train_config_yaml

        cfg = load_train_config_yaml(config_path)
        enc = encoder if encoder is not None else cfg.encoder
        model_cfg = {**MODEL_CONFIGS[enc], "max_depth": cfg.max_depth}
        model = DepthAnythingV2(**model_cfg).to(device)
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = unwrap_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
        model.load_state_dict(state, strict=True)
        model.eval()
        settings = VisualizeDataSettings(
            val_list=cfg.val_list,
            list_root=cfg.list_root,
            depth_scale=cfg.depth_scale,
            img_size=cfg.img_size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
        )
        return cls(model, settings, device)

    def _forward(self, image_bchw: torch.Tensor) -> torch.Tensor:
        return self.model(image_bchw)


class ZoeDepthPredictor(DepthPredictor):
    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
        *,
        encoder: str | None = None,
    ) -> ZoeDepthPredictor:
        from zoedepth.checkpoints import unwrap_state_dict
        from zoedepth.yaml_config import load_train_config_yaml
        from zoedepth.zoe_model import build_zoe_model

        cfg = load_train_config_yaml(config_path)
        model = build_zoe_model(cfg).to(device)
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = unwrap_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
        model.load_state_dict(state, strict=True)
        model.eval()
        settings = VisualizeDataSettings(
            val_list=cfg.val_list,
            list_root=cfg.list_root,
            depth_scale=cfg.depth_scale,
            img_size=cfg.img_size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
        )
        return cls(model, settings, device)

    def _forward(self, image_bchw: torch.Tensor) -> torch.Tensor:
        out = self.model(image_bchw, denorm=True)
        return out["metric_depth"]


class AdaBinsPredictor(DepthPredictor):
    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
        *,
        encoder: str | None = None,
    ) -> AdaBinsPredictor:
        from adabins.checkpoints import unwrap_state_dict
        from adabins.model import UnetAdaptiveBins
        from adabins.yaml_config import load_train_config_yaml

        cfg = load_train_config_yaml(config_path)
        model = UnetAdaptiveBins.build(
            n_bins=cfg.n_bins,
            min_val=cfg.min_depth,
            max_val=cfg.max_depth,
            norm=cfg.norm,
            backend_name=cfg.backend_name,
            backend_pretrained=False,
        ).to(device)
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = unwrap_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
        model.load_state_dict(state, strict=True)
        model.eval()
        settings = VisualizeDataSettings(
            val_list=cfg.val_list,
            list_root=cfg.list_root,
            depth_scale=cfg.depth_scale,
            img_size=cfg.img_size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
        )
        return cls(model, settings, device)

    def _forward(self, image_bchw: torch.Tensor) -> torch.Tensor:
        _, pred = self.model(image_bchw)
        return pred


_PREDICTOR_TYPES: dict[str, type[DepthPredictor]] = {
    "depth_anything_v2": DepthAnythingV2Predictor,
    "zoedepth": ZoeDepthPredictor,
    "adabins": AdaBinsPredictor,
}


def build_predictor(
    backend: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    encoder: str | None = None,
) -> DepthPredictor:
    key = backend.strip().lower()
    if key not in _PREDICTOR_TYPES:
        raise ValueError(f"Unknown backend {backend!r}. Choose from: {', '.join(BACKEND_CHOICES)}")
    return _PREDICTOR_TYPES[key].from_checkpoint(config_path, checkpoint_path, device, encoder=encoder)


def median_scale_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_median = torch.median(pred)
    target_median = torch.median(target)
    scale = target_median / pred_median.clamp_min(1e-6)
    return pred * scale
