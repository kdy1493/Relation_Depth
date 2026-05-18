"""Build ZoeDepth from YAML config via official ``get_config`` / ``build_model``."""

from __future__ import annotations

import torch.nn as nn

from ._vendor import ensure_zoedepth_on_path
from .yaml_config import ZoeDepthTrainConfig


def _normalize_pretrained_resource(s: str) -> str:
    s = s.strip()
    if s.startswith("url::") or s.startswith("local::"):
        return s
    return f"local::{s}"


def build_zoe_model(cfg: ZoeDepthTrainConfig) -> nn.Module:
    """Instantiate ZoeDepth. Uses ImageNet-normalized inputs from ``CustomDepthPairDataset`` with ``denorm=True`` at forward."""
    ensure_zoedepth_on_path()
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    z = get_config("zoedepth", "infer", version_name=cfg.version_name)

    z.midas_model_type = cfg.midas_model_type
    z.n_bins = cfg.n_bins
    z.bin_centers_type = cfg.bin_centers_type
    z.bin_embedding_dim = cfg.bin_embedding_dim
    z.n_attractors = cfg.n_attractors
    z.attractor_alpha = cfg.attractor_alpha
    z.attractor_gamma = cfg.attractor_gamma
    z.attractor_kind = cfg.attractor_kind
    z.attractor_type = cfg.attractor_type
    z.min_depth = cfg.min_depth
    z.max_depth = cfg.max_depth
    z.min_temp = cfg.min_temp
    z.max_temp = cfg.max_temp
    z.inverse_midas = cfg.inverse_midas
    z.memory_efficient = cfg.memory_efficient
    z.output_distribution = cfg.output_distribution
    z.train_midas = cfg.train_midas
    z.freeze_midas_bn = cfg.freeze_midas_bn
    z.midas_lr_factor = cfg.midas_lr_factor
    z.encoder_lr_factor = cfg.encoder_lr_factor
    z.pos_enc_lr_factor = cfg.pos_enc_lr_factor
    z.force_keep_ar = cfg.force_keep_ar
    z.img_size = [cfg.img_size, cfg.img_size]

    if cfg.pretrained_from:
        z.pretrained_resource = _normalize_pretrained_resource(cfg.pretrained_from)
        z.use_pretrained_midas = False
    else:
        z.pretrained_resource = None
        z.use_pretrained_midas = cfg.use_pretrained_midas

    return build_model(z)
