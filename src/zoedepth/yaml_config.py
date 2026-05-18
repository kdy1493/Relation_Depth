"""YAML + CLI configuration for ZoeDepth fine-tuning (SynTable-style layout)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

_VALID_METRICS = {"d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse boolean from string: {value!r}")
    return bool(value)


@dataclass
class ZoeDepthTrainConfig:
    train_list: str
    val_list: str
    list_root: str | None
    depth_scale: float
    version_name: str
    midas_model_type: str
    n_bins: int
    bin_centers_type: str
    bin_embedding_dim: int
    n_attractors: list[int]
    attractor_alpha: float
    attractor_gamma: float
    attractor_kind: str
    attractor_type: str
    min_depth: float
    max_depth: float
    min_temp: float
    max_temp: float
    inverse_midas: bool
    memory_efficient: bool
    output_distribution: str
    pretrained_from: str | None
    train_midas: bool
    use_pretrained_midas: bool
    freeze_midas_bn: bool
    midas_lr_factor: float
    encoder_lr_factor: float
    pos_enc_lr_factor: float
    force_keep_ar: bool
    img_size: int
    epochs: int
    bs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int | None
    find_unused_parameters: bool
    eval_freq: int
    grad_accum_steps: int
    amp: bool
    patience: int | None
    ema_decay: float
    monitor_metric: str
    monitor_mode: str
    auto_batch_size: bool
    scale_align_eval: bool
    save_path: str
    port: int | None
    cuda_visible_devices: list[int] | None
    cuda_indices: list[int] | None
    resume_from: str | None


def _require(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(".".join(path))
        cur = cur[p]
    return cur


def load_train_config_yaml(path: str | Path) -> ZoeDepthTrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    data = raw.get("data") or {}
    model = raw.get("model") or {}
    training = raw.get("training") or {}
    output = raw.get("output") or {}
    dist = raw.get("distributed") or {}
    device = raw.get("device") or {}

    monitor_metric = str(training.get("monitor_metric", "abs_rel"))
    if monitor_metric not in _VALID_METRICS:
        raise ValueError(f"training.monitor_metric must be one of {_VALID_METRICS}, got {monitor_metric!r}")

    monitor_mode = str(training.get("monitor_mode", "min"))
    if monitor_mode not in ("min", "max"):
        raise ValueError(f"training.monitor_mode must be 'min' or 'max', got {monitor_mode!r}")

    na = model.get("n_attractors", [16, 8, 4, 1])
    if isinstance(na, str):
        n_attractors = [int(x.strip()) for x in na.split(",")]
    else:
        n_attractors = [int(x) for x in na]

    cuda_visible_devices = (
        [int(x) for x in device["cuda_visible_devices"]]
        if device.get("cuda_visible_devices") not in (None, "")
        else None
    )
    cuda_indices = (
        [int(x) for x in device["cuda_indices"]]
        if device.get("cuda_indices") not in (None, "")
        else None
    )
    if cuda_visible_devices is not None and cuda_indices is None:
        cuda_indices = list(range(len(cuda_visible_devices)))
    if cuda_visible_devices is not None and cuda_indices is not None:
        visible_set = set(cuda_visible_devices)
        if all(i in visible_set for i in cuda_indices):
            index_map = {physical: local for local, physical in enumerate(cuda_visible_devices)}
            cuda_indices = [index_map[i] for i in cuda_indices]

    return ZoeDepthTrainConfig(
        train_list=str(_require(data, "train_list")),
        val_list=str(_require(data, "val_list")),
        list_root=(str(data["list_root"]) if data.get("list_root") not in (None, "") else None),
        depth_scale=float(data.get("depth_scale", 1.0)),
        version_name=str(model.get("version_name", "v1")),
        midas_model_type=str(model.get("midas_model_type", "DPT_BEiT_L_384")),
        n_bins=int(model.get("n_bins", 64)),
        bin_centers_type=str(model.get("bin_centers_type", "softplus")),
        bin_embedding_dim=int(model.get("bin_embedding_dim", 128)),
        n_attractors=n_attractors,
        attractor_alpha=float(model.get("attractor_alpha", 1000.0)),
        attractor_gamma=float(model.get("attractor_gamma", 2.0)),
        attractor_kind=str(model.get("attractor_kind", "mean")),
        attractor_type=str(model.get("attractor_type", "inv")),
        min_depth=float(model.get("min_depth", 1e-3)),
        max_depth=float(model.get("max_depth", 10.0)),
        min_temp=float(model.get("min_temp", 0.0212)),
        max_temp=float(model.get("max_temp", 50.0)),
        inverse_midas=_parse_bool(model.get("inverse_midas", False)),
        memory_efficient=_parse_bool(model.get("memory_efficient", True)),
        output_distribution=str(model.get("output_distribution", "logbinomial")),
        pretrained_from=(str(model["pretrained_from"]) if model.get("pretrained_from") not in (None, "") else None),
        train_midas=_parse_bool(model.get("train_midas", True)),
        use_pretrained_midas=_parse_bool(model.get("use_pretrained_midas", True)),
        freeze_midas_bn=_parse_bool(model.get("freeze_midas_bn", True)),
        midas_lr_factor=float(model.get("midas_lr_factor", 1.0)),
        encoder_lr_factor=float(model.get("encoder_lr_factor", 10.0)),
        pos_enc_lr_factor=float(model.get("pos_enc_lr_factor", 10.0)),
        force_keep_ar=_parse_bool(model.get("force_keep_ar", True)),
        img_size=int(training.get("img_size", 512)),
        epochs=int(training.get("epochs", 50)),
        bs=int(training.get("batch_size", training.get("bs", 2))),
        lr=float(training.get("lr", 1.61e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        num_workers=int(training.get("num_workers", 4)),
        seed=(int(training["seed"]) if training.get("seed") is not None else None),
        find_unused_parameters=_parse_bool(training.get("find_unused_parameters", True)),
        eval_freq=int(training.get("eval_freq", 2)),
        grad_accum_steps=max(1, int(training.get("grad_accum_steps", 1))),
        amp=_parse_bool(training.get("amp", True)),
        patience=(int(training["patience"]) if training.get("patience") is not None else None),
        ema_decay=float(training.get("ema_decay", 0.9998)),
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        auto_batch_size=_parse_bool(training.get("auto_batch_size", False)),
        scale_align_eval=_parse_bool(training.get("scale_align_eval", True)),
        save_path=str(_require(output, "save_path")),
        port=(int(dist["port"]) if dist.get("port") not in (None, "") else None),
        cuda_visible_devices=cuda_visible_devices,
        cuda_indices=cuda_indices,
        resume_from=(str(model["resume_from"]) if model.get("resume_from") not in (None, "") else None),
    )


def build_train_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ZoeDepth from YAML config")
    p.add_argument("--config", "-c", type=str, required=True)
    for flag in ("train_list", "val_list", "list_root", "save_path", "resume_from", "pretrained_from"):
        p.add_argument(f"--{flag.replace('_', '-')}", type=str, default=None)
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--bs", "--batch-size", type=int, default=None, dest="bs")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--find-unused-parameters", type=_parse_bool, default=None, metavar="{true,false}")
    p.add_argument("--eval-freq", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--amp", type=_parse_bool, default=None, metavar="{true,false}")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--monitor-metric", type=str, choices=sorted(_VALID_METRICS), default=None)
    p.add_argument("--monitor-mode", type=str, choices=["min", "max"], default=None)
    p.add_argument("--auto-batch-size", type=_parse_bool, default=None, metavar="{true,false}")
    p.add_argument("--scale-align-eval", type=_parse_bool, default=None, metavar="{true,false}")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--cuda-visible-devices", type=int, nargs="+", default=None)
    p.add_argument("--cuda-indices", type=int, nargs="+", default=None)
    return p


def apply_cli_overrides(cfg: ZoeDepthTrainConfig, args: argparse.Namespace) -> ZoeDepthTrainConfig:
    overrides: dict[str, Any] = {}
    for f in fields(ZoeDepthTrainConfig):
        if hasattr(args, f.name):
            v = getattr(args, f.name)
            if v is not None:
                overrides[f.name] = v
    out = replace(cfg, **overrides)
    if out.eval_freq < 1:
        raise ValueError("eval_freq must be >= 1")
    if out.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    return out


def train_config_from_args(args: argparse.Namespace) -> ZoeDepthTrainConfig:
    return apply_cli_overrides(load_train_config_yaml(args.config), args)
