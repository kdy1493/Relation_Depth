"""Load training configuration from YAML with optional CLI overrides."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

from depthanything.model_configs import MODEL_CONFIGS

_VALID_METRICS = {"d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse boolean from string: {value!r}")
    return bool(value)


@dataclass
class TrainConfig:
    """Flat training settings (YAML + CLI overrides merged into this)."""

    # data
    train_list: str
    val_list: str
    list_root: str | None
    depth_scale: float
    # model
    encoder: str
    max_depth: float
    pretrained_from: str | None
    resume_from: str | None
    # training – core
    img_size: int
    min_depth: float
    epochs: int
    bs: int
    lr: float
    num_workers: int
    seed: int | None
    find_unused_parameters: bool
    # training – new features
    eval_freq: int           # evaluate every N epochs
    grad_accum_steps: int    # gradient accumulation (effective_bs = bs × grad_accum_steps)
    amp: bool                # automatic mixed precision
    patience: int | None     # early stopping patience in evals (None = disabled)
    ema_decay: float         # EMA decay (0.0 = disabled)
    monitor_metric: str      # metric to watch for best ckpt / early stopping
    monitor_mode: str        # "min" or "max"
    auto_batch_size: bool    # OOM probe + auto grad_accum adjustment
    scale_align_eval: bool   # median-scale align prediction to GT before metrics
    # output / distributed
    save_path: str
    port: int | None
    cuda_visible_devices: list[int] | None
    cuda_indices: list[int] | None


def _require(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(".".join(path))
        cur = cur[p]
    return cur


def load_train_config_yaml(path: str | Path) -> TrainConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    data     = raw.get("data")        or {}
    model    = raw.get("model")       or {}
    training = raw.get("training")    or {}
    output   = raw.get("output")      or {}
    dist     = raw.get("distributed") or {}
    device   = raw.get("device")      or {}

    train_list = _require(data, "train_list")
    val_list   = _require(data, "val_list")
    if not isinstance(train_list, str) or not isinstance(val_list, str):
        raise TypeError("data.train_list and data.val_list must be strings")

    encoder = str(model.get("encoder", "vitl"))
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"model.encoder must be one of {list(MODEL_CONFIGS)}, got {encoder!r}")

    monitor_metric = str(training.get("monitor_metric", "rmse"))
    if monitor_metric not in _VALID_METRICS:
        raise ValueError(f"training.monitor_metric must be one of {_VALID_METRICS}, got {monitor_metric!r}")

    monitor_mode = str(training.get("monitor_mode", "min"))
    if monitor_mode not in ("min", "max"):
        raise ValueError(f"training.monitor_mode must be 'min' or 'max', got {monitor_mode!r}")

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
    if cuda_visible_devices is not None:
        if cuda_indices is None:
            # When visibility is fixed in YAML, default to local PyTorch indices.
            cuda_indices = list(range(len(cuda_visible_devices)))
        else:
            # Accept both local indices ([0,1]) and physical indices ([3,4]) when
            # cuda_visible_devices is set. Normalize to local PyTorch indices.
            visible_set = set(cuda_visible_devices)
            if all(i in visible_set for i in cuda_indices):
                index_map = {physical: local for local, physical in enumerate(cuda_visible_devices)}
                cuda_indices = [index_map[i] for i in cuda_indices]

    return TrainConfig(
        # data
        train_list=str(train_list),
        val_list=str(val_list),
        list_root=(str(data["list_root"]) if data.get("list_root") not in (None, "") else None),
        depth_scale=float(data.get("depth_scale", 1.0)),
        # model
        encoder=encoder,
        max_depth=float(model.get("max_depth", 20.0)),
        pretrained_from=(str(model["pretrained_from"]) if model.get("pretrained_from") not in (None, "") else None),
        resume_from=(str(model["resume_from"]) if model.get("resume_from") not in (None, "") else None),
        # training – core
        img_size=int(training.get("img_size", 518)),
        min_depth=float(training.get("min_depth", 0.001)),
        epochs=int(training.get("epochs", 40)),
        bs=int(training.get("batch_size", training.get("bs", 2))),
        lr=float(training.get("lr", 5e-6)),
        num_workers=int(training.get("num_workers", 4)),
        seed=(int(training["seed"]) if training.get("seed") is not None else None),
        find_unused_parameters=_parse_bool(training.get("find_unused_parameters", True)),
        # training – new features
        eval_freq=int(training.get("eval_freq", 1)),
        grad_accum_steps=max(1, int(training.get("grad_accum_steps", 1))),
        amp=_parse_bool(training.get("amp", False)),
        patience=(int(training["patience"]) if training.get("patience") is not None else None),
        ema_decay=float(training.get("ema_decay", 0.0)),
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        auto_batch_size=_parse_bool(training.get("auto_batch_size", False)),
        scale_align_eval=_parse_bool(training.get("scale_align_eval", False)),
        # output / distributed
        save_path=str(_require(output, "save_path")),
        port=(int(dist["port"]) if dist.get("port") not in (None, "") else None),
        cuda_visible_devices=cuda_visible_devices,
        cuda_indices=cuda_indices,
    )


def build_train_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune Depth Anything V2 from a YAML config")
    p.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    # data
    p.add_argument("--train-list",   type=str,   default=None)
    p.add_argument("--val-list",     type=str,   default=None)
    p.add_argument("--list-root",    type=str,   default=None)
    p.add_argument("--depth-scale",  type=float, default=None)
    # model
    p.add_argument("--encoder",          type=str,   default=None, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--max-depth",        type=float, default=None)
    p.add_argument("--pretrained-from",  type=str,   default=None)
    p.add_argument("--resume-from",      type=str,   default=None)
    # training – core
    p.add_argument("--img-size",     type=int,   default=None)
    p.add_argument("--min-depth",    type=float, default=None)
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--bs", "--batch-size", type=int, default=None, dest="bs")
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--num-workers",  type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=None)
    # training – new features
    p.add_argument("--eval-freq",        type=int,   default=None, help="Evaluate every N epochs")
    p.add_argument("--grad-accum-steps", type=int,   default=None, help="Gradient accumulation steps")
    p.add_argument("--amp",              type=_parse_bool, default=None, metavar="{true,false}")
    p.add_argument("--patience",         type=int,   default=None, help="Early stopping patience (evals)")
    p.add_argument("--ema-decay",        type=float, default=None)
    p.add_argument("--monitor-metric",   type=str,   default=None, choices=sorted(_VALID_METRICS))
    p.add_argument("--monitor-mode",     type=str,   default=None, choices=["min", "max"])
    p.add_argument("--auto-batch-size",  type=_parse_bool, default=None, metavar="{true,false}")
    p.add_argument("--scale-align-eval", type=_parse_bool, default=None, metavar="{true,false}")
    # output / distributed
    p.add_argument("--save-path",  type=str, default=None)
    p.add_argument("--port",       type=int, default=None)
    p.add_argument(
        "--cuda-visible-devices", type=int, nargs="+", default=None,
        help="Physical GPUs to expose via CUDA_VISIBLE_DEVICES (e.g. 3 4)",
    )
    p.add_argument(
        "--cuda-indices", type=int, nargs="+", default=None,
        help="PyTorch CUDA indices, e.g. --cuda-indices 0 1",
    )
    return p


def apply_cli_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Replace TrainConfig fields with non-None CLI values."""
    overrides: dict[str, Any] = {}
    for f in fields(TrainConfig):
        name = f.name
        if not hasattr(args, name):
            continue
        v = getattr(args, name)
        if v is not None:
            overrides[name] = v
    out = replace(cfg, **overrides)

    # post-override validation
    if out.encoder not in MODEL_CONFIGS:
        raise ValueError(f"encoder must be one of {list(MODEL_CONFIGS)}, got {out.encoder!r}")
    if out.monitor_metric not in _VALID_METRICS:
        raise ValueError(f"monitor_metric must be one of {_VALID_METRICS}, got {out.monitor_metric!r}")
    if out.monitor_mode not in ("min", "max"):
        raise ValueError(f"monitor_mode must be 'min' or 'max', got {out.monitor_mode!r}")
    if out.cuda_visible_devices is not None:
        if len(out.cuda_visible_devices) == 0:
            raise ValueError("device.cuda_visible_devices must contain at least one index")
        if any(i < 0 for i in out.cuda_visible_devices):
            raise ValueError(
                f"device.cuda_visible_devices must be non-negative, got {out.cuda_visible_devices}"
            )
    if out.cuda_indices is not None:
        if len(out.cuda_indices) == 0:
            raise ValueError("device.cuda_indices must contain at least one index")
        if any(i < 0 for i in out.cuda_indices):
            raise ValueError(f"device.cuda_indices must be non-negative, got {out.cuda_indices}")
    if out.cuda_visible_devices is not None and out.cuda_indices is not None:
        max_local = len(out.cuda_visible_devices) - 1
        if any(i > max_local for i in out.cuda_indices):
            raise ValueError(
                "When device.cuda_visible_devices is set, device.cuda_indices must use local "
                f"PyTorch indices in [0..{max_local}], got {out.cuda_indices}"
            )
    if out.grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {out.grad_accum_steps}")
    if out.eval_freq < 1:
        raise ValueError(f"eval_freq must be >= 1, got {out.eval_freq}")
    return out


def train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    base = load_train_config_yaml(args.config)
    return apply_cli_overrides(base, args)
