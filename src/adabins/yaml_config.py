"""Load AdaBins training configuration from YAML with CLI overrides."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

_VALID_METRICS = {"d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"}
_VALID_NORMS = {"linear", "softmax", "sigmoid"}


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
class AdaBinsTrainConfig:
    train_list: str
    val_list: str
    list_root: str | None
    depth_scale: float
    n_bins: int
    norm: str
    max_depth: float
    min_depth: float
    backend_name: str
    backend_pretrained: bool
    pretrained_from: str | None
    resume_from: str | None
    img_size: int
    epochs: int
    bs: int
    lr: float
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


def _require(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(".".join(path))
        cur = cur[p]
    return cur


def load_train_config_yaml(path: str | Path) -> AdaBinsTrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    data = raw.get("data") or {}
    model = raw.get("model") or {}
    training = raw.get("training") or {}
    output = raw.get("output") or {}
    dist = raw.get("distributed") or {}
    device = raw.get("device") or {}

    norm = str(model.get("norm", "linear"))
    if norm not in _VALID_NORMS:
        raise ValueError(f"model.norm must be one of {_VALID_NORMS}, got {norm!r}")

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
    if cuda_visible_devices is not None and cuda_indices is None:
        cuda_indices = list(range(len(cuda_visible_devices)))
    if cuda_visible_devices is not None and cuda_indices is not None:
        visible_set = set(cuda_visible_devices)
        if all(i in visible_set for i in cuda_indices):
            index_map = {physical: local for local, physical in enumerate(cuda_visible_devices)}
            cuda_indices = [index_map[i] for i in cuda_indices]

    return AdaBinsTrainConfig(
        train_list=str(_require(data, "train_list")),
        val_list=str(_require(data, "val_list")),
        list_root=(str(data["list_root"]) if data.get("list_root") not in (None, "") else None),
        depth_scale=float(data.get("depth_scale", 1.0)),
        n_bins=int(model.get("n_bins", 256)),
        norm=norm,
        max_depth=float(model.get("max_depth", 10.0)),
        min_depth=float(model.get("min_depth", 1e-3)),
        backend_name=str(model.get("backend_name", "tf_efficientnet_b5_ap")),
        backend_pretrained=_parse_bool(model.get("backend_pretrained", True)),
        pretrained_from=(str(model["pretrained_from"]) if model.get("pretrained_from") not in (None, "") else None),
        resume_from=(str(model["resume_from"]) if model.get("resume_from") not in (None, "") else None),
        img_size=int(training.get("img_size", 480)),
        epochs=int(training.get("epochs", 40)),
        bs=int(training.get("batch_size", training.get("bs", 2))),
        lr=float(training.get("lr", 3.57e-4)),
        num_workers=int(training.get("num_workers", 4)),
        seed=(int(training["seed"]) if training.get("seed") is not None else None),
        find_unused_parameters=_parse_bool(training.get("find_unused_parameters", True)),
        eval_freq=int(training.get("eval_freq", 1)),
        grad_accum_steps=max(1, int(training.get("grad_accum_steps", 1))),
        amp=_parse_bool(training.get("amp", True)),
        patience=(int(training["patience"]) if training.get("patience") is not None else None),
        ema_decay=float(training.get("ema_decay", 0.0)),
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        auto_batch_size=_parse_bool(training.get("auto_batch_size", False)),
        scale_align_eval=_parse_bool(training.get("scale_align_eval", False)),
        save_path=str(_require(output, "save_path")),
        port=(int(dist["port"]) if dist.get("port") not in (None, "") else None),
        cuda_visible_devices=cuda_visible_devices,
        cuda_indices=cuda_indices,
    )


def build_train_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train AdaBins from YAML config")
    p.add_argument("--config", "-c", type=str, required=True)
    for flag in ("train_list", "val_list", "list_root", "pretrained_from", "resume_from", "save_path"):
        p.add_argument(f"--{flag.replace('_', '-')}", type=str, default=None)
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--n-bins", type=int, default=None)
    p.add_argument("--norm", type=str, choices=sorted(_VALID_NORMS), default=None)
    p.add_argument("--max-depth", type=float, default=None)
    p.add_argument("--min-depth", type=float, default=None)
    p.add_argument("--backend-name", type=str, default=None)
    p.add_argument("--backend-pretrained", type=_parse_bool, default=None, metavar="{true,false}")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--bs", "--batch-size", type=int, default=None, dest="bs")
    p.add_argument("--lr", type=float, default=None)
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


def apply_cli_overrides(cfg: AdaBinsTrainConfig, args: argparse.Namespace) -> AdaBinsTrainConfig:
    overrides: dict[str, Any] = {}
    for f in fields(AdaBinsTrainConfig):
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


def train_config_from_args(args: argparse.Namespace) -> AdaBinsTrainConfig:
    return apply_cli_overrides(load_train_config_yaml(args.config), args)
