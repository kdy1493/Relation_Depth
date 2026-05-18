"""Evaluate a fine-tuned Depth Anything V2 (metric) checkpoint on a pair list."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from depthanything.checkpoints import unwrap_state_dict
from depthanything.custom_dataset import CustomDepthPairDataset
from depthanything.da2_vendor.depth_anything_v2.dpt import DepthAnythingV2
from depthanything.da2_vendor.metric_train.metric import eval_depth
from depthanything.model_configs import MODEL_CONFIGS
from depthanything.yaml_config import load_train_config_yaml


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _median_scale_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_median = torch.median(pred)
    target_median = torch.median(target)
    scale = target_median / pred_median.clamp_min(1e-6)
    return pred * scale


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> tuple[dict[str, float], int]:
    cfg = load_train_config_yaml(args.config)

    device = _device()
    img_size = args.img_size if args.img_size is not None else cfg.img_size
    size = (img_size, img_size)
    valset = CustomDepthPairDataset(
        args.val_list or cfg.val_list,
        "val",
        size=size,
        depth_scale=args.depth_scale if args.depth_scale is not None else cfg.depth_scale,
        list_root=args.list_root if args.list_root is not None else cfg.list_root,
    )
    loader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        shuffle=False,
    )

    encoder = args.encoder if args.encoder is not None else cfg.encoder
    max_depth = args.max_depth if args.max_depth is not None else cfg.max_depth
    model_cfg = {**MODEL_CONFIGS[encoder], "max_depth": max_depth}
    model = DepthAnythingV2(**model_cfg).to(device)
    blob = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(blob, dict) and "model" in blob:
        state = unwrap_state_dict(blob["model"])
    else:
        state = unwrap_state_dict(blob)
    model.load_state_dict(state, strict=True)
    model.eval()

    min_depth = args.min_depth if args.min_depth is not None else cfg.min_depth
    scale_align_eval = bool(args.scale_align_eval) or cfg.scale_align_eval

    sums = {k: 0.0 for k in ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]}
    n = 0

    for sample in loader:
        img = sample["image"].to(device).float()
        depth = sample["depth"].to(device)[0]
        valid_mask = sample["valid_mask"].to(device)[0]

        pred = model(img)
        pred = F.interpolate(pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True)[0, 0]

        mask = valid_mask & (depth >= min_depth) & (depth <= max_depth)
        if mask.sum() < 10:
            continue

        pred_eval = pred[mask]
        depth_eval = depth[mask]
        if scale_align_eval:
            pred_eval = _median_scale_align(pred_eval, depth_eval)
        cur = eval_depth(pred_eval, depth_eval)
        for k in sums:
            sums[k] += float(cur[k])
        n += 1

    if n == 0:
        raise RuntimeError("No valid evaluation samples (check masks and depth range).")

    metrics = {k: sums[k] / n for k in sums}
    return metrics, n


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Depth Anything V2 on a pair list (YAML supplies defaults).")
    p.add_argument("--config", "-c", type=str, required=True, help="Training YAML (data/model/training fields used).")
    p.add_argument("--encoder", default=None, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--val-list", type=str, default=None, help="Override data.val_list from config.")
    p.add_argument("--list-root", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True, help="latest.pth or any state dict from this repo")
    p.add_argument("--img-size", default=None, type=int)
    p.add_argument("--depth-scale", default=None, type=float)
    p.add_argument("--min-depth", default=None, type=float)
    p.add_argument("--max-depth", default=None, type=float)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--scale-align-eval", action="store_true", help="Median-scale align prediction before metrics")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics, n = run_eval(args)
    print(f"Evaluated samples (with valid depth): {n}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
