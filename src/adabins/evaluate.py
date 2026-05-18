"""Evaluate AdaBins on custom RGB-depth pair lists."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adabins.model import UnetAdaptiveBins
from adabins.yaml_config import load_train_config_yaml
from adabins.checkpoints import unwrap_state_dict
from adabins.dataset import CustomDepthPairDataset
from adabins.metrics import eval_depth


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
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
    val_list = args.val_list or cfg.val_list
    depth_scale = args.depth_scale if args.depth_scale is not None else cfg.depth_scale
    list_root = args.list_root if args.list_root is not None else cfg.list_root
    min_depth = args.min_depth if args.min_depth is not None else cfg.min_depth
    max_depth = args.max_depth if args.max_depth is not None else cfg.max_depth
    n_bins = args.n_bins if args.n_bins is not None else cfg.n_bins
    norm = args.norm if args.norm is not None else cfg.norm
    backend_name = args.backend_name if args.backend_name is not None else cfg.backend_name
    scale_align_eval = bool(args.scale_align_eval) or cfg.scale_align_eval

    valset = CustomDepthPairDataset(
        list_path=val_list,
        mode="val",
        size=(img_size, img_size),
        depth_scale=depth_scale,
        list_root=list_root,
    )
    loader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = UnetAdaptiveBins.build(
        n_bins=n_bins,
        min_val=min_depth,
        max_val=max_depth,
        norm=norm,
        backend_name=backend_name,
        backend_pretrained=False,
    ).to(device)
    blob = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(blob, dict) and "model" in blob:
        state = unwrap_state_dict(blob["model"])
    else:
        state = unwrap_state_dict(blob)
    model.load_state_dict(state, strict=True)
    model.eval()

    metric_keys = ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]
    sums = {k: 0.0 for k in metric_keys}
    n = 0
    for sample in loader:
        img = sample["image"].to(device).float()
        depth = sample["depth"].to(device)[0]
        valid_mask = sample["valid_mask"].to(device)[0]
        _, pred = model(img)
        pred = F.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)[0, 0]
        mask = valid_mask & (depth >= min_depth) & (depth <= max_depth)
        if mask.sum() < 10:
            continue
        pred_eval = pred[mask].clamp(min=1e-6)
        depth_eval = depth[mask]
        if scale_align_eval:
            pred_eval = _median_scale_align(pred_eval, depth_eval).clamp(min=1e-6)
        if not torch.isfinite(pred_eval).all():
            continue
        cur = eval_depth(pred_eval, depth_eval)
        for k in metric_keys:
            sums[k] += float(cur[k])
        n += 1

    if n == 0:
        raise RuntimeError("No valid evaluation samples (check depth range and pair list).")
    return ({k: sums[k] / n for k in metric_keys}, n)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate AdaBins on a pair list (YAML supplies defaults).")
    p.add_argument("--config", "-c", type=str, required=True, help="Training YAML (data/model/training used).")
    p.add_argument("--val-list", type=str, default=None, help="Override data.val_list from config.")
    p.add_argument("--list-root", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--min-depth", type=float, default=None)
    p.add_argument("--max-depth", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-bins", type=int, default=None)
    p.add_argument("--norm", type=str, choices=["linear", "softmax", "sigmoid"], default=None)
    p.add_argument("--backend-name", type=str, default=None)
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
