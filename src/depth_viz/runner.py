"""Batch export of GT / predicted depth as Open3D point clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from depth_viz.data_settings import load_data_settings
from depth_viz.dataset import build_val_dataset
from depth_viz.intrinsics import syntable_intrinsics
from depth_viz.pointcloud import (
    capture_point_cloud_png,
    rgb_depth_to_pointcloud,
    show_point_cloud,
    write_point_cloud,
)
from depth_viz.predictors import (
    BACKEND_CHOICES,
    DepthPredictor,
    build_predictor,
    median_scale_align,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_stem(image_path: str) -> str:
    return Path(image_path).stem.replace("/", "_").replace("\\", "_")


def _load_rgb_float(rgb_path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _auto_depth_trunc(depth_m: np.ndarray, valid: np.ndarray, fallback: float) -> float:
    vals = depth_m[valid & (depth_m > 0)]
    if vals.size == 0:
        return fallback
    return float(np.percentile(vals, 95) * 1.2)


def _default_save_dir(checkpoint: Path | None, backend: str | None, *, gt_only: bool = False) -> Path:
    if checkpoint is not None:
        ckpt_dir = checkpoint.parent
        run_root = ckpt_dir.parent if ckpt_dir.name == "checkpoints" else ckpt_dir
        return run_root / "figures" / "pointcloud_vis"
    if gt_only:
        return Path("outputs") / "figures" / "pointcloud_vis_gt"
    return Path("outputs") / "figures" / f"pointcloud_vis_{backend or 'unknown'}"


def _export_one(
    *,
    tag: str,
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    valid_np: np.ndarray,
    out_dir: Path,
    stem: str,
    depth_trunc: float,
    intrinsics,
    write_png: bool,
    point_size: float,
    interactive: bool,
) -> dict[str, Any]:
    depth_np = np.where(valid_np, np.clip(depth_np, 0.0, depth_trunc), 0.0)
    rgb_np = np.where(valid_np[..., None], rgb_np, 0.0)
    pcd = rgb_depth_to_pointcloud(
        rgb_np, depth_np, intrinsics, depth_trunc=depth_trunc, valid_mask=valid_np
    )
    ply_path = out_dir / f"{stem}_{tag}.ply"
    write_point_cloud(ply_path, pcd)

    png_path = out_dir / f"{stem}_{tag}.png"
    png_err = None
    if write_png:
        png_err = capture_point_cloud_png(pcd, png_path, point_size=point_size)

    if interactive:
        show_point_cloud(pcd)

    return {"ply": str(ply_path), "png": str(png_path) if write_png else None, "png_error": png_err}


@torch.no_grad()
def run_pointcloud_visualize(args: argparse.Namespace) -> dict[str, Any]:
    device = _device()
    predictor: DepthPredictor | None = None

    if not args.gt_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --gt-only is set")
        predictor = build_predictor(
            args.backend,
            args.config,
            args.checkpoint,
            device,
            encoder=args.encoder,
        )
        settings = predictor.settings
    else:
        settings = load_data_settings(
            args.config,
            val_list=args.val_list,
            list_root=args.list_root,
            depth_scale=args.depth_scale,
            img_size=args.img_size,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

    val_list = args.val_list or settings.val_list
    list_root = args.list_root if args.list_root is not None else settings.list_root
    img_size = args.img_size if args.img_size is not None else settings.img_size
    depth_scale = args.depth_scale if args.depth_scale is not None else settings.depth_scale
    min_depth = args.min_depth if args.min_depth is not None else settings.min_depth
    max_depth = args.max_depth if args.max_depth is not None else settings.max_depth

    valset = build_val_dataset(
        val_list,
        img_size=img_size,
        depth_scale=depth_scale,
        list_root=list_root,
    )

    if args.indices:
        sample_indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        sample_indices = list(range(len(valset)))
    sample_indices = sample_indices[: args.max_samples]

    out_dir = Path(args.save_dir) if args.save_dir else _default_save_dir(
        Path(args.checkpoint) if args.checkpoint else None,
        args.backend,
        gt_only=bool(args.gt_only),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    exported = 0
    for idx in tqdm(sample_indices, desc="pointcloud-vis"):
        sample = valset[idx]
        image_path = sample["image_path"]
        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0]
        stem = _safe_stem(image_path)
        gt_depth = sample["depth"]
        valid_mask = sample["valid_mask"]
        if gt_depth.ndim == 3:
            gt_depth = gt_depth[0]
            valid_mask = valid_mask[0]
        h, w = gt_depth.shape[-2:]

        valid_t = valid_mask & (gt_depth >= min_depth) & (gt_depth <= max_depth)
        if int(valid_t.sum().item()) < 10:
            continue

        rgb_np = _load_rgb_float(image_path)
        if rgb_np.shape[0] != h or rgb_np.shape[1] != w:
            rgb_np = cv2.resize(rgb_np, (w, h), interpolation=cv2.INTER_LINEAR)

        gt_np = gt_depth.numpy().astype(np.float32)
        valid_np = valid_t.numpy()
        intrinsics = syntable_intrinsics(w, h)
        depth_trunc = args.depth_trunc if args.depth_trunc is not None else _auto_depth_trunc(gt_np, valid_np, max_depth)

        record: dict[str, Any] = {"index": idx, "image_name": Path(image_path).name, "image_path": image_path}

        if args.export_gt:
            record["gt"] = _export_one(
                tag="gt",
                rgb_np=rgb_np,
                depth_np=gt_np,
                valid_np=valid_np,
                out_dir=out_dir,
                stem=stem,
                depth_trunc=depth_trunc,
                intrinsics=intrinsics,
                write_png=args.write_png,
                point_size=args.point_size,
                interactive=args.interactive and exported == 0,
            )

        if predictor is not None:
            image_batch = sample["image"]
            if image_batch.ndim == 3:
                image_batch = image_batch.unsqueeze(0)
            pred = predictor.predict(image_batch, (h, w))
            if args.scale_align:
                pred = median_scale_align(pred[valid_t], gt_depth[valid_t])
            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            pred_valid = valid_np & np.isfinite(pred_np) & (pred_np > 0)
            record["pred"] = _export_one(
                tag="pred",
                rgb_np=rgb_np,
                depth_np=pred_np,
                valid_np=pred_valid,
                out_dir=out_dir,
                stem=stem,
                depth_trunc=depth_trunc,
                intrinsics=intrinsics,
                write_png=args.write_png,
                point_size=args.point_size,
                interactive=False,
            )

        items.append(record)
        exported += 1

    if exported == 0:
        raise RuntimeError("No samples exported (check indices, depth masks, and paths).")

    summary = {
        "backend": args.backend,
        "gt_only": bool(args.gt_only),
        "exported": exported,
        "save_dir": str(out_dir),
        "scale_align": bool(args.scale_align),
        "items": items,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export GT and/or predicted metric depth as Open3D point clouds (.ply)."
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=BACKEND_CHOICES,
        default="depth_anything_v2",
        help="Model family (used with --checkpoint).",
    )
    p.add_argument("--config", "-c", type=str, required=True, help="Training YAML for data paths and model defaults.")
    p.add_argument("--checkpoint", type=str, default=None, help="Fine-tuned checkpoint (.pth). Omit with --gt-only.")
    p.add_argument("--gt-only", action="store_true", help="Export GT depth only (no model).")
    p.add_argument("--encoder", default=None, help="depth_anything_v2 only: vitb / vitl / …")
    p.add_argument("--val-list", type=str, default=None)
    p.add_argument("--list-root", type=str, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--min-depth", type=float, default=None)
    p.add_argument("--max-depth", type=float, default=None)
    p.add_argument("--depth-trunc", type=float, default=None, help="PLY depth cutoff in meters (auto if omitted).")
    p.add_argument("--max-samples", type=int, default=20, help="Max images to export (ignored if --indices is set).")
    p.add_argument("--indices", type=str, default=None, help="Comma-separated dataset indices (e.g. 0,5,12).")
    p.add_argument("--scale-align", action="store_true", help="Median-scale align pred to GT before export.")
    p.add_argument("--no-gt", action="store_true", help="Skip GT point clouds (pred only).")
    p.add_argument("--write-png", action="store_true", help="Also save off-screen PNG renders.")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--interactive", action="store_true", help="Open Open3D viewer for the first cloud.")
    p.add_argument("--save-dir", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.export_gt = not args.no_gt
    if args.gt_only:
        args.export_gt = True
    summary = run_pointcloud_visualize(args)
    print(f"exported: {summary['exported']}")
    print(f"save_dir: {summary['save_dir']}")
    if args.write_png:
        png_failed = any(
            (item.get("gt") or {}).get("png_error") or (item.get("pred") or {}).get("png_error")
            for item in summary.get("items", [])
        )
        if png_failed:
            print(
                "note: PNG export failed (headless server). Use .ply locally, or: "
                "apt install libosmesa6-dev && re-run, or omit --write-png."
            )
