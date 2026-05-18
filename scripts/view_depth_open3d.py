#!/usr/bin/env python3
"""[Local] View .ply point clouds with Open3D GUI viewer.

Opens exported .ply files from scripts/export_depth_open3d.py on a local PC.
No GPU, model, or data required — just open3d.

Requires: pip install open3d

Usage:
  # Single file
  python scripts/view_depth_open3d.py path/to/0_0_pred.ply

  # GT vs Pred side by side
  python scripts/view_depth_open3d.py path/to/0_0_gt.ply path/to/0_0_pred.ply

  # Auto-detect: pass a directory to view all GT/Pred pairs
  python scripts/view_depth_open3d.py outputs/syntable_DA_0/figures/pointcloud_vis/

  # Adjust spacing between GT and Pred
  python scripts/view_depth_open3d.py dir/ --spacing 0.8

  # Pick specific pair by stem name
  python scripts/view_depth_open3d.py dir/ --name 0_0
"""

from __future__ import annotations

import argparse
import sys


from pathlib import Path


def _require_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        print("open3d is not installed. Run:  pip install open3d", file=sys.stderr)
        sys.exit(1)


def view_single(path: str) -> None:
    o3d = _require_open3d()
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print(f"Empty or invalid point cloud: {path}", file=sys.stderr)
        return
    name = Path(path).stem
    print(f"Viewing: {path}  ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries([pcd], window_name=name, width=1280, height=720)


def view_pair(gt_path: str, pred_path: str, spacing: float = 0.6) -> None:
    o3d = _require_open3d()
    gt = o3d.io.read_point_cloud(gt_path)
    pred = o3d.io.read_point_cloud(pred_path)
    if gt.is_empty() and pred.is_empty():
        print("Both point clouds are empty.", file=sys.stderr)
        return

    pred.translate((spacing, 0, 0))
    title = f"GT (left)  vs  Pred (right) — spacing {spacing}m"
    total = len(gt.points) + len(pred.points)
    print(f"GT:   {gt_path}  ({len(gt.points)} pts)")
    print(f"Pred: {pred_path}  ({len(pred.points)} pts)")
    o3d.visualization.draw_geometries([gt, pred], window_name=title, width=1600, height=800)


def view_directory(directory: str, spacing: float = 0.6, name: str | None = None) -> None:
    d = Path(directory)
    if not d.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    gt_files = sorted(d.glob("*_gt.ply"))
    pred_files = sorted(d.glob("*_pred.ply"))

    stems_gt = {f.name.removesuffix("_gt.ply"): f for f in gt_files}
    stems_pred = {f.name.removesuffix("_pred.ply"): f for f in pred_files}
    all_stems = sorted(set(stems_gt) | set(stems_pred))

    if not all_stems:
        print(f"No *_gt.ply or *_pred.ply found in {directory}", file=sys.stderr)
        sys.exit(1)

    if name is not None:
        if name not in all_stems:
            print(f"Stem '{name}' not found. Available: {', '.join(all_stems)}", file=sys.stderr)
            sys.exit(1)
        all_stems = [name]

    print(f"Found {len(all_stems)} sample(s) in {directory}")
    for stem in all_stems:
        gt_f = stems_gt.get(stem)
        pred_f = stems_pred.get(stem)
        if gt_f and pred_f:
            print(f"\n--- {stem}: GT + Pred ---")
            view_pair(str(gt_f), str(pred_f), spacing=spacing)
        elif gt_f:
            print(f"\n--- {stem}: GT only ---")
            view_single(str(gt_f))
        elif pred_f:
            print(f"\n--- {stem}: Pred only ---")
            view_single(str(pred_f))


def main() -> None:
    parser = argparse.ArgumentParser(description="View .ply point clouds with Open3D.")
    parser.add_argument("paths", nargs="+", help=".ply file(s) or a directory containing *_gt.ply / *_pred.ply")
    parser.add_argument("--spacing", type=float, default=0.6, help="X-axis offset between GT and Pred (meters).")
    parser.add_argument("--name", type=str, default=None, help="Stem name to view (e.g. '0_0'). Directory mode only.")
    args = parser.parse_args()

    paths = args.paths

    if len(paths) == 1:
        p = Path(paths[0])
        if p.is_dir():
            view_directory(str(p), spacing=args.spacing, name=args.name)
        elif p.suffix == ".ply":
            view_single(str(p))
        else:
            print(f"Unknown input: {p}", file=sys.stderr)
            sys.exit(1)
    elif len(paths) == 2 and all(Path(p).suffix == ".ply" for p in paths):
        view_pair(paths[0], paths[1], spacing=args.spacing)
    else:
        for p in paths:
            view_single(p)


if __name__ == "__main__":
    main()
