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

  # Adjust gap between GT and Pred (default 2m)
  python scripts/view_depth_open3d.py dir/ --gap 3.0

  # Pick specific pair by stem name
  python scripts/view_depth_open3d.py dir/ --name 0_0

  # Error heatmap (blue=accurate, red=large error)
  python scripts/view_depth_open3d.py gt.ply pred.ply --error-map
"""

from __future__ import annotations

import argparse
import sys


from pathlib import Path

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        print("open3d is not installed. Run:  pip install open3d", file=sys.stderr)
        sys.exit(1)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map normalized [0,1] values to jet colormap (blue→cyan→green→yellow→red)."""
    r = np.clip(1.5 - np.abs(values * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(values * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(values * 4 - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def view_error_map(gt_path: str, pred_path: str, percentile: float = 95.0) -> None:
    """Visualize per-point depth error as a heatmap on the GT geometry."""
    o3d = _require_open3d()
    gt = o3d.io.read_point_cloud(gt_path)
    pred = o3d.io.read_point_cloud(pred_path)

    gt_pts = np.asarray(gt.points)
    pred_pts = np.asarray(pred.points)

    if len(gt_pts) != len(pred_pts):
        print(f"Point count mismatch: GT {len(gt_pts)} vs Pred {len(pred_pts)}", file=sys.stderr)
        return
    if len(gt_pts) == 0:
        print("Empty point clouds.", file=sys.stderr)
        return

    errors = np.linalg.norm(gt_pts - pred_pts, axis=1)
    vmax = np.percentile(errors, percentile)
    vmin = 0.0
    norm = np.clip((errors - vmin) / max(vmax - vmin, 1e-8), 0, 1)

    colors = _jet_colormap(norm)
    gt.colors = o3d.utility.Vector3dVector(colors)

    print(f"GT:   {gt_path}  ({len(gt_pts)} pts)")
    print(f"Pred: {pred_path}  ({len(pred_pts)} pts)")
    print(f"Error — mean: {errors.mean():.4f}m, median: {np.median(errors):.4f}m, "
          f"p{percentile:.0f}: {vmax:.4f}m, max: {errors.max():.4f}m")
    print("Color: blue=accurate → red=large error")

    title = f"Error Map — {Path(gt_path).stem} (blue=good, red=bad)"
    o3d.visualization.draw_geometries([gt], window_name=title, width=1280, height=720)


def view_single(path: str) -> None:
    o3d = _require_open3d()
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print(f"Empty or invalid point cloud: {path}", file=sys.stderr)
        return
    name = Path(path).stem
    print(f"Viewing: {path}  ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries([pcd], window_name=name, width=1280, height=720)


def _create_vis(o3d, pcd, title: str, left: int, top: int, width: int = 780, height: int = 720):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=width, height=height, left=left, top=top)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    return vis


def _get_cam_params(vis):
    ctr = vis.get_view_control()
    return ctr.convert_to_pinhole_camera_parameters()


def _set_cam_params(vis, params):
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def _cam_changed(a, b) -> bool:
    """Check if two PinholeCameraParameters differ (extrinsic matrix)."""
    return not np.allclose(a.extrinsic, b.extrinsic, atol=1e-10)


def view_pair(gt_path: str, pred_path: str, gap: float = 2.0) -> None:
    o3d = _require_open3d()
    gt = o3d.io.read_point_cloud(gt_path)
    pred = o3d.io.read_point_cloud(pred_path)
    if gt.is_empty() and pred.is_empty():
        print("Both point clouds are empty.", file=sys.stderr)
        return

    print(f"GT:   {gt_path}  ({len(gt.points)} pts)")
    print(f"Pred: {pred_path}  ({len(pred.points)} pts)")
    gt_stem = Path(gt_path).stem
    pred_stem = Path(pred_path).stem

    gt_colors_orig = np.asarray(gt.colors).copy() if gt.has_colors() else None
    pred_colors_orig = np.asarray(pred.colors).copy() if pred.has_colors() else None

    def _depth_colormap(pcd):
        """Color by depth (Z) with jet colormap. Near=red, far=blue."""
        pts = np.asarray(pcd.points)
        z = pts[:, 2]
        zmin, zmax = z.min(), z.max()
        norm = np.clip((z - zmin) / max(zmax - zmin, 1e-8), 0, 1)
        return _jet_colormap(norm)

    gt_depth_colors = _depth_colormap(gt)
    pred_depth_colors = _depth_colormap(pred)

    state = {"sync": True, "rgb": True}

    def _toggle_sync(vis):
        state["sync"] = not state["sync"]
        mode = "ON (synced)" if state["sync"] else "OFF (individual)"
        print(f"Camera sync: {mode}")
        return False

    def _toggle_rgb(vis):
        state["rgb"] = not state["rgb"]
        if state["rgb"] and gt_colors_orig is not None:
            gt.colors = o3d.utility.Vector3dVector(gt_colors_orig)
            pred.colors = o3d.utility.Vector3dVector(pred_colors_orig)
        else:
            gt.colors = o3d.utility.Vector3dVector(gt_depth_colors)
            pred.colors = o3d.utility.Vector3dVector(pred_depth_colors)
        vis1.update_geometry(gt)
        vis2.update_geometry(pred)
        print(f"Color: {'RGB' if state['rgb'] else 'Depth jet (red=near, blue=far)'}")
        return False

    vis1 = _create_vis(o3d, gt, f"GT — {gt_stem}", left=50, top=100)
    vis2 = _create_vis(o3d, pred, f"Pred — {pred_stem}", left=850, top=100)

    vis1.register_key_callback(ord("S"), _toggle_sync)
    vis2.register_key_callback(ord("S"), _toggle_sync)
    vis1.register_key_callback(ord("C"), _toggle_rgb)
    vis2.register_key_callback(ord("C"), _toggle_rgb)

    prev1 = _get_cam_params(vis1)
    prev2 = _get_cam_params(vis2)

    print("Keys:  S = toggle camera sync  |  C = toggle RGB / Depth colormap")

    alive = True
    while alive:
        if not vis1.poll_events():
            alive = False
            break
        vis1.update_renderer()
        if not vis2.poll_events():
            alive = False
            break
        vis2.update_renderer()

        if not state["sync"]:
            prev1 = _get_cam_params(vis1)
            prev2 = _get_cam_params(vis2)
            continue

        cur1 = _get_cam_params(vis1)
        cur2 = _get_cam_params(vis2)

        if _cam_changed(cur1, prev1):
            _set_cam_params(vis2, cur1)
            vis2.update_renderer()
            prev1 = cur1
            prev2 = cur1
        elif _cam_changed(cur2, prev2):
            _set_cam_params(vis1, cur2)
            vis1.update_renderer()
            prev1 = cur2
            prev2 = cur2
        else:
            prev1 = cur1
            prev2 = cur2

    vis1.destroy_window()
    vis2.destroy_window()


def view_directory(directory: str, gap: float = 2.0, name: str | None = None) -> None:
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
            view_pair(str(gt_f), str(pred_f), gap=gap)
        elif gt_f:
            print(f"\n--- {stem}: GT only ---")
            view_single(str(gt_f))
        elif pred_f:
            print(f"\n--- {stem}: Pred only ---")
            view_single(str(pred_f))


def main() -> None:
    parser = argparse.ArgumentParser(description="View .ply point clouds with Open3D.")
    parser.add_argument("paths", nargs="+", help=".ply file(s) or a directory containing *_gt.ply / *_pred.ply")
    parser.add_argument("--gap", type=float, default=2.0, help="Gap (meters) between GT and Pred point clouds.")
    parser.add_argument("--name", type=str, default=None, help="Stem name to view (e.g. '0_0'). Directory mode only.")
    parser.add_argument("--error-map", action="store_true", help="Show depth error heatmap (blue=accurate, red=large error). Requires GT+Pred pair.")
    parser.add_argument("--percentile", type=float, default=95.0, help="Error colormap saturation percentile (default 95).")
    args = parser.parse_args()

    paths = args.paths

    if len(paths) == 1:
        p = Path(paths[0])
        if p.is_dir():
            view_directory(str(p), gap=args.gap, name=args.name)
        elif p.suffix == ".ply":
            view_single(str(p))
        else:
            print(f"Unknown input: {p}", file=sys.stderr)
            sys.exit(1)
    elif len(paths) == 2 and all(Path(p).suffix == ".ply" for p in paths):
        if args.error_map:
            view_error_map(paths[0], paths[1], percentile=args.percentile)
        else:
            view_pair(paths[0], paths[1], gap=args.gap)
    else:
        for p in paths:
            view_single(p)


if __name__ == "__main__":
    main()
