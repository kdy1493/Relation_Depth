"""RGB-D → Open3D point cloud export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from depth_viz.intrinsics import PinholeIntrinsics

_RGBD_TO_WORLD = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# Set after first failed off-screen render (no display / OSMesa on server).
_headless_png_disabled = False


def rgb_depth_to_pointcloud(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    depth_trunc: float,
    valid_mask: np.ndarray | None = None,
):
    """Build an Open3D ``PointCloud`` from metric depth (meters) and RGB float [0,1] or uint8."""
    import open3d as o3d

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
    if depth_m.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_m.shape}")
    if depth_m.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"rgb/depth shape mismatch: {rgb.shape[:2]} vs {depth_m.shape}")

    h, w = depth_m.shape
    intr = intrinsics.scaled(w, h)

    depth_use = depth_m.astype(np.float32, copy=True)
    if valid_mask is not None:
        depth_use = np.where(valid_mask, depth_use, 0.0)

    if np.issubdtype(rgb.dtype, np.floating):
        color_u8 = np.ascontiguousarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    else:
        color_u8 = np.ascontiguousarray(rgb)

    depth_mm = np.clip(depth_use * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_u8),
        o3d.geometry.Image(depth_mm),
        depth_scale=1000.0,
        depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False,
    )
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
    pcd.transform(_RGBD_TO_WORLD)
    return pcd


def write_point_cloud(path: Path, pcd) -> None:
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def capture_point_cloud_png(pcd, path: Path, *, width: int = 1280, height: int = 720, point_size: float = 2.0) -> str | None:
    global _headless_png_disabled

    if _headless_png_disabled:
        return "Open3D headless rendering unavailable (skipped)"

    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(visible=False, width=width, height=height):
        vis.destroy_window()
        _headless_png_disabled = True
        return "Open3D headless window creation failed (no display/OSMesa); .ply still saved"
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = point_size
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(path), do_render=True)
    vis.destroy_window()
    return None


def show_point_cloud(pcd) -> None:
    import open3d as o3d

    o3d.visualization.draw_geometries([pcd])
