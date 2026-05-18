#!/usr/bin/env python3
"""[Server] Export GT / predicted depth as Open3D point clouds (.ply).

Runs model inference (GPU) and saves .ply files. No display required.
View the exported .ply on a local PC with: scripts/view_depth_open3d.py

Requires: open3d, torch, model packages (uv sync && uv pip install -e .)

Examples (default save: results/pointcloud/<subdir>/):
  # Depth Anything V2 → results/pointcloud/syntable_DA_0/
  python scripts/export_depth_open3d.py --backend depth_anything_v2 -c configs/syntable_DA_0.yaml --checkpoint runs/syntable_DA_0/checkpoints/best.pth --max-samples 5

  # ZoeDepth → results/pointcloud/syntable_zoedepth_0/
  python scripts/export_depth_open3d.py --backend zoedepth -c configs/syntable_zoedepth_0.yaml --checkpoint runs/syntable_zoedepth_0/checkpoints/best.pth --indices 0,1,2

  # GT only → results/pointcloud/gt_only/
  python scripts/export_depth_open3d.py --gt-only -c configs/syntable_DA_0.yaml --max-samples 3

  # Custom tag
  python scripts/export_depth_open3d.py --gt-only -c configs/syntable_DA_0.yaml --indices 923,902 --save-dir results/pointcloud/high_occ
"""

from __future__ import annotations

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

try:
    from depth_viz.runner import main
except ImportError as e:
    raise ImportError(
        "Install the package first: pip install -e .  "
        "(installs depth_viz and model packages from src/.)"
    ) from e

if __name__ == "__main__":
    main()
