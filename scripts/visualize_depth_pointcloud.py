#!/usr/bin/env python3
"""Export GT / predicted depth as Open3D point clouds (all model backends).

Examples:
  # Depth Anything V2
  uv run python scripts/visualize_depth_pointcloud.py \\
    --backend depth_anything_v2 \\
    -c configs/syntable_DA_0.yaml \\
    --checkpoint outputs/syntable_DA_0/checkpoints/best.pth \\
    --max-samples 5 --write-png

  # ZoeDepth
  uv run python scripts/visualize_depth_pointcloud.py \\
    --backend zoedepth \\
    -c configs/syntable_zoedepth_0.yaml \\
    --checkpoint outputs/syntable_zoedepth_0/checkpoints/best.pth \\
    --indices 0,1,2

  # AdaBins
  uv run python scripts/visualize_depth_pointcloud.py \\
    --backend adabins \\
    -c configs/syntable_adabins_0.yaml \\
    --checkpoint outputs/syntable_adabins_0/checkpoints/best.pth

  # GT only (no checkpoint)
  uv run python scripts/visualize_depth_pointcloud.py \\
    --gt-only -c configs/syntable_DA_0.yaml --max-samples 3
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
