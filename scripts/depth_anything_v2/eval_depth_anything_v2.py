#!/usr/bin/env python3
"""Evaluate a Depth Anything V2 checkpoint on a validation pair list.

Example:
  python scripts/depth_anything_v2/eval_depth_anything_v2.py -c configs/syntable_DA_0.yaml \\
    --checkpoint runs/syntable_DA_0/checkpoints/best.pth
"""

from __future__ import annotations

import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

try:
    from depthanything.evaluate import main
except ImportError as e:
    raise ImportError(
        "Run once: pip install -e .   "
        "(pyrootutils adds only the repo root to sys.path; packages live under src/.)"
    ) from e

if __name__ == "__main__":
    main()
