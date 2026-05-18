#!/usr/bin/env python3
"""Evaluate AdaBins checkpoints on pair lists.

Example:
  python scripts/adabins/eval_adabins.py -c configs/syntable_adabins_0.yaml \\
    --checkpoint runs/syntable_adabins_0/checkpoints/best.pth
"""

from __future__ import annotations

import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

try:
    from adabins.evaluate import main
except ImportError as e:
    raise ImportError(
        "Run once: pip install -e .   "
        "(pyrootutils adds only repo root; packages live under src/.)"
    ) from e


if __name__ == "__main__":
    main()
