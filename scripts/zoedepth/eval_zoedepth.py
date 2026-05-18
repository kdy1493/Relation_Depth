#!/usr/bin/env python3
"""Evaluate ZoeDepth checkpoints on pair lists (YAML + checkpoint)."""

from __future__ import annotations

import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

try:
    from zoedepth.evaluate import main
except ImportError as e:
    raise ImportError(
        "Run once: pip install -e .   "
        "(pyrootutils adds only repo root; packages live under src/.)"
    ) from e


if __name__ == "__main__":
    main()
