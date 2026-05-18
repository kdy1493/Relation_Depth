#!/usr/bin/env python3
"""Launch Depth Anything V2 fine-tuning from a YAML config.

Examples:
  python scripts/depth_anything_v2/train_depth_anything_v2.py --config configs/train.yaml
  python scripts/depth_anything_v2/train_depth_anything_v2.py -c configs/train.yaml --epochs 80 --lr 1e-5
  torchrun --nproc_per_node=4 scripts/depth_anything_v2/train_depth_anything_v2.py -c configs/train.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pyrootutils
import yaml

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)


def _extract_config_path(argv: list[str]) -> Path | None:
    for i, token in enumerate(argv):
        if token == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if token == "-c" and i + 1 < len(argv):
            return Path(argv[i + 1])
    return None


def _apply_cuda_visible_devices_from_yaml(argv: list[str]) -> None:
    config_path = _extract_config_path(argv)
    if config_path is None or not config_path.exists():
        return
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return
    device = raw.get("device") or {}
    visible = device.get("cuda_visible_devices")
    if visible in (None, ""):
        return
    ids = ",".join(str(int(x)) for x in visible)
    os.environ["CUDA_VISIBLE_DEVICES"] = ids
    print(f"[launcher] CUDA_VISIBLE_DEVICES set from YAML: {ids}")


_apply_cuda_visible_devices_from_yaml(sys.argv[1:])

try:
    from depthanything.train import main
except ImportError as e:
    raise ImportError(
        "Run once: pip install -e .   "
        "(pyrootutils adds only the repo root to sys.path; packages live under src/.)"
    ) from e

if __name__ == "__main__":
    main()
