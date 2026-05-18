#!/usr/bin/env python3
"""Launch AdaBins fine-tuning from a YAML config.

Examples:
  python scripts/adabins/train_adabins.py -c configs/syntable_adabins_0.yaml 2>&1 | tee syntable_adabins_0.log
  torchrun --nproc_per_node=2 scripts/adabins/train_adabins.py -c configs/syntable_adabins_0.yaml 2>&1 | tee syntable_adabins_0.log
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
        if token in {"--config", "-c"} and i + 1 < len(argv):
            return Path(argv[i + 1])
    return None


def _apply_cuda_visible_devices_from_yaml(argv: list[str]) -> None:
    cfg_path = _extract_config_path(argv)
    if cfg_path is None or not cfg_path.exists():
        return
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
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
    from adabins.train import main
except ImportError as e:
    raise ImportError(
        "Run once: pip install -e .   "
        "(pyrootutils adds only repo root; packages live under src/.)"
    ) from e


if __name__ == "__main__":
    main()
