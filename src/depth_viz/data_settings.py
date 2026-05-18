"""Load shared data paths / depth range from any training YAML in this repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from depth_viz.predictors import VisualizeDataSettings


def load_data_settings(config_path: str | Path, **overrides: Any) -> VisualizeDataSettings:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    def pick(key: str, *sections: str, default: Any = None) -> Any:
        if key in overrides and overrides[key] is not None:
            return overrides[key]
        for sec in sections:
            block = raw.get(sec) or {}
            if key in block and block[key] is not None:
                return block[key]
        return default

    val_list = pick("val_list", "data")
    if val_list is None:
        raise KeyError("data.val_list missing in config")

    return VisualizeDataSettings(
        val_list=str(val_list),
        list_root=pick("list_root", "data"),
        depth_scale=float(pick("depth_scale", "data", default=1.0)),
        img_size=int(pick("img_size", "training", default=518)),
        min_depth=float(pick("min_depth", "model", "training", default=0.001)),
        max_depth=float(pick("max_depth", "model", "training", default=20.0)),
    )
