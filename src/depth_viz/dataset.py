"""Validation dataset for visualization (shared RGB–depth pair list format)."""

from __future__ import annotations

from pathlib import Path

from depthanything.custom_dataset import CustomDepthPairDataset


def build_val_dataset(
    val_list: str | Path,
    *,
    img_size: int,
    depth_scale: float,
    list_root: str | Path | None,
) -> CustomDepthPairDataset:
    return CustomDepthPairDataset(
        val_list,
        "val",
        size=(img_size, img_size),
        depth_scale=depth_scale,
        list_root=list_root,
    )
