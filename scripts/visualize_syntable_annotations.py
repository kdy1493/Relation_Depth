#!/usr/bin/env python3
"""Run SynTable annotation visualizer on local syntable data.

Uses src/syntable_viz/visualize_annotations.py (vendored from upstream SynTable).

Example (26_10.png, occlusion order graph + masks):
  uv run python scripts/visualize_syntable_annotations.py \\
    --image-ids 1311 --occ-order

Output: data/syntable/validation/visualise_dataset/<image_id>/
  - occlusion_order_<id>.png
  - occlusion_order_adjacency_matrix_<id>.png
  - rgb_occlusion_<id>.png, rgb_visible_mask_<id>.png, amodal_masks_<id>.png, ...

Requires: pip install -e .  and viz deps (matplotlib, networkx, pycocotools, seaborn).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

try:
    from syntable_viz.visualize_annotations import main as run_visualize
except ImportError as e:
    raise ImportError(
        "Install the package first: pip install -e .  "
        "(installs syntable_viz from src/.)"
    ) from e

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "syntable" / "validation"
DEFAULT_JSON = DEFAULT_DATASET / "uoais_val.json"


def main() -> None:
    p = argparse.ArgumentParser(description="SynTable annotation visualization.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--ann-json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--image-ids", type=str, required=True, help="COCO image_id(s), e.g. 1311 for 26_10.png")
    p.add_argument("--occ-order", action="store_true", help="Include OOAM directed graph + heatmap")
    args = p.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")

    viz_argv = [
        "--dataset",
        str(args.dataset),
        "--ann_json",
        str(args.ann_json),
        "--image-ids",
        args.image_ids,
    ]
    if args.occ_order:
        viz_argv.append("--occ-order")

    run_visualize(viz_argv)
    out = args.dataset / "visualise_dataset"
    print(f"done. see: {out}/")


if __name__ == "__main__":
    main()
