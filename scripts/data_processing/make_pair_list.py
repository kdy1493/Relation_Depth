#!/usr/bin/env python3
"""Generate tab-separated RGB-depth pair list files from a SynTable split directory.

Each output line:
    <rgb_path>\\t<depth_path>

Paths are written relative to the repository root so that ``list_root: null``
works in the YAML config.

Examples:
    python scripts/data_processing/make_pair_list.py --split train
    python scripts/data_processing/make_pair_list.py --split val
    python scripts/data_processing/make_pair_list.py --split train --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (scripts/data_processing/ → ../..)

SPLIT_DIRS = {
    "train": Path("data/syntable/train/data/mono"),
    "val":   Path("data/syntable/validation/data/mono"),
}

OUT_NAMES = {
    "train": "syntable_train.txt",
    "val":   "syntable_val.txt",
}


def make_list(split: str, out_dir: Path) -> Path:
    mono = SPLIT_DIRS[split]
    rgb_dir   = mono / "rgb"
    depth_dir = mono / "depth"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    names = sorted(rgb_dir.glob("*.png"), key=lambda p: p.name)
    missing = [n.name for n in names if not (depth_dir / n.name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} depth files missing for split '{split}': {missing[:5]}..."
        )

    out_path = out_dir / OUT_NAMES[split]
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"{(ROOT / rgb_dir / n.name).relative_to(ROOT)}\t{(ROOT / depth_dir / n.name).relative_to(ROOT)}"
        for n in names
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{split:>5}] {len(lines):,} pairs → {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SynTable pair list files")
    p.add_argument(
        "--split", choices=list(SPLIT_DIRS), nargs="+", default=list(SPLIT_DIRS),
        help="Which splits to generate (default: both)",
    )
    p.add_argument(
        "--out-dir", default="data/syntable/lists", type=Path,
        help="Output directory for pair list files (default: data/syntable/lists)",
    )
    args = p.parse_args()

    for split in args.split:
        make_list(split, args.out_dir)


if __name__ == "__main__":
    main()
