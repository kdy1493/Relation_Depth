"""Load depth maps and file lists for custom datasets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_depth_map(path: str | Path, *, depth_scale: float = 1.0) -> np.ndarray:
    """Load a metric depth map as float32 HxW.

    - ``.npy`` / ``.npz``: first array used for npz
    - image: single-channel or multi-channel uint8/uint16 (first channel used)
    - all formats divide by ``depth_scale`` (default 1.0 = no-op)
      set ``depth_scale=256.0`` for uint16 PNG fixed-point encoding, etc.
    """
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".npy":
        d = np.load(str(path)).astype(np.float32) / float(depth_scale)
    elif suf == ".npz":
        z = np.load(str(path))
        key = sorted(z.files)[0]
        d = z[key].astype(np.float32) / float(depth_scale)
    else:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        if im.ndim == 3:
            im = im[:, :, 0]
        d = im.astype(np.float32) / float(depth_scale)

    if d.ndim != 2:
        raise ValueError(f"Expected HxW depth, got shape {d.shape} for {path}")

    return d.astype(np.float32)


def read_pair_list(
    list_path: str | Path,
    *,
    root: str | Path | None = None,
) -> list[tuple[Path, Path]]:
    """Parse a text file: one pair per line ``rgb_path<TAB>depth_path``.

    Columns must be separated by a single tab so paths with spaces are supported.
    Empty lines and lines starting with ``#`` are skipped. Relative paths resolve under
    ``root`` when given.
    """
    root = Path(root) if root is not None else None
    pairs: list[tuple[Path, Path]] = []
    text = Path(list_path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(
                f"Need two tab-separated columns (rgb<TAB>depth), got: {line!r}"
            )
        rgb, dep = parts[0].strip(), parts[1].strip()
        p_rgb = Path(rgb)
        p_dep = Path(dep)
        if root is not None:
            if not p_rgb.is_absolute():
                p_rgb = root / p_rgb
            if not p_dep.is_absolute():
                p_dep = root / p_dep
        pairs.append((p_rgb, p_dep))
    if not pairs:
        raise ValueError(f"No samples parsed from {list_path}")
    return pairs
