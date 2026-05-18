"""Put official [ZoeDepth](https://github.com/isl-org/ZoeDepth) on ``sys.path`` (vendored under ``src/zoedepth/vendor``)."""

from __future__ import annotations

import sys
from pathlib import Path


def zoedepth_repo_root() -> Path:
    """Directory that contains the upstream ``zoedepth/`` package (ZoeDepth repo root)."""
    return Path(__file__).resolve().parent / "vendor"


def ensure_zoedepth_on_path() -> Path:
    root = zoedepth_repo_root()
    legacy = Path(__file__).resolve().parent.parent.parent / "third_party" / "ZoeDepth"
    if not (root / "zoedepth").is_dir() and (legacy / "zoedepth").is_dir():
        root = legacy
    if not (root / "zoedepth").is_dir():
        raise RuntimeError(
            f"ZoeDepth not found at {zoedepth_repo_root()} (or legacy {legacy}). Clone once:\n"
            "  git clone https://github.com/isl-org/ZoeDepth.git src/zoedepth/vendor"
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    # Our local package is also named "zoedepth". Extend its module search path
    # so imports like `zoedepth.models.*` resolve to the vendored upstream tree.
    pkg = sys.modules.get("zoedepth")
    vendor_pkg = str(root / "zoedepth")
    if pkg is not None and hasattr(pkg, "__path__"):
        if vendor_pkg not in pkg.__path__:
            pkg.__path__.append(vendor_pkg)
    return root
