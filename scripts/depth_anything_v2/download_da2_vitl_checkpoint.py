#!/usr/bin/env python3
"""Download Depth Anything V2 ViT-L checkpoint in one command.

Examples:
  python scripts/depth_anything_v2/download_da2_vitb_checkpoint.py
  python scripts/depth_anything_v2/download_da2_vitb_checkpoint.py --force
  python scripts/depth_anything_v2/download_da2_vitb_checkpoint.py --out checkpoints/depth_anything_v2_vitb.pth
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/"
    "resolve/main/depth_anything_v2_vitb.pth?download=true"
)
KNOWN_SHA256 = "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345"
DEFAULT_OUT = Path("checkpoints/depth_anything_v2_vitb.pth")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urlopen(url) as resp, tmp_path.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Download Depth Anything V2 ViT-B checkpoint")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output path (default: {DEFAULT_OUT})")
    p.add_argument("--force", action="store_true", help="Overwrite existing file if present")
    args = p.parse_args()

    out_path = args.out
    if out_path.exists() and not args.force:
        print(f"[skip] checkpoint already exists: {out_path}")
        print("       use --force to re-download")
    else:
        if out_path.exists():
            out_path.unlink()
        print(f"[download] {CHECKPOINT_URL}")
        print(f"           -> {out_path}")
        download_file(CHECKPOINT_URL, out_path)
        print("[done] download complete")

    digest = sha256_of(out_path)
    print(f"[sha256] {digest}")
    if digest == KNOWN_SHA256:
        print("[ok] checksum matches official file")
    else:
        print("[warn] checksum does not match expected official hash")
        print(f"       expected: {KNOWN_SHA256}")


if __name__ == "__main__":
    main()
