"""RGB-depth pair dataset for AdaBins training."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from depthanything.io_util import load_depth_map, read_pair_list
from depthanything.transforms import Crop, NormalizeImage, PrepareForNet, Resize


class CustomDepthPairDataset(Dataset):
    """Paired RGB image and metric depth for AdaBins training."""

    def __init__(
        self,
        list_path: str | Path,
        mode: str,
        size: tuple[int, int] = (518, 518),
        *,
        depth_scale: float = 1.0,
        list_root: str | Path | None = None,
    ) -> None:
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")

        self.pairs = read_pair_list(list_path, root=list_root)
        self.mode = mode
        self.size = size
        self.depth_scale = depth_scale

        net_w, net_h = size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if mode == "train" else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        rgb_path, depth_path = self.pairs[idx]

        image = cv2.imread(str(rgb_path))
        if image is None:
            raise FileNotFoundError(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = load_depth_map(depth_path, depth_scale=self.depth_scale)

        sample = self.transform({"image": image, "depth": depth})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])

        d = sample["depth"]
        sample["valid_mask"] = torch.isfinite(d) & (d > 0)
        sample["depth"] = torch.where(sample["valid_mask"], d, torch.zeros_like(d))

        sample["image_path"] = str(rgb_path)
        return sample
