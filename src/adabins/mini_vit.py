"""Mini vision transformer block used by AdaBins."""

from __future__ import annotations

import torch
import torch.nn as nn

from adabins.layers import PatchTransformerEncoder, PixelWiseDotProduct


class MiniViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_query_channels: int = 128,
        patch_size: int = 16,
        dim_out: int = 256,
        embedding_dim: int = 128,
        num_heads: int = 4,
        norm: str = "linear",
    ) -> None:
        super().__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.dot_product_layer = PixelWiseDotProduct()
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, dim_out),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_transformer(x.clone())
        x = self.conv3x3(x)

        regression_head = tokens[0, ...]
        queries = tokens[1 : self.n_query_channels + 1, ...].permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)

        y = self.regressor(regression_head)
        if self.norm == "linear":
            y = torch.relu(y) + 0.1
        elif self.norm == "softmax":
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)

        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps
