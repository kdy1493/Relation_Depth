"""Core layers used by AdaBins."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTransformerEncoder(nn.Module):
    """Transformer over patch embeddings from decoder features."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int = 10,
        embedding_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        self.embedding_convPxP = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_convPxP(x).flatten(2)
        embeddings = embeddings + self.positional_encodings[: embeddings.shape[2], :].T.unsqueeze(0)
        embeddings = embeddings.permute(2, 0, 1)  # S, N, E
        # Run transformer in float32 to avoid attention-score overflow in float16.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out = self.transformer_encoder(embeddings.float())
        return out.to(embeddings.dtype)


class PixelWiseDotProduct(nn.Module):
    """Compute per-pixel dot-product attention maps."""

    def forward(self, x: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        _, n_queries, q_dim = queries.size()
        if c != q_dim:
            raise ValueError(
                "Channel mismatch between feature map and query embeddings: "
                f"{c} vs {q_dim}."
            )
        y = torch.matmul(
            x.view(n, c, h * w).permute(0, 2, 1),
            queries.permute(0, 2, 1),
        )
        return y.permute(0, 2, 1).view(n, n_queries, h, w)
