"""AdaBins model definition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adabins.mini_vit import MiniViT


class UpSampleBN(nn.Module):
    def __init__(self, skip_input: int, output_features: int) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor, concat_with: torch.Tensor) -> torch.Tensor:
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode="bilinear",
            align_corners=True,
        )
        return self._net(torch.cat([up_x, concat_with], dim=1))


class DecoderBN(nn.Module):
    def __init__(self, num_features: int = 2048, num_classes: int = 1, bottleneck_features: int = 2048) -> None:
        super().__init__()
        features = int(num_features)
        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        self.up1 = UpSampleBN(skip_input=features + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=(features // 2) + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=(features // 4) + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=(features // 8) + 16 + 8, output_features=features // 16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class Encoder(nn.Module):
    def __init__(self, backend: nn.Module) -> None:
        super().__init__()
        self.original_model = backend

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [x]
        for key, module in self.original_model._modules.items():
            if key == "blocks":
                for _, block in module._modules.items():
                    features.append(block(features[-1]))
            else:
                features.append(module(features[-1]))
        return features


class UnetAdaptiveBins(nn.Module):
    def __init__(
        self,
        backend: nn.Module,
        n_bins: int = 100,
        min_val: float = 0.1,
        max_val: float = 10.0,
        norm: str = "linear",
    ) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = MiniViT(
            128,
            n_query_channels=128,
            patch_size=16,
            dim_out=n_bins,
            embedding_dim=128,
            norm=norm,
        )
        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unet_out = self.decoder(self.encoder(x))
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode="constant", value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, n_bins = centers.size()
        centers = centers.view(n, n_bins, 1, 1)
        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return bin_edges, pred

    def get_1x_lr_params(self):
        return self.encoder.parameters()

    def get_10x_lr_params(self):
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for module in modules:
            yield from module.parameters()

    @classmethod
    def build(
        cls,
        n_bins: int,
        *,
        min_val: float,
        max_val: float,
        norm: str,
        backend_name: str = "tf_efficientnet_b5_ap",
        backend_pretrained: bool = True,
    ) -> "UnetAdaptiveBins":
        backend = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            backend_name,
            pretrained=backend_pretrained,
        )
        backend.global_pool = nn.Identity()
        backend.classifier = nn.Identity()
        return cls(backend, n_bins=n_bins, min_val=min_val, max_val=max_val, norm=norm)
