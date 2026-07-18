"""Error-aware residual adapter — the only trainable module during feedback."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _group_count(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ErrorAwareResidualAdapter(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        error_channels: int = 1,
        error_embed_channels: int = 16,
        bottleneck_channels: int = 64,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.error_embed_channels = error_embed_channels
        self.feature_norm = nn.GroupNorm(_group_count(channels), channels)
        self.error_dropout = nn.Dropout(p=0.3)
        self.error_encoder = nn.Sequential(
            nn.Conv2d(error_channels, error_embed_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(error_embed_channels, error_embed_channels, 3, padding=1),
            nn.GELU(),
        )
        self.delta_net = nn.Sequential(
            nn.Conv2d(channels + error_embed_channels, bottleneck_channels, 1),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, channels, 1),
        )
        if zero_init:
            nn.init.zeros_(self.delta_net[-1].weight)
            nn.init.zeros_(self.delta_net[-1].bias)

    def _encode_error(self, features: Tensor, error_map: Optional[Tensor]) -> Tensor:
        B, _, H, W = features.shape
        if error_map is None:
            return features.new_zeros(B, self.error_embed_channels, H, W)
        resized = F.interpolate(error_map, size=(H, W), mode="bilinear", align_corners=False)
        return self.error_dropout(self.error_encoder(resized))

    def forward(self, features: Tensor, error_map: Optional[Tensor] = None):
        error_embed = self._encode_error(features, error_map)
        delta = self.delta_net(torch.cat([self.feature_norm(features), error_embed], dim=1))
        if error_map is None:
            gate = 1.0
        else:
            error_strength = error_map.abs().mean(dim=(1, 2, 3), keepdim=True)
            gate = 1.0 - torch.exp(-5.0 * error_strength)
        delta = gate * delta
        return features + delta, delta