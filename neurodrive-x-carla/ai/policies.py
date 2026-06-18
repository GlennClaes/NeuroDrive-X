"""Custom Stable-Baselines3 feature extractors for NeuroDrive X."""

from __future__ import annotations

from collections import OrderedDict

import gymnasium as gym
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NeuroDriveCNNExtractor(BaseFeaturesExtractor):
    """Multi-modal CNN extractor for CARLA camera, BEV, LiDAR, and scalar state.

    Stable-Baselines3's default MultiInputPolicy is strong, but an explicit
    extractor makes the research intent clear: visual observations are learned
    through convolutional branches, while LiDAR sectors and vehicle-state
    scalars flow through a compact MLP before fusion.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 384) -> None:
        super().__init__(observation_space, features_dim)
        self.visual_extractors = nn.ModuleDict()
        self.scalar_keys: list[str] = []
        visual_output_dim = 0
        scalar_input_dim = 0

        for key, subspace in observation_space.spaces.items():
            shape = subspace.shape or ()
            if key in {"camera_features", "bev_map"} and len(shape) == 3:
                channels = int(shape[0])
                self.visual_extractors[key] = _visual_branch(channels)
                visual_output_dim += 128
            else:
                self.scalar_keys.append(key)
                scalar_input_dim += int(_shape_product(shape))

        self.scalar_branch = nn.Sequential(
            nn.Linear(max(1, scalar_input_dim), 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        fused_dim = visual_output_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation dictionaries into one latent feature vector."""

        encoded: list[torch.Tensor] = []
        batch_size = next(iter(observations.values())).shape[0]
        for key, extractor in self.visual_extractors.items():
            encoded.append(extractor(observations[key].float()))

        scalar_tensors: list[torch.Tensor] = []
        for key in self.scalar_keys:
            value = observations[key].float()
            scalar_tensors.append(value.view(batch_size, -1))
        if scalar_tensors:
            scalar_input = torch.cat(scalar_tensors, dim=1)
        else:
            scalar_input = torch.zeros((batch_size, 1), device=next(iter(observations.values())).device)
        encoded.append(self.scalar_branch(scalar_input))
        return self.fusion(torch.cat(encoded, dim=1))


def _visual_branch(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(96, 128),
        nn.ReLU(),
    )


def _shape_product(shape: tuple[int, ...]) -> int:
    total = 1
    for value in shape:
        total *= int(value)
    return total

