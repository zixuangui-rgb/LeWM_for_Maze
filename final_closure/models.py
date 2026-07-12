"""Model definitions frozen for the final baseline addendum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hdwm.config import LEWMCNNConfig, ProcgenMazeConfig
from scripts.train.train_dim256 import Unisize256


@dataclass(frozen=True)
class BCPolicyConfig:
    observation_channels: int = 5
    stem_channels: int = 64
    hidden_channels: int = 128
    dropout: float = 0.3
    action_count: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BCPolicyConfig:
        return cls(**value)


class ResidualBlock(nn.Module):
    """The residual block used by the historical DeepCNN BC baseline."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(inputs) + inputs)


class DeepCNNPolicy(nn.Module):
    """Historical BC backbone with a four-direction protocol-aligned output."""

    def __init__(self, config: BCPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or BCPolicyConfig()
        stem = self.config.stem_channels
        hidden = self.config.hidden_channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.observation_channels, stem, 3, padding=1),
            nn.BatchNorm2d(stem),
            nn.ReLU(),
        )
        self.res1 = nn.Sequential(ResidualBlock(stem), ResidualBlock(stem))
        self.down = nn.Sequential(
            nn.Conv2d(stem, hidden, 3, padding=1, stride=2),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            ResidualBlock(hidden),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, self.config.action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(observations)
        hidden = self.res1(hidden)
        hidden = self.down(hidden)
        hidden = self.pool(hidden).flatten(1)
        return self.mlp(hidden)


def make_lewm_config(train_config: dict[str, Any]) -> LEWMCNNConfig:
    """Build the exact variable-size Unisize256 configuration used historically."""

    environment = ProcgenMazeConfig(
        height=25,
        width=25,
        observation_channels=5,
        p_noise=0.0,
        p_noop=0.0,
        p_action_turn=0.0,
        p_action_stay=0.0,
        resample_maze_per_sequence=False,
    )
    return LEWMCNNConfig(
        env_config=environment,
        latent_dim=int(train_config["latent_dim"]),
        cnn_channels=tuple(int(value) for value in train_config["cnn_channels"]),
        latent_batch_norm=bool(train_config["latent_batch_norm"]),
        embedding_stage=str(train_config["embedding_stage"]),
        sigreg_stage=str(train_config["sigreg_stage"]),
        predictor_heads=int(train_config["predictor_heads"]),
    )


def serialize_lewm_config(config: LEWMCNNConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def deserialize_lewm_config(value: dict[str, Any]) -> LEWMCNNConfig:
    return LEWMCNNConfig.model_validate(value)


def build_lewm(train_config: dict[str, Any]) -> tuple[Unisize256, LEWMCNNConfig]:
    config = make_lewm_config(train_config)
    return Unisize256(config, max_size=int(train_config["max_size_embedding"])), config


__all__ = [
    "BCPolicyConfig",
    "DeepCNNPolicy",
    "ResidualBlock",
    "build_lewm",
    "deserialize_lewm_config",
    "make_lewm_config",
    "serialize_lewm_config",
]
