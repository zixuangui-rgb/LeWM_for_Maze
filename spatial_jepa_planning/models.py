"""Full-resolution Spatial-JEPA and weight-shared maze planners."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        groups = _group_count(channels)
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(inputs + self.net(inputs))


@dataclass(frozen=True)
class SpatialRepresentationConfig:
    observation_channels: int = 5
    spatial_dim: int = 64
    planning_dim: int = 64
    encoder_blocks: int = 3
    predictor_blocks: int = 2
    action_vocab_size: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialRepresentationConfig:
        names = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in names if key in data})


class SpatialEncoder(nn.Module):
    """Stride-one encoder that keeps one latent token per maze cell."""

    def __init__(self, config: SpatialRepresentationConfig) -> None:
        super().__init__()
        channels = config.spatial_dim
        self.stem = nn.Sequential(
            nn.Conv2d(config.observation_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(channels) for _ in range(config.encoder_blocks)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(observations)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class SpatialProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ActionConditionedSpatialPredictor(nn.Module):
    """Predict the next spatial latent from a global discrete action."""

    def __init__(self, config: SpatialRepresentationConfig) -> None:
        super().__init__()
        channels = config.spatial_dim
        self.action_vocab_size = config.action_vocab_size
        self.input = nn.Conv2d(
            channels + config.action_vocab_size, channels, 3, padding=1
        )
        self.blocks = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(config.predictor_blocks))
        )
        self.output = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(f"expected latent [B,C,H,W], got {tuple(latent.shape)}")
        if actions.ndim != 1 or actions.shape[0] != latent.shape[0]:
            raise ValueError(
                f"expected actions [B] for batch {latent.shape[0]}, "
                f"got {tuple(actions.shape)}"
            )
        if bool(((actions < 0) | (actions >= self.action_vocab_size)).any()):
            raise ValueError("action id is outside the configured vocabulary")
        one_hot = F.one_hot(actions.long(), self.action_vocab_size).to(latent.dtype)
        planes = one_hot[:, :, None, None].expand(
            -1, -1, latent.shape[-2], latent.shape[-1]
        )
        delta = self.output(
            self.blocks(F.gelu(self.input(torch.cat([latent, planes], dim=1))))
        )
        return latent + delta


class SpatialMapDecoder(nn.Module):
    """Decode planning-relevant fields without an oracle map at inference."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels, 32)
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden),
        )
        self.wall = nn.Conv2d(hidden, 1, 1)
        self.agent = nn.Conv2d(hidden, 1, 1)
        self.goal = nn.Conv2d(hidden, 1, 1)
        self.valid = nn.Conv2d(hidden, 4, 1)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(latent)
        return {
            "wall_logits": self.wall(hidden).squeeze(1),
            "agent_logits": self.agent(hidden).squeeze(1),
            "goal_logits": self.goal(hidden).squeeze(1),
            "valid_logits": self.valid(hidden),
        }


class SpatialRepresentation(nn.Module):
    """Online Spatial-JEPA with separate dynamics and planning projectors."""

    def __init__(self, config: SpatialRepresentationConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SpatialEncoder(config)
        self.dynamics_projector = SpatialProjector(
            config.spatial_dim, config.spatial_dim
        )
        self.planning_projector = SpatialProjector(
            config.spatial_dim, config.planning_dim
        )
        self.predictor = ActionConditionedSpatialPredictor(config)
        self.map_decoder = SpatialMapDecoder(config.planning_dim)

    @staticmethod
    def _to_channels_first(
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if observations.ndim == 4:
            batch, height, width, channels = observations.shape
            return observations.permute(0, 3, 1, 2), (batch, height, width, channels)
        if observations.ndim == 5:
            batch, time, height, width, channels = observations.shape
            flat = observations.reshape(batch * time, height, width, channels)
            return flat.permute(0, 3, 1, 2), (batch, time, height, width, channels)
        raise ValueError(
            "expected observations [B,H,W,C] or [B,T,H,W,C], "
            f"got {tuple(observations.shape)}"
        )

    @staticmethod
    def _restore_time(latent: torch.Tensor, original: tuple[int, ...]) -> torch.Tensor:
        if len(original) == 4:
            return latent
        batch, time = original[:2]
        return latent.reshape(batch, time, *latent.shape[1:])

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        flat, original = self._to_channels_first(observations)
        return self._restore_time(self.encoder(flat), original)

    def dynamics_latent(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(observations)
        if encoded.ndim == 4:
            return self.dynamics_projector(encoded)
        batch, time = encoded.shape[:2]
        flat = encoded.reshape(batch * time, *encoded.shape[2:])
        projected = self.dynamics_projector(flat)
        return projected.reshape(batch, time, *projected.shape[1:])

    def planning_latent(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(observations)
        if encoded.ndim == 4:
            return self.planning_projector(encoded)
        batch, time = encoded.shape[:2]
        flat = encoded.reshape(batch * time, *encoded.shape[2:])
        projected = self.planning_projector(flat)
        return projected.reshape(batch, time, *projected.shape[1:])


@torch.no_grad()
def make_ema_target(online: SpatialRepresentation) -> SpatialRepresentation:
    target = copy.deepcopy(online)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad = False
    return target


@torch.no_grad()
def update_ema_target(
    online: SpatialRepresentation,
    target: SpatialRepresentation,
    momentum: float,
) -> None:
    if not 0.0 <= momentum < 1.0:
        raise ValueError("EMA momentum must be in [0, 1)")
    online_parameters = dict(online.named_parameters())
    for name, target_parameter in target.named_parameters():
        target_parameter.mul_(momentum).add_(
            online_parameters[name], alpha=1.0 - momentum
        )
    online_buffers = dict(online.named_buffers())
    for name, target_buffer in target.named_buffers():
        target_buffer.copy_(online_buffers[name])


@dataclass(frozen=True)
class PlannerConfig:
    input_channels: int = 5
    hidden_dim: int = 64
    planner_type: Literal["feedforward", "feedforward_dilated", "iterative"] = (
        "iterative"
    )
    depth: int = 8
    recall: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannerConfig:
        names = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in names if key in data})


@dataclass
class PlannerOutput:
    value: torch.Tensor
    policy_logits: torch.Tensor
    valid_logits: torch.Tensor
    hidden: torch.Tensor
    iterations: int


class PlannerReadout(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.value = nn.Conv2d(hidden_dim, 1, 1)
        self.policy = nn.Conv2d(hidden_dim, 4, 1)
        self.valid = nn.Conv2d(hidden_dim, 4, 1)

    def forward(self, hidden: torch.Tensor, iterations: int) -> PlannerOutput:
        return PlannerOutput(
            value=F.softplus(self.value(hidden).squeeze(1)),
            policy_logits=self.policy(hidden),
            valid_logits=self.valid(hidden),
            hidden=hidden,
            iterations=iterations,
        )


class FeedForwardPlanner(nn.Module):
    """Capacity-matched non-recurrent control for the recurrent planner."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.input_channels, config.hidden_dim, 3, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(config.hidden_dim), config.hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden_dim) for _ in range(config.depth)
        )
        self.readout = PlannerReadout(config.hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        iterations: int | None = None,
        deep_supervision_every: int = 0,
    ) -> list[PlannerOutput]:
        del iterations
        hidden = self.stem(features)
        outputs: list[PlannerOutput] = []
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            if deep_supervision_every > 0 and index % deep_supervision_every == 0:
                outputs.append(self.readout(hidden, index))
        if not outputs or outputs[-1].iterations != len(self.blocks):
            outputs.append(self.readout(hidden, len(self.blocks)))
        return outputs


class DilatedFeedForwardPlanner(FeedForwardPlanner):
    """Non-recurrent full-receptive-field control with fixed dilation stages."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__(config)
        dilation_cycle = (1, 2, 4, 8)
        self.blocks = nn.ModuleList(
            ResidualBlock(
                config.hidden_dim,
                dilation=dilation_cycle[index % len(dilation_cycle)],
            )
            for index in range(config.depth)
        )


class ConvGRURecallCell(nn.Module):
    def __init__(self, hidden_dim: int, recall: bool) -> None:
        super().__init__()
        self.recall = recall
        input_dim = hidden_dim * (2 if recall else 1)
        self.gates = nn.Conv2d(input_dim, hidden_dim * 2, 3, padding=1)
        self.candidate = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)

    def forward(
        self, hidden: torch.Tensor, recall_features: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([hidden, recall_features], dim=1) if self.recall else hidden
        reset, update = self.gates(inputs).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate_inputs = (
            torch.cat([reset * hidden, recall_features], dim=1)
            if self.recall
            else reset * hidden
        )
        candidate = torch.tanh(self.candidate(candidate_inputs))
        return (1.0 - update) * hidden + update * candidate


class IterativePlanner(nn.Module):
    """Weight-shared gated planner with immutable input recall every iteration."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.recall_encoder = nn.Sequential(
            nn.Conv2d(
                config.input_channels, config.hidden_dim, 3, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(config.hidden_dim), config.hidden_dim),
            nn.GELU(),
            ResidualBlock(config.hidden_dim),
        )
        self.initial = nn.Conv2d(config.hidden_dim, config.hidden_dim, 1)
        self.cell = ConvGRURecallCell(config.hidden_dim, recall=config.recall)
        self.readout = PlannerReadout(config.hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        iterations: int | None = None,
        deep_supervision_every: int = 0,
    ) -> list[PlannerOutput]:
        count = int(iterations if iterations is not None else self.config.depth)
        if count <= 0:
            raise ValueError("planner iterations must be positive")
        recall_features = self.recall_encoder(features)
        hidden = torch.tanh(self.initial(recall_features))
        outputs: list[PlannerOutput] = []
        for index in range(1, count + 1):
            hidden = self.cell(hidden, recall_features)
            if deep_supervision_every > 0 and index % deep_supervision_every == 0:
                outputs.append(self.readout(hidden, index))
        if not outputs or outputs[-1].iterations != count:
            outputs.append(self.readout(hidden, count))
        return outputs


def build_planner(config: PlannerConfig) -> nn.Module:
    if config.planner_type == "feedforward":
        return FeedForwardPlanner(config)
    if config.planner_type == "feedforward_dilated":
        return DilatedFeedForwardPlanner(config)
    if config.planner_type == "iterative":
        return IterativePlanner(config)
    raise ValueError(f"unsupported planner_type: {config.planner_type}")


def neighbor_stack(values: torch.Tensor, fill: float) -> torch.Tensor:
    """Return UP, DOWN, LEFT, RIGHT neighbor values as [B,4,H,W]."""

    if values.ndim != 3:
        raise ValueError(f"expected values [B,H,W], got {tuple(values.shape)}")
    up = F.pad(values[:, :-1, :], (0, 0, 1, 0), value=fill)
    down = F.pad(values[:, 1:, :], (0, 0, 0, 1), value=fill)
    left = F.pad(values[:, :, :-1], (1, 0, 0, 0), value=fill)
    right = F.pad(values[:, :, 1:], (0, 1, 0, 0), value=fill)
    return torch.stack([up, down, left, right], dim=1)


class OracleValueIteration(nn.Module):
    """Hardcoded differentiable VI control; this is an oracle, not a learned model."""

    def forward(
        self,
        wall_mask: torch.Tensor,
        goal_mask: torch.Tensor,
        iterations: int,
    ) -> PlannerOutput:
        if wall_mask.shape != goal_mask.shape or wall_mask.ndim != 3:
            raise ValueError("wall_mask and goal_mask must both be [B,H,W]")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        wall = wall_mask.bool()
        goal = goal_mask.bool()
        large = float(wall.shape[-2] * wall.shape[-1] + iterations + 1)
        values = torch.full_like(wall_mask, large, dtype=torch.float32)
        values = torch.where(goal, torch.zeros_like(values), values)
        for _ in range(iterations):
            neighbors = neighbor_stack(values, large)
            update = 1.0 + neighbors.min(dim=1).values
            values = torch.minimum(values, update)
            values = torch.where(goal, torch.zeros_like(values), values)
            values = torch.where(wall, torch.full_like(values, large), values)
        neighbor_values = neighbor_stack(values, large)
        valid = neighbor_stack(wall.float(), 1.0) < 0.5
        policy_logits = -neighbor_values.masked_fill(~valid, large)
        valid_logits = torch.where(valid, torch.full_like(neighbor_values, 20.0), -20.0)
        return PlannerOutput(
            value=values,
            policy_logits=policy_logits,
            valid_logits=valid_logits,
            hidden=values.unsqueeze(1),
            iterations=iterations,
        )


__all__ = [
    "FeedForwardPlanner",
    "IterativePlanner",
    "OracleValueIteration",
    "PlannerConfig",
    "PlannerOutput",
    "SpatialRepresentation",
    "SpatialRepresentationConfig",
    "build_planner",
    "make_ema_target",
    "neighbor_stack",
    "update_ema_target",
]
