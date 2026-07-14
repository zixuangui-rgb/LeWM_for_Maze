"""Trainable planner heads that operate only on pooled JEPA vectors."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vector_jepa_planner_frontier import REACHABILITY_BINS
from vector_jepa_planner_frontier.schemas import PlannerKind, ProposalKind


def required_head_names(method: Any) -> set[str]:
    names: set[str] = set()
    if (
        method.scorer.verifier_weight > 0.0
        or method.planner.kind == PlannerKind.BIDIRECTIONAL
    ):
        names.add("verifier")
    if (
        method.scorer.reachability_weight > 0.0
        or method.planner.kind == PlannerKind.VECTOR_DTS
    ):
        names.add("reachability")
    if method.memory.enabled or method.planner.kind == PlannerKind.BIDIRECTIONAL:
        names.add("join")
    if method.proposal.learned_weight > 0.0:
        names.add(
            "denoising_proposal"
            if method.proposal.kind == ProposalKind.DISCRETE_DENOISING
            else "autoregressive_proposal"
        )
    if method.planner.kind == PlannerKind.VECTOR_DTS:
        names.add("dts")
    if method.scorer.counterexample_ranker_weight > 0.0:
        names.add("ranker")
    return names


def pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("latent pairs must share shape [batch, latent_dim]")
    return torch.cat([left, right, right - left, left * right], dim=-1)


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    if min(input_dim, hidden_dim, output_dim) <= 0:
        raise ValueError("MLP dimensions must be positive")
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class HeadConfig:
    latent_dim: int = 256
    hidden_dim: int = 512
    action_count: int = 4
    horizon: int = 12
    reachability_bins: tuple[int, ...] = REACHABILITY_BINS

    def __post_init__(self) -> None:
        if min(self.latent_dim, self.hidden_dim, self.action_count, self.horizon) <= 0:
            raise ValueError("head dimensions must be positive")
        if tuple(sorted(set(self.reachability_bins))) != self.reachability_bins:
            raise ValueError("reachability bins must be unique and increasing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HeadConfig:
        payload = dict(value)
        if "reachability_bins" in payload:
            payload["reachability_bins"] = tuple(payload["reachability_bins"])
        return cls(**payload)


class ActionConsistencyVerifier(nn.Module):
    """Inverse-dynamics verifier q(a | z_t, z_t+1)."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.net = make_mlp(
            4 * config.latent_dim, config.hidden_dim, config.action_count
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features(source, target))

    def action_nll(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        slots = action_ids.to(dtype=torch.long) - 1
        if bool(((slots < 0) | (slots >= self.config.action_count)).any()):
            raise ValueError("verifier action IDs must be in [1, 4]")
        return (
            -F.log_softmax(self(source, target), dim=-1)
            .gather(1, slots.reshape(-1, 1))
            .squeeze(1)
        )


class DistributionalReachability(nn.Module):
    """Monotone CDF P(D <= b | z, z_goal) over locked distance bins."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = make_mlp(
            4 * config.latent_dim,
            config.hidden_dim,
            config.hidden_dim,
        )
        self.base = nn.Linear(config.hidden_dim, 1)
        self.positive_increments = nn.Linear(
            config.hidden_dim, len(config.reachability_bins) - 1
        )

    def logits(self, source: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(pair_features(source, goal))
        base = self.base(hidden)
        if len(self.config.reachability_bins) == 1:
            return base
        increments = F.softplus(self.positive_increments(hidden))
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)

    def forward(self, source: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(source, goal))

    def probability_for_budget(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        remaining_budget: int,
    ) -> torch.Tensor:
        budget = max(1, min(int(remaining_budget), self.config.reachability_bins[-1]))
        index = next(
            index
            for index, boundary in enumerate(self.config.reachability_bins)
            if boundary >= budget
        )
        return self(source, goal)[:, index]

    def loss(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        bfs_distance: torch.Tensor,
        *,
        monotonic_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.logits(source, goal)
        bins = torch.tensor(
            self.config.reachability_bins,
            dtype=bfs_distance.dtype,
            device=bfs_distance.device,
        )
        targets = (bfs_distance.reshape(-1, 1) <= bins.reshape(1, -1)).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probabilities = torch.sigmoid(logits)
        violations = F.relu(probabilities[:, :-1] - probabilities[:, 1:])
        monotonic = violations.mean() if violations.numel() else logits.new_zeros(())
        total = bce + float(monotonic_weight) * monotonic
        return total, {"reachability_bce": bce, "monotonic_penalty": monotonic}


class StateJoinHead(nn.Module):
    """Binary same-state or k-step join classifier for transposition memory."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.net = make_mlp(4 * config.latent_dim, config.hidden_dim, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features(left, right)).squeeze(-1)

    def probability(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(left, right))


class AutoregressiveProposal(nn.Module):
    """Goal-conditioned action-chunk proposal; search still performs selection."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.condition = make_mlp(
            4 * config.latent_dim, config.hidden_dim, config.hidden_dim
        )
        self.action_embedding = nn.Embedding(config.action_count + 1, config.hidden_dim)
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.output = nn.Linear(config.hidden_dim, config.action_count)

    def forward(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_actions.ndim != 2:
            raise ValueError("teacher actions must have shape [batch, horizon]")
        slots = teacher_actions.to(dtype=torch.long) - 1
        if bool(((slots < 0) | (slots >= self.config.action_count)).any()):
            raise ValueError("proposal teacher action IDs must be in [1, 4]")
        start_token = torch.full(
            (slots.shape[0], 1),
            self.config.action_count,
            dtype=torch.long,
            device=slots.device,
        )
        shifted = torch.cat([start_token, slots[:, :-1]], dim=1)
        inputs = self.action_embedding(shifted)
        hidden = self.condition(pair_features(source, goal)).unsqueeze(0)
        output, _ = self.gru(inputs, hidden)
        return self.output(output)

    def next_logits(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        prefix: torch.Tensor,
    ) -> torch.Tensor:
        if prefix.ndim != 2:
            raise ValueError("proposal prefix must have shape [batch, length]")
        batch = source.shape[0]
        start_token = torch.full(
            (batch, 1),
            self.config.action_count,
            dtype=torch.long,
            device=source.device,
        )
        if prefix.shape[1] == 0:
            tokens = start_token
        else:
            slots = prefix.to(dtype=torch.long) - 1
            if bool(((slots < 0) | (slots >= self.config.action_count)).any()):
                raise ValueError("proposal prefix action IDs must be in [1, 4]")
            tokens = torch.cat([start_token, slots], dim=1)
        hidden = self.condition(pair_features(source, goal)).unsqueeze(0)
        output, _ = self.gru(self.action_embedding(tokens), hidden)
        return self.output(output[:, -1])


class DiscreteDenoisingProposal(nn.Module):
    """Mask-and-denoise action-chunk generator for multimodal proposals."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_dim
        self.mask_slot = config.action_count
        self.action_embedding = nn.Embedding(config.action_count + 1, width)
        self.position_embedding = nn.Embedding(config.horizon, width)
        self.condition = make_mlp(4 * config.latent_dim, width, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=8,
            dim_feedforward=2 * width,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(width, config.action_count)

    def forward(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        noisy_slots: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_slots.ndim != 2 or noisy_slots.shape[1] != self.config.horizon:
            raise ValueError("denoising input has the wrong action-chunk shape")
        if bool(((noisy_slots < 0) | (noisy_slots > self.mask_slot)).any()):
            raise ValueError("denoising slots are outside the token vocabulary")
        positions = torch.arange(self.config.horizon, device=noisy_slots.device)
        hidden = self.action_embedding(noisy_slots)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden + self.condition(pair_features(source, goal)).unsqueeze(1)
        return self.output(self.transformer(hidden))


class VectorDTSHead(nn.Module):
    """Learned expansion policy and distributional value for Vector-DTS."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = make_mlp(
            4 * config.latent_dim, config.hidden_dim, config.hidden_dim
        )
        self.policy = nn.Linear(config.hidden_dim, config.action_count)
        self.value = nn.Linear(config.hidden_dim, len(config.reachability_bins))
        self.uncertainty = nn.Linear(config.hidden_dim, 1)

    def forward(
        self, source: torch.Tensor, goal: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = self.trunk(pair_features(source, goal))
        return {
            "policy_logits": self.policy(hidden),
            "value_logits": self.value(hidden),
            "uncertainty": F.softplus(self.uncertainty(hidden)).squeeze(-1),
        }


class CounterexampleRanker(nn.Module):
    """Score complete imagined chunks for pairwise hard-negative ranking."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = 4 * config.latent_dim + config.action_count
        self.net = make_mlp(input_dim, config.hidden_dim, 1)

    def forward(
        self,
        source: torch.Tensor,
        terminal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if actions.ndim != 2:
            raise ValueError("ranker actions must have shape [batch, horizon]")
        slots = actions.to(dtype=torch.long) - 1
        histogram = F.one_hot(slots, self.config.action_count).float().mean(dim=1)
        features = torch.cat([pair_features(source, terminal), histogram], dim=-1)
        return self.net(features).squeeze(-1)


def pairwise_ranking_loss(
    good_score: torch.Tensor, bad_score: torch.Tensor
) -> torch.Tensor:
    if good_score.shape != bad_score.shape:
        raise ValueError("paired ranker scores must share shape")
    return -F.logsigmoid(good_score - bad_score).mean()


def expected_distance_from_logits(
    logits: torch.Tensor,
    bins: tuple[int, ...] = REACHABILITY_BINS,
) -> torch.Tensor:
    if logits.shape[-1] != len(bins):
        raise ValueError("value logits do not match reachability bins")
    weights = torch.softmax(logits, dim=-1)
    values = torch.tensor(bins, dtype=logits.dtype, device=logits.device)
    return (weights * values).sum(dim=-1)


def inverse_sqrt_parameter_init(module: nn.Module) -> None:
    """Deterministic, non-pathological initialization for planner heads."""

    if isinstance(module, nn.Linear):
        bound = 1.0 / math.sqrt(max(module.in_features, 1))
        nn.init.uniform_(module.weight, -bound, bound)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


__all__ = [
    "ActionConsistencyVerifier",
    "AutoregressiveProposal",
    "CounterexampleRanker",
    "DiscreteDenoisingProposal",
    "DistributionalReachability",
    "HeadConfig",
    "StateJoinHead",
    "VectorDTSHead",
    "expected_distance_from_logits",
    "inverse_sqrt_parameter_init",
    "pair_features",
    "pairwise_ranking_loss",
    "required_head_names",
]
