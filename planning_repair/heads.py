#!/usr/bin/env python3
"""Auxiliary heads used by the planning-repair experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class EmbeddingAuxConfig:
    latent_dim: int = 256
    hidden_dim: int = 256
    action_slots: int = 4
    reach_budgets: tuple[int, ...] = (1, 3, 5, 8, 12)
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reach_budgets"] = list(self.reach_budgets)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingAuxConfig":
        return cls(
            latent_dim=int(data.get("latent_dim", 256)),
            hidden_dim=int(data.get("hidden_dim", 256)),
            action_slots=int(data.get("action_slots", 4)),
            reach_budgets=tuple(int(x) for x in data.get("reach_budgets", [1, 3, 5, 8, 12])),
            dropout=float(data.get("dropout", 0.0)),
        )


class EmbeddingAuxHeads(nn.Module):
    """Planning-relevant probes attached directly to post-projector embeddings."""

    def __init__(self, config: EmbeddingAuxConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        dropout = config.dropout
        self.trunk = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.agent_xy = nn.Linear(hidden, 2)
        self.goal_xy = nn.Linear(hidden, 2)
        self.valid_action = nn.Linear(hidden, config.action_slots)
        self.action_logits = nn.Linear(hidden, config.action_slots)
        self.bfs_distance_norm = nn.Linear(hidden, 1)
        self.reachability = nn.Linear(hidden, len(config.reach_budgets))

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(embedding)
        return {
            "agent_xy": torch.sigmoid(self.agent_xy(hidden)),
            "goal_xy": torch.sigmoid(self.goal_xy(hidden)),
            "valid_action_logits": self.valid_action(hidden),
            "action_logits": self.action_logits(hidden),
            "bfs_distance_norm": torch.sigmoid(self.bfs_distance_norm(hidden)).squeeze(-1),
            "reachability_logits": self.reachability(hidden),
        }


@dataclass(frozen=True)
class PrefixPredictorConfig:
    latent_dim: int = 256
    hidden_dim: int = 256
    action_vocab_size: int = 5
    max_horizon: int = 12
    num_layers: int = 1
    dropout: float = 0.0
    residual_prediction: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrefixPredictorConfig":
        return cls(
            latent_dim=int(data.get("latent_dim", 256)),
            hidden_dim=int(data.get("hidden_dim", 256)),
            action_vocab_size=int(data.get("action_vocab_size", 5)),
            max_horizon=int(data.get("max_horizon", 12)),
            num_layers=int(data.get("num_layers", 1)),
            dropout=float(data.get("dropout", 0.0)),
            residual_prediction=bool(data.get("residual_prediction", True)),
        )


class ActionPrefixPredictor(nn.Module):
    """Predict future latents for all prefixes of a candidate action sequence.

    This is a compact Fast-LeWM-style auxiliary model. It does not replace the
    original one-step LeWM predictor; it provides a direct multi-horizon head
    that can be trained and evaluated without breaking old checkpoints.
    """

    def __init__(self, config: PrefixPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.action_embedding = nn.Embedding(config.action_vocab_size, config.hidden_dim)
        self.initial_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def forward(self, initial_embedding: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if initial_embedding.ndim != 2:
            raise ValueError(
                f"expected initial_embedding [B,D], got {tuple(initial_embedding.shape)}"
            )
        if actions.ndim != 2:
            raise ValueError(f"expected actions [B,H], got {tuple(actions.shape)}")
        if actions.shape[1] > self.config.max_horizon:
            raise ValueError(
                f"horizon {actions.shape[1]} exceeds max_horizon {self.config.max_horizon}"
            )

        action_tokens = self.action_embedding(actions.long())
        h0 = torch.tanh(self.initial_projection(initial_embedding)).unsqueeze(0)
        h0 = h0.repeat(self.config.num_layers, 1, 1).contiguous()
        hidden, _ = self.gru(action_tokens, h0)
        delta_or_latent = self.output(hidden)
        if self.config.residual_prediction:
            return initial_embedding.unsqueeze(1) + delta_or_latent
        return delta_or_latent


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy against a possibly tied set of optimal actions."""

    if logits.shape != target_mask.shape:
        raise ValueError(
            f"logits/target shape mismatch: {tuple(logits.shape)} vs {tuple(target_mask.shape)}"
        )
    valid = target_mask.sum(dim=-1) > 0
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    targets = target_mask[valid]
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1.0)
    log_probs = F.log_softmax(logits[valid], dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def load_aux_heads(data: dict[str, Any], device: torch.device) -> EmbeddingAuxHeads | None:
    if "aux_state_dict" not in data or "aux_config" not in data:
        return None
    cfg = EmbeddingAuxConfig.from_dict(data["aux_config"])
    heads = EmbeddingAuxHeads(cfg).to(device)
    heads.load_state_dict(data["aux_state_dict"], strict=True)
    heads.eval()
    for param in heads.parameters():
        param.requires_grad = False
    return heads


def load_prefix_predictor(
    data: dict[str, Any],
    device: torch.device,
) -> ActionPrefixPredictor | None:
    if "prefix_state_dict" not in data or "prefix_config" not in data:
        return None
    cfg = PrefixPredictorConfig.from_dict(data["prefix_config"])
    predictor = ActionPrefixPredictor(cfg).to(device)
    predictor.load_state_dict(data["prefix_state_dict"], strict=True)
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad = False
    return predictor

