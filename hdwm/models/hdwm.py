"""Minimal HDWM architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hdwm.config import HDWMConfig
from hdwm.models.shared import (
    ActionConditionedCausalTransformer,
    ObservationEncoder,
    action_indices,
    add_temporal_position_embedding,
    make_action_embedding,
    make_temporal_position_embedding,
)


@dataclass(frozen=True)
class HDWMOutput:
    """Forward outputs with explicit shape contracts.

    Shapes:
        encoded: [B, T, L, M]
        prior: [B, T, M]
        evidence: [B, T, M]
        posterior: [B, T, M]
        readout_attention: [B, T, L]
    """

    encoded: torch.Tensor
    prior: torch.Tensor
    evidence: torch.Tensor
    posterior: torch.Tensor
    readout_attention: torch.Tensor


class CausalPriorTransformer(nn.Module):
    """Causal temporal transformer that predicts prior from history only.

    Shapes:
        encoded: [B, T, L, M]
        actions: [B, T-1]
        output: [B, T, M]
    """

    def __init__(self, config: HDWMConfig) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        self.action_embedding = make_action_embedding(config, model_dim)
        self.temporal_position_embedding = make_temporal_position_embedding(
            config,
            config.max_sequence_length,
            model_dim,
        )
        self.prior_transformer = ActionConditionedCausalTransformer(
            model_dim=model_dim,
            layers=config.prior_layers,
            heads=config.prior_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            rotary=config.uses_rotary_temporal_position_encoding,
        )
        self.initial_prior = nn.Parameter(torch.zeros(model_dim))

    def forward(self, encoded: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 4:
            raise ValueError(f"expected encoded rank 4, got {encoded.ndim}")
        batch_size, sequence_length, _, model_dim = encoded.shape
        if model_dim != self.config.effective_model_dim:
            raise ValueError(
                f"expected model dim {self.config.effective_model_dim}, got {model_dim}"
            )
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )
        # initial_prior: [B, 1, M]
        initial_prior = self.initial_prior.view(1, 1, model_dim).expand(
            batch_size, 1, -1
        )
        if sequence_length == 1:
            return initial_prior

        if actions.shape != (batch_size, sequence_length - 1):
            raise ValueError(
                f"expected actions shape {(batch_size, sequence_length - 1)}, "
                f"got {tuple(actions.shape)}"
            )

        # frame_summary: [B, T-1, M]
        frame_summary = encoded[:, :-1].mean(dim=2)
        # action_condition: [B, T-1, M]
        action_condition = self.action_embedding(action_indices(self.config, actions))
        # prior_inputs: [B, T-1, M]
        prior_inputs = add_temporal_position_embedding(
            frame_summary,
            self.temporal_position_embedding,
        )
        # prior_context: [B, T-1, M]
        prior_context = self.prior_transformer(prior_inputs, action_condition)
        return torch.cat([initial_prior, prior_context], dim=1)


class PriorGuidedReadout(nn.Module):
    """Cross-attention readout using standard Q/K/V/O projections.

    Shapes:
        prior: [B, T, M]
        encoded: [B, T, L, M]
        evidence: [B, T, M]
        attention: [B, T, L]
    """

    def __init__(self, config: HDWMConfig) -> None:
        super().__init__()
        model_dim = config.effective_model_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=config.readout_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        prior: torch.Tensor,
        encoded: torch.Tensor,
        normalize_query: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded.ndim != 4:
            raise ValueError(f"expected encoded rank 4, got {encoded.ndim}")
        batch_size, sequence_length, observation_size, model_dim = encoded.shape
        if prior.shape != (batch_size, sequence_length, model_dim):
            raise ValueError(
                f"expected prior shape {(batch_size, sequence_length, model_dim)}, "
                f"got {tuple(prior.shape)}"
            )
        # query: [B*T, 1, M]
        query = prior.reshape(batch_size * sequence_length, 1, model_dim)
        if normalize_query:
            query = F.normalize(query, dim=-1)
        # key_value: [B*T, L, M]
        key_value = encoded.reshape(
            batch_size * sequence_length, observation_size, model_dim
        )
        # attended: [B*T, 1, M], attention: [B*T, 1, L]
        attended, attention = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=True,
        )
        # evidence: [B, T, M]
        evidence = self.output_norm(attended.squeeze(dim=1)).view(
            batch_size,
            sequence_length,
            model_dim,
        )
        return evidence, attention.squeeze(dim=1).view(
            batch_size, sequence_length, observation_size
        )

    def pairwise(
        self,
        prior: torch.Tensor,
        encoded: torch.Tensor,
        normalize_query: bool = False,
    ) -> torch.Tensor:
        """Read every encoded observation with every prior.

        Shapes:
            prior: [N, M]
            encoded: [N, L, M]
            output: [N, N, M]
        """

        if encoded.ndim != 3:
            raise ValueError(f"expected encoded rank 3, got {encoded.ndim}")
        num_items, observation_size, model_dim = encoded.shape
        if prior.shape != (num_items, model_dim):
            raise ValueError(
                f"expected prior shape {(num_items, model_dim)}, "
                f"got {tuple(prior.shape)}"
            )
        # query: [N*N, 1, M]
        query = (
            prior.view(num_items, 1, 1, model_dim)
            .expand(num_items, num_items, 1, model_dim)
            .reshape(num_items * num_items, 1, model_dim)
        )
        if normalize_query:
            query = F.normalize(query, dim=-1)
        # key_value: [N*N, L, M]
        key_value = (
            encoded.view(1, num_items, observation_size, model_dim)
            .expand(num_items, num_items, observation_size, model_dim)
            .reshape(num_items * num_items, observation_size, model_dim)
        )
        # attended: [N*N, 1, M]
        attended, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        # evidence: [N, N, M]
        evidence = self.output_norm(attended.squeeze(dim=1))
        return evidence.view(num_items, num_items, model_dim)


class PosteriorProjector(nn.Module):
    """Project readout evidence into posterior model space.

    Shapes:
        evidence: [B, T, M]
        posterior: [B, T, M]
    """

    def __init__(self, config: HDWMConfig) -> None:
        super().__init__()
        model_dim = config.effective_model_dim
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * config.mlp_ratio),
            nn.GELU(),
            nn.Linear(model_dim * config.mlp_ratio, model_dim),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence)


class HDWM(nn.Module):
    """Minimal HDWM model."""

    def __init__(self, config: HDWMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.prior_transformer = CausalPriorTransformer(config)
        self.readout = PriorGuidedReadout(config)
        self.posterior_projector = PosteriorProjector(config)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        normalize_prior_for_readout: bool = False,
    ) -> HDWMOutput:
        # observations: [B, T, L]
        # actions: [B, T-1]
        encoded = self.encoder(observations)
        prior = self.prior_transformer(encoded, actions)
        evidence, readout_attention = self.readout(
            prior, encoded, normalize_query=normalize_prior_for_readout
        )
        posterior = self.posterior_projector(evidence)
        return HDWMOutput(
            encoded=encoded,
            prior=prior,
            evidence=evidence,
            posterior=posterior,
            readout_attention=readout_attention,
        )

    def conditional_mi_logits(
        self,
        prior: torch.Tensor,
        encoded: torch.Tensor,
        posterior: torch.Tensor,
        temperature: float,
        normalize_prior: bool = False,
        normalize_posterior: bool = True,
        normalize_evidence: bool = True,
    ) -> torch.Tensor:
        """Compute pairwise InfoNCE logits per sequence using shared readout attention.

        Shapes:
            prior: [B, T, M]
            encoded: [B, T, L, M]
            posterior: [B, T, M]
            output: [B, T, T]
        """

        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if encoded.ndim != 4:
            raise ValueError(f"expected encoded rank 4, got {encoded.ndim}")
        batch_size, sequence_length, observation_size, model_dim = encoded.shape
        expected_model_shape = (batch_size, sequence_length, model_dim)
        if prior.shape != expected_model_shape:
            raise ValueError(
                f"expected prior shape {expected_model_shape}, got {tuple(prior.shape)}"
            )
        if posterior.shape != expected_model_shape:
            raise ValueError(
                f"expected posterior shape {expected_model_shape}, "
                f"got {tuple(posterior.shape)}"
            )
        # Build pairwise queries and keys within each batch.
        # query: [B, T, T, M] -> [B*T*T, 1, M]
        query = prior.unsqueeze(2).expand(
            batch_size, sequence_length, sequence_length, model_dim
        )
        if normalize_prior:
            query = F.normalize(query, dim=-1)
        query = query.reshape(
            batch_size * sequence_length * sequence_length,
            1,
            model_dim,
        )
        # key_value: [B, T, T, L, M] -> [B*T*T, L, M]
        key_value = encoded.unsqueeze(1).expand(
            batch_size, sequence_length, sequence_length, observation_size, model_dim
        )
        key_value = key_value.reshape(
            batch_size * sequence_length * sequence_length, observation_size, model_dim
        )
        # attended: [B*T*T, 1, M]
        attended, _ = self.readout.attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        # pairwise_evidence: [B, T, T, M]
        evidence = self.readout.output_norm(attended.squeeze(dim=1))
        pairwise_evidence = evidence.view(
            batch_size, sequence_length, sequence_length, model_dim
        )
        # logits: [B, T, T]
        if normalize_posterior:
            posterior = F.normalize(posterior, dim=-1)
        if normalize_evidence:
            pairwise_evidence = F.normalize(pairwise_evidence, dim=-1)
        logits = torch.einsum("btd,btud->btu", posterior, pairwise_evidence)
        return logits / temperature
