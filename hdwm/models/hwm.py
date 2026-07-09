"""HWM architecture from PLAN_v2 with heteroscedastic next-latent loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hdwm.config import HWMConfig
from hdwm.models.shared import (
    ActionConditionedCausalTransformer,
    LatentEmbeddingProjector,
    ObservationEncoder,
    action_indices,
    add_temporal_position_embedding,
    make_action_embedding,
    make_temporal_position_embedding,
)


@dataclass(frozen=True)
class HWMOutput:
    """Forward outputs with explicit shape contracts.

    Shapes:
        encoded: [B, T, L, M]
        embedding: [B, T, D]
        sigreg_embedding: [B, T, D]
        prediction: [B, T-1, D]
        logvar: [B, T-1, D]
        target: [B, T-1, D]
    """

    encoded: torch.Tensor
    embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    prediction: torch.Tensor
    logvar: torch.Tensor
    target: torch.Tensor


class HeteroscedasticNextLatentPredictor(nn.Module):
    """Autoregressive next-latent predictor with per-dimension uncertainty."""

    def __init__(self, config: HWMConfig) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        self.input_projection = nn.Linear(config.latent_dim, model_dim)
        self.action_embedding = make_action_embedding(config, model_dim)
        self.temporal_position_embedding = make_temporal_position_embedding(
            config,
            config.max_sequence_length,
            model_dim,
        )
        self.transformer = ActionConditionedCausalTransformer(
            model_dim=model_dim,
            layers=config.predictor_layers,
            heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            rotary=config.uses_rotary_temporal_position_encoding,
        )
        self.z_head = nn.Linear(model_dim, config.latent_dim)
        self.global_logvar = nn.Parameter(torch.zeros(config.latent_dim))

    def forward(
        self,
        embedding: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 3:
            raise ValueError(f"expected embedding rank 3, got {embedding.ndim}")
        batch_size, sequence_length, latent_dim = embedding.shape
        if latent_dim != self.config.latent_dim:
            raise ValueError(
                f"expected latent dim {self.config.latent_dim}, got {latent_dim}"
            )
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )
        if sequence_length == 1:
            empty = embedding[:, :0]
            return empty, empty
        if actions.shape != (batch_size, sequence_length - 1):
            raise ValueError(
                f"expected actions shape {(batch_size, sequence_length - 1)}, "
                f"got {tuple(actions.shape)}"
            )

        inputs = self.input_projection(embedding[:, :-1])
        inputs = add_temporal_position_embedding(
            inputs,
            self.temporal_position_embedding,
        )
        action_condition = self.action_embedding(action_indices(self.config, actions))
        hidden = self.transformer(inputs, action_condition)
        prediction = self.z_head(hidden)
        logvar = self.global_logvar.view(1, 1, latent_dim).expand_as(prediction)
        logvar = torch.clamp(
            logvar,
            min=self.config.logvar_min,
            max=self.config.logvar_max,
        )
        return prediction, logvar


class HWM(nn.Module):
    """Heteroscedastic next-latent world model described in PLAN_v2."""

    def __init__(self, config: HWMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.predictor = HeteroscedasticNextLatentPredictor(config)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> HWMOutput:
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        prediction, logvar = self.predictor(embedding, actions)
        target = embedding[:, 1:]
        return HWMOutput(
            encoded=encoded,
            embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            prediction=prediction,
            logvar=logvar,
            target=target,
        )
