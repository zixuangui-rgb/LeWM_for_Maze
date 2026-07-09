"""PRISM architecture from PLAN_v4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hdwm.config import PRISMConfig, PRISMV2Config
from hdwm.models.shared import (
    CausalTransformer,
    LatentEmbeddingProjector,
    ObservationEncoder,
    ProjectionStage,
    action_indices,
    add_temporal_position_embedding,
    make_action_embedding,
    make_temporal_position_embedding,
)


@dataclass(frozen=True)
class PRISMOutput:
    """Forward outputs with explicit shape contracts.

    Shapes:
        encoded: [B, T, L, M]
        embedding: [B, T, D]
        sigreg_embedding: [B, T, D]
        clean_state: [B, T, D]
        prediction: [B, T-1, D]
        clean_prediction: [B, T-1, D]
        target: [B, T-1, D]
    """

    encoded: torch.Tensor
    embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    clean_state: torch.Tensor
    prediction: torch.Tensor
    clean_prediction: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class PRISMV2Output:
    """Forward outputs with explicit shape contracts.

    Shapes:
        encoded: [B, T, L, M]
        embedding: [B, T, D]
        sigreg_embedding: [B, T, D]
        posterior_belief_sigreg: [B, T, K]
        posterior_belief: [B, T, K]
        prior_belief: [B, T-1, K]
        posterior_obs_prediction: [B, T, D]
        prior_obs_prediction: [B, T-1, D]
        obs_target: [B, T, D]
    """

    encoded: torch.Tensor
    embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    posterior_belief_sigreg: torch.Tensor
    posterior_belief: torch.Tensor
    prior_belief: torch.Tensor
    posterior_obs_prediction: torch.Tensor
    prior_obs_prediction: torch.Tensor
    obs_target: torch.Tensor


class InterleavedPredictiveTransformer(nn.Module):
    """Causal transformer over alternating state and action tokens."""

    def __init__(self, config: PRISMConfig) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        max_token_length = config.max_sequence_length * 2 - 1
        self.state_projection = nn.Linear(config.latent_dim, model_dim)
        self.action_embedding = make_action_embedding(config, model_dim)
        self.token_type_embedding = nn.Embedding(2, model_dim)
        self.temporal_position_embedding = make_temporal_position_embedding(
            config,
            max_token_length,
            model_dim,
        )
        self.transformer = CausalTransformer(
            model_dim=model_dim,
            layers=config.predictor_layers,
            heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            rotary=config.uses_rotary_temporal_position_encoding,
        )
        self.output_projection = nn.Linear(model_dim, config.latent_dim)

    def forward(
        self,
        embedding: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 3:
            raise ValueError(f"expected embedding rank 3, got {embedding.ndim}")
        batch_size, sequence_length, latent_dim = embedding.shape
        if sequence_length <= 0:
            raise ValueError("sequence length must be positive")
        if latent_dim != self.config.latent_dim:
            raise ValueError(
                f"expected latent dim {self.config.latent_dim}, got {latent_dim}"
            )
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )
        expected_actions_shape = (batch_size, max(sequence_length - 1, 0))
        if actions.shape != expected_actions_shape:
            raise ValueError(
                f"expected actions shape {expected_actions_shape}, "
                f"got {tuple(actions.shape)}"
            )

        token_length = sequence_length * 2 - 1
        tokens = embedding.new_empty(
            batch_size,
            token_length,
            self.config.effective_model_dim,
        )
        tokens[:, 0::2] = self.state_projection(embedding)
        if sequence_length > 1:
            action_tokens = self.action_embedding(action_indices(self.config, actions))
            tokens[:, 1::2] = action_tokens

        positions = torch.arange(token_length, device=embedding.device)
        token_types = positions.remainder(2)
        tokens = add_temporal_position_embedding(
            tokens, self.temporal_position_embedding
        )
        tokens = tokens + self.token_type_embedding(token_types).view(
            1, token_length, -1
        )
        outputs = self.output_projection(self.transformer(tokens))
        return outputs[:, 0::2], outputs[:, 1::2]


class InterleavedBeliefTransformer(nn.Module):
    """Causal transformer whose observation/action hidden states are beliefs."""

    def __init__(self, config: PRISMV2Config) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        max_token_length = config.max_sequence_length * 2 - 1
        self.state_projection = nn.Linear(config.latent_dim, model_dim)
        self.action_embedding = make_action_embedding(config, model_dim)
        self.token_type_embedding = nn.Embedding(2, model_dim)
        self.temporal_position_embedding = make_temporal_position_embedding(
            config,
            max_token_length,
            model_dim,
        )
        self.transformer = CausalTransformer(
            model_dim=model_dim,
            layers=config.predictor_layers,
            heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            rotary=config.uses_rotary_temporal_position_encoding,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 3:
            raise ValueError(f"expected embedding rank 3, got {embedding.ndim}")
        batch_size, sequence_length, latent_dim = embedding.shape
        if sequence_length <= 0:
            raise ValueError("sequence length must be positive")
        if latent_dim != self.config.latent_dim:
            raise ValueError(
                f"expected latent dim {self.config.latent_dim}, got {latent_dim}"
            )
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )
        expected_actions_shape = (batch_size, max(sequence_length - 1, 0))
        if actions.shape != expected_actions_shape:
            raise ValueError(
                f"expected actions shape {expected_actions_shape}, "
                f"got {tuple(actions.shape)}"
            )

        token_length = sequence_length * 2 - 1
        tokens = embedding.new_empty(
            batch_size,
            token_length,
            self.config.effective_model_dim,
        )
        tokens[:, 0::2] = self.state_projection(embedding)
        if sequence_length > 1:
            tokens[:, 1::2] = self.action_embedding(
                action_indices(self.config, actions)
            )

        positions = torch.arange(token_length, device=embedding.device)
        token_types = positions.remainder(2)
        tokens = add_temporal_position_embedding(
            tokens, self.temporal_position_embedding
        )
        tokens = tokens + self.token_type_embedding(token_types).view(
            1, token_length, -1
        )
        beliefs = self.transformer(tokens)
        return beliefs[:, 0::2], beliefs[:, 1::2]


class PRISM(nn.Module):
    """Predictive internal state model with action-position prediction loss."""

    def __init__(self, config: PRISMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.predictor = InterleavedPredictiveTransformer(config)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> PRISMOutput:
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        clean_state, prediction = self.predictor(embedding, actions)
        if self.config.embedding_stage == ProjectionStage.POST_L2:
            prediction = torch.nn.functional.normalize(prediction, dim=-1)
            clean_state = torch.nn.functional.normalize(clean_state, dim=-1)
        _, clean_prediction = self.predictor(clean_state, actions)
        if self.config.embedding_stage == ProjectionStage.POST_L2:
            clean_prediction = torch.nn.functional.normalize(clean_prediction, dim=-1)
        target = embedding[:, 1:]
        return PRISMOutput(
            encoded=encoded,
            embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            clean_state=clean_state,
            prediction=prediction,
            clean_prediction=clean_prediction,
            target=target,
        )


class PRISMV2(nn.Module):
    """PLAN_v5 belief-space causal Transformer with z-space SIGReg."""

    def __init__(self, config: PRISMV2Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.predictor = InterleavedBeliefTransformer(config)
        self.obs_bottleneck_projector = nn.Linear(
            config.effective_model_dim,
            config.effective_bottleneck_dim,
        )
        self.obs_decoder = self._build_obs_decoder(config)

    @staticmethod
    def _build_obs_decoder(config: PRISMV2Config) -> nn.Module:
        if config.obs_decoder_layers == 1:
            return nn.Linear(
                config.effective_bottleneck_dim,
                config.latent_dim,
            )

        layers: list[nn.Module] = []
        for _ in range(config.obs_decoder_layers - 1):
            layers.extend(
                [
                    nn.Linear(
                        config.effective_bottleneck_dim,
                        config.effective_bottleneck_dim,
                    ),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(config.effective_bottleneck_dim, config.latent_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> PRISMV2Output:
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        raw_posterior, raw_prior = self.predictor(embedding, actions)
        posterior_belief_sigreg = self.obs_bottleneck_projector(raw_posterior)
        posterior_belief = torch.nn.functional.normalize(
            posterior_belief_sigreg,
            dim=-1,
        )
        prior_belief = torch.nn.functional.normalize(
            self.obs_bottleneck_projector(raw_prior),
            dim=-1,
        )
        posterior_obs_prediction = torch.nn.functional.normalize(
            self.obs_decoder(posterior_belief),
            dim=-1,
        )
        prior_obs_prediction = torch.nn.functional.normalize(
            self.obs_decoder(prior_belief),
            dim=-1,
        )
        return PRISMV2Output(
            encoded=encoded,
            embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            posterior_belief_sigreg=posterior_belief_sigreg,
            posterior_belief=posterior_belief,
            prior_belief=prior_belief,
            posterior_obs_prediction=posterior_obs_prediction,
            prior_obs_prediction=prior_obs_prediction,
            obs_target=embedding,
        )
