"""Li-group world model for position extrapolation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hdwm.config import LIWMConfig
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
class LIWMOutput:
    """Forward outputs with LE-WM embedding and homogeneous pos branches.

    Shapes:
        encoded: [B, T, L, M]
        embedding: [B, T, D]
        sigreg_embedding: [B, T, D]
        pos_embedding: [B, T, P]
        prediction: [B, T-1, D]
        target: [B, T-1, D]
        pos_prediction: [B, T-1, P]
        pos_target: [B, T-1, P]
    """

    encoded: torch.Tensor
    embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    pos_embedding: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    pos_prediction: torch.Tensor
    pos_target: torch.Tensor


class HomogeneousPosProjector(nn.Module):
    """Project encoded observations into low-dimensional homogeneous positions."""

    def __init__(self, config: LIWMConfig) -> None:
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.effective_model_dim, config.pos_dim - 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim == 4:
            processed = encoded.mean(dim=2)
        elif encoded.ndim == 3:
            processed = encoded
        else:
            raise ValueError(f"expected encoded rank 3 or 4, got {encoded.ndim}")
        if processed.shape[-1] != self.config.effective_model_dim:
            raise ValueError(
                f"expected encoded dim {self.config.effective_model_dim}, "
                f"got {processed.shape[-1]}"
            )
        coordinates = self.linear(processed)
        homogeneous_coordinate = torch.ones_like(coordinates[..., :1])
        return torch.cat([coordinates, homogeneous_coordinate], dim=-1)


class LIWMNextPredictor(nn.Module):
    """Predict both branches while allowing cross-branch conditioning."""

    def __init__(self, config: LIWMConfig) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        self.input_projection = nn.Linear(
            config.latent_dim + config.pos_dim,
            model_dim,
        )
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
        self.output_projection = nn.Linear(model_dim, config.latent_dim)
        self.pos_output_projection = nn.Linear(model_dim, config.pos_dim - 1)

    def forward(
        self,
        embedding: torch.Tensor,
        pos_embedding: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 3:
            raise ValueError(f"expected embedding rank 3, got {embedding.ndim}")
        if pos_embedding.ndim != 3:
            raise ValueError(f"expected pos_embedding rank 3, got {pos_embedding.ndim}")
        batch_size, sequence_length, latent_dim = embedding.shape
        if latent_dim != self.config.latent_dim:
            raise ValueError(
                f"expected latent dim {self.config.latent_dim}, got {latent_dim}"
            )
        if pos_embedding.shape != (
            batch_size,
            sequence_length,
            self.config.pos_dim,
        ):
            raise ValueError(
                "expected pos shape "
                f"{(batch_size, sequence_length, self.config.pos_dim)}, "
                f"got {tuple(pos_embedding.shape)}"
            )
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )
        if sequence_length == 1:
            return embedding[:, :0], pos_embedding[:, :0]
        if actions.shape != (batch_size, sequence_length - 1):
            raise ValueError(
                f"expected actions shape {(batch_size, sequence_length - 1)}, "
                f"got {tuple(actions.shape)}"
            )

        inputs = torch.cat(
            [embedding[:, :-1], pos_embedding[:, :-1]],
            dim=-1,
        )
        inputs = self.input_projection(inputs)
        inputs = add_temporal_position_embedding(
            inputs,
            self.temporal_position_embedding,
        )
        action_condition = self.action_embedding(action_indices(self.config, actions))
        hidden = self.transformer(inputs, action_condition)
        prediction = self.output_projection(hidden)
        pos_coordinates = self.pos_output_projection(hidden)
        homogeneous_coordinate = torch.ones_like(pos_coordinates[..., :1])
        pos_prediction = torch.cat([pos_coordinates, homogeneous_coordinate], dim=-1)
        return prediction, pos_prediction


class LIWM(nn.Module):
    """LE-WM add-on with invariant embedding and equivariant pos branches."""

    def __init__(self, config: LIWMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.pos_projector = HomogeneousPosProjector(config)
        self.predictor = LIWMNextPredictor(config)
        self.raw_generators = nn.Parameter(
            torch.empty(config.num_generators, config.pos_dim, config.pos_dim)
        )
        nn.init.normal_(self.raw_generators, mean=0.0, std=0.02)
        generator_mask = torch.ones(config.pos_dim, config.pos_dim)
        generator_mask[-1] = 0.0
        self.register_buffer("generator_mask", generator_mask)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> LIWMOutput:
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        pos_embedding = self.pos_projector(encoded)
        prediction, pos_prediction = self.predictor(
            embedding,
            pos_embedding,
            actions,
        )
        return LIWMOutput(
            encoded=encoded,
            embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            pos_embedding=pos_embedding,
            prediction=prediction,
            target=embedding[:, 1:],
            pos_prediction=pos_prediction,
            pos_target=pos_embedding[:, 1:],
        )

    def effective_generators(self) -> torch.Tensor:
        """Return affine generators with a fixed zero homogeneous output row."""

        return self.raw_generators * self.generator_mask

    def normalized_generators(self) -> torch.Tensor:
        """Return Frobenius-normalized generators for infinitesimal transforms."""

        generators = self.effective_generators()
        return F.normalize(generators.flatten(start_dim=1), dim=-1).view_as(generators)

    def generator_group_lasso(self) -> torch.Tensor:
        """Return group-lasso over effective learned generator matrices."""

        return self.effective_generators().flatten(start_dim=1).norm(dim=-1).sum()

    def equivariance_loss(
        self,
        output: LIWMOutput,
        actions: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """Measure infinitesimal equivariance of the pos predictor."""

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if output.pos_prediction.numel() == 0:
            return output.pos_embedding.new_tensor(0.0)

        losses = []
        for generator in self.normalized_generators():
            pos_delta = torch.einsum("ij,btj->bti", generator, output.pos_embedding)
            perturbed_pos = output.pos_embedding + epsilon * pos_delta
            _, perturbed_prediction = self.predictor(
                output.embedding,
                perturbed_pos,
                actions,
            )
            prediction_delta = torch.einsum(
                "ij,btj->bti",
                generator,
                output.pos_prediction,
            )
            expected_prediction = output.pos_prediction + epsilon * prediction_delta
            losses.append(F.mse_loss(perturbed_prediction, expected_prediction))
        return torch.stack(losses).sum()
