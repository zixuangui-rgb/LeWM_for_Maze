"""LE-WM baseline architecture for HDWM comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hdwm.config import (
    ICWMConfig,
    LEWMCNNConfig,
    LEWMConfig,
    LEWMV2Config,
    LEWMV3Config,
    ProjectionStage,
)
from hdwm.models.shared import (
    ActionConditionedCausalTransformer,
    CausalTransformer,
    LatentEmbeddingProjector,
    ObservationEncoder,
    action_indices,
    add_temporal_position_embedding,
    make_action_embedding,
    make_temporal_position_embedding,
)


@dataclass(frozen=True)
class LEWMOutput:
    """Forward outputs with explicit shape contracts.

    Shapes:
        encoded: [B, T, L, M]
        embedding: [B, T, D]
        sigreg_embedding: [B, T, D]
        prediction: [B, T-1, D]
        target: [B, T-1, D]
    """

    encoded: torch.Tensor
    embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class ICWMOutput:
    """Forward outputs for in-context trajectory packing.

    Shapes:
        encoded: [B, C, T, L, M]
        observation_embedding: [B, C*T, D]
        sigreg_embedding: [B, C*T, D]
        packed_embedding: [B, P, D]
        prediction: [B, P-1, D]
        target: [B, P-1, D]
        valid_prediction_mask: [B, P-1]
    """

    encoded: torch.Tensor
    observation_embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    packed_embedding: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    valid_prediction_mask: torch.Tensor


@dataclass(frozen=True)
class LEWMV2Output:
    """Forward outputs with explicit shape contracts.

    Shapes:
        encoded: [B, T, L, M]
        embedding: [B, T, D]
        sigreg_embedding: [B, T, D]
        hidden: [B, T, H]
        hidden_prediction: [B, T-1, H]
        hidden_target: [B, T-1, H]
        decoded_hidden: [B, T, D]
        decoded_prediction: [B, T-1, D]
        obs_target: [B, T, D]
        rollout_hidden_predictions: horizon-indexed [B, T-k, H]
        rollout_hidden_targets: horizon-indexed [B, T-k, H]
        rollout_decoded_predictions: horizon-indexed [B, T-k, D]
        rollout_obs_targets: horizon-indexed [B, T-k, D]
    """

    encoded: torch.Tensor
    embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    hidden: torch.Tensor
    hidden_prediction: torch.Tensor
    hidden_target: torch.Tensor
    decoded_hidden: torch.Tensor
    decoded_prediction: torch.Tensor
    obs_target: torch.Tensor
    rollout_hidden_predictions: tuple[torch.Tensor, ...]
    rollout_hidden_targets: tuple[torch.Tensor, ...]
    rollout_decoded_predictions: tuple[torch.Tensor, ...]
    rollout_obs_targets: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class LEWMV3Output(LEWMV2Output):
    """LE-WMv3 outputs including concept-conditioned dynamics internals.

    Shapes:
        concepts: [B, T-1, C]
        transitions: [B, T-1, H]
        rotation_angles: [B, T-1, H / 2] for rotation mode, else [B, T-1, 0]
        rotation_matrices: [B, T-1, H, H] for rotation mode, else [B, T-1, 0, 0]
        rollout_concepts: horizon-indexed [B, T-k, C]
        rollout_transitions: horizon-indexed [B, T-k, H]
        rollout_rotation_angles: horizon-indexed rotation params, empty for transition
        rollout_rotation_matrices: horizon-indexed rotation matrices,
            empty for transition
    """

    concepts: torch.Tensor
    transitions: torch.Tensor
    rotation_angles: torch.Tensor
    rotation_matrices: torch.Tensor
    rollout_concepts: tuple[torch.Tensor, ...]
    rollout_transitions: tuple[torch.Tensor, ...]
    rollout_rotation_angles: tuple[torch.Tensor, ...]
    rollout_rotation_matrices: tuple[torch.Tensor, ...]


class NextEmbeddingPredictor(nn.Module):
    """Autoregressive next-embedding predictor conditioned on actions."""

    def __init__(self, config: LEWMConfig) -> None:
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
        self.output_projection = nn.Linear(model_dim, config.latent_dim)

    def forward(self, embedding: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = self._validate_embedding(embedding)
        if sequence_length == 1:
            return embedding[:, :0]
        if actions.shape != (batch_size, sequence_length - 1):
            raise ValueError(
                f"expected actions shape {(batch_size, sequence_length - 1)}, "
                f"got {tuple(actions.shape)}"
            )

        action_condition = self.action_condition(actions)
        return self.forward_with_action_condition(embedding, action_condition)

    def action_condition(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_embedding(action_indices(self.config, actions))

    def forward_with_action_condition(
        self,
        embedding: torch.Tensor,
        action_condition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = self._validate_embedding(embedding)
        if sequence_length == 1:
            return embedding[:, :0]
        expected_condition_shape = (
            batch_size,
            sequence_length - 1,
            self.config.effective_model_dim,
        )
        if action_condition.shape != expected_condition_shape:
            raise ValueError(
                f"expected action condition shape {expected_condition_shape}, "
                f"got {tuple(action_condition.shape)}"
            )

        inputs = self.input_projection(embedding[:, :-1])
        inputs = add_temporal_position_embedding(
            inputs,
            self.temporal_position_embedding,
        )
        hidden = self.transformer(inputs, action_condition)
        return self.output_projection(hidden)

    def _validate_embedding(self, embedding: torch.Tensor) -> tuple[int, int, int]:
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
        return batch_size, sequence_length, latent_dim


class LEWMHiddenEncoder(nn.Module):
    """Causal Transformer that maps observation embeddings into dynamics states."""

    def __init__(self, config: LEWMV2Config) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        self.input_projection = nn.Linear(config.latent_dim, model_dim)
        self.temporal_position_embedding = make_temporal_position_embedding(
            config,
            config.max_sequence_length,
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

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 3:
            raise ValueError(f"expected embedding rank 3, got {embedding.ndim}")
        _, sequence_length, latent_dim = embedding.shape
        if latent_dim != self.config.latent_dim:
            raise ValueError(
                f"expected latent dim {self.config.latent_dim}, got {latent_dim}"
            )
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )

        inputs = self.input_projection(embedding)
        inputs = add_temporal_position_embedding(
            inputs,
            self.temporal_position_embedding,
        )
        return self.transformer(inputs)


class NextHiddenPredictor(nn.Module):
    """Predict the next dynamics state from the current state and action."""

    def __init__(self, config: LEWMV2Config) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        self.action_embedding = make_action_embedding(config, model_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim * config.mlp_ratio),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(model_dim * config.mlp_ratio, model_dim),
        )

    def forward(self, hidden: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"expected hidden rank 3, got {hidden.ndim}")
        batch_size, sequence_length, hidden_dim = hidden.shape
        if hidden_dim != self.config.effective_model_dim:
            raise ValueError(
                f"expected hidden dim {self.config.effective_model_dim}, "
                f"got {hidden_dim}"
            )
        if sequence_length == 1:
            return hidden[:, :0]
        if actions.shape != (batch_size, sequence_length - 1):
            raise ValueError(
                f"expected actions shape {(batch_size, sequence_length - 1)}, "
                f"got {tuple(actions.shape)}"
            )

        action_condition = self.action_embedding(action_indices(self.config, actions))
        return self.net(torch.cat([hidden[:, :-1], action_condition], dim=-1))


class ConceptConditionedDynamicsPredictor(nn.Module):
    """Predict concept-conditioned additive transitions or rotations."""

    def __init__(self, config: LEWMV3Config) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        self.action_embedding = make_action_embedding(config, model_dim)
        concept_hidden_dim = config.concept_hidden_dim or model_dim
        transition_hidden_dim = config.transition_hidden_dim or model_dim
        rotation_hidden_dim = config.rotation_hidden_dim or model_dim
        if config.dynamics_mode == "rotation" and model_dim % 2 != 0:
            raise ValueError("model_dim must be even for paired-plane rotations")
        self.rotation_angle_dim = (
            model_dim // 2 if config.dynamics_mode == "rotation" else 0
        )
        plane_rows = torch.arange(0, model_dim, 2)
        plane_cols = torch.arange(1, model_dim, 2)
        self.register_buffer("plane_rows", plane_rows, persistent=False)
        self.register_buffer("plane_cols", plane_cols, persistent=False)
        self.concept_net = self._build_mlp(
            input_dim=model_dim * 2,
            hidden_dim=concept_hidden_dim,
            output_dim=config.concept_dim,
            layers=config.concept_mlp_layers,
            input_layer_norm=True,
        )
        dynamics_input_dim = config.concept_dim + model_dim
        self.transition_net = nn.Sequential()
        self.rotation_angle_net = nn.Sequential()
        if config.dynamics_mode == "transition":
            self.transition_net = self._build_mlp(
                input_dim=dynamics_input_dim,
                hidden_dim=transition_hidden_dim,
                output_dim=model_dim,
                layers=config.transition_mlp_layers,
                input_layer_norm=False,
            )
            self._zero_init_output(self.transition_net, "transition_net")
        else:
            self.rotation_angle_net = self._build_mlp(
                input_dim=dynamics_input_dim,
                hidden_dim=rotation_hidden_dim,
                output_dim=self.rotation_angle_dim,
                layers=config.rotation_mlp_layers,
                input_layer_norm=False,
            )
            self._zero_init_output(self.rotation_angle_net, "rotation_angle_net")

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        input_layer_norm: bool,
    ) -> nn.Sequential:
        if layers <= 0:
            raise ValueError("layers must be positive")
        modules: list[nn.Module] = []
        if input_layer_norm:
            modules.append(nn.LayerNorm(input_dim))
        if layers == 1:
            modules.append(nn.Linear(input_dim, output_dim))
            return nn.Sequential(*modules)

        modules.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            ]
        )
        for _ in range(layers - 2):
            modules.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
        modules.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*modules)

    @staticmethod
    def _zero_init_output(net: nn.Sequential, name: str) -> None:
        for module in reversed(net):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                return
        raise RuntimeError(f"{name} is missing a Linear output layer")

    def _angles_to_rotation_matrices(self, angles: torch.Tensor) -> torch.Tensor:
        if self.config.dynamics_mode != "rotation":
            raise ValueError("rotation matrices require dynamics_mode='rotation'")
        if angles.ndim != 3:
            raise ValueError(f"expected angles rank 3, got {angles.ndim}")
        if angles.shape[-1] != self.rotation_angle_dim:
            raise ValueError(
                f"expected {self.rotation_angle_dim} rotation angles, "
                f"got {angles.shape[-1]}"
            )

        *batch_shape, _ = angles.shape
        model_dim = self.config.effective_model_dim
        rotation_matrices = (
            torch.eye(
                model_dim,
                device=angles.device,
                dtype=angles.dtype,
            )
            .expand(*batch_shape, model_dim, model_dim)
            .clone()
        )
        cos_angles = angles.cos()
        sin_angles = angles.sin()
        rotation_matrices[..., self.plane_rows, self.plane_rows] = cos_angles
        rotation_matrices[..., self.plane_rows, self.plane_cols] = sin_angles
        rotation_matrices[..., self.plane_cols, self.plane_rows] = -sin_angles
        rotation_matrices[..., self.plane_cols, self.plane_cols] = cos_angles
        return rotation_matrices

    def forward(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if hidden.ndim != 3:
            raise ValueError(f"expected hidden rank 3, got {hidden.ndim}")
        batch_size, sequence_length, hidden_dim = hidden.shape
        if hidden_dim != self.config.effective_model_dim:
            raise ValueError(
                f"expected hidden dim {self.config.effective_model_dim}, "
                f"got {hidden_dim}"
            )
        if sequence_length == 1:
            matrix_shape = (
                (batch_size, 0, hidden_dim, hidden_dim)
                if self.config.dynamics_mode == "rotation"
                else (batch_size, 0, 0, 0)
            )
            empty_matrices = hidden.new_empty(*matrix_shape)
            return (
                hidden[:, :0],
                hidden.new_empty(batch_size, 0, self.config.concept_dim),
                hidden[:, :0],
                hidden.new_empty(batch_size, 0, self.rotation_angle_dim),
                empty_matrices,
            )
        if actions.shape != (batch_size, sequence_length - 1):
            raise ValueError(
                f"expected actions shape {(batch_size, sequence_length - 1)}, "
                f"got {tuple(actions.shape)}"
            )

        action_condition = self.action_embedding(action_indices(self.config, actions))
        concept_inputs = torch.cat([hidden[:, :-1], action_condition], dim=-1)
        concepts = F.normalize(self.concept_net(concept_inputs), dim=-1)
        dynamics_inputs = torch.cat([concepts, action_condition], dim=-1)
        if self.config.dynamics_mode == "transition":
            transitions = self.transition_net(dynamics_inputs)
            rotation_angles = hidden.new_empty(batch_size, sequence_length - 1, 0)
            rotation_matrices = hidden.new_empty(batch_size, sequence_length - 1, 0, 0)
            return (
                hidden[:, :-1] + transitions,
                concepts,
                transitions,
                rotation_angles,
                rotation_matrices,
            )

        rotation_angles = self.rotation_angle_net(dynamics_inputs)
        rotation_matrices = self._angles_to_rotation_matrices(rotation_angles)
        rotated_hidden = torch.einsum(
            "...ij,...j->...i",
            rotation_matrices,
            hidden[:, :-1],
        )
        transitions = rotated_hidden - hidden[:, :-1]
        return rotated_hidden, concepts, transitions, rotation_angles, rotation_matrices


class LEWM(nn.Module):
    """LE-WM next-embedding prediction baseline with SIGReg training loss."""

    def __init__(self, config: LEWMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.predictor = NextEmbeddingPredictor(config)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> LEWMOutput:
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        prediction = self.predictor(embedding, actions)
        target = embedding[:, 1:]
        return LEWMOutput(
            encoded=encoded,
            embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            prediction=prediction,
            target=target,
        )


class CNNEncoder(nn.Module):
    """CNN-based observation encoder for 2D grid-world observations.

    Replaces the per-cell Transformer ``ObservationEncoder`` with a
    configurable convolutional backbone.  Each frame is processed
    independently and the output is a per-frame feature vector.

    Shapes:
        observations: [B, T, H, W, C]
        output: [B, T, M]  where M = config.effective_model_dim
    """

    def __init__(self, config: LEWMCNNConfig) -> None:
        super().__init__()
        self.config = config
        in_channels = config.observation_channels
        out_dim = config.effective_model_dim
        padding = (
            config.cnn_padding
            if config.cnn_padding is not None
            else config.cnn_kernel_size // 2
        )

        layers: list[nn.Module] = []
        prev_channels = in_channels
        for channels in config.cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        prev_channels,
                        channels,
                        kernel_size=config.cnn_kernel_size,
                        stride=config.cnn_stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_channels = channels

        self.conv = nn.Sequential(*layers) if layers else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_projection = nn.Linear(prev_channels, out_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim < 4:
            raise ValueError(
                f"expected observations with [B, T, H, W, C], got rank "
                f"{observations.ndim}"
            )
        batch_size, sequence_length = observations.shape[:2]
        # Flatten batch and time dims for per-frame CNN processing.
        x = observations.reshape(
            batch_size * sequence_length,
            *observations.shape[2:],
        )
        # Rearrange: [B*T, H, W, C] -> [B*T, C, H, W].
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.pool(x)[..., 0, 0]  # [B*T, last_channels]
        x = self.output_projection(x)  # [B*T, M]
        return x.view(batch_size, sequence_length, -1)


class LEWMCNN(nn.Module):
    """LE-WM variant with a CNN encoder instead of the per-cell Transformer.

    Uses the same projector, predictor, and LEWMOutput format as ``LEWM``
    so it is fully compatible with the existing training pipeline.
    """

    def __init__(self, config: LEWMCNNConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CNNEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.predictor = NextEmbeddingPredictor(config)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> LEWMOutput:
        # CNN encoder outputs [B, T, M] (3D) which the projector
        # passes through directly (no mean-pooling needed).
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        prediction = self.predictor(embedding, actions)
        target = embedding[:, 1:]
        return LEWMOutput(
            encoded=encoded,
            embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            prediction=prediction,
            target=target,
        )


class ICWM(nn.Module):
    """In-context LE-WM with BOS/RESET/EOS trajectory packing."""

    BOS_TOKEN = 0
    RESET_TOKEN = 1
    EOS_TOKEN = 2

    def __init__(self, config: ICWMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.special_embedding = nn.Embedding(3, config.latent_dim)
        self.predictor = NextEmbeddingPredictor(config)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> ICWMOutput:
        if observations.ndim < 4:
            raise ValueError(
                f"expected observations with shape [B, C, T, ...], "
                f"got rank {observations.ndim}"
            )
        if actions.ndim != 3:
            raise ValueError(f"expected actions rank 3, got {actions.ndim}")
        batch_size, context_length, sequence_length = observations.shape[:3]
        if context_length != self.config.context_length:
            raise ValueError(
                f"expected context length {self.config.context_length}, "
                f"got {context_length}"
            )
        expected_actions_shape = (
            batch_size,
            context_length,
            max(sequence_length - 1, 0),
        )
        if actions.shape != expected_actions_shape:
            raise ValueError(
                f"expected actions shape {expected_actions_shape}, "
                f"got {tuple(actions.shape)}"
            )

        flat_observations = observations.reshape(
            batch_size * context_length,
            sequence_length,
            *observations.shape[3:],
        )
        encoded = self.encoder(flat_observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        packed_embedding, action_condition, valid_prediction_mask = self._pack_context(
            embedding=embedding.reshape(
                batch_size,
                context_length,
                sequence_length,
                self.config.latent_dim,
            ),
            actions=actions,
        )
        prediction = self.predictor.forward_with_action_condition(
            packed_embedding,
            action_condition,
        )
        target = packed_embedding[:, 1:]
        encoded = encoded.reshape(
            batch_size,
            context_length,
            sequence_length,
            *encoded.shape[2:],
        )
        return ICWMOutput(
            encoded=encoded,
            observation_embedding=embedding.reshape(
                batch_size,
                context_length * sequence_length,
                self.config.latent_dim,
            ),
            sigreg_embedding=sigreg_embedding.reshape(
                batch_size,
                context_length * sequence_length,
                self.config.latent_dim,
            ),
            packed_embedding=packed_embedding,
            prediction=prediction,
            target=target,
            valid_prediction_mask=valid_prediction_mask,
        )

    def _pack_context(
        self,
        embedding: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embedding.ndim != 4:
            raise ValueError(f"expected embedding rank 4, got {embedding.ndim}")
        batch_size, context_length, sequence_length, latent_dim = embedding.shape
        if latent_dim != self.config.latent_dim:
            raise ValueError(
                f"expected latent dim {self.config.latent_dim}, got {latent_dim}"
            )
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        packed_length = context_length * sequence_length + context_length + 1
        if packed_length > self.config.max_sequence_length:
            raise ValueError(
                f"packed sequence length {packed_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )

        packed = embedding.new_empty(batch_size, packed_length, latent_dim)
        action_condition = embedding.new_zeros(
            batch_size,
            max(packed_length - 1, 0),
            self.config.effective_model_dim,
        )
        valid_mask = torch.zeros(
            batch_size,
            max(packed_length - 1, 0),
            dtype=torch.bool,
            device=embedding.device,
        )
        special_tokens = self.special_embedding.weight.to(
            device=embedding.device,
            dtype=embedding.dtype,
        )

        write_index = 0
        packed[:, write_index] = special_tokens[self.BOS_TOKEN]
        write_index += 1
        for context_index in range(context_length):
            trajectory_start = write_index
            packed[:, trajectory_start : trajectory_start + sequence_length] = (
                embedding[:, context_index]
            )
            if sequence_length > 1:
                real_actions = actions[:, context_index]
                action_condition[
                    :,
                    trajectory_start : trajectory_start + sequence_length - 1,
                ] = self.predictor.action_condition(real_actions)
                valid_mask[
                    :,
                    trajectory_start : trajectory_start + sequence_length - 1,
                ] = True
            write_index += sequence_length
            if context_index < context_length - 1:
                packed[:, write_index] = special_tokens[self.RESET_TOKEN]
                write_index += 1
        packed[:, write_index] = special_tokens[self.EOS_TOKEN]
        if write_index != packed_length - 1:
            raise RuntimeError("ICWM packing wrote an unexpected number of tokens")
        return packed, action_condition, valid_mask


class LEWMV2(nn.Module):
    """LE-WMv2 with dynamics-space prediction and z-space decoding losses."""

    def __init__(self, config: LEWMV2Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.hidden_encoder = LEWMHiddenEncoder(config)
        self.predictor = NextHiddenPredictor(config)
        self.decoder = self._build_obs_decoder(config)

    @staticmethod
    def _build_obs_decoder(config: LEWMV2Config) -> nn.Module:
        if config.obs_decoder_layers == 1:
            return nn.Linear(config.effective_model_dim, config.latent_dim)

        layers: list[nn.Module] = []
        for _ in range(config.obs_decoder_layers - 1):
            layers.extend(
                [
                    nn.Linear(config.effective_model_dim, config.effective_model_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(config.effective_model_dim, config.latent_dim))
        return nn.Sequential(*layers)

    def _normalize_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.config.h_l2_norm:
            return torch.nn.functional.normalize(hidden, dim=-1)
        return hidden

    def _decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(hidden)
        if self.config.embedding_stage == ProjectionStage.POST_L2:
            return torch.nn.functional.normalize(decoded, dim=-1)
        return decoded

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        prediction_steps: int = 1,
    ) -> LEWMV2Output:
        if prediction_steps <= 0:
            raise ValueError("prediction_steps must be positive")
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        expected_actions_shape = (embedding.shape[0], max(embedding.shape[1] - 1, 0))
        if actions.shape != expected_actions_shape:
            raise ValueError(
                f"expected actions shape {expected_actions_shape}, "
                f"got {tuple(actions.shape)}"
            )
        hidden = self._normalize_hidden(self.hidden_encoder(embedding))
        decoded_hidden = self._decode_hidden(hidden)

        rollout_hidden_predictions: list[torch.Tensor] = []
        rollout_hidden_targets: list[torch.Tensor] = []
        rollout_decoded_predictions: list[torch.Tensor] = []
        rollout_obs_targets: list[torch.Tensor] = []
        current_hidden = hidden
        max_prediction_steps = min(prediction_steps, actions.shape[1])
        for step in range(1, max_prediction_steps + 1):
            current_hidden = self.predictor(current_hidden, actions[:, step - 1 :])
            current_hidden = self._normalize_hidden(current_hidden)
            rollout_hidden_predictions.append(current_hidden)
            rollout_hidden_targets.append(hidden[:, step:])
            rollout_decoded_predictions.append(self._decode_hidden(current_hidden))
            rollout_obs_targets.append(embedding[:, step:])

        if rollout_hidden_predictions:
            hidden_prediction = rollout_hidden_predictions[0]
            hidden_target = rollout_hidden_targets[0]
            decoded_prediction = rollout_decoded_predictions[0]
        else:
            hidden_prediction = hidden[:, :0]
            hidden_target = hidden[:, :0]
            decoded_prediction = decoded_hidden[:, :0]

        return LEWMV2Output(
            encoded=encoded,
            embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            hidden=hidden,
            hidden_prediction=hidden_prediction,
            hidden_target=hidden_target,
            decoded_hidden=decoded_hidden,
            decoded_prediction=decoded_prediction,
            obs_target=embedding,
            rollout_hidden_predictions=tuple(rollout_hidden_predictions),
            rollout_hidden_targets=tuple(rollout_hidden_targets),
            rollout_decoded_predictions=tuple(rollout_decoded_predictions),
            rollout_obs_targets=tuple(rollout_obs_targets),
        )


class LEWMV3(LEWMV2):
    """LE-WMv3 concept-conditioned transition or rotation dynamics model."""

    config: LEWMV3Config
    predictor: ConceptConditionedDynamicsPredictor

    def __init__(self, config: LEWMV3Config) -> None:
        super().__init__(config)
        self.config = config
        self.predictor = ConceptConditionedDynamicsPredictor(config)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        prediction_steps: int = 1,
    ) -> LEWMV3Output:
        if prediction_steps <= 0:
            raise ValueError("prediction_steps must be positive")
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        expected_actions_shape = (embedding.shape[0], max(embedding.shape[1] - 1, 0))
        if actions.shape != expected_actions_shape:
            raise ValueError(
                f"expected actions shape {expected_actions_shape}, "
                f"got {tuple(actions.shape)}"
            )
        hidden = self._normalize_hidden(self.hidden_encoder(embedding))
        decoded_hidden = self._decode_hidden(hidden)

        rollout_hidden_predictions: list[torch.Tensor] = []
        rollout_hidden_targets: list[torch.Tensor] = []
        rollout_decoded_predictions: list[torch.Tensor] = []
        rollout_obs_targets: list[torch.Tensor] = []
        rollout_concepts: list[torch.Tensor] = []
        rollout_transitions: list[torch.Tensor] = []
        rollout_rotation_angles: list[torch.Tensor] = []
        rollout_rotation_matrices: list[torch.Tensor] = []
        current_hidden = hidden
        max_prediction_steps = min(prediction_steps, actions.shape[1])
        for step in range(1, max_prediction_steps + 1):
            (
                current_hidden,
                concepts,
                transitions,
                rotation_angles,
                rotation_matrices,
            ) = self.predictor(
                current_hidden,
                actions[:, step - 1 :],
            )
            current_hidden = self._normalize_hidden(current_hidden)
            rollout_hidden_predictions.append(current_hidden)
            rollout_hidden_targets.append(hidden[:, step:])
            rollout_decoded_predictions.append(self._decode_hidden(current_hidden))
            rollout_obs_targets.append(embedding[:, step:])
            rollout_concepts.append(concepts)
            rollout_transitions.append(transitions)
            rollout_rotation_angles.append(rotation_angles)
            rollout_rotation_matrices.append(rotation_matrices)

        if rollout_hidden_predictions:
            hidden_prediction = rollout_hidden_predictions[0]
            hidden_target = rollout_hidden_targets[0]
            decoded_prediction = rollout_decoded_predictions[0]
            concepts = rollout_concepts[0]
            transitions = rollout_transitions[0]
            rotation_angles = rollout_rotation_angles[0]
            rotation_matrices = rollout_rotation_matrices[0]
        else:
            hidden_prediction = hidden[:, :0]
            hidden_target = hidden[:, :0]
            decoded_prediction = decoded_hidden[:, :0]
            concepts = hidden.new_empty(hidden.shape[0], 0, self.config.concept_dim)
            transitions = hidden[:, :0]
            rotation_angles = hidden.new_empty(
                hidden.shape[0],
                0,
                self.predictor.rotation_angle_dim,
            )
            rotation_matrices = hidden.new_empty(
                *(
                    (hidden.shape[0], 0, hidden.shape[-1], hidden.shape[-1])
                    if self.config.dynamics_mode == "rotation"
                    else (hidden.shape[0], 0, 0, 0)
                )
            )

        return LEWMV3Output(
            encoded=encoded,
            embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            hidden=hidden,
            hidden_prediction=hidden_prediction,
            hidden_target=hidden_target,
            decoded_hidden=decoded_hidden,
            decoded_prediction=decoded_prediction,
            obs_target=embedding,
            rollout_hidden_predictions=tuple(rollout_hidden_predictions),
            rollout_hidden_targets=tuple(rollout_hidden_targets),
            rollout_decoded_predictions=tuple(rollout_decoded_predictions),
            rollout_obs_targets=tuple(rollout_obs_targets),
            concepts=concepts,
            transitions=transitions,
            rotation_angles=rotation_angles,
            rotation_matrices=rotation_matrices,
            rollout_concepts=tuple(rollout_concepts),
            rollout_transitions=tuple(rollout_transitions),
            rollout_rotation_angles=tuple(rollout_rotation_angles),
            rollout_rotation_matrices=tuple(rollout_rotation_matrices),
        )


# ── ViT Encoder ───────────────────────────────────────────────────────────────


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for grid-based observations.

    Splits the HxW grid into patch_size x patch_size patches, embeds each patch
    with a linear projection, prepends a CLS token, adds learnable position
    embeddings, and processes through a Transformer encoder.

    Supports variable input sizes via bicubic interpolation of position embeddings.

    Shapes:
        observations: [B, T, H, W, C]
        output: [B, T, M]  where M = config.effective_model_dim
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.patch_size = getattr(config, 'vit_patch_size', 3)
        self.model_dim = config.effective_model_dim

        H, W = config.effective_observation_spatial_shape
        C = config.observation_channels
        self.num_patches_h = H // self.patch_size
        self.num_patches_w = W // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        patch_dim = self.patch_size * self.patch_size * C

        self.patch_proj = nn.Linear(patch_dim, self.model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim) * 0.02)
        # Position embeddings sized for the max expected grid
        max_patches = 13 * 13  # Supports up to 13x13 with patch_size=3 → ~5x5 = 25 patches
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches + 1, self.model_dim) * 0.02)

        n_layers = getattr(config, 'vit_layers', 4)
        n_heads = getattr(config, 'vit_heads', 4)
        mlp_ratio = getattr(config, 'vit_mlp_ratio', 4)
        dropout = getattr(config, 'vit_dropout', 0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=n_heads,
            dim_feedforward=self.model_dim * mlp_ratio,
            dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(self.model_dim)

    def _interpolate_pos_embed(self, num_patches):
        """Interpolate position embeddings for a different number of patches."""
        if num_patches == self.num_patches:
            return self.pos_embed
        # Simple truncation approach — for different sizes, just take first N patches
        total_needed = num_patches + 1  # +1 for CLS token
        if total_needed <= self.pos_embed.shape[1]:
            return self.pos_embed[:, :total_needed]
        # Need more positions than max — use linear interpolation
        old = self.pos_embed.shape[1] - 1
        new = num_patches
        # Interpolate only the patch positions (skip CLS token at index 0)
        cls = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:old+1].permute(0, 2, 1)  # [1, D, old]
        patch_pos = F.interpolate(patch_pos, size=new, mode='linear', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 1)  # [1, new, D]
        return torch.cat([cls, patch_pos], dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: [B, T, H, W, C]
        returns: [B, T, M]
        """
        B, T, H, W, C = observations.shape
        x = observations.reshape(B * T, H, W, C)

        # Extract patches with padding for non-divisible sizes
        ph, pw = H // self.patch_size, W // self.patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            # Pad to next multiple of patch_size
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
            ph = (H + pad_h) // self.patch_size
            pw = (W + pad_w) // self.patch_size

        ps = self.patch_size
        x = x.reshape(B * T, ph, ps, pw, ps, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * T, ph * pw, ps * ps * C)

        # Project patches
        x = self.patch_proj(x)  # [B*T, num_patches, model_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B*T, num_patches+1, model_dim]

        # Add position embeddings
        pos = self._interpolate_pos_embed(ph * pw)
        x = x + pos[:, :x.shape[1]]

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Return CLS token
        out = x[:, 0]  # [B*T, model_dim]
        return out.view(B, T, -1)


class LEWMViT(nn.Module):
    """LE-WM variant with ViT encoder instead of CNN or per-cell Transformer."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = ViTEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.predictor = NextEmbeddingPredictor(config)

    def forward(self, observations, actions):
        encoded = self.encoder(observations)
        embedding, sigreg_embedding = self.embedding_projector(encoded)
        prediction = self.predictor(embedding, actions)
        target = embedding[:, 1:]
        return LEWMOutput(
            encoded=encoded, embedding=embedding,
            sigreg_embedding=sigreg_embedding,
            prediction=prediction, target=target)
