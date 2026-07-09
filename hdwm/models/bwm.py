"""Bayes world model architectures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hdwm.config import (
    BWMConfig,
    BWMPriorForm,
    BWMV2Config,
    ObsCMINegativeSource,
    PriorCMINegativeSource,
)
from hdwm.models.shared import (
    ActionConditionedCausalTransformer,
    AdaLNTransformerBlock,
    LatentEmbeddingProjector,
    action_indices,
    add_temporal_position_embedding,
    make_action_embedding,
    make_temporal_position_embedding,
    prepare_observation_values,
)

BWMModelConfig = BWMConfig | BWMV2Config


@dataclass(frozen=True)
class BWMV2Output:
    """BWMv2 outputs with LE-WM main-branch names.

    Shapes:
        shared_encoded: [B, T, L, M]
        encoded: [B, T, L, M]
        embedding: [B, T, D]
        sigreg_embedding: [B, T, D]
        prediction: [B, T-1, D]
        target: [B, T-1, D]
        prior: [B, T-1, K]
        modulated_encoded: [B, T-1, L, M]
        modulated_embedding: [B, T-1, D]
        modulated_sigreg_embedding: [B, T-1, D]
        observations: [B, T, L]
    """

    shared_encoded: torch.Tensor
    encoded: torch.Tensor
    embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    prior: torch.Tensor
    modulated_encoded: torch.Tensor
    modulated_embedding: torch.Tensor
    modulated_sigreg_embedding: torch.Tensor
    observations: torch.Tensor

    @property
    def bwm_branch_available(self) -> bool:
        """Whether the prior-modulated BWM branch was computed."""

        return self.modulated_embedding.shape[1] == self.prediction.shape[1]

    @property
    def encoded_tokens(self) -> torch.Tensor:
        """Explicit name for the LE-WM encoded token branch."""

        return self.encoded

    @property
    def encoded_z(self) -> torch.Tensor:
        """Explicit name for the LE-WM embedding space."""

        return self.embedding

    @property
    def encoded_sigreg_z(self) -> torch.Tensor:
        """Explicit name for the LE-WM SIGReg embedding space."""

        return self.sigreg_embedding

    @property
    def encoded_target_z(self) -> torch.Tensor:
        """Explicit name for the LE-WM next-step prediction target."""

        return self.target

    @property
    def pred_z(self) -> torch.Tensor:
        """Explicit name for predicted z."""

        return self.prediction

    @property
    def modulated_z(self) -> torch.Tensor:
        """Explicit name for prior-modulated z."""

        return self.modulated_embedding

    @property
    def modulated_sigreg_z(self) -> torch.Tensor:
        """Explicit name for prior-modulated SIGReg z."""

        return self.modulated_sigreg_embedding


@dataclass(frozen=True)
class BWMOutput:
    """Backward-compatible original BWM output.

    Original BWM keeps the prior-modulated branch as `target` and exposes the
    LE-WM branch under plain/encoded aliases.
    """

    shared_encoded: torch.Tensor
    plain_encoded: torch.Tensor
    embedding: torch.Tensor
    plain_sigreg_embedding: torch.Tensor
    prior: torch.Tensor
    prediction: torch.Tensor
    modulated_encoded: torch.Tensor
    modulated_embedding: torch.Tensor
    sigreg_embedding: torch.Tensor
    target: torch.Tensor
    observations: torch.Tensor

    @property
    def bwm_branch_available(self) -> bool:
        """Whether the prior-modulated BWM branch was computed."""

        return self.modulated_embedding.shape[1] == self.prediction.shape[1]

    @property
    def encoded(self) -> torch.Tensor:
        """LE-WM-style name for the plain encoded token branch."""

        return self.plain_encoded

    @property
    def encoded_tokens(self) -> torch.Tensor:
        """Explicit name for the plain encoded token branch."""

        return self.plain_encoded

    @property
    def encoded_z(self) -> torch.Tensor:
        """Explicit name for the plain embedding branch."""

        return self.embedding

    @property
    def encoded_sigreg_z(self) -> torch.Tensor:
        """Explicit name for the plain SIGReg embedding branch."""

        return self.plain_sigreg_embedding

    @property
    def encoded_target_z(self) -> torch.Tensor:
        """LE-WM branch target z aligned to one-step predictions."""

        return self.embedding[:, 1:]

    @property
    def pred_z(self) -> torch.Tensor:
        """Explicit name for predicted z."""

        return self.prediction

    @property
    def modulated_z(self) -> torch.Tensor:
        """Explicit name for prior-modulated z."""

        return self.modulated_embedding

    @property
    def modulated_sigreg_z(self) -> torch.Tensor:
        """Explicit name for prior-modulated SIGReg z."""

        return self.sigreg_embedding


class SplitObservationEncoder(nn.Module):
    """Observation encoder with shared lower layers and an encoded-z branch."""

    def __init__(self, config: BWMModelConfig) -> None:
        super().__init__()
        self.config = config
        model_dim = config.effective_model_dim
        self.bit_embedding = nn.Embedding(2, model_dim)
        self.value_projection = (
            nn.Linear(config.observation_channels, model_dim)
            if config.observation_channels != 1
            else None
        )
        spatial_shape = config.effective_observation_spatial_shape
        if len(spatial_shape) == 1:
            self.position_embedding = nn.Embedding(config.observation_size, model_dim)
            self.y_position_embedding = None
            self.x_position_embedding = None
        else:
            self.position_embedding = None
            self.y_position_embedding = nn.Embedding(spatial_shape[0], model_dim)
            self.x_position_embedding = nn.Embedding(spatial_shape[1], model_dim)
        self.shared_layers = nn.ModuleList(
            [
                self._make_encoder_layer(config)
                for _ in range(config.shared_encoder_layers)
            ]
        )
        self.plain_layers = nn.ModuleList(
            [self._make_encoder_layer(config) for _ in range(config.encoder_layers)]
        )
        self.plain_norm = nn.LayerNorm(model_dim)

    def forward_shared(
        self,
        observations: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = prepare_observation_values(
            self.config,
            observations,
            self.bit_embedding.weight.dtype,
        )
        batch_size, sequence_length, observation_size, _ = values.shape
        bit_tokens = self._embed_observations(values)
        if observation_mask is not None:
            expected_mask_shape = (batch_size, sequence_length, observation_size)
            if observation_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"expected observation_mask shape {expected_mask_shape}, "
                    f"got {tuple(observation_mask.shape)}"
                )
            bit_tokens = bit_tokens * observation_mask.unsqueeze(-1)
        position_tokens = self._position_tokens(observations.device)
        tokens = bit_tokens + position_tokens.view(1, 1, observation_size, -1)
        flat_tokens = tokens.view(batch_size * sequence_length, observation_size, -1)
        for layer in self.shared_layers:
            flat_tokens = layer(flat_tokens)
        return flat_tokens.view(batch_size, sequence_length, observation_size, -1)

    def forward_plain(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        if shared_encoded.ndim != 4:
            raise ValueError(
                f"expected shared_encoded rank 4, got {shared_encoded.ndim}"
            )
        batch_size, sequence_length, observation_size, model_dim = shared_encoded.shape
        flat_tokens = shared_encoded.reshape(
            batch_size * sequence_length,
            observation_size,
            model_dim,
        )
        for layer in self.plain_layers:
            flat_tokens = layer(flat_tokens)
        flat_tokens = self.plain_norm(flat_tokens)
        return flat_tokens.view(
            batch_size, sequence_length, observation_size, model_dim
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_encoded = self.forward_shared(observations)
        plain_encoded = self.forward_plain(shared_encoded)
        return shared_encoded, plain_encoded

    def _make_encoder_layer(self, config: BWMModelConfig) -> nn.TransformerEncoderLayer:
        model_dim = config.effective_model_dim
        return nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=config.encoder_heads,
            dim_feedforward=model_dim * config.mlp_ratio,
            dropout=config.dropout,
            batch_first=True,
        )

    def _embed_observations(self, values: torch.Tensor) -> torch.Tensor:
        if self.value_projection is not None:
            return self.value_projection(values)
        off_embedding, on_embedding = self.bit_embedding.weight
        return off_embedding + values * (on_embedding - off_embedding)

    def _position_tokens(self, device: torch.device) -> torch.Tensor:
        spatial_shape = self.config.effective_observation_spatial_shape
        if len(spatial_shape) == 1:
            if self.position_embedding is None:
                raise RuntimeError("1D observation encoder is missing positions")
            positions = torch.arange(self.config.observation_size, device=device)
            return self.position_embedding(positions)

        if self.y_position_embedding is None or self.x_position_embedding is None:
            raise RuntimeError("2D observation encoder is missing x/y positions")
        height, width = spatial_shape
        y_positions = torch.arange(height, device=device).repeat_interleave(width)
        x_positions = torch.arange(width, device=device).repeat(height)
        return self.y_position_embedding(y_positions) + self.x_position_embedding(
            x_positions
        )


class BWMNextPredictor(nn.Module):
    """Autoregressive predictor producing prior and predicted z."""

    def __init__(self, config: BWMModelConfig) -> None:
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
        self.prior_head = nn.Linear(model_dim, config.prior_dim)
        self.post_head = nn.Linear(model_dim, config.latent_dim)

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
            return empty.new_empty(batch_size, 0, self.config.prior_dim), empty
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
        return self.prior_head(hidden), self.post_head(hidden)


class PriorModulatedEncoder(nn.Module):
    """Upper observation encoder branch producing prior-modulated z."""

    def __init__(self, config: BWMModelConfig) -> None:
        super().__init__()
        self.config = config
        self.prior_form = BWMPriorForm(config.prior_form)
        model_dim = config.effective_model_dim
        self.prior_dropout = nn.Dropout(config.prior_dropout)
        if self.prior_form == BWMPriorForm.ADALN:
            self.prior_projection = nn.Linear(config.prior_dim, model_dim)
            self.layers = nn.ModuleList(
                [
                    AdaLNTransformerBlock(
                        dim=model_dim,
                        heads=config.encoder_heads,
                        mlp_dim=model_dim * config.mlp_ratio,
                        dropout=config.dropout,
                        causal=False,
                        use_shift=config.modulate_shift,
                    )
                    for _ in range(config.encoder_layers)
                ]
            )
        elif self.prior_form == BWMPriorForm.OBSERVATION_SOFTMAX_MASK:
            self.prior_projection = nn.Linear(config.prior_dim, config.observation_size)
            self.layers = self._make_plain_layers(config)
        else:
            self.prior_projection = nn.Linear(
                config.prior_dim,
                config.observation_size * model_dim,
            )
            self.layers = self._make_plain_layers(config)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self, shared_encoded: torch.Tensor, prior: torch.Tensor
    ) -> torch.Tensor:
        if shared_encoded.ndim != 4:
            raise ValueError(
                f"expected shared_encoded rank 4, got {shared_encoded.ndim}"
            )
        batch_size, sequence_length, observation_size, model_dim = shared_encoded.shape
        expected_prior_shape = (batch_size, sequence_length, self.config.prior_dim)
        if prior.shape != expected_prior_shape:
            raise ValueError(
                f"expected prior shape {expected_prior_shape}, got {tuple(prior.shape)}"
            )

        dropped_prior = self.prior_dropout(prior)
        flat_tokens = self._inject_prior(shared_encoded, dropped_prior)
        flat_tokens = self.output_norm(flat_tokens)
        return flat_tokens.view(
            batch_size, sequence_length, observation_size, model_dim
        )

    def _inject_prior(
        self,
        shared_encoded: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, observation_size, model_dim = shared_encoded.shape
        flat_tokens = shared_encoded.reshape(
            batch_size * sequence_length,
            observation_size,
            model_dim,
        )
        if self.prior_form == BWMPriorForm.ADALN:
            condition = (
                self.prior_projection(prior)
                .unsqueeze(2)
                .expand(
                    batch_size,
                    sequence_length,
                    observation_size,
                    model_dim,
                )
            )
            flat_condition = condition.reshape(
                batch_size * sequence_length,
                observation_size,
                model_dim,
            )
            for layer in self.layers:
                flat_tokens = layer(flat_tokens, flat_condition)
            return flat_tokens

        if self.prior_form == BWMPriorForm.OBSERVATION_SOFTMAX_MASK:
            gated = shared_encoded
        elif self.prior_form == BWMPriorForm.FEATURE_SOFTMAX_MULTIPLIER:
            logits = self.prior_projection(prior).view(
                batch_size,
                sequence_length,
                observation_size,
                model_dim,
            )
            multiplier = F.softmax(logits.flatten(start_dim=2), dim=-1).view_as(logits)
            gated = shared_encoded * multiplier.mul(observation_size * model_dim)
        else:
            raise ValueError(f"unsupported BWM prior form: {self.prior_form}")

        flat_tokens = gated.reshape(
            batch_size * sequence_length,
            observation_size,
            model_dim,
        )
        for layer in self.layers:
            flat_tokens = layer(flat_tokens)
        return flat_tokens

    def _make_plain_layers(self, config: BWMModelConfig) -> nn.ModuleList:
        model_dim = config.effective_model_dim
        return nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=config.encoder_heads,
                    dim_feedforward=model_dim * config.mlp_ratio,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.encoder_layers)
            ]
        )

    def observation_mask_from_prior(
        self,
        observations: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        if self.prior_form != BWMPriorForm.OBSERVATION_SOFTMAX_MASK:
            raise ValueError("observation_mask_from_prior requires observation mask")
        values = prepare_observation_values(
            self.config,
            observations,
            self.prior_projection.weight.dtype,
        )
        batch_size, sequence_length, observation_size, _ = values.shape
        expected_prior_shape = (batch_size, sequence_length, self.config.prior_dim)
        if prior.shape != expected_prior_shape:
            raise ValueError(
                f"expected prior shape {expected_prior_shape}, got {tuple(prior.shape)}"
            )

        dropped_prior = self.prior_dropout(prior)
        return F.softmax(self.prior_projection(dropped_prior), dim=-1)


class BWMV2(nn.Module):
    """BWMv2 with LE-WM as the main branch and optional BWM branch."""

    def __init__(self, config: BWMModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SplitObservationEncoder(config)
        self.embedding_projector = LatentEmbeddingProjector(config)
        self.predictor = BWMNextPredictor(config)
        bwm_branch_enabled = getattr(config, "bwm_branch_enabled", True)
        self.modulated_encoder = (
            PriorModulatedEncoder(config) if bwm_branch_enabled else None
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> BWMV2Output:
        shared_encoded, plain_encoded = self.encoder(observations)
        encoded_z, encoded_sigreg_z = self.embedding_projector(plain_encoded)
        raw_prior, pred_z = self.predictor(encoded_z, actions)
        prior = self._normalize_prior(raw_prior)
        encoded_target_z = encoded_z[:, 1:]
        modulated_encoded, modulated_z, modulated_sigreg_z = self._optional_bwm_branch(
            target_shared_encoded=shared_encoded[:, 1:],
            target_observations=observations[:, 1:],
            prior=prior,
            reference_z=encoded_target_z,
        )
        return self._build_output(
            shared_encoded=shared_encoded,
            encoded_tokens=plain_encoded,
            encoded_z=encoded_z,
            encoded_sigreg_z=encoded_sigreg_z,
            prior=prior,
            pred_z=pred_z,
            modulated_encoded=modulated_encoded,
            modulated_z=modulated_z,
            modulated_sigreg_z=modulated_sigreg_z,
            observations=observations,
        )

    def _build_output(
        self,
        shared_encoded: torch.Tensor,
        encoded_tokens: torch.Tensor,
        encoded_z: torch.Tensor,
        encoded_sigreg_z: torch.Tensor,
        prior: torch.Tensor,
        pred_z: torch.Tensor,
        modulated_encoded: torch.Tensor,
        modulated_z: torch.Tensor,
        modulated_sigreg_z: torch.Tensor,
        observations: torch.Tensor,
    ) -> BWMV2Output:
        return BWMV2Output(
            shared_encoded=shared_encoded,
            encoded=encoded_tokens,
            embedding=encoded_z,
            sigreg_embedding=encoded_sigreg_z,
            prediction=pred_z,
            target=encoded_z[:, 1:],
            prior=prior,
            modulated_encoded=modulated_encoded,
            modulated_embedding=modulated_z,
            modulated_sigreg_embedding=modulated_sigreg_z,
            observations=observations,
        )

    def encode_target_with_prior(
        self,
        target_shared_encoded: torch.Tensor,
        prior: torch.Tensor,
        target_observations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode target observations through the prior-modulated branch."""

        if self.modulated_encoder is None:
            raise ValueError("BWM branch is disabled by model.bwm_branch_enabled=false")
        prior = self._normalize_prior(prior)
        if self.modulated_encoder.prior_form == BWMPriorForm.OBSERVATION_SOFTMAX_MASK:
            if target_observations is None:
                raise ValueError(
                    "target_observations is required for observation_softmax_mask"
                )
            observation_mask = self.modulated_encoder.observation_mask_from_prior(
                target_observations,
                prior,
            )
            target_shared_encoded = self.encoder.forward_shared(
                target_observations,
                observation_mask=observation_mask,
            )
        modulated_encoded = self.modulated_encoder(
            target_shared_encoded,
            prior,
        )
        modulated_embedding, sigreg_embedding = self.embedding_projector(
            modulated_encoded
        )
        return modulated_encoded, modulated_embedding, sigreg_embedding

    def _optional_bwm_branch(
        self,
        target_shared_encoded: torch.Tensor,
        target_observations: torch.Tensor,
        prior: torch.Tensor,
        reference_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prior.shape[1] == 0 or self.modulated_encoder is None:
            return (
                target_shared_encoded[:, :0],
                reference_z[:, :0],
                reference_z[:, :0],
            )
        return self.encode_target_with_prior(
            target_shared_encoded,
            prior,
            target_observations=target_observations,
        )

    def obs_cmi_logits(
        self,
        output: BWMV2Output,
        fixed_negatives: int,
        temperature: float,
        negative_source: ObsCMINegativeSource | str = ObsCMINegativeSource.BATCH,
        noise_ratio: float = 0.05,
        normalize_prediction: bool = True,
        normalize_target: bool = True,
    ) -> torch.Tensor:
        """Contrast predictions against observation targets or noisy batch features."""

        if fixed_negatives < 0:
            raise ValueError("fixed_negatives must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if noise_ratio < 0.0:
            raise ValueError("noise_ratio must be non-negative")
        negative_source = ObsCMINegativeSource(negative_source)
        if output.prediction.numel() == 0:
            return output.prediction.new_empty(0, 0)
        if not output.bwm_branch_available:
            raise ValueError("observation CMI requires the BWM branch to be enabled")

        batch_size, steps, latent_dim = output.prediction.shape
        if output.modulated_embedding.shape != (batch_size, steps, latent_dim):
            raise ValueError(
                f"expected modulated target shape {(batch_size, steps, latent_dim)}, "
                f"got {tuple(output.modulated_embedding.shape)}"
            )
        target_shared = output.shared_encoded[:, 1:]
        if target_shared.shape[:2] != (batch_size, steps):
            raise ValueError(
                f"expected next shared encoding shape prefix {(batch_size, steps)}, "
                f"got {tuple(target_shared.shape[:2])}"
            )
        target_observations = output.observations[:, 1:]
        if target_observations.shape[:2] != (batch_size, steps):
            raise ValueError(
                f"expected next observation shape prefix {(batch_size, steps)}, "
                f"got {tuple(target_observations.shape[:2])}"
            )
        if (
            self.modulated_encoder is not None
            and self.modulated_encoder.prior_form
            == BWMPriorForm.OBSERVATION_SOFTMAX_MASK
            and negative_source == ObsCMINegativeSource.NOISE
        ):
            raise ValueError(
                "obs_cmi_negative_source=noise is incompatible with "
                "observation_softmax_mask because it samples shared features, "
                "not raw observations"
            )

        num_items = batch_size * steps
        batch_candidate_shared = target_shared.reshape(
            num_items,
            target_shared.shape[2],
            target_shared.shape[3],
        )
        batch_candidate_observations = target_observations.reshape(
            num_items,
            target_observations.shape[2],
        )
        if negative_source == ObsCMINegativeSource.BATCH:
            candidate_indices = self._sample_candidate_indices(
                num_items=num_items,
                fixed_negatives=fixed_negatives,
                device=output.prediction.device,
            )
            candidate_count = candidate_indices.shape[1]
            candidate_shared = batch_candidate_shared[candidate_indices]
            candidate_observations = batch_candidate_observations[candidate_indices]
        else:
            candidate_count = fixed_negatives + 1
            positive_shared = batch_candidate_shared.unsqueeze(1)
            positive_observations = batch_candidate_observations.unsqueeze(1)
            if fixed_negatives > 0:
                negative_indices = self._sample_negative_candidate_indices(
                    num_items=num_items,
                    fixed_negatives=fixed_negatives,
                    device=output.prediction.device,
                )
                negative_shared = batch_candidate_shared[negative_indices]
                negative_observations = batch_candidate_observations[negative_indices]
                if noise_ratio > 0.0:
                    negative_shared = negative_shared + self._scaled_shared_noise(
                        reference_shared=negative_shared,
                        noise_ratio=noise_ratio,
                    )
                candidate_shared = torch.cat([positive_shared, negative_shared], dim=1)
                candidate_observations = torch.cat(
                    [positive_observations, negative_observations],
                    dim=1,
                )
            else:
                candidate_shared = positive_shared
                candidate_observations = positive_observations
        prior = output.prior.reshape(num_items, self.config.prior_dim)
        pairwise_prior = (
            prior.view(num_items, 1, self.config.prior_dim)
            .expand(num_items, candidate_count, self.config.prior_dim)
            .reshape(num_items, candidate_count, self.config.prior_dim)
        )
        _, pairwise_target, _ = self.encode_target_with_prior(
            candidate_shared,
            pairwise_prior,
            target_observations=candidate_observations,
        )
        prediction = output.prediction.reshape(num_items, latent_dim)
        if normalize_prediction:
            prediction = F.normalize(prediction, dim=-1)
        if normalize_target:
            pairwise_target = F.normalize(pairwise_target, dim=-1)
        logits = torch.einsum("nd,nud->nu", prediction, pairwise_target)
        return logits / temperature

    def _sample_candidate_indices(
        self,
        num_items: int,
        fixed_negatives: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_items <= 0:
            raise ValueError("num_items must be positive")

        positive = torch.arange(num_items, device=device).view(num_items, 1)
        negative_count = min(fixed_negatives, num_items - 1)
        if negative_count == 0:
            return positive
        if negative_count == num_items - 1:
            all_indices = torch.arange(num_items, device=device).view(1, num_items)
            all_indices = all_indices.expand(num_items, num_items)
            negative = all_indices[all_indices != positive].view(num_items, -1)
        else:
            negative = torch.randint(
                low=0,
                high=num_items - 1,
                size=(num_items, negative_count),
                device=device,
            )
            negative = negative + (negative >= positive).long()
        return torch.cat([positive, negative], dim=1)

    def _sample_negative_candidate_indices(
        self,
        num_items: int,
        fixed_negatives: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if fixed_negatives < 0:
            raise ValueError("fixed_negatives must be non-negative")
        if fixed_negatives == 0:
            return torch.empty(num_items, 0, dtype=torch.long, device=device)
        if num_items == 1:
            return torch.zeros(
                num_items,
                fixed_negatives,
                dtype=torch.long,
                device=device,
            )

        positive = torch.arange(num_items, device=device).view(num_items, 1)
        negative = torch.randint(
            low=0,
            high=num_items - 1,
            size=(num_items, fixed_negatives),
            device=device,
        )
        return negative + (negative >= positive).long()

    def _scaled_shared_noise(
        self,
        reference_shared: torch.Tensor,
        noise_ratio: float,
    ) -> torch.Tensor:
        if reference_shared.ndim < 1:
            raise ValueError(
                "expected reference_shared rank at least 1, "
                f"got {reference_shared.ndim}"
            )
        if noise_ratio < 0.0:
            raise ValueError("noise_ratio must be non-negative")

        noise = torch.randn_like(reference_shared)
        reference_norm = reference_shared.norm(dim=-1, keepdim=True)
        return F.normalize(noise, dim=-1) * reference_norm * noise_ratio

    def prior_cmi_logits(
        self,
        output: BWMV2Output,
        fixed_negatives: int,
        temperature: float,
        negative_source: PriorCMINegativeSource | str = (
            PriorCMINegativeSource.ONE_HOT
        ),
        normalize_prediction: bool = True,
        normalize_target: bool = True,
    ) -> torch.Tensor:
        """Contrast correct priors against one-hot or random prior candidates."""

        if fixed_negatives < 0:
            raise ValueError("fixed_negatives must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        negative_source = PriorCMINegativeSource(negative_source)
        if output.prediction.numel() == 0:
            return output.prediction.new_empty(0, 0)
        if not output.bwm_branch_available:
            raise ValueError("prior CMI requires the BWM branch to be enabled")

        batch_size, steps, latent_dim = output.prediction.shape
        target_shared = output.shared_encoded[:, 1:]
        if target_shared.shape[:2] != (batch_size, steps):
            raise ValueError(
                f"expected next shared encoding shape prefix {(batch_size, steps)}, "
                f"got {tuple(target_shared.shape[:2])}"
            )
        target_observations = output.observations[:, 1:]

        num_items = batch_size * steps
        candidate_count = fixed_negatives + 1
        correct_shared = target_shared.reshape(
            num_items,
            target_shared.shape[2],
            target_shared.shape[3],
        )
        correct_prior = self._normalize_prior(
            output.prior.reshape(num_items, self.config.prior_dim)
        )
        if fixed_negatives > 0:
            if negative_source == PriorCMINegativeSource.ONE_HOT:
                if fixed_negatives <= self.config.prior_dim:
                    indices = torch.randperm(
                        self.config.prior_dim,
                        device=output.prior.device,
                    )[:fixed_negatives]
                else:
                    indices = torch.randint(
                        low=0,
                        high=self.config.prior_dim,
                        size=(fixed_negatives,),
                        device=output.prior.device,
                    )
                negative_priors = F.one_hot(
                    indices,
                    num_classes=self.config.prior_dim,
                ).to(dtype=output.prior.dtype)
                negative_priors = negative_priors.view(
                    1, fixed_negatives, self.config.prior_dim
                ).expand(num_items, fixed_negatives, self.config.prior_dim)
            else:
                negative_priors = torch.randn(
                    num_items,
                    fixed_negatives,
                    self.config.prior_dim,
                    dtype=output.prior.dtype,
                    device=output.prior.device,
                )
            candidate_prior = torch.cat(
                [
                    correct_prior.unsqueeze(1),
                    negative_priors,
                ],
                dim=1,
            )
        else:
            candidate_prior = correct_prior.unsqueeze(1)
        candidate_shared = (
            correct_shared.view(num_items, 1, *correct_shared.shape[1:])
            .expand(num_items, candidate_count, *correct_shared.shape[1:])
            .reshape(num_items, candidate_count, *correct_shared.shape[1:])
        )
        correct_observations = target_observations.reshape(
            num_items,
            target_observations.shape[2],
        )
        candidate_observations = (
            correct_observations.view(num_items, 1, *correct_observations.shape[1:])
            .expand(num_items, candidate_count, *correct_observations.shape[1:])
            .reshape(num_items, candidate_count, *correct_observations.shape[1:])
        )
        _, candidate_target, _ = self.encode_target_with_prior(
            candidate_shared,
            candidate_prior,
            target_observations=candidate_observations,
        )
        prediction = output.prediction.reshape(num_items, latent_dim)
        if normalize_prediction:
            prediction = F.normalize(prediction, dim=-1)
        if normalize_target:
            candidate_target = F.normalize(candidate_target, dim=-1)
        logits = torch.einsum("nd,ncd->nc", prediction, candidate_target)
        return logits / temperature

    def _normalize_prior(self, prior: torch.Tensor) -> torch.Tensor:
        """Keep prior modulation scale stable across predictor states."""

        return F.normalize(prior, dim=-1)

    def shuffled_target(self, output: BWMV2Output) -> torch.Tensor:
        """Build a wrong-observation target using the same predictor prior.

        The shuffle rolls target observations across the batch when possible, so
        each prior is paired with a target observation from another trajectory at
        the same timestep. With batch size 1, it falls back to a temporal roll.
        """

        shuffled_shared_encoded = self._shuffled_target_shared_encoded(output)
        shuffled_observations = self._shuffled_target_observations(output)
        _, shuffled_target, _ = self.encode_target_with_prior(
            shuffled_shared_encoded,
            output.prior,
            target_observations=shuffled_observations,
        )
        return shuffled_target

    def shuffled_target_without_prior(self, output: BWMV2Output) -> torch.Tensor:
        """Build a wrong-observation target with no predictor prior injected."""

        shuffled_shared_encoded = self._shuffled_target_shared_encoded(output)
        shuffled_observations = self._shuffled_target_observations(output)
        zero_prior = torch.zeros_like(output.prior)
        _, shuffled_target, _ = self.encode_target_with_prior(
            shuffled_shared_encoded,
            zero_prior,
            target_observations=shuffled_observations,
        )
        return shuffled_target

    def target_without_prior(self, output: BWMV2Output) -> torch.Tensor:
        """Build the true-observation target with no predictor prior injected."""

        zero_prior = torch.zeros_like(output.prior)
        _, target, _ = self.encode_target_with_prior(
            output.shared_encoded[:, 1:],
            zero_prior,
            target_observations=output.observations[:, 1:],
        )
        return target

    def _shuffled_target_shared_encoded(self, output: BWMV2Output) -> torch.Tensor:
        target_shared_encoded = output.shared_encoded[:, 1:]
        if target_shared_encoded.shape[0] > 1:
            return target_shared_encoded.roll(shifts=1, dims=0)
        elif target_shared_encoded.shape[1] > 1:
            return target_shared_encoded.roll(shifts=1, dims=1)
        return target_shared_encoded

    def _shuffled_target_observations(self, output: BWMV2Output) -> torch.Tensor:
        target_observations = output.observations[:, 1:]
        if target_observations.shape[0] > 1:
            return target_observations.roll(shifts=1, dims=0)
        elif target_observations.shape[1] > 1:
            return target_observations.roll(shifts=1, dims=1)
        return target_observations


class BWM(BWMV2):
    """Compatibility wrapper for the original BWM config and output type."""

    def _build_output(
        self,
        shared_encoded: torch.Tensor,
        encoded_tokens: torch.Tensor,
        encoded_z: torch.Tensor,
        encoded_sigreg_z: torch.Tensor,
        prior: torch.Tensor,
        pred_z: torch.Tensor,
        modulated_encoded: torch.Tensor,
        modulated_z: torch.Tensor,
        modulated_sigreg_z: torch.Tensor,
        observations: torch.Tensor,
    ) -> BWMOutput:
        return BWMOutput(
            shared_encoded=shared_encoded,
            plain_encoded=encoded_tokens,
            embedding=encoded_z,
            plain_sigreg_embedding=encoded_sigreg_z,
            prior=prior,
            prediction=pred_z,
            modulated_encoded=modulated_encoded,
            modulated_embedding=modulated_z,
            sigreg_embedding=modulated_sigreg_z,
            target=modulated_z,
            observations=observations,
        )
