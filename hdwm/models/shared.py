"""Shared neural network modules for HDWM model variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hdwm.config import EncoderModelConfig, ProjectionStage


def prepare_observation_values(
    config: EncoderModelConfig,
    observations: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Flatten per-frame observations into cell tokens with channel values."""

    if observations.ndim < 3:
        raise ValueError(
            f"expected observations rank at least 3, got {observations.ndim}"
        )

    batch_size, sequence_length = observations.shape[:2]
    spatial_shape = config.effective_observation_spatial_shape
    channels = config.observation_channels
    frame_shape = tuple(observations.shape[2:])

    if channels == 1 and frame_shape == spatial_shape:
        values = observations.reshape(
            batch_size,
            sequence_length,
            config.observation_size,
            1,
        )
    elif frame_shape == (*spatial_shape, channels):
        values = observations.reshape(
            batch_size,
            sequence_length,
            config.observation_size,
            channels,
        )
    else:
        expected = (
            f"{spatial_shape} or {(*spatial_shape, 1)}"
            if channels == 1
            else f"{(*spatial_shape, channels)}"
        )
        raise ValueError(
            f"expected observation frame shape {expected}, got {frame_shape}"
        )
    return values.to(dtype=dtype).clamp(0.0, 1.0)


def action_indices(config: EncoderModelConfig, actions: torch.Tensor) -> torch.Tensor:
    """Map environment action ids into embedding indices."""

    indices = actions.long()
    if ((indices < 0) | (indices >= config.num_actions)).any():
        raise ValueError(f"actions must be in [0, {config.num_actions - 1}]")
    return indices


class ObservationEncoder(nn.Module):
    """Per-frame self-attention encoder.

    Shapes:
        observations: [B, T, L]
        output: [B, T, L, M]
    """

    def __init__(self, config: EncoderModelConfig) -> None:
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=config.encoder_heads,
            dim_feedforward=model_dim * config.mlp_ratio,
            dropout=config.dropout,
            batch_first=True,
        )
        self.frame_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.encoder_layers,
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        values = prepare_observation_values(
            self.config,
            observations,
            self.bit_embedding.weight.dtype,
        )
        batch_size, sequence_length, observation_size, _ = values.shape
        # bit_tokens: [B, T, L, M]
        bit_tokens = self._embed_values(values)
        # position_tokens: [L, M]
        position_tokens = self._position_tokens(observations.device)
        # frame_tokens: [B, T, L, M]
        frame_tokens = bit_tokens + position_tokens.view(1, 1, observation_size, -1)
        # flat_tokens: [B*T, L, M]
        flat_tokens = frame_tokens.view(
            batch_size * sequence_length, observation_size, -1
        )
        # encoded: [B*T, L, M]
        encoded = self.frame_encoder(flat_tokens)
        encoded = self.output_norm(encoded)
        return encoded.view(batch_size, sequence_length, observation_size, -1)

    def _embed_values(self, values: torch.Tensor) -> torch.Tensor:
        if self.value_projection is not None:
            return self.value_projection(values)
        return self.bit_embedding(values.squeeze(dim=-1).long().clamp(0, 1))

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


def make_action_embedding(config: EncoderModelConfig, model_dim: int) -> nn.Embedding:
    return nn.Embedding(config.action_vocab_size, model_dim)


def make_temporal_position_embedding(
    config: EncoderModelConfig,
    max_length: int,
    model_dim: int,
) -> nn.Embedding | None:
    if config.uses_rotary_temporal_position_encoding:
        return None
    return nn.Embedding(max_length, model_dim)


def add_temporal_position_embedding(
    inputs: torch.Tensor,
    position_embedding: nn.Embedding | None,
) -> torch.Tensor:
    if position_embedding is None:
        return inputs
    positions = torch.arange(inputs.shape[1], device=inputs.device)
    return inputs + position_embedding(positions).view(1, inputs.shape[1], -1)


class LastDimBatchNorm1d(nn.Module):
    """Apply BatchNorm1d to the final tensor dimension."""

    def __init__(self, dim: int, affine: bool = True) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.norm = nn.BatchNorm1d(dim, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"expected input rank at least 2, got {x.ndim}")
        feature_dim = x.shape[-1]
        flat = x.reshape(-1, feature_dim)
        normalized = self.norm(flat)
        return normalized.view_as(x)


class LatentEmbeddingProjector(nn.Module):
    """Project processed observation features into latent embedding space.

    Pipeline: x1 -> LN -> x2 -> BN -> x3 -> L2 -> x4
    Returns (embedding, sigreg_embedding) selected from different stages.
    """

    def __init__(self, config: EncoderModelConfig) -> None:
        super().__init__()
        self.config = config
        temporal_fusion_enabled = config.latent_temporal_fusion_enabled
        temporal_context_window = config.latent_temporal_context_window
        if temporal_context_window is not None and temporal_context_window < 0:
            raise ValueError("temporal_context_window must be non-negative")
        self.temporal_fusion_enabled = temporal_fusion_enabled
        self.temporal_context_window = temporal_context_window
        self.linear = nn.Linear(config.effective_model_dim, config.latent_dim)
        if temporal_fusion_enabled:
            self.temporal_fusion = nn.Sequential(
                nn.Linear(
                    config.latent_dim * 2,
                    config.latent_dim * config.mlp_ratio,
                ),
                nn.GELU(),
                nn.Linear(config.latent_dim * config.mlp_ratio, config.latent_dim),
            )
        else:
            self.temporal_fusion = None
        if config.latent_layer_norm:
            self.layer_norm = nn.LayerNorm(config.latent_dim)
        else:
            self.layer_norm = None
        if config.latent_batch_norm:
            self.batch_norm = LastDimBatchNorm1d(
                config.latent_dim, affine=config.latent_batch_norm_affine
            )
        else:
            self.batch_norm = None

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self._process_encoded_observations(encoded)
        if processed.shape[-1] != self.config.effective_model_dim:
            raise ValueError(
                f"expected processed encoded dim {self.config.effective_model_dim}, "
                f"got {processed.shape[-1]}"
            )

        # Pipeline: x1 -> LN -> x2 -> BN -> x3 -> L2 -> x4
        x1 = self.linear(processed)
        if self.temporal_fusion is not None:
            pooled = self._causal_temporal_mean_pool(x1)
            x1 = self.temporal_fusion(torch.cat([x1, pooled], dim=-1))
        x2 = self.layer_norm(x1) if self.layer_norm is not None else x1
        x3 = self.batch_norm(x2) if self.batch_norm is not None else x2
        x4 = F.normalize(x3, dim=-1) if self.config.latent_l2_norm else x3

        # Select output stages
        stages = {
            ProjectionStage.PRE_NORM: x1,
            ProjectionStage.POST_LN: x2,
            ProjectionStage.POST_BN: x3,
            ProjectionStage.POST_L2: x4,
        }
        embedding = stages[self.config.embedding_stage]
        sigreg_embedding = stages[self.config.sigreg_stage]
        return embedding, sigreg_embedding

    def _process_encoded_observations(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim == 4:
            return encoded.mean(dim=2)
        if encoded.ndim == 3:
            return encoded
        raise ValueError(f"expected encoded rank 3 or 4, got {encoded.ndim}")

    def _causal_temporal_mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected temporal pooling input rank 3, got {x.ndim}")
        batch_size, sequence_length, latent_dim = x.shape
        if sequence_length == 0:
            return x

        cumulative = x.cumsum(dim=1)
        positions = torch.arange(sequence_length, device=x.device)
        starts = torch.zeros_like(positions)
        if self.temporal_context_window is not None:
            starts = (positions - self.temporal_context_window).clamp_min(0)
        previous = torch.zeros(
            batch_size,
            sequence_length,
            latent_dim,
            dtype=x.dtype,
            device=x.device,
        )
        has_previous = starts > 0
        if has_previous.any():
            previous[:, has_previous] = cumulative[:, starts[has_previous] - 1]
        counts = (
            (positions - starts + 1)
            .to(dtype=x.dtype)
            .view(
                1,
                sequence_length,
                1,
            )
        )
        return (cumulative - previous) / counts


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation."""

    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """Feed-forward network used inside AdaLN transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention with optional causal masking."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        causal: bool,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.dim_head = dim // heads
        if rotary and self.dim_head % 2 != 0:
            raise ValueError("attention head dim must be even for rotary positions")
        self.dropout = dropout
        self.causal = causal
        self.rotary = rotary
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected attention input rank 3, got {x.ndim}")
        batch_size, sequence_length, dim = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        query, key, value = (
            tensor.view(batch_size, sequence_length, self.heads, self.dim_head)
            .transpose(1, 2)
            .contiguous()
            for tensor in qkv
        )
        if self.rotary:
            query, key = apply_rotary_position_embedding(query, key)
        dropout = self.dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            is_causal=self.causal,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                dim,
            )
        )
        return self.to_out(attended)


class CausalSelfAttention(SelfAttention):
    """Scaled dot-product self-attention with causal masking."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        rotary: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            heads=heads,
            dropout=dropout,
            causal=True,
            rotary=rotary,
        )


def apply_rotary_position_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape != key.shape:
        raise ValueError(
            f"expected query/key shape match, got {tuple(query.shape)} and "
            f"{tuple(key.shape)}"
        )
    if query.ndim != 4:
        raise ValueError(f"expected query/key rank 4, got {query.ndim}")
    head_dim = query.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("rotary position embedding requires even head dim")

    sequence_length = query.shape[-2]
    half_dim = head_dim // 2
    frequencies = torch.arange(half_dim, device=query.device, dtype=torch.float32)
    frequencies = 1.0 / (10000.0 ** (frequencies / half_dim))
    positions = torch.arange(sequence_length, device=query.device, dtype=torch.float32)
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos().to(dtype=query.dtype).view(1, 1, sequence_length, half_dim)
    sin = angles.sin().to(dtype=query.dtype).view(1, 1, sequence_length, half_dim)
    return _apply_rotary(query, cos, sin), _apply_rotary(key, cos, sin)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack(
        (
            even * cos - odd * sin,
            even * sin + odd * cos,
        ),
        dim=-1,
    )
    return rotated.flatten(start_dim=-2)


class CausalTransformerBlock(nn.Module):
    """Standard causal transformer block for token sequences."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            rotary=rotary,
        )
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x


class CausalTransformer(nn.Module):
    """Reusable causal transformer over already embedded tokens."""

    def __init__(
        self,
        *,
        model_dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    dim=model_dim,
                    heads=heads,
                    mlp_dim=model_dim * mlp_ratio,
                    dropout=dropout,
                    rotary=rotary,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected transformer input rank 3, got {x.ndim}")
        for layer in self.layers:
            x = layer(x)
        return self.output_projection(self.output_norm(x))


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with shared AdaLN-zero conditioning logic."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        causal: bool,
        rotary: bool = False,
        use_shift: bool = True,
    ) -> None:
        super().__init__()
        self.use_shift = use_shift
        self.attention = SelfAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            causal=causal,
            rotary=rotary,
        )
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape != x.shape:
            raise ValueError(
                f"expected condition shape {tuple(x.shape)}, got "
                f"{tuple(condition.shape)}"
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(condition).chunk(6, dim=-1)
        )
        if not self.use_shift:
            shift_msa = torch.zeros_like(shift_msa)
            shift_mlp = torch.zeros_like(shift_mlp)
        x = x + gate_msa * self.attention(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class AdaLNCausalBlock(AdaLNTransformerBlock):
    """Causal transformer block with AdaLN-zero conditioning."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        rotary: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            causal=True,
            rotary=rotary,
        )


class ActionConditionedCausalTransformer(nn.Module):
    """Causal transformer whose blocks are conditioned through AdaLN-zero."""

    def __init__(
        self,
        *,
        model_dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AdaLNCausalBlock(
                    dim=model_dim,
                    heads=heads,
                    mlp_dim=model_dim * mlp_ratio,
                    dropout=dropout,
                    rotary=rotary,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if x.shape != condition.shape:
            raise ValueError(
                f"expected condition shape {tuple(x.shape)}, got "
                f"{tuple(condition.shape)}"
            )
        for layer in self.layers:
            x = layer(x, condition)
        return self.output_projection(self.output_norm(x))
