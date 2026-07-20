"""Shared recurrent token-workspace model for AIR-JEPA Stage 0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from air_jepa.stage0_workspace.schemas import ModelSpec


def _channel_layer_norm(inputs: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    return norm(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LocalNeighborhoodAttention(nn.Module):
    """Four-neighbor-plus-self attention with learned relative bias."""

    _UNFOLD_INDICES = (1, 3, 4, 5, 7)

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.query = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.key = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.output = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 5))

    def _neighbor_mask(
        self,
        valid_mask: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        unfolded = F.unfold(
            valid_mask[:, None].to(dtype=torch.float32),
            kernel_size=3,
            padding=1,
        )
        indices = torch.as_tensor(
            self._UNFOLD_INDICES,
            dtype=torch.long,
            device=valid_mask.device,
        )
        return (
            unfolded.index_select(1, indices).reshape(
                valid_mask.shape[0], 1, 5, height * width
            )
            > 0.5
        )

    def forward(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("local attention expects [B,C,H,W]")
        batch, channels, height, width = inputs.shape
        if channels != self.hidden_dim:
            raise ValueError("local attention channel mismatch")
        if valid_mask.shape != (batch, height, width):
            raise ValueError("valid_mask must have shape [B,H,W]")
        count = height * width
        query = self.query(inputs).reshape(batch, self.heads, self.head_dim, count)
        index = torch.as_tensor(
            self._UNFOLD_INDICES,
            dtype=torch.long,
            device=inputs.device,
        )

        def unfold(projected: torch.Tensor) -> torch.Tensor:
            values = F.unfold(projected, kernel_size=3, padding=1)
            values = values.reshape(batch, channels, 9, count)
            values = values.index_select(2, index)
            return values.reshape(batch, self.heads, self.head_dim, 5, count)

        key = unfold(self.key(inputs))
        value = unfold(self.value(inputs))
        logits = (query.unsqueeze(3) * key).sum(dim=2) / math.sqrt(self.head_dim)
        logits = logits + self.relative_bias[None, :, :, None]
        neighbor_mask = self._neighbor_mask(valid_mask, height, width)
        logits = logits.masked_fill(~neighbor_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=2)
        attended = (weights.unsqueeze(2) * value).sum(dim=3)
        attended = attended.reshape(batch, channels, height, width)
        attended = self.output(attended)
        return attended * valid_mask[:, None].to(attended.dtype)


class GatedResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if residual.shape != update.shape:
            raise ValueError("gated residual shape mismatch")
        return residual + torch.sigmoid(self.logit) * update


class SharedReasonerBlock(nn.Module):
    """One block reused for every reasoning iteration."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        dim = spec.hidden_dim
        expansion = dim * spec.ffn_expansion
        self.state_norm_local = nn.LayerNorm(dim)
        self.state_norm_cross = nn.LayerNorm(dim)
        self.state_norm_ffn = nn.LayerNorm(dim)
        self.workspace_norm_self = nn.LayerNorm(dim)
        self.workspace_norm_cross = nn.LayerNorm(dim)
        self.workspace_norm_ffn = nn.LayerNorm(dim)
        self.local_attention = LocalNeighborhoodAttention(
            hidden_dim=dim,
            heads=spec.attention_heads,
        )
        self.workspace_self_attention = nn.MultiheadAttention(
            dim,
            spec.attention_heads,
            dropout=spec.dropout,
            batch_first=True,
        )
        self.workspace_from_state = nn.MultiheadAttention(
            dim,
            spec.attention_heads,
            dropout=spec.dropout,
            batch_first=True,
        )
        self.state_from_workspace = nn.MultiheadAttention(
            dim,
            spec.attention_heads,
            dropout=spec.dropout,
            batch_first=True,
        )
        self.state_ffn = nn.Sequential(
            nn.Conv2d(dim, expansion, 1),
            nn.GELU(),
            nn.Conv2d(expansion, dim, 1),
        )
        self.workspace_ffn = nn.Sequential(
            nn.Linear(dim, expansion),
            nn.GELU(),
            nn.Linear(expansion, dim),
        )
        self.local_residual = GatedResidual()
        self.workspace_self_residual = GatedResidual()
        self.workspace_cross_residual = GatedResidual()
        self.state_cross_residual = GatedResidual()
        self.state_ffn_residual = GatedResidual()
        self.workspace_ffn_residual = GatedResidual()
        self.recall_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        workspace: torch.Tensor,
        recall: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape != recall.shape:
            raise ValueError("state and immutable recall must have identical shape")
        local_input = state + torch.sigmoid(self.recall_logit) * recall
        local_update = self.local_attention(
            _channel_layer_norm(local_input, self.state_norm_local),
            valid_mask,
        )
        state = self.local_residual(state, local_update)

        normalized_workspace = self.workspace_norm_self(workspace)
        self_update, _ = self.workspace_self_attention(
            normalized_workspace,
            normalized_workspace,
            normalized_workspace,
            need_weights=False,
        )
        workspace = self.workspace_self_residual(workspace, self_update)

        batch, channels, height, width = state.shape
        state_tokens = state.flatten(2).transpose(1, 2)
        valid_tokens = valid_mask.flatten(1)
        workspace_query = self.workspace_norm_cross(workspace)
        state_values = self.state_norm_cross(state_tokens)
        workspace_update, _ = self.workspace_from_state(
            workspace_query,
            state_values,
            state_values,
            key_padding_mask=~valid_tokens,
            need_weights=False,
        )
        workspace = self.workspace_cross_residual(workspace, workspace_update)

        state_query = self.state_norm_cross(state_tokens)
        workspace_values = self.workspace_norm_cross(workspace)
        state_update, _ = self.state_from_workspace(
            state_query,
            workspace_values,
            workspace_values,
            need_weights=False,
        )
        state_tokens = self.state_cross_residual(state_tokens, state_update)
        state = state_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        state = state * valid_mask[:, None].to(state.dtype)

        state_update = self.state_ffn(_channel_layer_norm(state, self.state_norm_ffn))
        state = self.state_ffn_residual(state, state_update)
        state = state * valid_mask[:, None].to(state.dtype)
        workspace = self.workspace_ffn_residual(
            workspace,
            self.workspace_ffn(self.workspace_norm_ffn(workspace)),
        )
        return state, workspace


class ActionConditionedFutureDecoder(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        dim = spec.hidden_dim
        self.dim = dim
        self.base = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.condition = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, spec.input_dim, 3, padding=1),
        )

    def forward(
        self,
        reasoned_state: torch.Tensor,
        source_latent: torch.Tensor,
        goal: torch.Tensor,
        actions: torch.Tensor,
        futures: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = reasoned_state.shape
        if actions.shape != futures.shape or actions.shape[1:] != (4, self.dim):
            raise ValueError("expected four action and future tokens")
        if goal.shape != (batch, 1, self.dim):
            raise ValueError("goal token shape mismatch")
        goal_expanded = goal.expand(-1, 4, -1)
        gamma, beta = self.condition(
            torch.cat([goal_expanded, actions, futures], dim=-1)
        ).chunk(2, dim=-1)
        base = self.base(reasoned_state)[:, None]
        conditioned = base * (1.0 + 0.1 * torch.tanh(gamma)[..., None, None])
        conditioned = conditioned + beta[..., None, None]
        delta = self.refine(
            conditioned.reshape(batch * 4, self.dim, height, width)
        ).reshape(batch, 4, source_latent.shape[1], height, width)
        predicted = source_latent[:, None] + delta
        return predicted * valid_mask[:, None, None].to(predicted.dtype)


class DistributionalEnergyHead(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        dim = spec.hidden_dim
        self.dim = dim
        self.cost_bins = spec.cost_bins
        self.future_projection = nn.Conv2d(spec.input_dim, dim, 1)
        self.pool = nn.MultiheadAttention(
            dim,
            spec.attention_heads,
            dropout=spec.dropout,
            batch_first=True,
        )
        self.query = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.output = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, spec.cost_bins),
        )
        self.register_buffer(
            "distance_support",
            torch.arange(spec.cost_bins, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        future_fields: torch.Tensor,
        goal: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if future_fields.ndim != 5 or future_fields.shape[1] != 4:
            raise ValueError("future fields must have shape [B,4,C,H,W]")
        batch, _, channels, height, width = future_fields.shape
        if valid_mask.shape != (batch, height, width):
            raise ValueError("energy valid_mask shape mismatch")
        fields = self.future_projection(
            future_fields.reshape(batch * 4, channels, height, width)
        )
        fields = fields.flatten(2).transpose(1, 2)
        goal_expanded = goal.expand(-1, 4, -1)
        query = self.query(torch.cat([goal_expanded, actions], dim=-1)).reshape(
            batch * 4, 1, self.dim
        )
        padding_mask = (~valid_mask.flatten(1))[:, None].expand(-1, 4, -1)
        padding_mask = padding_mask.reshape(batch * 4, height * width)
        pooled, _ = self.pool(
            query,
            fields,
            fields,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        logits = self.output(pooled[:, 0]).reshape(batch, 4, self.cost_bins)
        probabilities = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        energy = (probabilities * self.distance_support.to(logits.dtype)).sum(dim=-1)
        return logits, energy


@dataclass
class AIRWorkspaceOutput:
    cost_logits: torch.Tensor
    energy: torch.Tensor
    iterations: int
    goal_token: torch.Tensor
    action_tokens: torch.Tensor
    future_tokens: torch.Tensor
    predicted_future: torch.Tensor | None


class AIRWorkspaceModel(nn.Module):
    """Frozen-latent AIR core with a shared recurrent reasoning block."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        if spec.input_dim != spec.hidden_dim:
            raise ValueError("AIR0-v1 requires input_dim == hidden_dim")
        self.spec = spec
        dim = spec.hidden_dim
        self.input_adapter = nn.Sequential(
            nn.Conv2d(spec.input_dim, dim, 1, bias=False),
            nn.GroupNorm(8 if dim % 8 == 0 else 1, dim),
            nn.GELU(),
        )
        self.initial_state = nn.Conv2d(dim, dim, 1)
        self.goal_query = nn.Parameter(torch.empty(1, 1, dim))
        self.goal_attention = nn.MultiheadAttention(
            dim,
            spec.attention_heads,
            dropout=spec.dropout,
            batch_first=True,
        )
        self.action_embeddings = nn.Parameter(torch.empty(1, 4, dim))
        self.future_embeddings = nn.Parameter(torch.empty(1, 4, dim))
        self.reasoner = SharedReasonerBlock(spec)
        self.future_decoder = ActionConditionedFutureDecoder(spec)
        self.energy_head = DistributionalEnergyHead(spec)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.goal_query, mean=0.0, std=0.02)
        nn.init.normal_(self.action_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.future_embeddings, mean=0.0, std=0.02)

    @staticmethod
    def supervision_points(iterations: int, every: int) -> tuple[int, ...]:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        points = {int(iterations)}
        if every > 0:
            points.update(range(every, iterations, every))
        return tuple(sorted(points))

    def _initialize(
        self,
        latent: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent.ndim != 4:
            raise ValueError("AIR input latent must have shape [B,C,H,W]")
        batch, _, height, width = latent.shape
        if valid_mask.shape != (batch, height, width):
            raise ValueError("valid_mask must have shape [B,H,W]")
        recall = self.input_adapter(latent)
        recall = recall * valid_mask[:, None].to(recall.dtype)
        state = torch.tanh(self.initial_state(recall))
        tokens = recall.flatten(2).transpose(1, 2)
        goal_query = self.goal_query.expand(batch, -1, -1)
        goal, _ = self.goal_attention(
            goal_query,
            tokens,
            tokens,
            key_padding_mask=~valid_mask.flatten(1),
            need_weights=False,
        )
        actions = self.action_embeddings.expand(batch, -1, -1)
        futures = self.future_embeddings.expand(batch, -1, -1) + actions
        workspace = torch.cat([goal, actions, futures], dim=1)
        return state, workspace, recall

    @staticmethod
    def _split_workspace(
        workspace: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if workspace.ndim != 3 or workspace.shape[1] != 9:
            raise ValueError("workspace must contain 1 goal, 4 action, 4 future tokens")
        return workspace[:, :1], workspace[:, 1:5], workspace[:, 5:9]

    def _readout(
        self,
        state: torch.Tensor,
        source_latent: torch.Tensor,
        workspace: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int,
        *,
        keep_future: bool,
    ) -> AIRWorkspaceOutput:
        goal, actions, futures = self._split_workspace(workspace)
        predicted = self.future_decoder(
            state,
            source_latent,
            goal,
            actions,
            futures,
            valid_mask,
        )
        logits, energy = self.energy_head(
            predicted,
            goal,
            actions,
            valid_mask,
        )
        return AIRWorkspaceOutput(
            cost_logits=logits,
            energy=energy,
            iterations=int(iterations),
            goal_token=goal,
            action_tokens=actions,
            future_tokens=futures,
            predicted_future=predicted if keep_future else None,
        )

    def forward(
        self,
        latent: torch.Tensor,
        *,
        iterations: int,
        deep_supervision_every: int = 0,
        valid_mask: torch.Tensor | None = None,
    ) -> list[AIRWorkspaceOutput]:
        if valid_mask is None:
            valid_mask = torch.ones(
                latent.shape[0],
                latent.shape[-2],
                latent.shape[-1],
                dtype=torch.bool,
                device=latent.device,
            )
        else:
            valid_mask = valid_mask.bool()
        points = set(self.supervision_points(iterations, deep_supervision_every))
        state, workspace, recall = self._initialize(latent, valid_mask)
        outputs: list[AIRWorkspaceOutput] = []
        for index in range(1, iterations + 1):
            state, workspace = self.reasoner(
                state,
                workspace,
                recall,
                valid_mask,
            )
            if index in points:
                outputs.append(
                    self._readout(
                        state,
                        latent,
                        workspace,
                        valid_mask,
                        index,
                        keep_future=index == iterations,
                    )
                )
        if not outputs or outputs[-1].iterations != iterations:
            raise RuntimeError("AIR reasoner failed to produce its final readout")
        return outputs

    def score_external_futures(
        self,
        output: AIRWorkspaceOutput,
        future_fields: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.energy_head(
            future_fields,
            output.goal_token,
            output.action_tokens,
            valid_mask.bool(),
        )

    def analytical_mac_breakdown(
        self,
        maze_size: int,
        iterations: int,
    ) -> dict[str, int]:
        """Count inference Conv/Linear/attention MACs by architectural component."""

        if maze_size <= 0 or iterations <= 0:
            raise ValueError("maze_size and iterations must be positive")
        n = maze_size * maze_size
        d = self.spec.hidden_dim
        input_dim = self.spec.input_dim
        w = 9
        expansion = d * self.spec.ffn_expansion
        local = 4 * n * d * d + 2 * n * 5 * d
        workspace_self = 4 * w * d * d + 2 * w * w * d
        workspace_from_state = w * d * d + 2 * n * d * d + w * d * d + 2 * w * n * d
        state_from_workspace = n * d * d + 2 * w * d * d + n * d * d + 2 * n * w * d
        ffn = 2 * n * d * expansion + 2 * w * d * expansion
        repeated = iterations * (
            local + workspace_self + workspace_from_state + state_from_workspace + ffn
        )
        adapter = n * input_dim * d + n * d * d
        goal_pool = 2 * n * d * d + 2 * n * d + 2 * d * d
        future_base = 9 * n * d * d
        future_condition = 4 * (10 * d * d)
        future_refine = 4 * n * (9 * d * d + 9 * d * input_dim)
        energy_query = 4 * (3 * d * d)
        energy_pool = 4 * (n * input_dim * d + 2 * n * d * d + 2 * n * d + 2 * d * d)
        energy_output = 4 * (2 * d * d + 2 * d * self.spec.cost_bins)
        return {
            "adapter_and_goal": int(adapter + goal_pool),
            "shared_reasoner": int(repeated),
            "future_decoder": int(future_base + future_condition + future_refine),
            "energy_head": int(energy_query + energy_pool + energy_output),
        }

    def analytical_macs(self, maze_size: int, iterations: int) -> int:
        """Return the locked total for the component-wise analytical MAC count."""

        return int(sum(self.analytical_mac_breakdown(maze_size, iterations).values()))


def require_finite_output(output: AIRWorkspaceOutput) -> None:
    values: list[tuple[str, torch.Tensor | None]] = [
        ("cost_logits", output.cost_logits),
        ("energy", output.energy),
        ("goal_token", output.goal_token),
        ("action_tokens", output.action_tokens),
        ("future_tokens", output.future_tokens),
        ("predicted_future", output.predicted_future),
    ]
    for name, value in values:
        if value is not None and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite AIR output: {name}")


def model_config_dict(spec: ModelSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json")


__all__ = [
    "AIRWorkspaceModel",
    "AIRWorkspaceOutput",
    "LocalNeighborhoodAttention",
    "model_config_dict",
    "require_finite_output",
]
