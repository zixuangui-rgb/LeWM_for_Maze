"""Distance/cost heads with one scoring contract for training and planning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from distance_head_study.schemas import ArchitectureKind, HeadSpec, OutputKind


def _mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(width, int(hidden)), nn.ReLU()))
        width = int(hidden)
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class HeadOutput:
    """All optional outputs plus the scalar cost used by a planner."""

    score: torch.Tensor
    scalar: torch.Tensor | None = None
    ordinal_logits: torch.Tensor | None = None
    distribution_logits: torch.Tensor | None = None
    quantiles: torch.Tensor | None = None
    reachability_logits: torch.Tensor | None = None
    log_variance: torch.Tensor | None = None

    def validate(self, batch_size: int) -> None:
        if self.score.shape != (batch_size,):
            raise ValueError("DistanceHead score must have shape [batch]")
        tensors = (
            self.score,
            self.scalar,
            self.ordinal_logits,
            self.distribution_logits,
            self.quantiles,
            self.reachability_logits,
            self.log_variance,
        )
        if any(
            value is not None and not torch.isfinite(value).all() for value in tensors
        ):
            raise FloatingPointError("DistanceHead produced a non-finite output")


class DistanceHeadModel(nn.Module):
    """Strict superset of the historical concat MLP DistanceHead.

    `score` is always lower-is-better. Scalar heads emit transformed target units;
    ordinal/distributional/quantile heads emit expected raw BFS steps.
    """

    def __init__(self, spec: HeadSpec) -> None:
        super().__init__()
        self.spec = spec
        latent_dim = int(spec.latent_dim)
        horizon_dim = 1 if spec.horizon_conditioned else 0
        self._quasimetric = spec.architecture == ArchitectureKind.QUASIMETRIC
        self._hierarchical = spec.architecture == ArchitectureKind.HIERARCHICAL

        if self._quasimetric:
            feature_dim = int(spec.hidden_dims[-1])
            self.source_features = _mlp(latent_dim, spec.hidden_dims[:-1], feature_dim)
            self.goal_features = None
            self.trunk = None
            trunk_dim = 1
        else:
            if spec.architecture == ArchitectureKind.HISTORICAL_CONCAT:
                input_dim = latent_dim * 2 + horizon_dim
            elif spec.architecture == ArchitectureKind.ASYMMETRIC:
                input_dim = latent_dim * 3 + horizon_dim
            elif spec.architecture in (
                ArchitectureKind.HORIZON_CONDITIONED,
                ArchitectureKind.HIERARCHICAL,
            ):
                input_dim = latent_dim * 3 + 1
            else:
                raise ValueError(f"unsupported head architecture: {spec.architecture}")
            trunk_dim = int(spec.hidden_dims[-1])
            self.trunk = _mlp(input_dim, spec.hidden_dims[:-1], trunk_dim)
            self.source_features = None
            self.goal_features = None

        output_dim = self._primary_output_dim(spec)
        if self._hierarchical:
            self.primary = nn.Identity()
            self.hierarchical_experts: nn.ModuleList | None = nn.ModuleList(
                nn.Linear(trunk_dim, output_dim) for _ in range(3)
            )
            self.hierarchical_gate: nn.Module | None = nn.Linear(trunk_dim, 3)
            self.register_buffer(
                "hierarchical_log_centers",
                torch.log1p(torch.tensor([1.0, 4.0, 12.0])),
                persistent=True,
            )
        else:
            self.primary = (
                nn.Identity() if self._quasimetric else nn.Linear(trunk_dim, output_dim)
            )
            self.hierarchical_experts = None
            self.hierarchical_gate = None
        self.reachability = (
            nn.Linear(trunk_dim, len(spec.reachability_budgets))
            if spec.output == OutputKind.MULTITASK
            else None
        )
        self.log_variance = nn.Linear(trunk_dim, 1) if spec.uncertainty else None
        self.domain_adapter = (
            nn.Embedding(2, trunk_dim) if spec.domain_adapter else None
        )
        self.register_buffer(
            "distribution_centers",
            self._distribution_centers(spec.distribution_edges),
            persistent=True,
        )
        self.register_buffer(
            "ordinal_centers",
            self._ordinal_centers(spec.ordinal_thresholds),
            persistent=True,
        )

    @staticmethod
    def _primary_output_dim(spec: HeadSpec) -> int:
        if spec.output in (OutputKind.SCALAR, OutputKind.MULTITASK):
            return 1
        if spec.output == OutputKind.ORDINAL:
            return len(spec.ordinal_thresholds)
        if spec.output == OutputKind.DISTRIBUTION:
            return len(spec.distribution_edges) + 1
        if spec.output == OutputKind.QUANTILE:
            return len(spec.quantiles)
        raise ValueError(f"unsupported output kind: {spec.output}")

    @staticmethod
    def _distribution_centers(edges: tuple[int, ...]) -> torch.Tensor:
        if not edges:
            raise ValueError("distribution edges cannot be empty")
        boundaries = (0, *edges)
        centers = [
            0.5 * (boundaries[index] + boundaries[index + 1])
            for index in range(len(edges))
        ]
        centers.append(
            float(edges[-1] + max(edges[-1] - edges[-2], 1))
            if len(edges) > 1
            else float(edges[-1] + 1)
        )
        return torch.tensor(centers, dtype=torch.float32)

    @staticmethod
    def _ordinal_centers(thresholds: tuple[int, ...]) -> torch.Tensor:
        if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError("ordinal thresholds must be strictly increasing")
        centers = [0.5 * float(thresholds[0])]
        centers.extend(
            0.5 * float(left + right)
            for left, right in zip(thresholds[:-1], thresholds[1:], strict=True)
        )
        tail_width = thresholds[-1] - thresholds[-2] if len(thresholds) > 1 else 1
        centers.append(float(thresholds[-1]) + 0.5 * float(tail_width))
        return torch.tensor(centers, dtype=torch.float32)

    def _features(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        *,
        horizon: torch.Tensor | None,
        predicted_domain: bool,
    ) -> torch.Tensor:
        if source.ndim != 2 or goal.shape != source.shape:
            raise ValueError("source and goal latents must share shape [batch, dim]")
        if source.shape[-1] != self.spec.latent_dim:
            raise ValueError("latent dimension differs from the head specification")
        if self._quasimetric:
            assert self.source_features is not None
            # Positive-part coordinate distance is directed, non-negative, zero on
            # the diagonal, and obeys triangle inequality by construction.
            source_features = self.source_features(source)
            goal_features = self.source_features(goal)
            features = F.relu(goal_features - source_features).mean(
                dim=-1, keepdim=True
            )
        else:
            assert self.trunk is not None
            pieces = [source, goal]
            if self.spec.architecture != ArchitectureKind.HISTORICAL_CONCAT:
                pieces.append(source - goal)
            if self.spec.horizon_conditioned:
                if horizon is None:
                    raise ValueError("horizon-conditioned head requires horizon")
                horizon = horizon.reshape(-1, 1).to(source)
                if horizon.shape[0] == 1 and source.shape[0] > 1:
                    horizon = horizon.expand(source.shape[0], -1)
                if horizon.shape[0] != source.shape[0]:
                    raise ValueError("horizon batch does not match latent batch")
                pieces.append(horizon / 128.0)
            features = self.trunk(torch.cat(pieces, dim=-1))
        if self.domain_adapter is not None:
            domain = torch.full(
                (source.shape[0],),
                int(predicted_domain),
                dtype=torch.long,
                device=source.device,
            )
            features = features + self.domain_adapter(domain)
        return features

    def forward(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        *,
        horizon: torch.Tensor | None = None,
        predicted_domain: bool = False,
    ) -> HeadOutput:
        features = self._features(
            source,
            goal,
            horizon=horizon,
            predicted_domain=predicted_domain,
        )
        if self._hierarchical:
            if horizon is None:
                raise ValueError("hierarchical head requires a horizon")
            assert self.hierarchical_experts is not None
            assert self.hierarchical_gate is not None
            expert_outputs = torch.stack(
                [expert(features) for expert in self.hierarchical_experts], dim=1
            )
            horizon_value = horizon.reshape(-1, 1).to(features).clamp_min(0.0)
            if horizon_value.shape[0] == 1 and features.shape[0] > 1:
                horizon_value = horizon_value.expand(features.shape[0], -1)
            if horizon_value.shape[0] != features.shape[0]:
                raise ValueError("hierarchical horizon batch does not match features")
            scale_prior = -(
                torch.log1p(horizon_value)
                - self.hierarchical_log_centers.to(features)[None, :]
            ).abs()
            mixture = F.softmax(
                self.hierarchical_gate(features) + scale_prior,
                dim=-1,
            )
            primary = (expert_outputs * mixture[:, :, None]).sum(dim=1)
        else:
            primary = self.primary(features)
        scalar: torch.Tensor | None = None
        ordinal: torch.Tensor | None = None
        distribution: torch.Tensor | None = None
        quantiles: torch.Tensor | None = None
        if self.spec.output in (OutputKind.SCALAR, OutputKind.MULTITASK):
            scalar = (
                primary.squeeze(-1)
                if self._quasimetric
                else F.softplus(primary.squeeze(-1))
            )
            score = scalar
        elif self.spec.output == OutputKind.ORDINAL:
            ordinal = primary
            survival = torch.cummin(torch.sigmoid(ordinal), dim=-1).values
            category_probabilities = torch.cat(
                [
                    1.0 - survival[:, :1],
                    survival[:, :-1] - survival[:, 1:],
                    survival[:, -1:],
                ],
                dim=-1,
            )
            score = (category_probabilities * self.ordinal_centers.to(primary)).sum(
                dim=-1
            )
        elif self.spec.output == OutputKind.DISTRIBUTION:
            distribution = primary
            centers = self.distribution_centers.to(primary)
            score = (F.softmax(distribution, dim=-1) * centers).sum(dim=-1)
        elif self.spec.output == OutputKind.QUANTILE:
            quantiles = torch.sort(F.softplus(primary), dim=-1).values
            median_index = min(
                range(len(self.spec.quantiles)),
                key=lambda index: abs(self.spec.quantiles[index] - 0.5),
            )
            score = quantiles[:, median_index]
        else:
            raise AssertionError("validated output kind became unsupported")
        reachability = self.reachability(features) if self.reachability else None
        log_variance = (
            self.log_variance(features).squeeze(-1).clamp(-8.0, 8.0)
            if self.log_variance
            else None
        )
        output = HeadOutput(
            score=score,
            scalar=scalar,
            ordinal_logits=ordinal,
            distribution_logits=distribution,
            quantiles=quantiles,
            reachability_logits=reachability,
            log_variance=log_variance,
        )
        output.validate(source.shape[0])
        return output


def build_distance_head(spec: HeadSpec) -> DistanceHeadModel:
    return DistanceHeadModel(spec)


__all__ = ["DistanceHeadModel", "HeadOutput", "build_distance_head"]
