"""Planning-aligned DistanceHead objectives in explicit raw/transformed units."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from distance_head_study.data import TrainingBatch
from distance_head_study.models import DistanceHeadModel, HeadOutput
from distance_head_study.schemas import (
    LabelMode,
    OutputKind,
    RegressionLoss,
    ResolvedMethod,
)
from distance_head_study.transforms import inverse_distance, transform_distance


@dataclass(frozen=True)
class TrajectoryBatch:
    predicted_terminal: torch.Tensor
    goals: torch.Tensor
    max_distance: torch.Tensor
    true_endpoint_distance: torch.Tensor
    horizon: int

    def validate(self) -> None:
        if self.predicted_terminal.ndim != 3:
            raise ValueError("trajectory terminals need [context,candidate,latent]")
        if self.goals.shape != (
            self.predicted_terminal.shape[0],
            self.predicted_terminal.shape[2],
        ):
            raise ValueError("trajectory goal shape mismatch")
        if self.true_endpoint_distance.shape != self.predicted_terminal.shape[:2]:
            raise ValueError("trajectory label shape mismatch")
        if self.max_distance.shape != (self.predicted_terminal.shape[0],):
            raise ValueError("trajectory max-distance shape mismatch")
        if not bool(torch.isfinite(self.max_distance).all()) or not bool(
            (self.max_distance > 0).all()
        ):
            raise ValueError("trajectory max distances must be finite and positive")


def score_in_raw_steps(
    output: HeadOutput,
    head: DistanceHeadModel,
    *,
    max_distance: torch.Tensor,
) -> torch.Tensor:
    if head.spec.output in (
        OutputKind.ORDINAL,
        OutputKind.DISTRIBUTION,
        OutputKind.QUANTILE,
    ):
        return output.score
    return inverse_distance(
        output.score,
        head.spec.target,
        max_distance=max_distance,
        global_scale=head.spec.global_distance_scale,
    )


def _regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kind: RegressionLoss,
) -> torch.Tensor:
    if kind == RegressionLoss.MSE:
        return F.mse_loss(prediction, target)
    if kind == RegressionLoss.MAE:
        return F.l1_loss(prediction, target)
    if kind == RegressionLoss.HUBER:
        return F.smooth_l1_loss(prediction, target)
    if kind == RegressionLoss.ASYMMETRIC:
        residual = prediction - target
        weight = torch.where(residual > 0, 2.0, 1.0)
        return (weight * F.smooth_l1_loss(prediction, target, reduction="none")).mean()
    raise ValueError(f"unsupported regression loss: {kind}")


def _absolute_loss(
    output: HeadOutput,
    head: DistanceHeadModel,
    raw_distance: torch.Tensor,
    max_distance: torch.Tensor,
) -> torch.Tensor:
    spec = head.spec
    if spec.output in (OutputKind.SCALAR, OutputKind.MULTITASK):
        assert output.scalar is not None
        target = transform_distance(
            raw_distance,
            spec.target,
            max_distance=max_distance,
            global_scale=spec.global_distance_scale,
        )
        return _regression_loss(output.scalar, target, spec.regression_loss)
    if spec.output == OutputKind.ORDINAL:
        assert output.ordinal_logits is not None
        thresholds = torch.tensor(
            spec.ordinal_thresholds,
            device=raw_distance.device,
            dtype=raw_distance.dtype,
        )
        labels = (raw_distance[:, None] > thresholds[None, :]).to(raw_distance)
        return F.binary_cross_entropy_with_logits(output.ordinal_logits, labels)
    if spec.output == OutputKind.DISTRIBUTION:
        assert output.distribution_logits is not None
        edges = torch.tensor(
            spec.distribution_edges,
            device=raw_distance.device,
            dtype=raw_distance.dtype,
        )
        labels = torch.bucketize(raw_distance, edges)
        return F.cross_entropy(output.distribution_logits, labels)
    if spec.output == OutputKind.QUANTILE:
        assert output.quantiles is not None
        quantiles = torch.tensor(
            spec.quantiles, device=raw_distance.device, dtype=raw_distance.dtype
        )
        residual = raw_distance[:, None] - output.quantiles
        return torch.maximum((quantiles - 1.0) * residual, quantiles * residual).mean()
    raise ValueError(f"unsupported output kind: {spec.output}")


def _head_call(
    head: DistanceHeadModel,
    source: torch.Tensor,
    goal: torch.Tensor,
    *,
    horizon: int = 12,
    predicted_domain: bool = False,
) -> HeadOutput:
    horizon_tensor = (
        torch.full((source.shape[0],), float(horizon), device=source.device)
        if head.spec.horizon_conditioned
        else None
    )
    return head(
        source,
        goal,
        horizon=horizon_tensor,
        predicted_domain=predicted_domain,
    )


def _horizon_grid(head: DistanceHeadModel) -> tuple[int, ...]:
    if not head.spec.horizon_conditioned:
        return (12,)
    if head.spec.output == OutputKind.MULTITASK:
        return head.spec.reachability_budgets
    # Scalar horizon-conditioned heads are queried locally at 1 and on legacy
    # rollout slots action_horizon + 1 for the preregistered trajectory grid.
    return (1, 2, 4, 6, 9, 12)


def reachability_logits_by_budget(
    head: DistanceHeadModel,
    source: torch.Tensor,
    goal: torch.Tensor,
    *,
    predicted_domain: bool = False,
) -> torch.Tensor:
    """Evaluate each reachability channel at its matching horizon input."""

    if head.spec.output != OutputKind.MULTITASK:
        raise ValueError("budgeted reachability requires a multitask head")
    columns = []
    for index, budget in enumerate(head.spec.reachability_budgets):
        output = _head_call(
            head,
            source,
            goal,
            horizon=int(budget),
            predicted_domain=predicted_domain,
        )
        if output.reachability_logits is None:
            raise ValueError("multitask head omitted reachability logits")
        columns.append(output.reachability_logits[:, index])
    return torch.stack(columns, dim=1)


def _masked_listwise(
    scores: torch.Tensor, valid: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if scores.shape != valid.shape or target.shape != valid.shape:
        raise ValueError("listwise score/mask/target shapes differ")
    if bool((target.sum(dim=1) <= 0).any()):
        raise ValueError("listwise target has no positive action")
    logits = -scores.masked_fill(~valid, float("inf"))
    log_probabilities = F.log_softmax(logits, dim=-1)
    distribution = target.float() / target.sum(dim=-1, keepdim=True)
    return (
        -(distribution * log_probabilities.masked_fill(~target, 0.0)).sum(dim=-1).mean()
    )


def compute_objective_terms(
    head: DistanceHeadModel,
    method: ResolvedMethod,
    batch: TrainingBatch,
    *,
    predicted_next: torch.Tensor | None = None,
    trajectory: TrajectoryBatch | None = None,
) -> dict[str, torch.Tensor]:
    """Return unweighted active terms; weighting is locked by the trainer."""

    if method.objectives is None:
        raise ValueError("trainable DistanceHead method has no objectives")
    weights = method.objectives
    current = _head_call(head, batch.source, batch.goal)
    current_raw = score_in_raw_steps(current, head, max_distance=batch.max_distance)
    batch_size, action_count, latent_dim = batch.next_latents.shape
    goal_actions = (
        batch.goal[:, None, :].expand(-1, action_count, -1).reshape(-1, latent_dim)
    )
    max_actions = batch.max_distance[:, None].expand(-1, action_count).reshape(-1)
    true_next_output = _head_call(
        head,
        batch.next_latents.reshape(-1, latent_dim),
        goal_actions,
        horizon=1,
    )
    true_next_raw = score_in_raw_steps(
        true_next_output, head, max_distance=max_actions
    ).reshape(batch_size, action_count)
    terms: dict[str, torch.Tensor] = {}

    if weights.absolute > 0:
        horizons = _horizon_grid(head)
        if horizons == (12,):
            terms["absolute"] = _absolute_loss(
                current, head, batch.raw_distance, batch.max_distance
            )
        else:
            repeated_source = batch.source.repeat_interleave(len(horizons), dim=0)
            repeated_goal = batch.goal.repeat_interleave(len(horizons), dim=0)
            horizon_tensor = torch.tensor(
                horizons, dtype=batch.source.dtype, device=batch.source.device
            ).repeat(batch_size)
            augmented = head(
                repeated_source,
                repeated_goal,
                horizon=horizon_tensor,
                predicted_domain=False,
            )
            terms["absolute"] = _absolute_loss(
                augmented,
                head,
                batch.raw_distance.repeat_interleave(len(horizons)),
                batch.max_distance.repeat_interleave(len(horizons)),
            )
    if weights.anchor > 0:
        anchor_values = [
            _head_call(head, batch.goal, batch.goal, horizon=horizon).score
            for horizon in _horizon_grid(head)
        ]
        terms["anchor"] = torch.stack(anchor_values).abs().mean()
    if weights.pairwise > 0:
        best = (
            true_next_raw.masked_fill(~batch.optimal_actions, float("inf"))
            .min(dim=1)
            .values
        )
        worse_mask = batch.valid_actions & ~batch.optimal_actions
        worse = true_next_raw.masked_fill(~worse_mask, float("inf")).min(dim=1).values
        eligible = torch.isfinite(worse)
        terms["pairwise"] = (
            F.relu(1.0 + best[eligible] - worse[eligible]).mean()
            if bool(eligible.any())
            else current_raw.new_zeros(())
        )
    if weights.listwise > 0:
        terms["listwise"] = _masked_listwise(
            true_next_raw, batch.valid_actions, batch.optimal_actions
        )
    if weights.all_action > 0:
        mask = batch.valid_actions
        terms["all_action"] = F.smooth_l1_loss(
            true_next_raw[mask], batch.next_distances[mask]
        )
    if weights.delta > 0:
        mask = batch.valid_actions
        predicted_delta = true_next_raw - current_raw[:, None]
        true_delta = batch.next_distances - batch.raw_distance[:, None]
        terms["delta"] = F.smooth_l1_loss(predicted_delta[mask], true_delta[mask])
    if weights.bellman > 0 or weights.eikonal > 0:
        minimum = (
            true_next_raw.masked_fill(~batch.valid_actions, float("inf"))
            .min(dim=1)
            .values
        )
        bellman = F.smooth_l1_loss(current_raw, 1.0 + minimum)
        if weights.bellman > 0:
            terms["bellman"] = bellman
        if weights.eikonal > 0:
            local_change = (true_next_raw - current_raw[:, None]).abs()
            lipschitz = F.relu(local_change[batch.valid_actions] - 1.0).mean()
            terms["eikonal"] = bellman + lipschitz
    if weights.multistep > 0:
        horizons = (1, 3, 5, 8, 12)
        repeated_goal = batch.goal[:, None, :].expand(-1, len(horizons), -1)
        path_latents = batch.path_latents[:, horizons]
        path_output = _head_call(
            head,
            path_latents.reshape(-1, latent_dim),
            repeated_goal.reshape(-1, latent_dim),
        )
        path_max = batch.max_distance[:, None].expand(-1, len(horizons)).reshape(-1)
        path_raw = score_in_raw_steps(path_output, head, max_distance=path_max).reshape(
            batch_size, len(horizons)
        )
        true_steps = batch.raw_distance[:, None] - batch.path_distances[:, horizons]
        terms["multistep"] = F.smooth_l1_loss(
            current_raw[:, None].expand_as(path_raw), true_steps + path_raw
        )
    if weights.triangle > 0:
        waypoint_output = _head_call(head, batch.triangle_latent, batch.goal)
        waypoint_raw = score_in_raw_steps(
            waypoint_output,
            head,
            max_distance=batch.max_distance,
        )
        terms["triangle"] = F.relu(
            current_raw - batch.triangle_source_distance - waypoint_raw
        ).mean()
    if weights.successor_contrastive > 0:
        successor_horizons = (1, 3, 5, 8, 12)
        successor_predictions = []
        for horizon in successor_horizons:
            output = _head_call(
                head,
                batch.source,
                batch.path_latents[:, horizon],
                horizon=horizon,
            )
            successor_predictions.append(
                score_in_raw_steps(output, head, max_distance=batch.max_distance)
            )
        successor_raw = torch.stack(successor_predictions, dim=1)
        successor_truth = (
            batch.raw_distance[:, None] - batch.path_distances[:, successor_horizons]
        )
        contrastive_terms = []
        for left in range(len(successor_horizons)):
            for right in range(left + 1, len(successor_horizons)):
                ordered = successor_truth[:, right] > successor_truth[:, left]
                if bool(ordered.any()):
                    contrastive_terms.append(
                        F.softplus(
                            0.5
                            + successor_raw[ordered, left]
                            - successor_raw[ordered, right]
                        ).mean()
                    )
        waypoint_output = _head_call(
            head,
            batch.source,
            batch.triangle_latent,
        )
        waypoint_raw = score_in_raw_steps(
            waypoint_output,
            head,
            max_distance=batch.max_distance,
        )
        for index in range(len(successor_horizons)):
            truth_delta = batch.triangle_source_distance - successor_truth[:, index]
            comparable = truth_delta != 0
            if bool(comparable.any()):
                direction = truth_delta[comparable].sign()
                predicted_delta = (
                    waypoint_raw[comparable] - successor_raw[comparable, index]
                )
                contrastive_terms.append(
                    F.softplus(0.5 - direction * predicted_delta).mean()
                )
        terms["successor_contrastive"] = (
            torch.stack(contrastive_terms).mean()
            if contrastive_terms
            else current_raw.sum() * 0.0
        )

    predicted_raw: torch.Tensor | None = None
    if weights.predicted_listwise > 0 or weights.predicted_consistency > 0:
        if predicted_next is None or predicted_next.shape != batch.next_latents.shape:
            raise ValueError(
                "predicted-latent objectives require the full model action vocabulary"
            )
        predicted_output = _head_call(
            head,
            predicted_next.reshape(-1, latent_dim),
            goal_actions,
            horizon=1,
            predicted_domain=True,
        )
        predicted_raw = score_in_raw_steps(
            predicted_output, head, max_distance=max_actions
        ).reshape(batch_size, action_count)
        if weights.predicted_listwise > 0:
            terms["predicted_listwise"] = _masked_listwise(
                predicted_raw, batch.valid_actions, batch.optimal_actions
            )
        if weights.predicted_consistency > 0:
            terms["predicted_consistency"] = F.smooth_l1_loss(
                predicted_raw[batch.valid_actions],
                true_next_raw.detach()[batch.valid_actions],
            )
    if weights.trajectory_listwise > 0:
        if trajectory is None:
            raise ValueError("trajectory objective requires a trajectory batch")
        trajectory.validate()
        contexts, candidates, dimension = trajectory.predicted_terminal.shape
        goals = trajectory.goals[:, None, :].expand(-1, candidates, -1)
        trajectory_output = _head_call(
            head,
            trajectory.predicted_terminal.reshape(-1, dimension),
            goals.reshape(-1, dimension),
            horizon=trajectory.horizon,
            predicted_domain=True,
        )
        maximum = trajectory.max_distance[:, None].expand(-1, candidates).reshape(-1)
        scores = score_in_raw_steps(
            trajectory_output, head, max_distance=maximum
        ).reshape(contexts, candidates)
        labels = trajectory.true_endpoint_distance
        if method.label_mode == LabelMode.SHUFFLED:
            labels = torch.roll(labels, shifts=1, dims=1)
        elif method.label_mode == LabelMode.RANDOM:
            labels = torch.flip(labels, dims=(1,))
        target = labels == labels.min(dim=1, keepdim=True).values
        terms["trajectory_listwise"] = _masked_listwise(
            scores, torch.ones_like(target), target
        )
    if weights.reachability > 0:
        logits = reachability_logits_by_budget(head, batch.source, batch.goal)
        budgets = torch.tensor(
            head.spec.reachability_budgets,
            dtype=batch.raw_distance.dtype,
            device=batch.raw_distance.device,
        )
        labels = (batch.raw_distance[:, None] <= budgets[None, :]).float()
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        probabilities = torch.sigmoid(logits)
        monotonic = F.relu(probabilities[:, :-1] - probabilities[:, 1:]).mean()
        terms["reachability"] = bce + monotonic
    if weights.uncertainty > 0:
        if current.log_variance is None:
            raise ValueError("uncertainty objective requires uncertainty output")
        residual = current_raw - batch.raw_distance
        terms["uncertainty"] = (
            0.5
            * (
                torch.exp(-current.log_variance) * residual.square()
                + current.log_variance
            )
        ).mean()
    return terms


def gradient_calibrated_weights(
    terms: dict[str, torch.Tensor],
    base_weights: dict[str, float],
    parameters: list[torch.nn.Parameter],
    *,
    target_ratio: float,
    clip: tuple[float, float],
) -> dict[str, float]:
    """Scale auxiliary losses to a fixed fraction of absolute-loss gradient norm."""

    if "absolute" not in terms or base_weights.get("absolute", 0.0) <= 0:
        return {name: float(base_weights[name]) for name in terms}

    def norm(loss: torch.Tensor) -> float:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared = sum(
            float(gradient.detach().float().square().sum())
            for gradient in gradients
            if gradient is not None
        )
        return squared**0.5

    reference = max(norm(terms["absolute"]), 1e-12)
    calibrated: dict[str, float] = {}
    for name, term in terms.items():
        base = float(base_weights[name])
        if name == "absolute" or base <= 0:
            calibrated[name] = base
            continue
        multiplier = target_ratio * reference / max(norm(term), 1e-12)
        calibrated[name] = base * min(max(multiplier, clip[0]), clip[1])
    return calibrated


def weighted_total(
    terms: dict[str, torch.Tensor], calibrated_weights: dict[str, float]
) -> torch.Tensor:
    if set(terms) != set(calibrated_weights):
        raise ValueError("loss terms and calibrated weights differ")
    total = sum(calibrated_weights[name] * value for name, value in terms.items())
    if not torch.isfinite(total):
        raise FloatingPointError("weighted DistanceHead loss is non-finite")
    return total


__all__ = [
    "TrajectoryBatch",
    "compute_objective_terms",
    "gradient_calibrated_weights",
    "reachability_logits_by_budget",
    "score_in_raw_steps",
    "weighted_total",
]
