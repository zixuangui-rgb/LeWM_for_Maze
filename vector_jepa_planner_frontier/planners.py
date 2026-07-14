"""Budgeted search algorithms sharing one pooled-vector planning contract."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hdwm.planning import cem_plan
from vector_jepa_planner_frontier import ACTION_IDS, INVERSE_ACTION
from vector_jepa_planner_frontier.common import ComputeLedger
from vector_jepa_planner_frontier.heads import (
    ActionConsistencyVerifier,
    CounterexampleRanker,
    DistributionalReachability,
    StateJoinHead,
    VectorDTSHead,
    expected_distance_from_logits,
)
from vector_jepa_planner_frontier.proposals import CandidateProposal, UniformProposal
from vector_jepa_planner_frontier.schemas import (
    MemoryConfig,
    PlannerConfig,
    PlannerKind,
    ScorerConfig,
)
from vector_jepa_planner_frontier.world_model import (
    RolloutBatch,
    VectorContext,
    VectorWorldModel,
)


@dataclass(frozen=True)
class ScoreBatch:
    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def validate(self, batch_size: int) -> None:
        if self.total.shape != (batch_size,):
            raise ValueError("candidate score must have one value per sequence")
        if not torch.isfinite(self.total).all():
            raise FloatingPointError("candidate scorer produced non-finite costs")
        for name, value in self.components.items():
            if value.shape != (batch_size,) or not torch.isfinite(value).all():
                raise FloatingPointError(f"invalid scorer component: {name}")


class CompositeScorer:
    """Latent goal cost plus optional verifier/reachability/ranker terms."""

    def __init__(
        self,
        config: ScorerConfig,
        *,
        verifier: ActionConsistencyVerifier | None = None,
        reachability: DistributionalReachability | None = None,
        ranker: CounterexampleRanker | None = None,
        shuffle_candidate_association: bool = False,
    ) -> None:
        self.config = config
        self.verifier = verifier
        self.reachability = reachability
        self.ranker = ranker
        self.shuffle_candidate_association = bool(shuffle_candidate_association)
        if config.verifier_weight > 0.0 and verifier is None:
            raise ValueError("verifier score is enabled but its head is missing")
        if config.reachability_weight > 0.0 and reachability is None:
            raise ValueError("reachability score is enabled but its head is missing")
        if config.counterexample_ranker_weight > 0.0 and ranker is None:
            raise ValueError("counterexample score is enabled but its head is missing")

    def __call__(
        self,
        context: VectorContext,
        rollout: RolloutBatch,
        *,
        remaining_budget: int,
        ledger: ComputeLedger,
    ) -> ScoreBatch:
        batch_size, horizon = rollout.actions.shape
        if self.shuffle_candidate_association and batch_size > 1:
            rollout = RolloutBatch(
                states=torch.roll(rollout.states, shifts=1, dims=0),
                terminal=torch.roll(rollout.terminal, shifts=1, dims=0),
                actions=rollout.actions,
                semantics=rollout.semantics,
            )
        goal = context.goal.squeeze(1).expand(batch_size, -1)
        goal_l2 = F.mse_loss(rollout.terminal, goal, reduction="none").sum(dim=-1)
        total = self.config.goal_l2_weight * goal_l2
        components = {"goal_l2": goal_l2}

        if self.verifier is not None and self.config.verifier_weight > 0.0:
            action_cost = self._action_cost(context, rollout, ledger)
            scale = self._population_scale(goal_l2, action_cost)
            total = total + self.config.verifier_weight * scale * action_cost
            components["action_nll"] = action_cost
            components["action_scale"] = torch.full_like(action_cost, scale)

        if self.reachability is not None and self.config.reachability_weight > 0.0:
            probability = self.reachability.probability_for_budget(
                rollout.terminal,
                goal,
                remaining_budget=remaining_budget,
            )
            reachability_cost = -torch.log(probability.clamp_min(self.config.eps))
            ledger.reachability_forward_calls += 1
            total = total + self.config.reachability_weight * reachability_cost
            components["reachability_nll"] = reachability_cost

        if self.ranker is not None and self.config.counterexample_ranker_weight > 0.0:
            source = context.embeddings[:, -1].expand(batch_size, -1)
            rank_score = self.ranker(source, rollout.terminal, rollout.actions)
            ledger.ranker_forward_calls += 1
            rank_cost = -rank_score
            total = total + self.config.counterexample_ranker_weight * rank_cost
            components["counterexample_rank_cost"] = rank_cost

        result = ScoreBatch(total=total, components=components)
        result.validate(batch_size)
        return result

    def _action_cost(
        self,
        context: VectorContext,
        rollout: RolloutBatch,
        ledger: ComputeLedger,
    ) -> torch.Tensor:
        assert self.verifier is not None
        batch_size, horizon, latent_dim = rollout.states.shape
        if rollout.semantics.value == "legacy_warmup_v1":
            if horizon == 1:
                return rollout.states.new_zeros((batch_size,))
            sources = rollout.states[:, :-1]
            targets = rollout.states[:, 1:]
            actions = rollout.actions[:, :-1]
        else:
            root = context.embeddings[:, -1:].expand(batch_size, 1, latent_dim)
            sources = torch.cat([root, rollout.states[:, :-1]], dim=1)
            targets = rollout.states
            actions = rollout.actions
        edge_costs = self.verifier.action_nll(
            sources.reshape(-1, latent_dim),
            targets.reshape(-1, latent_dim),
            actions.reshape(-1),
        ).reshape(batch_size, -1)
        ledger.verifier_forward_calls += 1
        return edge_costs.mean(dim=1)

    def _population_scale(
        self, primary: torch.Tensor, auxiliary: torch.Tensor
    ) -> float:
        primary_std = float(primary.detach().std(unbiased=False))
        auxiliary_std = float(auxiliary.detach().std(unbiased=False))
        if primary_std < self.config.eps or auxiliary_std < self.config.eps:
            return 1.0
        ratio = primary_std / (auxiliary_std + self.config.eps)
        return float(
            np.clip(
                ratio,
                self.config.verifier_scale_min,
                self.config.verifier_scale_max,
            )
        )


@dataclass(frozen=True)
class CandidateBatch:
    """One planner-evaluated candidate population retained for offline diagnosis."""

    sequences: np.ndarray
    predicted_costs: np.ndarray | None
    source: str = "predictor_scored"

    def validate(self) -> None:
        if self.sequences.ndim != 2 or self.sequences.shape[0] == 0:
            raise ValueError("candidate batch must have shape [count, horizon]")
        if self.predicted_costs is not None and self.predicted_costs.shape != (
            self.sequences.shape[0],
        ):
            raise ValueError("candidate costs must have one value per sequence")
        if not np.isin(self.sequences, np.asarray(ACTION_IDS)).all():
            raise ValueError("candidate batch contains an out-of-protocol action")
        if (
            self.predicted_costs is not None
            and not np.isfinite(self.predicted_costs).all()
        ):
            raise FloatingPointError("candidate batch contains a non-finite cost")


@dataclass(frozen=True)
class PlannerResult:
    sequence: np.ndarray
    cost: float
    ledger: ComputeLedger
    diagnostics: dict[str, Any]
    candidate_batches: tuple[CandidateBatch, ...] = ()

    def validate(self, horizon: int) -> None:
        if self.sequence.shape != (horizon,):
            raise ValueError("planner result has the wrong horizon")
        if not np.isin(self.sequence, np.asarray(ACTION_IDS)).all():
            raise ValueError("planner result contains an out-of-protocol action")
        if not math.isfinite(self.cost):
            raise FloatingPointError("planner result cost is non-finite")
        if self.ledger.plan_transitions < 0:
            raise ValueError("planner result has negative imagined-transition compute")
        for batch in self.candidate_batches:
            batch.validate()


class BasePlanner:
    def __init__(
        self,
        world_model: VectorWorldModel,
        config: PlannerConfig,
        scorer: CompositeScorer,
        *,
        proposal: CandidateProposal | None = None,
    ) -> None:
        self.world_model = world_model
        self.config = config
        self.scorer = scorer
        self.proposal = proposal or UniformProposal()
        self._candidate_batches: list[CandidateBatch] = []

    def reset(self) -> None:
        """Clear episode-local planner state."""

    def observe_real_state(self, latent: torch.Tensor) -> None:
        """Optionally register one actually visited pooled vector."""

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        raise NotImplementedError

    def _begin_candidate_capture(self) -> None:
        self._candidate_batches.clear()

    def _capture_candidates(
        self,
        sequences: np.ndarray,
        predicted_costs: np.ndarray | None,
        *,
        source: str = "predictor_scored",
    ) -> None:
        batch = CandidateBatch(
            sequences=np.asarray(sequences, dtype=np.int64).copy(),
            predicted_costs=(
                None
                if predicted_costs is None
                else np.asarray(predicted_costs, dtype=np.float64).copy()
            ),
            source=source,
        )
        batch.validate()
        self._candidate_batches.append(batch)

    def _evaluate(
        self,
        context: VectorContext,
        sequences: np.ndarray,
        ledger: ComputeLedger,
    ) -> tuple[np.ndarray, RolloutBatch, ScoreBatch]:
        rollout = self.world_model.rollout(
            context,
            sequences,
            semantics=self.config.rollout_semantics,
            ledger=ledger,
        )
        remaining_budget = max(1, context.remaining_steps - int(sequences.shape[1]))
        scores = self.scorer(
            context,
            rollout,
            remaining_budget=remaining_budget,
            ledger=ledger,
        )
        costs = scores.total.detach().cpu().numpy()
        self._capture_candidates(sequences, costs)
        return costs, rollout, scores

    def diagnostic_costs(
        self, context: VectorContext, sequences: np.ndarray
    ) -> tuple[np.ndarray, ComputeLedger]:
        """Post-decision rescoring; its compute is excluded from formal planning."""

        ledger = ComputeLedger()
        rollout = self.world_model.rollout(
            context,
            sequences,
            semantics=self.config.rollout_semantics,
            ledger=ledger,
        )
        remaining_budget = max(1, context.remaining_steps - int(sequences.shape[1]))
        scores = self.scorer(
            context,
            rollout,
            remaining_budget=remaining_budget,
            ledger=ledger,
        )
        return scores.total.detach().cpu().numpy(), ledger

    def _pad(self, sequence: tuple[int, ...] | np.ndarray) -> np.ndarray:
        values = np.asarray(sequence, dtype=np.int64).reshape(-1)
        if values.size == 0:
            values = np.asarray([ACTION_IDS[0]], dtype=np.int64)
        if values.size < self.config.horizon:
            values = np.concatenate(
                [
                    values,
                    np.full(
                        self.config.horizon - values.size,
                        int(values[-1]),
                        dtype=np.int64,
                    ),
                ]
            )
        return values[: self.config.horizon]


class LegacyCEMPlanner(BasePlanner):
    """Exact wrapper around the historical hdwm.planning.cem_plan baseline."""

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        context.validate(self.config.history_size)

        def score_fn(terminal: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(terminal, goal, reduction="none").sum(dim=-1)

        sequence, cost, history = cem_plan(
            self.world_model.model,
            context.embeddings,
            context.actions,
            context.goal,
            horizon=self.config.horizon,
            history_size=self.config.history_size,
            num_candidates=self.config.num_candidates,
            num_elites=self.config.num_elites,
            cem_iters=self.config.cem_iters,
            momentum=self.config.momentum,
            num_actions=5,
            device=self.world_model.device,
            seed=int(seed),
            score_fn=score_fn,
            allowed_actions=np.asarray(ACTION_IDS, dtype=np.int64),
        )
        rng = np.random.default_rng(seed)
        candidates = np.stack(
            [
                rng.choice(
                    np.asarray(ACTION_IDS, dtype=np.int64),
                    size=self.config.num_candidates,
                    p=np.full(len(ACTION_IDS), 1.0 / len(ACTION_IDS)),
                )
                for _ in range(self.config.horizon)
            ],
            axis=1,
        )
        self._candidate_batches.append(
            CandidateBatch(
                sequences=candidates,
                predicted_costs=None,
                source="legacy_cem_candidates_scores_unavailable",
            )
        )
        ledger = ComputeLedger(
            plan_transitions=(
                self.config.num_candidates * self.config.horizon * self.config.cem_iters
            ),
            planner_forward_calls=self.config.horizon * self.config.cem_iters,
            planner_max_batch=self.config.num_candidates,
            candidate_sequences=self.config.num_candidates * self.config.cem_iters,
        )
        result = PlannerResult(
            sequence=sequence,
            cost=float(cost),
            ledger=ledger,
            diagnostics={
                "cost_history": [float(value) for value in history],
                "implementation": "hdwm.planning.cem_plan",
            },
            candidate_batches=tuple(self._candidate_batches),
        )
        result.validate(self.config.horizon)
        return result


class CategoricalCEMPlanner(BasePlanner):
    """Instrumented categorical CEM for non-baseline compute-frontier cells."""

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        context.validate(self.config.history_size)
        rng = np.random.default_rng(seed)
        ledger = ComputeLedger()
        actions = np.asarray(ACTION_IDS, dtype=np.int64)
        probabilities = np.full(
            (self.config.horizon, len(actions)),
            1.0 / len(actions),
            dtype=np.float64,
        )
        best_sequence: np.ndarray | None = None
        best_cost = float("inf")
        history: list[float] = []
        for _ in range(self.config.cem_iters):
            candidates = np.stack(
                [
                    rng.choice(
                        actions,
                        size=self.config.num_candidates,
                        p=probabilities[step],
                    )
                    for step in range(self.config.horizon)
                ],
                axis=1,
            ).astype(np.int64)
            costs, _, _ = self._evaluate(context, candidates, ledger)
            elite_indices = np.argsort(costs)[: self.config.num_elites]
            elites = candidates[elite_indices]
            if float(costs[elite_indices[0]]) < best_cost:
                best_cost = float(costs[elite_indices[0]])
                best_sequence = elites[0].copy()
            history.append(best_cost)
            frequencies = np.zeros_like(probabilities)
            for step in range(self.config.horizon):
                for slot, action in enumerate(actions):
                    frequencies[step, slot] = np.mean(elites[:, step] == action)
            probabilities = (
                self.config.momentum * probabilities
                + (1.0 - self.config.momentum) * frequencies
            )
        if best_sequence is None:
            raise RuntimeError("categorical CEM evaluated no candidate")
        result = PlannerResult(
            sequence=best_sequence,
            cost=best_cost,
            ledger=ledger,
            diagnostics={
                "cost_history": history,
                "implementation": (
                    "vector_jepa_planner_frontier.CategoricalCEMPlanner"
                ),
                "transition_limit": self.config.budget.transition_limit,
            },
            candidate_batches=tuple(self._candidate_batches),
        )
        result.validate(self.config.horizon)
        return result


class ProposalOnlyPlanner(BasePlanner):
    """Negative control: execute proposal consensus without predictor/search."""

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        rng = np.random.default_rng(seed)
        candidates = self.proposal.sample(
            context.embeddings[:, -1],
            context.goal.squeeze(1),
            count=self.config.num_candidates,
            horizon=self.config.horizon,
            rng=rng,
        )
        ledger = ComputeLedger(
            candidate_sequences=self.config.num_candidates,
            proposal_forward_calls=1,
        )
        first_actions, counts = np.unique(candidates[:, 0], return_counts=True)
        selected_action = int(first_actions[np.argmax(counts)])
        selected_index = int(np.flatnonzero(candidates[:, 0] == selected_action)[0])
        self._capture_candidates(
            candidates,
            np.zeros(len(candidates), dtype=np.float64),
            source="proposal_only_unscored_control",
        )
        result = PlannerResult(
            sequence=candidates[selected_index],
            cost=0.0,
            ledger=ledger,
            diagnostics={
                "proposal_only": True,
                "first_action_counts": {
                    str(action): int(count)
                    for action, count in zip(first_actions, counts, strict=True)
                },
            },
            candidate_batches=tuple(self._candidate_batches),
        )
        result.validate(self.config.horizon)
        return result


class DirectDTSPlanner(BasePlanner):
    """Search-disabled direct-head control for Vector-DTS."""

    def __init__(
        self,
        *args: Any,
        dts_head: VectorDTSHead,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dts_head = dts_head

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        del seed
        ledger = ComputeLedger(dts_forward_calls=1)
        with torch.no_grad():
            output = self.dts_head(context.embeddings[:, -1], context.goal.squeeze(1))
        action = int(output["policy_logits"].argmax(dim=-1)[0]) + 1
        value = float(expected_distance_from_logits(output["value_logits"])[0])
        sequence = np.full(self.config.horizon, action, dtype=np.int64)
        self._capture_candidates(
            sequence[None, :],
            np.asarray([value], dtype=np.float64),
            source="direct_dts_control",
        )
        result = PlannerResult(
            sequence=sequence,
            cost=value,
            ledger=ledger,
            diagnostics={"search_disabled_direct_head": True},
            candidate_batches=tuple(self._candidate_batches),
        )
        result.validate(self.config.horizon)
        return result


class ICEMPlanner(BasePlanner):
    """Categorical iCEM with receding warm start and elite reuse."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._previous_elites: np.ndarray | None = None
        self._previous_best: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_elites = None
        self._previous_best = None

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        rng = np.random.default_rng(seed)
        ledger = ComputeLedger()
        action_values = np.asarray(ACTION_IDS, dtype=np.int64)
        probabilities = np.full(
            (self.config.horizon, len(ACTION_IDS)),
            1.0 / len(ACTION_IDS),
            dtype=np.float64,
        )
        if self._previous_best is not None:
            shifted = np.concatenate(
                [self._previous_best[1:], self._previous_best[-1:]]
            )
            probabilities.fill(0.1 / (len(ACTION_IDS) - 1))
            for step, action in enumerate(shifted.tolist()):
                probabilities[step, action - 1] = 0.9

        best_sequence: np.ndarray | None = None
        best_cost = float("inf")
        history: list[float] = []
        final_elites: np.ndarray | None = None
        transition_limit = self.config.budget.transition_limit
        for iteration in range(self.config.cem_iters):
            remaining = transition_limit - ledger.plan_transitions
            fresh_count = min(
                self.config.num_candidates, remaining // self.config.horizon
            )
            if fresh_count <= 0:
                break
            candidates = np.stack(
                [
                    rng.choice(action_values, size=fresh_count, p=probabilities[step])
                    for step in range(self.config.horizon)
                ],
                axis=1,
            )
            if iteration == 0 and self._previous_elites is not None:
                reuse_target = int(
                    round(fresh_count * self.config.elite_reuse_fraction)
                )
                reusable = self._previous_elites[:reuse_target]
                if reusable.size:
                    shifted = np.concatenate(
                        [reusable[:, 1:], reusable[:, -1:]], axis=1
                    )
                    candidates[: len(shifted)] = shifted
            proposal_count = min(fresh_count // 4, fresh_count)
            if proposal_count:
                candidates[:proposal_count] = self.proposal.sample(
                    context.embeddings[:, -1],
                    context.goal.squeeze(1),
                    count=proposal_count,
                    horizon=self.config.horizon,
                    rng=rng,
                )
                ledger.proposal_forward_calls += 1
            costs, _, _ = self._evaluate(context, candidates, ledger)
            elite_count = min(self.config.num_elites, fresh_count)
            elite_indices = np.argsort(costs, kind="stable")[:elite_count]
            elites = candidates[elite_indices]
            final_elites = elites
            if float(costs[elite_indices[0]]) < best_cost:
                best_cost = float(costs[elite_indices[0]])
                best_sequence = elites[0].copy()
            history.append(best_cost)
            frequencies = np.zeros_like(probabilities)
            for step in range(self.config.horizon):
                for slot, action in enumerate(action_values):
                    frequencies[step, slot] = np.mean(elites[:, step] == action)
            probabilities = (
                self.config.momentum * probabilities
                + (1.0 - self.config.momentum) * frequencies
            )
        if best_sequence is None or final_elites is None:
            raise RuntimeError("iCEM exhausted its budget before evaluating candidates")
        self._previous_best = best_sequence.copy()
        self._previous_elites = final_elites.copy()
        result = PlannerResult(
            sequence=best_sequence,
            cost=best_cost,
            ledger=ledger,
            diagnostics={
                "cost_history": history,
                "iterations_completed": len(history),
                "transition_limit": transition_limit,
            },
            candidate_batches=tuple(self._candidate_batches),
        )
        result.validate(self.config.horizon)
        return result


class BeamPlanner(BasePlanner):
    """Diverse beam search over action prefixes under an exact transition cap."""

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        del seed
        ledger = ComputeLedger()
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        best: tuple[tuple[int, ...], float] | None = None
        depths_completed = 0
        for depth in range(1, self.config.horizon + 1):
            expanded = [
                prefix + (action,) for prefix, _ in beams for action in ACTION_IDS
            ]
            remaining = self.config.budget.transition_limit - ledger.plan_transitions
            maximum = remaining // depth
            complete = min(
                len(expanded), (maximum // len(ACTION_IDS)) * len(ACTION_IDS)
            )
            if complete <= 0:
                break
            expanded = expanded[:complete]
            sequences = np.asarray(expanded, dtype=np.int64)
            costs, _, _ = self._evaluate(context, sequences, ledger)
            ranked = sorted(
                zip(expanded, costs.tolist(), strict=True),
                key=lambda item: (item[1], item[0]),
            )
            selected: list[tuple[tuple[int, ...], float]] = []
            for prefix, raw_cost in ranked:
                diversity = sum(
                    sum(
                        left == right
                        for left, right in zip(prefix, existing, strict=True)
                    )
                    / depth
                    for existing, _ in selected
                )
                adjusted = float(raw_cost) + self.config.diversity_penalty * diversity
                selected.append((prefix, adjusted))
                if len(selected) >= self.config.beam_width:
                    break
            beams = selected
            best = min(beams, key=lambda item: (item[1], item[0]))
            depths_completed = depth
            ledger.node_expansions += len(expanded)
        if best is None:
            raise RuntimeError("beam search could not evaluate one action")
        result = PlannerResult(
            sequence=self._pad(best[0]),
            cost=float(best[1]),
            ledger=ledger,
            diagnostics={
                "depths_completed": depths_completed,
                "beam_width": self.config.beam_width,
                "transition_limit": self.config.budget.transition_limit,
            },
            candidate_batches=tuple(self._candidate_batches),
        )
        result.validate(self.config.horizon)
        return result


class DTSBreadthPlanner(BeamPlanner):
    """Fixed breadth expansion with the learned DTS value retained."""

    def __init__(
        self,
        *args: Any,
        dts_head: VectorDTSHead,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dts_head = dts_head

    def _evaluate(
        self,
        context: VectorContext,
        sequences: np.ndarray,
        ledger: ComputeLedger,
    ) -> tuple[np.ndarray, RolloutBatch, ScoreBatch]:
        costs, rollout, scores = super()._evaluate(context, sequences, ledger)
        goal = context.goal.squeeze(1).expand(rollout.terminal.shape[0], -1)
        with torch.no_grad():
            output = self.dts_head(rollout.terminal, goal)
            value = expected_distance_from_logits(output["value_logits"])
        ledger.dts_forward_calls += 1
        combined = scores.total + value
        combined_scores = ScoreBatch(
            total=combined,
            components={**scores.components, "dts_value": value},
        )
        combined_scores.validate(rollout.terminal.shape[0])
        self._candidate_batches[-1] = CandidateBatch(
            sequences=np.asarray(sequences, dtype=np.int64).copy(),
            predicted_costs=combined.detach().cpu().numpy().astype(np.float64),
            source="predictor_scored_with_dts_value",
        )
        return combined.detach().cpu().numpy(), rollout, combined_scores


class TranspositionMemory:
    """Precision-gated episode/search memory; no coordinates are stored."""

    def __init__(
        self,
        config: MemoryConfig,
        join_head: StateJoinHead | None,
        *,
        validated_precision: float | None,
    ) -> None:
        self.config = config
        self.join_head = join_head
        self.validated_precision = validated_precision
        self.entries: list[tuple[torch.Tensor, float]] = []
        self.real_entries: list[torch.Tensor] = []
        if config.enabled and join_head is None:
            raise ValueError("memory is enabled but no join head was loaded")
        if config.hard_pruning and (
            validated_precision is None
            or validated_precision < config.required_validation_precision
        ):
            raise ValueError("hard pruning requires a preregistered precision gate")

    def dominated(
        self,
        latent: torch.Tensor,
        path_cost: float,
        ledger: ComputeLedger,
    ) -> bool:
        if not self.config.enabled or not self.entries:
            return False
        assert self.join_head is not None
        current = latent.reshape(1, -1).expand(len(self.entries), -1)
        stored = torch.stack([item[0] for item in self.entries]).to(current.device)
        with torch.no_grad():
            probabilities = self.join_head.probability(current, stored)
        ledger.join_forward_calls += 1
        for probability, (_, old_cost) in zip(
            probabilities.tolist(), self.entries, strict=True
        ):
            same = probability >= self.config.join_threshold
            dominated = path_cost >= old_cost + self.config.dominance_delta
            if same and dominated and self.config.hard_pruning:
                return True
        return False

    def priority_penalty(
        self,
        latent: torch.Tensor,
        path_cost: float,
        ledger: ComputeLedger,
    ) -> float:
        if not self.config.enabled:
            return 0.0
        assert self.join_head is not None
        if not self.config.hard_pruning and self.entries:
            current = latent.reshape(1, -1).expand(len(self.entries), -1)
            stored = torch.stack([item[0] for item in self.entries]).to(current.device)
            with torch.no_grad():
                probabilities = self.join_head.probability(current, stored)
            ledger.join_forward_calls += 1
            for probability, (_, old_cost) in zip(
                probabilities.tolist(), self.entries, strict=True
            ):
                if (
                    probability >= self.config.join_threshold
                    and path_cost >= old_cost + self.config.dominance_delta
                ):
                    return self.config.soft_priority_penalty
        if self.real_entries:
            current = latent.reshape(1, -1).expand(len(self.real_entries), -1)
            stored = torch.stack(self.real_entries).to(current.device)
            with torch.no_grad():
                probabilities = self.join_head.probability(current, stored)
            ledger.join_forward_calls += 1
            if bool((probabilities >= self.config.join_threshold).any()):
                return self.config.soft_priority_penalty
        return 0.0

    def add(self, latent: torch.Tensor, path_cost: float) -> None:
        if self.config.enabled:
            self.entries.append((latent.detach().cpu().reshape(-1), float(path_cost)))

    def add_real(self, latent: torch.Tensor) -> None:
        if self.config.enabled:
            self.real_entries.append(latent.detach().cpu().reshape(-1))

    def begin_search(self) -> None:
        self.entries.clear()


class BestFirstPlanner(BasePlanner):
    def __init__(
        self,
        *args: Any,
        memory_config: MemoryConfig | None = None,
        join_head: StateJoinHead | None = None,
        join_precision: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory = TranspositionMemory(
            memory_config or MemoryConfig(),
            join_head,
            validated_precision=join_precision,
        )

    def reset(self) -> None:
        self.memory.entries.clear()
        self.memory.real_entries.clear()

    def observe_real_state(self, latent: torch.Tensor) -> None:
        self.memory.add_real(latent)

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        rng = np.random.default_rng(seed)
        ledger = ComputeLedger()
        self.memory.begin_search()
        queue: list[tuple[float, int, tuple[int, ...]]] = [(0.0, 0, ())]
        counter = 1
        best: tuple[tuple[int, ...], float] | None = None
        pruned = 0
        soft_penalties = 0
        transition_limit = self.config.budget.transition_limit
        proposal_transition_budget = int(
            transition_limit * self.config.proposal_seed_fraction
        )
        proposal_capacity = max(
            1,
            int(round(self.config.num_candidates * self.config.budget.multiplier)),
        )
        proposal_count = min(
            proposal_capacity,
            proposal_transition_budget // self.config.horizon,
        )
        proposal_guidance: dict[tuple[int, ...], float] = {}
        if proposal_count:
            proposed = self.proposal.sample(
                context.embeddings[:, -1],
                context.goal.squeeze(1),
                count=proposal_count,
                horizon=self.config.horizon,
                rng=rng,
            )
            ledger.proposal_forward_calls += 1
            proposal_costs, proposal_rollout, _ = self._evaluate(
                context, proposed, ledger
            )
            ledger.node_expansions += proposal_count
            for sequence_array, raw_cost, latent in zip(
                proposed,
                proposal_costs.tolist(),
                proposal_rollout.terminal,
                strict=True,
            ):
                sequence = tuple(int(action) for action in sequence_array.tolist())
                cost = float(raw_cost)
                self.memory.add(latent, float(self.config.horizon))
                if best is None or (cost, sequence) < (best[1], best[0]):
                    best = (sequence, cost)
                for depth in range(1, self.config.horizon + 1):
                    prefix = sequence[:depth]
                    proposal_guidance[prefix] = min(
                        proposal_guidance.get(prefix, float("inf")),
                        cost,
                    )
        while queue:
            _, _, prefix = heapq.heappop(queue)
            if len(prefix) >= self.config.horizon:
                continue
            child_depth = len(prefix) + 1
            remaining = transition_limit - ledger.plan_transitions
            if remaining < len(ACTION_IDS) * child_depth:
                break
            children = [prefix + (action,) for action in ACTION_IDS]
            costs, rollout, _ = self._evaluate(
                context, np.asarray(children, dtype=np.int64), ledger
            )
            for child_index, child in enumerate(children):
                cost = float(costs[child_index])
                latent = rollout.terminal[child_index]
                ledger.node_expansions += 1
                path_cost = float(child_depth)
                if self.memory.dominated(latent, path_cost, ledger):
                    pruned += 1
                    continue
                penalty = self.memory.priority_penalty(latent, path_cost, ledger)
                priority = cost
                if penalty > 0.0:
                    priority += penalty
                    soft_penalties += 1
                if child in proposal_guidance:
                    priority = 0.5 * (priority + proposal_guidance[child])
                self.memory.add(latent, path_cost)
                if best is None or (cost, child) < (best[1], best[0]):
                    best = (child, cost)
                if len(child) < self.config.horizon:
                    heapq.heappush(queue, (priority, counter, child))
                    counter += 1
            if ledger.plan_transitions >= transition_limit:
                break
        if best is None:
            raise RuntimeError("best-first search could not evaluate one child")
        result = PlannerResult(
            sequence=self._pad(best[0]),
            cost=best[1],
            ledger=ledger,
            diagnostics={
                "memory_entries": len(self.memory.entries),
                "episode_real_memory_entries": len(self.memory.real_entries),
                "dominated_pruned": pruned,
                "soft_priority_penalties": soft_penalties,
                "proposal_seed_candidates": proposal_count,
                "proposal_seed_transitions": proposal_count * self.config.horizon,
                "proposal_guided_prefixes": len(proposal_guidance),
                "transition_limit": transition_limit,
            },
            candidate_batches=tuple(self._candidate_batches),
        )
        result.validate(self.config.horizon)
        return result


@dataclass
class _TreeNode:
    prefix: tuple[int, ...]
    visits: int = 0
    value_sum: float = 0.0
    leaf_reward: float = 0.0
    prior: float = 1.0
    children: dict[int, _TreeNode] = field(default_factory=dict)
    action_priors: dict[int, float] = field(
        default_factory=lambda: {action: 1.0 / len(ACTION_IDS) for action in ACTION_IDS}
    )

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTSPlanner(BasePlanner):
    def __init__(
        self,
        *args: Any,
        dts_head: VectorDTSHead | None = None,
        dts_policy_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dts_head = dts_head
        self.dts_policy_enabled = bool(dts_policy_enabled)

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        rng = np.random.default_rng(seed)
        ledger = ComputeLedger()
        root = _TreeNode(prefix=())
        best_by_root: dict[int, tuple[tuple[int, ...], float]] = {}
        simulations = 0
        max_simulations = max(1, 10 * self.config.budget.transition_limit)
        termination_reason = "simulation_limit"
        while True:
            node = root
            path = [node]
            while (
                len(node.children) == len(ACTION_IDS)
                and len(node.prefix) < self.config.horizon
            ):
                node = self._select_child(node, rng)
                path.append(node)
            if len(node.prefix) >= self.config.horizon:
                leaf = node
                reward = leaf.leaf_reward
            else:
                unexpanded = [
                    action for action in ACTION_IDS if action not in node.children
                ]
                if not unexpanded:
                    continue
                action = int(rng.choice(unexpanded))
                prefix = node.prefix + (action,)
                remaining = (
                    self.config.budget.transition_limit - ledger.plan_transitions
                )
                if remaining < len(prefix):
                    break
                costs, rollout, _ = self._evaluate(
                    context, np.asarray([prefix], dtype=np.int64), ledger
                )
                cost = float(costs[0])
                priors, learned_value = self._leaf_guidance(
                    rollout.terminal[0:1], context.goal.squeeze(1), ledger
                )
                reward = -cost - learned_value
                leaf = _TreeNode(
                    prefix=prefix,
                    leaf_reward=reward,
                    prior=float(node.action_priors.get(action, 1.0 / len(ACTION_IDS))),
                    action_priors={
                        child_action: float(prior)
                        for child_action, prior in zip(ACTION_IDS, priors, strict=True)
                    },
                )
                node.children[action] = leaf
                path.append(leaf)
                ledger.node_expansions += 1
                root_action = prefix[0]
                incumbent = best_by_root.get(root_action)
                if incumbent is None or (cost, prefix) < (
                    incumbent[1],
                    incumbent[0],
                ):
                    best_by_root[root_action] = (prefix, cost)
            for current in reversed(path):
                current.visits += 1
                current.value_sum += reward
            simulations += 1
            if ledger.plan_transitions >= self.config.budget.transition_limit:
                termination_reason = "transition_limit"
                break
            if simulations >= max_simulations:
                break
        if not best_by_root or not root.children:
            raise RuntimeError("MCTS could not evaluate one action")
        root_action, _ = max(
            root.children.items(), key=lambda item: (item[1].visits, -item[0])
        )
        sequence, selected_cost = best_by_root[root_action]
        result = PlannerResult(
            sequence=self._pad(sequence),
            cost=float(selected_cost),
            ledger=ledger,
            diagnostics={
                "root_visits": root.visits,
                "simulations": simulations,
                "max_simulations": max_simulations,
                "termination_reason": termination_reason,
                "selected_root_action": root_action,
                "selected_evaluated_prefix": list(sequence),
                "root_action_visits": {
                    str(action): child.visits for action, child in root.children.items()
                },
                "transition_limit": self.config.budget.transition_limit,
                "learned_guidance": self.dts_head is not None,
                "learned_expansion_policy": (
                    self.dts_head is not None and self.dts_policy_enabled
                ),
            },
            candidate_batches=tuple(self._candidate_batches),
        )
        result.validate(self.config.horizon)
        return result

    def _select_child(self, node: _TreeNode, rng: np.random.Generator) -> _TreeNode:
        parent_visits = max(node.visits, 1)
        values: list[tuple[float, _TreeNode]] = []
        for child in node.children.values():
            bonus = (
                self.config.exploration_constant
                * child.prior
                * math.sqrt(parent_visits)
                / (1 + child.visits)
            )
            values.append((child.mean_value + bonus, child))
        maximum = max(value for value, _ in values)
        tied = [child for value, child in values if abs(value - maximum) < 1e-12]
        return tied[int(rng.integers(len(tied)))]

    def _leaf_guidance(
        self,
        latent: torch.Tensor,
        goal: torch.Tensor,
        ledger: ComputeLedger,
    ) -> tuple[np.ndarray, float]:
        if self.dts_head is None:
            return np.full(len(ACTION_IDS), 1.0 / len(ACTION_IDS)), 0.0
        with torch.no_grad():
            output = self.dts_head(latent, goal)
            priors = (
                torch.softmax(output["policy_logits"], dim=-1)[0]
                if self.dts_policy_enabled
                else torch.full(
                    (len(ACTION_IDS),),
                    1.0 / len(ACTION_IDS),
                    dtype=latent.dtype,
                    device=latent.device,
                )
            )
            value = expected_distance_from_logits(output["value_logits"])[0]
        ledger.dts_forward_calls += 1
        return priors.detach().cpu().numpy(), float(value)


class BidirectionalPlanner(BasePlanner):
    """Diagnostic cycle-verified forward/goal-side vector search."""

    def __init__(
        self,
        *args: Any,
        join_head: StateJoinHead,
        join_threshold: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.join_head = join_head
        self.join_threshold = float(join_threshold)

    def plan(self, context: VectorContext, *, seed: int) -> PlannerResult:
        self._begin_candidate_capture()
        rng = np.random.default_rng(seed)
        ledger = ComputeLedger()
        half = max(1, self.config.horizon // 2)
        per_side = max(
            1,
            min(
                self.config.num_candidates,
                self.config.budget.transition_limit // (2 * half),
            ),
        )
        forward_actions = self.proposal.sample(
            context.embeddings[:, -1],
            context.goal.squeeze(1),
            count=per_side,
            horizon=half,
            rng=rng,
        )
        ledger.proposal_forward_calls += 1
        forward = self.world_model.rollout(
            context,
            forward_actions,
            semantics=self.config.rollout_semantics,
            ledger=ledger,
        )
        goal_context = VectorContext(
            embeddings=context.goal.repeat(1, self.config.history_size, 1),
            actions=torch.full_like(context.actions, 4),
            goal=context.embeddings[:, -1:],
            maze_size=context.maze_size,
        )
        backward_actions = self.proposal.sample(
            context.goal.squeeze(1),
            context.embeddings[:, -1],
            count=per_side,
            horizon=half,
            rng=rng,
        )
        ledger.proposal_forward_calls += 1
        backward = self.world_model.rollout(
            goal_context,
            backward_actions,
            semantics=self.config.rollout_semantics,
            ledger=ledger,
        )
        left = forward.terminal[:, None, :].expand(-1, per_side, -1)
        right = backward.terminal[None, :, :].expand(per_side, -1, -1)
        with torch.no_grad():
            joins = self.join_head.probability(
                left.reshape(-1, left.shape[-1]),
                right.reshape(-1, right.shape[-1]),
            ).reshape(per_side, per_side)
        ledger.join_forward_calls += 1
        flat_order = torch.argsort(joins.reshape(-1), descending=True)
        chosen: np.ndarray | None = None
        chosen_probability = 0.0
        for flat in flat_order.tolist():
            i, j = divmod(flat, per_side)
            probability = float(joins[i, j])
            if probability < self.join_threshold and chosen is not None:
                break
            inverse = [
                INVERSE_ACTION[int(action)] for action in backward_actions[j, ::-1]
            ]
            candidate = self._pad(
                np.concatenate(
                    [forward_actions[i], np.asarray(inverse, dtype=np.int64)]
                )
            )
            if (
                ledger.plan_transitions + self.config.horizon
                > self.config.budget.transition_limit
            ):
                break
            costs, _, _ = self._evaluate(context, candidate[None, :], ledger)
            if chosen is None or float(costs[0]) < chosen_probability:
                chosen = candidate
                chosen_probability = float(costs[0])
                chosen_join = probability
        if chosen is None:
            index = int(flat_order[0])
            i, j = divmod(index, per_side)
            inverse = [
                INVERSE_ACTION[int(action)] for action in backward_actions[j, ::-1]
            ]
            chosen = self._pad(
                np.concatenate(
                    [forward_actions[i], np.asarray(inverse, dtype=np.int64)]
                )
            )
            chosen_probability = float("inf")
            chosen_join = float(joins[i, j])
        result = PlannerResult(
            sequence=chosen,
            cost=chosen_probability,
            ledger=ledger,
            diagnostics={
                "join_probability": chosen_join,
                "join_threshold": self.join_threshold,
                "per_side_candidates": per_side,
                "diagnostic_only": True,
            },
            candidate_batches=tuple(self._candidate_batches),
        )
        if not math.isfinite(result.cost):
            # The stitched candidate could not be rerolled within budget. Use a
            # finite diagnostic cost without pretending it passed final scoring.
            result = PlannerResult(
                sequence=result.sequence,
                cost=1e30,
                ledger=result.ledger,
                diagnostics={**result.diagnostics, "reranked": False},
                candidate_batches=result.candidate_batches,
            )
        result.validate(self.config.horizon)
        return result


def build_planner(
    world_model: VectorWorldModel,
    config: PlannerConfig,
    scorer: CompositeScorer,
    *,
    proposal: CandidateProposal | None = None,
    memory_config: MemoryConfig | None = None,
    join_head: StateJoinHead | None = None,
    join_precision: float | None = None,
    dts_head: VectorDTSHead | None = None,
    proposal_only: bool = False,
    dts_expansion: str = "learned",
) -> BasePlanner:
    common = {
        "world_model": world_model,
        "config": config,
        "scorer": scorer,
        "proposal": proposal,
    }
    if proposal_only:
        return ProposalOnlyPlanner(**common)
    if config.kind == PlannerKind.LEGACY_CEM:
        return LegacyCEMPlanner(**common)
    if config.kind == PlannerKind.CATEGORICAL_CEM:
        return CategoricalCEMPlanner(**common)
    if config.kind == PlannerKind.ICEM:
        return ICEMPlanner(**common)
    if config.kind == PlannerKind.BEAM:
        return BeamPlanner(**common)
    if config.kind == PlannerKind.BEST_FIRST:
        return BestFirstPlanner(
            **common,
            memory_config=memory_config,
            join_head=join_head,
            join_precision=join_precision,
        )
    if config.kind in (PlannerKind.MCTS, PlannerKind.VECTOR_DTS):
        if config.kind == PlannerKind.VECTOR_DTS and dts_head is None:
            raise ValueError("Vector-DTS requires a trained DTS head")
        if config.kind == PlannerKind.VECTOR_DTS and dts_expansion == "direct":
            assert dts_head is not None
            return DirectDTSPlanner(**common, dts_head=dts_head)
        if config.kind == PlannerKind.VECTOR_DTS and dts_expansion == "random":
            assert dts_head is not None
            return MCTSPlanner(**common, dts_head=dts_head, dts_policy_enabled=False)
        if config.kind == PlannerKind.VECTOR_DTS and dts_expansion == "fixed_breadth":
            assert dts_head is not None
            return DTSBreadthPlanner(**common, dts_head=dts_head)
        return MCTSPlanner(**common, dts_head=dts_head)
    if config.kind == PlannerKind.BIDIRECTIONAL:
        if join_head is None:
            raise ValueError("bidirectional search requires a trained join head")
        return BidirectionalPlanner(
            **common,
            join_head=join_head,
            join_threshold=(memory_config or MemoryConfig()).join_threshold,
        )
    raise ValueError(f"unsupported planner kind: {config.kind}")


__all__ = [
    "BasePlanner",
    "BeamPlanner",
    "BestFirstPlanner",
    "BidirectionalPlanner",
    "CandidateBatch",
    "CategoricalCEMPlanner",
    "CompositeScorer",
    "DirectDTSPlanner",
    "DTSBreadthPlanner",
    "ICEMPlanner",
    "LegacyCEMPlanner",
    "MCTSPlanner",
    "PlannerResult",
    "ProposalOnlyPlanner",
    "ScoreBatch",
    "TranspositionMemory",
    "build_planner",
]
