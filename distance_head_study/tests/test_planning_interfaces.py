from __future__ import annotations

import numpy as np
import pytest
import torch

from distance_head_study import ACTION_IDS, MODEL_ACTION_VOCAB_SIZE
from distance_head_study.common import load_study_config
from distance_head_study.evaluate import (
    DistancePlanningScorer,
    _exact_cem,
    _frontier_planner,
    _greedy_proposal,
)
from distance_head_study.methods import resolve_method
from distance_head_study.models import build_distance_head
from vector_jepa_planner_frontier.common import ComputeLedger
from vector_jepa_planner_frontier.schemas import RolloutSemantics
from vector_jepa_planner_frontier.world_model import (
    RolloutBatch,
    VectorContext,
    VectorWorldModel,
)


class _DeterministicWorldModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def rollout(
        self,
        context: VectorContext,
        candidate_actions,
        *,
        semantics,
        ledger: ComputeLedger | None = None,
        gradients: bool = False,
    ) -> RolloutBatch:
        del gradients
        actions = torch.as_tensor(candidate_actions, dtype=torch.long)
        batch, horizon = actions.shape
        source = context.embeddings[:, -1].expand(batch, -1)
        offsets = actions.float().cumsum(dim=1)[..., None] / 100.0
        states = source[:, None, :].expand(-1, horizon, -1).clone() + offsets
        if ledger is not None:
            ledger.record_plan(
                transitions=batch * horizon, batch_size=batch, calls=horizon
            )
        return RolloutBatch(
            states=states,
            terminal=states[:, -1],
            actions=actions,
            semantics=semantics,
        )


def _context() -> VectorContext:
    return VectorContext(
        embeddings=torch.zeros(1, 3, 256),
        actions=torch.full((1, 3), 4, dtype=torch.long),
        goal=torch.ones(1, 1, 256),
        maze_size=11,
        remaining_steps=128,
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "p_path_integrated",
        "p_hybrid_l2",
        "p_reachability",
        "p_risk_loop",
    ],
)
def test_all_custom_costs_return_finite_candidate_scores(
    method_name, method_catalog, decision_root
) -> None:
    method, _ = resolve_method(method_catalog, method_name, decision_root=decision_root)
    assert method.head is not None
    head = build_distance_head(method.head)
    context = _context()
    world_model = _DeterministicWorldModel()
    actions = np.tile(np.arange(1, 5), (12, 1)).T
    rollout = world_model.rollout(
        context, actions, semantics=RolloutSemantics.LEGACY_WARMUP_V1
    )
    scores = DistancePlanningScorer(method, head)(
        context,
        rollout,
        remaining_budget=64,
        ledger=ComputeLedger(),
    )
    scores.validate(4)


def test_risk_loop_cost_uses_episode_real_state_memory(
    method_catalog, decision_root
) -> None:
    method, _ = resolve_method(
        method_catalog, "p_risk_loop", decision_root=decision_root
    )
    assert method.head is not None
    scorer = DistancePlanningScorer(method, build_distance_head(method.head))
    scorer.observe_real_state(torch.zeros(1, 1, 256))
    offsets = torch.arange(12, dtype=torch.float32)[:, None] * 0.01
    states = torch.stack(
        [offsets.expand(-1, 256), 0.5 + offsets.expand(-1, 256)], dim=0
    )
    rollout = RolloutBatch(
        states=states,
        terminal=states[:, -1],
        actions=torch.ones(2, 12, dtype=torch.long),
        semantics=RolloutSemantics.LEGACY_WARMUP_V1,
    )
    scores = scorer(_context(), rollout, remaining_budget=64, ledger=ComputeLedger())
    assert (
        scores.components["real_loop_risk"][0] > scores.components["real_loop_risk"][1]
    )
    assert scores.components["loop_risk"][0] > scores.components["loop_risk"][1]


@pytest.mark.parametrize("method_name", ["p_icem", "p_beam", "p_best_first"])
def test_search_planners_share_the_locked_interface_and_budget(
    method_name, method_catalog, decision_root
) -> None:
    method, _ = resolve_method(method_catalog, method_name, decision_root=decision_root)
    assert method.head is not None
    config = load_study_config("distance_head_study/configs/default.json")
    world_model = _DeterministicWorldModel()
    scorer = DistancePlanningScorer(method, build_distance_head(method.head))
    planner = _frontier_planner(world_model, method, scorer, config)
    planner.reset()
    result = planner.plan(_context(), seed=123)
    result.validate(config.planner.horizon)
    assert set(result.sequence.tolist()) <= {1, 2, 3, 4}
    assert result.ledger.plan_transitions > 0
    assert result.ledger.plan_transitions <= config.planner.reference_transitions
    planner.reset()
    repeated = planner.plan(_context(), seed=123)
    assert np.array_equal(result.sequence, repeated.sequence)
    assert result.cost == repeated.cost
    assert result.ledger.to_dict() == repeated.ledger.to_dict()
    if method_name == "p_icem":
        assert result.ledger.plan_transitions == config.planner.reference_transitions


@pytest.mark.parametrize(
    "method_name,expected_predicted_domain",
    [("b_dh_model_free", False), ("b_dh_predictor_greedy", True)],
)
def test_greedy_scoring_marks_true_and_predicted_latent_domains(
    method_name,
    expected_predicted_domain,
    method_catalog,
    decision_root,
    monkeypatch,
) -> None:
    method, _ = resolve_method(method_catalog, method_name, decision_root=decision_root)
    observed: list[bool] = []

    class WorldModel:
        device = torch.device("cpu")

        @staticmethod
        def encode(observation, maze_size):
            del observation, maze_size
            return torch.zeros(1, 1, 256)

        @staticmethod
        def one_step_all_actions(context):
            del context
            return torch.zeros(MODEL_ACTION_VOCAB_SIZE, 256)

    def terminal_cost(self, terminal, goal, *, horizon, predicted_domain=True):
        del self, goal, horizon
        observed.append(bool(predicted_domain))
        return torch.arange(terminal.shape[0], dtype=torch.float32)

    monkeypatch.setattr(DistancePlanningScorer, "terminal_cost", terminal_cost)
    monkeypatch.setattr(
        "distance_head_study.evaluate.next_state", lambda env, state, action: action
    )
    monkeypatch.setattr(
        "distance_head_study.evaluate.observe_state",
        lambda env, state: np.zeros((2, 2, 5), dtype=np.float32),
    )
    action, _, ledger = _greedy_proposal(
        WorldModel(), None, method, _context(), object(), 0
    )
    assert observed == [expected_predicted_domain]
    assert action in ACTION_IDS
    assert ledger.plan_transitions == (
        MODEL_ACTION_VOCAB_SIZE if expected_predicted_domain else 0
    )


class _Predictor(torch.nn.Module):
    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        output = embeddings[:, 1:].clone()
        output[:, -1] = output[:, -1] + actions[:, -1:].to(output) / 100.0
        return output


class _PredictorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = _Predictor()


def test_terminal_cem_and_instrumented_cem_are_candidate_exact(
    method_catalog, decision_root
) -> None:
    config = load_study_config("distance_head_study/configs/default.json")
    method, _ = resolve_method(method_catalog, "b_dh_cem", decision_root=decision_root)
    assert method.head is not None
    head = build_distance_head(method.head)
    world_model = VectorWorldModel(
        _PredictorModel(), device=torch.device("cpu"), history_size=3
    )
    scorer = DistancePlanningScorer(method, head)
    context = _context()
    exact_sequence, exact_cost, _ = _exact_cem(
        world_model, context, scorer, config, seed=19
    )
    instrumented = _frontier_planner(world_model, method, scorer, config).plan(
        context, seed=19
    )
    assert np.array_equal(instrumented.sequence, exact_sequence)
    assert instrumented.cost == pytest.approx(exact_cost, abs=1e-7)
