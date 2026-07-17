from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

import distance_head_study.losses as losses_module
from distance_head_study.losses import (
    TrajectoryBatch,
    _horizon_grid,
    compute_objective_terms,
    gradient_calibrated_weights,
    weighted_total,
)
from distance_head_study.methods import resolve_method
from distance_head_study.models import build_distance_head
from distance_head_study.schemas import ArchitectureKind, HeadSpec, TrainingScope
from distance_head_study.train_backbone import _initialize_backbone_rng
from distance_head_study.train_head import (
    _calibrate,
    _configure_backbone_scope,
    _initialize_from_parent,
    _predict_all_actions,
    _restore_rng_state,
    _rng_state,
)


def test_scalar_horizon_grid_matches_local_and_legacy_rollout_queries() -> None:
    head = build_distance_head(
        HeadSpec(
            architecture=ArchitectureKind.HIERARCHICAL,
            horizon_conditioned=True,
        )
    )
    assert _horizon_grid(head) == (1, 2, 4, 6, 9, 12)


def test_horizon_conditioned_local_objectives_query_exactly_one_step(
    monkeypatch,
    method_catalog,
    decision_root,
    synthetic_batch,
) -> None:
    method, _ = resolve_method(
        method_catalog, "j3_rcaux_reach", decision_root=decision_root
    )
    assert method.head is not None
    head = build_distance_head(method.head)
    predicted = torch.randn_like(synthetic_batch.next_latents)
    calls: list[tuple[int, int, bool]] = []
    original = losses_module._head_call

    def tracked_call(
        head,
        source,
        goal,
        *,
        horizon=12,
        predicted_domain=False,
    ):
        calls.append((source.shape[0], horizon, predicted_domain))
        return original(
            head,
            source,
            goal,
            horizon=horizon,
            predicted_domain=predicted_domain,
        )

    monkeypatch.setattr(losses_module, "_head_call", tracked_call)
    compute_objective_terms(
        head,
        method,
        synthetic_batch,
        predicted_next=predicted,
    )

    flattened_actions = (
        synthetic_batch.next_latents.shape[0] * synthetic_batch.next_latents.shape[1]
    )
    assert (flattened_actions, 1, False) in calls
    assert (flattened_actions, 1, True) in calls


@pytest.mark.parametrize(
    "method_name",
    [
        "b_dh_cem",
        "b1_listwise",
        "b2_bellman",
        "b3_multistep",
        "c1_predicted_listwise",
        "d2_trm_full",
        "d3_trm_shuffle",
        "d4_reachability",
        "r_output_ordinal",
        "r_output_distribution",
        "r_loss_mae",
        "r_pairwise",
        "r_delta",
        "r_eikonal",
        "r_quasimetric",
        "r_successor_contrastive",
        "r_arch_asymmetric",
        "r_arch_hierarchical_budget",
        "r_uncertainty",
    ],
)
def test_declared_objectives_have_finite_forward_and_backward(
    method_name,
    method_catalog,
    decision_root,
    synthetic_batch,
) -> None:
    method, _ = resolve_method(method_catalog, method_name, decision_root=decision_root)
    assert method.head is not None and method.objectives is not None
    head = build_distance_head(method.head)
    generator = torch.Generator().manual_seed(17)
    predicted = torch.randn(*synthetic_batch.next_latents.shape, generator=generator)
    contexts = 2
    candidates = 7
    labels = torch.tensor([[5.0, 4.0, 3.0, 3.0, 6.0, 8.0, 7.0]]).repeat(contexts, 1)
    trajectory = TrajectoryBatch(
        predicted_terminal=torch.randn(contexts, candidates, 256, generator=generator),
        goals=synthetic_batch.goal[:contexts],
        max_distance=synthetic_batch.max_distance[:contexts],
        true_endpoint_distance=labels,
        horizon=12,
    )
    terms = compute_objective_terms(
        head,
        method,
        synthetic_batch,
        predicted_next=predicted,
        trajectory=trajectory,
    )
    expected = {
        name
        for name, value in method.objectives.model_dump().items()
        if name != "original_jepa" and value > 0
    }
    assert set(terms) == expected
    assert all(torch.isfinite(term) for term in terms.values())
    weights = {name: float(getattr(method.objectives, name)) for name in terms}
    total = weighted_total(terms, weights)
    total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_gradient_calibration_is_bounded_and_key_exact(
    method_catalog, decision_root, synthetic_batch
) -> None:
    method, _ = resolve_method(
        method_catalog, "b2_bellman", decision_root=decision_root
    )
    assert method.head is not None and method.objectives is not None
    head = build_distance_head(method.head)
    terms = compute_objective_terms(head, method, synthetic_batch)
    base = {name: float(getattr(method.objectives, name)) for name in terms}
    calibrated = gradient_calibrated_weights(
        terms,
        base,
        list(head.parameters()),
        target_ratio=0.5,
        clip=(0.1, 10.0),
    )
    assert set(calibrated) == set(terms)
    for name, value in calibrated.items():
        if name != "absolute":
            assert base[name] * 0.1 <= value <= base[name] * 10.0


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.embedding_projector = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.predictor = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))


@pytest.mark.parametrize(
    "scope,expected_train",
    [
        (TrainingScope.FROZEN, set()),
        (TrainingScope.PREDICTOR, {"predictor"}),
        (
            TrainingScope.PROJECTOR_PREDICTOR,
            {"embedding_projector", "predictor"},
        ),
        (TrainingScope.FULL, {"encoder", "embedding_projector", "predictor"}),
    ],
)
def test_training_scope_freezes_parameters_and_module_state(
    scope: TrainingScope, expected_train: set[str]
) -> None:
    model = _Backbone()
    _configure_backbone_scope(model, scope)
    modules = {
        "encoder": model.encoder,
        "embedding_projector": model.embedding_projector,
        "predictor": model.predictor,
    }
    for name, module in modules.items():
        assert module.training is (name in expected_train)
        assert all(
            parameter.requires_grad is (name in expected_train)
            for parameter in module.parameters()
        )


def test_calibration_is_module_state_side_effect_free(monkeypatch) -> None:
    model = _Backbone()
    _configure_backbone_scope(model, TrainingScope.PREDICTOR)
    head = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    head.train()
    modules = list(model.modules()) + list(head.modules())
    expected_modes = [module.training for module in modules]

    def fake_impl(head, method, model, dataset, candidate_actions, **kwargs):
        del method, dataset, candidate_actions, kwargs
        assert all(not module.training for module in model.modules())
        assert all(not module.training for module in head.modules())
        return {"absolute": 1.0}

    monkeypatch.setattr("distance_head_study.train_head._calibrate_impl", fake_impl)
    result = _calibrate(
        head,
        object(),
        model,
        object(),
        torch.empty(0),
        config=object(),
        backbone_seed=42,
        device=torch.device("cpu"),
    )
    assert result == {"absolute": 1.0}
    assert [module.training for module in modules] == expected_modes


def test_rcaux_compatible_initialization_loads_shared_scoring_layers(
    method_catalog, decision_root
) -> None:
    parent_method, _ = resolve_method(
        method_catalog, "b_dh_cem", decision_root=decision_root
    )
    child_method, _ = resolve_method(
        method_catalog, "j3_rcaux_reach", decision_root=decision_root
    )
    assert parent_method.head is not None and child_method.head is not None
    parent = build_distance_head(parent_method.head)
    child = build_distance_head(child_method.head)
    with torch.no_grad():
        parent.primary.weight.fill_(2.0)
        parent.primary.bias.fill_(3.0)
    report = _initialize_from_parent(
        child,
        child_method,
        {
            "head_spec": parent_method.head.model_dump(mode="json"),
            "head_state_dict": parent.state_dict(),
        },
    )
    assert report["mode"] == "compatible_shared"
    assert "primary.weight" in report["loaded_keys"]
    assert torch.equal(child.primary.weight, parent.primary.weight)
    assert torch.equal(child.primary.bias, parent.primary.bias)


def test_rng_state_round_trip_reproduces_all_training_generators() -> None:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    state = _rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    random.random()
    np.random.rand()
    torch.rand(3)
    _restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_fresh_backbone_rng_reproduces_source_model_initialization() -> None:
    _initialize_backbone_rng(1001)
    first = nn.Linear(7, 5).state_dict()
    first_numpy = np.random.rand(3)
    first_python = [random.random() for _ in range(3)]
    _initialize_backbone_rng(1001)
    second = nn.Linear(7, 5).state_dict()
    second_numpy = np.random.rand(3)
    second_python = [random.random() for _ in range(3)]

    assert all(torch.equal(first[key], second[key]) for key in first)
    assert np.array_equal(first_numpy, second_numpy)
    assert first_python == second_python


class _SequencePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(256, 256, bias=False)

    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return (
            self.projection(embeddings[:, 1:])
            + actions[:, :, None].to(embeddings) / 100.0
        )


class _PredictorOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = _SequencePredictor()


def test_joint_control_blocks_only_distance_gradient_to_backbone(
    method_catalog, decision_root, synthetic_batch
) -> None:
    control, _ = resolve_method(
        method_catalog, "j0_cont_predictor", decision_root=decision_root
    )
    treatment, _ = resolve_method(
        method_catalog, "j0_dist_predictor", decision_root=decision_root
    )
    assert control.head is not None and control.objectives is not None
    assert treatment.head == control.head and treatment.objectives == control.objectives

    model_control = _PredictorOnly()
    model_treatment = _PredictorOnly()
    model_treatment.load_state_dict(model_control.state_dict())
    head_control = build_distance_head(control.head)
    head_treatment = build_distance_head(treatment.head)
    head_treatment.load_state_dict(head_control.state_dict())

    predicted_control = _predict_all_actions(
        model_control, synthetic_batch, gradients=control.distance_gradients_to_backbone
    )
    control_terms = compute_objective_terms(
        head_control,
        control,
        synthetic_batch,
        predicted_next=predicted_control,
    )
    sum(control_terms.values()).backward()
    assert any(parameter.grad is not None for parameter in head_control.parameters())
    assert all(parameter.grad is None for parameter in model_control.parameters())

    predicted_treatment = _predict_all_actions(
        model_treatment,
        synthetic_batch,
        gradients=treatment.distance_gradients_to_backbone,
    )
    treatment_terms = compute_objective_terms(
        head_treatment,
        treatment,
        synthetic_batch,
        predicted_next=predicted_treatment,
    )
    sum(treatment_terms.values()).backward()
    assert any(parameter.grad is not None for parameter in head_treatment.parameters())
    assert any(parameter.grad is not None for parameter in model_treatment.parameters())
