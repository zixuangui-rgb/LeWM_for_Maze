from __future__ import annotations

import pytest
import torch

from a1_quick_validation import NEW_METHODS
from a1_quick_validation.profile import load_profile
from distance_head_study.common import load_study_config
from distance_head_study.losses import compute_objective_terms, weighted_total
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.models import build_distance_head
from distance_head_study.protocol import verify_protocol_lock


@pytest.mark.parametrize("method_name", NEW_METHODS)
def test_every_new_head_has_finite_forward_backward_and_step(
    method_name: str, synthetic_batch
) -> None:
    profile = load_profile()
    config = load_study_config(profile.paths.quick_config)
    lock = verify_protocol_lock(config)
    method, _, _ = load_and_resolve_method(
        config.paths.method_catalog,
        method_name,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    assert method.head is not None and method.objectives is not None
    head = build_distance_head(method.head)
    predicted = torch.randn_like(synthetic_batch.next_latents)
    terms = compute_objective_terms(
        head,
        method,
        synthetic_batch,
        predicted_next=predicted,
    )
    expected = {
        name
        for name, value in method.objectives.model_dump().items()
        if name != "original_jepa" and value > 0
    }
    assert set(terms) == expected
    assert all(torch.isfinite(value) for value in terms.values())
    weights = {name: float(getattr(method.objectives, name)) for name in terms}
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    before = {name: value.detach().clone() for name, value in head.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    total = weighted_total(terms, weights)
    total.backward()
    optimizer.step()
    assert torch.isfinite(total)
    assert any(
        not torch.equal(before[name], value)
        for name, value in head.state_dict().items()
    )


def test_reachability_has_horizon_matched_control() -> None:
    profile = load_profile()
    config = load_study_config(profile.paths.quick_config)
    lock = verify_protocol_lock(config)
    control, _, _ = load_and_resolve_method(
        config.paths.method_catalog,
        "a1_hcond",
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    reach, _, _ = load_and_resolve_method(
        config.paths.method_catalog,
        "a1_reach",
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    assert control.head is not None and reach.head is not None
    assert control.head.horizon_conditioned
    assert reach.head.horizon_conditioned
    assert control.head.output.value == "scalar"
    assert reach.head.output.value == "multitask"
    assert control.objectives.reachability == 0.0
    assert reach.objectives.reachability == 1.0
    assert reach.planner == control.planner
    assert reach.planner.cost.value == "terminal_distance"
    assert reach.planner.reachability_weight == 0.0
