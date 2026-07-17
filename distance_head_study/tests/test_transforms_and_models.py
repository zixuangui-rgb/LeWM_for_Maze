from __future__ import annotations

import pytest
import torch

from distance_head_study.losses import reachability_logits_by_budget
from distance_head_study.models import build_distance_head
from distance_head_study.schemas import (
    ArchitectureKind,
    HeadSpec,
    OutputKind,
    TargetMode,
)
from distance_head_study.transforms import assert_round_trip


@pytest.mark.parametrize("mode", list(TargetMode))
def test_distance_transform_round_trip(mode: TargetMode) -> None:
    assert assert_round_trip(mode, tolerance=1e-9) < 1e-9


@pytest.mark.parametrize(
    "spec",
    [
        HeadSpec(),
        HeadSpec(output=OutputKind.ORDINAL),
        HeadSpec(output=OutputKind.DISTRIBUTION),
        HeadSpec(output=OutputKind.QUANTILE),
        HeadSpec(
            output=OutputKind.MULTITASK,
            architecture=ArchitectureKind.HORIZON_CONDITIONED,
            horizon_conditioned=True,
        ),
        HeadSpec(architecture=ArchitectureKind.ASYMMETRIC),
        HeadSpec(
            architecture=ArchitectureKind.HIERARCHICAL,
            horizon_conditioned=True,
        ),
    ],
)
def test_head_output_contracts(spec: HeadSpec) -> None:
    generator = torch.Generator().manual_seed(3)
    source = torch.randn(6, 256, generator=generator)
    goal = torch.randn(6, 256, generator=generator)
    head = build_distance_head(spec)
    horizon = torch.full((6,), 12.0) if spec.horizon_conditioned else None
    output = head(source, goal, horizon=horizon)
    output.validate(6)
    assert output.score.shape == (6,)
    assert bool((output.score >= 0).all())


def test_horizon_conditioned_head_fails_without_horizon() -> None:
    head = build_distance_head(
        HeadSpec(
            architecture=ArchitectureKind.HORIZON_CONDITIONED,
            horizon_conditioned=True,
        )
    )
    with pytest.raises(ValueError, match="requires horizon"):
        head(torch.zeros(2, 256), torch.zeros(2, 256))


def test_hierarchical_spec_cannot_omit_its_horizon_input() -> None:
    with pytest.raises(ValueError, match="needs its horizon input"):
        HeadSpec(
            architecture=ArchitectureKind.HIERARCHICAL,
            horizon_conditioned=False,
        )


def test_quasimetric_has_zero_diagonal_nonnegativity_and_triangle_inequality() -> None:
    generator = torch.Generator().manual_seed(5)
    head = build_distance_head(HeadSpec(architecture=ArchitectureKind.QUASIMETRIC))
    x = torch.randn(32, 256, generator=generator)
    y = torch.randn(32, 256, generator=generator)
    z = torch.randn(32, 256, generator=generator)
    d_xx = head(x, x).score
    d_xy = head(x, y).score
    d_yz = head(y, z).score
    d_xz = head(x, z).score
    assert torch.equal(d_xx, torch.zeros_like(d_xx))
    assert bool((d_xy >= 0).all())
    assert bool((d_xz <= d_xy + d_yz + 1e-6).all())


def test_quasimetric_spec_rejects_ignored_horizon_conditioning() -> None:
    with pytest.raises(ValueError, match="forbids horizon"):
        HeadSpec(
            architecture=ArchitectureKind.QUASIMETRIC,
            horizon_conditioned=True,
        )


def test_budgeted_reachability_uses_matching_horizon_and_backpropagates() -> None:
    spec = HeadSpec(
        output=OutputKind.MULTITASK,
        architecture=ArchitectureKind.HORIZON_CONDITIONED,
        horizon_conditioned=True,
    )
    head = build_distance_head(spec)
    source = torch.randn(4, 256)
    goal = torch.randn(4, 256)
    logits = reachability_logits_by_budget(head, source, goal)
    assert logits.shape == (4, len(spec.reachability_budgets))
    logits.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_hierarchical_head_has_three_real_scale_experts() -> None:
    spec = HeadSpec(
        architecture=ArchitectureKind.HIERARCHICAL,
        horizon_conditioned=True,
    )
    head = build_distance_head(spec)
    assert head.hierarchical_experts is not None
    assert len(head.hierarchical_experts) == 3
    source = torch.randn(4, 256)
    goal = torch.randn(4, 256)
    short = head(source, goal, horizon=torch.ones(4)).score
    long = head(source, goal, horizon=torch.full((4,), 12.0)).score
    assert torch.isfinite(short).all() and torch.isfinite(long).all()
    assert not torch.equal(short, long)


def test_ordinal_score_is_a_valid_expected_bin_center() -> None:
    spec = HeadSpec(output=OutputKind.ORDINAL)
    head = build_distance_head(spec)
    output = head(torch.zeros(3, 256), torch.ones(3, 256))
    assert bool((output.score >= head.ordinal_centers.min()).all())
    assert bool((output.score <= head.ordinal_centers.max()).all())
