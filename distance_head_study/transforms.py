"""Distance target transforms with explicit raw-unit inversion."""

from __future__ import annotations

import torch

from distance_head_study.schemas import TargetMode


def _validated_distance(distance: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(distance):
        distance = distance.float()
    if not torch.isfinite(distance).all() or bool((distance < 0).any()):
        raise ValueError("BFS distances must be finite and non-negative")
    return distance


def _validated_max_distance(
    distance: torch.Tensor, max_distance: torch.Tensor | None
) -> torch.Tensor:
    if max_distance is None:
        raise ValueError("legacy_log_norm requires per-topology max_distance")
    value = max_distance.to(device=distance.device, dtype=distance.dtype)
    if not torch.isfinite(value).all() or bool((value < 1).any()):
        raise ValueError("max_distance must be finite and at least one")
    return value


def transform_distance(
    distance: torch.Tensor,
    mode: TargetMode | str,
    *,
    max_distance: torch.Tensor | None = None,
    global_scale: float = 128.0,
) -> torch.Tensor:
    """Map raw BFS steps into the head's training unit."""

    value = _validated_distance(distance)
    selected = TargetMode(mode)
    if selected == TargetMode.RAW:
        return value
    if selected == TargetMode.GLOBAL_NORM:
        if global_scale <= 0:
            raise ValueError("global_scale must be positive")
        return value / float(global_scale)
    if selected == TargetMode.LOG1P:
        return torch.log1p(value)
    maximum = _validated_max_distance(value, max_distance)
    return torch.log1p(value) / torch.log1p(maximum)


def inverse_distance(
    transformed: torch.Tensor,
    mode: TargetMode | str,
    *,
    max_distance: torch.Tensor | None = None,
    global_scale: float = 128.0,
) -> torch.Tensor:
    """Convert a transformed score back to raw BFS-step units."""

    if not torch.is_floating_point(transformed):
        transformed = transformed.float()
    if not torch.isfinite(transformed).all():
        raise ValueError("transformed distances must be finite")
    selected = TargetMode(mode)
    if selected == TargetMode.RAW:
        return transformed
    if selected == TargetMode.GLOBAL_NORM:
        if global_scale <= 0:
            raise ValueError("global_scale must be positive")
        return transformed * float(global_scale)
    if selected == TargetMode.LOG1P:
        return torch.expm1(transformed)
    maximum = _validated_max_distance(transformed, max_distance)
    return torch.expm1(transformed * torch.log1p(maximum))


def assert_round_trip(
    mode: TargetMode | str,
    *,
    tolerance: float = 1e-5,
) -> float:
    """Run the protocol's transform/inverse numerical contract."""

    raw = torch.tensor([0.0, 1.0, 2.0, 7.0, 31.0, 128.0], dtype=torch.float64)
    maximum = torch.full_like(raw, 128.0)
    encoded = transform_distance(raw, mode, max_distance=maximum)
    decoded = inverse_distance(encoded, mode, max_distance=maximum)
    error = float((decoded - raw).abs().max())
    if error > tolerance:
        raise AssertionError(f"distance round-trip error {error} > {tolerance}")
    return error


__all__ = ["assert_round_trip", "inverse_distance", "transform_distance"]
