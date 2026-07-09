"""Shared action sampling helpers for toy environments."""

from __future__ import annotations

import numpy as np


def change_direction_ids(
    rng: np.random.Generator,
    current_directions: np.ndarray,
    direction_ids: tuple[int, ...],
) -> np.ndarray:
    """Change each direction to a different valid direction id."""

    if len(direction_ids) < 2:
        raise ValueError("direction_ids must contain at least two directions")
    choices = np.asarray(direction_ids, dtype=np.int64)
    changed = np.empty_like(current_directions, dtype=np.int64)
    for index, direction in enumerate(current_directions):
        candidates = choices[choices != int(direction)]
        changed[index] = rng.choice(candidates)
    return changed
