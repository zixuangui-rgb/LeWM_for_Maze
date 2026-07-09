"""Ice-world variant of the noisy 2D grid-world environment."""

from __future__ import annotations

import numpy as np

from hdwm.config import GridWorld2DAction, IceWorld2DConfig
from hdwm.envs.grid_world_2d import GridVirtualBorder, GridWorld2DEnv


class IceWorld2DEnv(GridWorld2DEnv):
    """2D grid-world where movement slides until a border or obstacle."""

    config: IceWorld2DConfig

    def __init__(self, config: IceWorld2DConfig, seed: int | None = None) -> None:
        super().__init__(config=config, seed=seed)

    def _sample_sequence_actions(
        self,
        batch_size: int,
        transition_count: int,
    ) -> np.ndarray:
        if transition_count < 0:
            raise ValueError("transition_count must be non-negative")
        if transition_count == 0:
            return np.empty((batch_size, 0), dtype=np.int64)

        actions = np.empty((batch_size, transition_count), dtype=np.int64)
        previous_actions = self.np_random.choice(
            np.asarray(self.DIRECTION_ACTIONS, dtype=np.int64),
            size=batch_size,
        )
        for step in range(transition_count):
            if step > 0:
                change_mask = (
                    self.np_random.random(batch_size) < self.config.p_action_turn
                )
                changed_actions = self._change_direction_ids(previous_actions)
                previous_actions = np.where(
                    change_mask,
                    changed_actions,
                    previous_actions,
                )
            stay_mask = self.np_random.random(batch_size) < self.config.p_action_stay
            actions[:, step] = np.where(
                stay_mask,
                GridWorld2DAction.STAY.value,
                previous_actions,
            )
            previous_actions = actions[:, step]
        return actions

    def _next_states(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        obstacle_mask: np.ndarray | None = None,
        virtual_border: GridVirtualBorder | None = None,
    ) -> np.ndarray:
        obstacle_mask = self._obstacle_mask if obstacle_mask is None else obstacle_mask
        if virtual_border is None:
            top, left, bottom, right = 0, 0, self.config.height, self.config.width
        else:
            top, left, bottom, right = virtual_border
        state_y = states // self.config.width
        state_x = states % self.config.width
        deltas = self._action_deltas(actions)
        next_y = state_y.copy()
        next_x = state_x.copy()
        active = actions != GridWorld2DAction.STAY.value

        while active.any():
            candidate_y = next_y + deltas[:, 0]
            candidate_x = next_x + deltas[:, 1]
            in_bounds = (
                (candidate_y >= top)
                & (candidate_y < bottom)
                & (candidate_x >= left)
                & (candidate_x < right)
            )
            obstacle_blocked = np.zeros_like(active, dtype=bool)
            obstacle_blocked[in_bounds] = obstacle_mask[
                candidate_y[in_bounds],
                candidate_x[in_bounds],
            ]
            can_move = active & in_bounds & ~obstacle_blocked
            next_y = np.where(can_move, candidate_y, next_y)
            next_x = np.where(can_move, candidate_x, next_x)
            active = can_move

        return next_y * self.config.width + next_x

    def _transition_deltas(
        self,
        previous_states: np.ndarray,
        next_states: np.ndarray,
        actual_actions: np.ndarray,
    ) -> np.ndarray:
        del actual_actions
        previous_y = previous_states // self.config.width
        previous_x = previous_states % self.config.width
        next_y = next_states // self.config.width
        next_x = next_states % self.config.width
        return np.stack((next_y - previous_y, next_x - previous_x), axis=1)

    def _step_actual_delta(
        self,
        previous_state: int,
        next_state: int,
        actual_action: GridWorld2DAction,
    ) -> tuple[int, int]:
        del actual_action
        previous_y, previous_x = divmod(previous_state, self.config.width)
        next_y, next_x = divmod(next_state, self.config.width)
        return next_y - previous_y, next_x - previous_x
