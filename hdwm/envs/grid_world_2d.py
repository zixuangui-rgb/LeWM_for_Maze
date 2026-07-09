"""Gymnasium-compatible noisy 2D grid-world environment."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding

from hdwm.config import GridNoisePlacement, GridWorld2DAction, GridWorld2DConfig
from hdwm.envs.action_utils import change_direction_ids
from hdwm.envs.ring_world import SequenceBatch

GridVirtualBorder = tuple[int, int, int, int]


class GridWorld2DEnv(gym.Env[np.ndarray, int]):
    """Noisy 2D grid-world with RGB observations and optional obstacles."""

    metadata = {"render_modes": ["ansi", "human"]}

    EMPTY_RGB = np.array([1.0, 1.0, 1.0])
    OBSTACLE_RGB = np.array([0.0, 0.0, 0.0])
    GREEN_RGB = np.array([0.0, 1.0, 0.0])
    ACTION_EFFECTS = {
        GridWorld2DAction.STAY: (0, 0),
        GridWorld2DAction.UP: (-1, 0),
        GridWorld2DAction.DOWN: (1, 0),
        GridWorld2DAction.LEFT: (0, -1),
        GridWorld2DAction.RIGHT: (0, 1),
    }
    DIRECTION_ACTIONS = (
        GridWorld2DAction.UP.value,
        GridWorld2DAction.DOWN.value,
        GridWorld2DAction.LEFT.value,
        GridWorld2DAction.RIGHT.value,
    )

    def __init__(self, config: GridWorld2DConfig, seed: int | None = None) -> None:
        self.config = config
        self.render_mode = config.render_mode
        self.action_space = spaces.Discrete(config.action_vocab_size)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(config.height, config.width, config.observation_channels),
            dtype=config.numpy_observation_dtype,
        )
        self._np_random, _ = seeding.np_random(seed)
        self._obstacle_mask = self._make_obstacle_mask()
        self._empty_positions = self._empty_positions_for_mask(self._obstacle_mask)
        if self._empty_positions.size == 0:
            raise ValueError("grid must contain at least one empty cell")
        self._state: int | None = None
        self._last_observation: np.ndarray | None = None
        self._last_noise_mask: np.ndarray | None = None
        self._elapsed_steps = 0

    @property
    def state(self) -> int:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return `(observation, info)`."""

        super().reset(seed=seed)
        options = options or {}
        self._state = self._make_start_state(options.get("start_state"))
        self._elapsed_steps = 0
        observation, noise_mask = self._observe_with_noise(np.array([self.state]))
        self._last_observation = observation[0]
        self._last_noise_mask = noise_mask[0]
        return self._last_observation.copy(), {"state": self.state}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action in {0, 1, 2, 3, 4} with no-op transition noise."""

        action_meaning = self._decode_action(action)

        noop = bool(self.np_random.random() < self.config.p_noop)
        actual_action = GridWorld2DAction.STAY if noop else action_meaning
        previous_state = self.state
        self._state = self._next_state(self.state, actual_action)
        self._elapsed_steps += 1

        terminated = False
        truncated = (
            self.config.max_episode_steps is not None
            and self._elapsed_steps >= self.config.max_episode_steps
        )
        info = {
            "state": self.state,
            "action": action_meaning.value,
            "action_name": action_meaning.name.lower(),
            "actual_action": actual_action.value,
            "actual_action_name": actual_action.name.lower(),
            "actual_delta": self._step_actual_delta(
                previous_state=previous_state,
                next_state=self.state,
                actual_action=actual_action,
            ),
            "noop": noop,
        }
        observation, noise_mask = self._observe_with_noise(np.array([self.state]))
        self._last_observation = observation[0]
        self._last_noise_mask = noise_mask[0]
        return self._last_observation.copy(), 0.0, terminated, truncated, info

    def render(
        self,
        batch: SequenceBatch | None = None,
        batch_index: int = 0,
    ) -> str | None:
        """Render the latest observation or one sampled sequence."""

        if batch is not None:
            frame = self._render_sequence(batch=batch, batch_index=batch_index)
            if self.render_mode == "human":
                print(frame)
                return None
            return frame
        if self._last_observation is None:
            raise RuntimeError("environment has not been reset")
        if self._last_noise_mask is None:
            raise RuntimeError("environment has not recorded observation noise")

        cells = self._render_grid(
            noise_mask=torch.as_tensor(self._last_noise_mask),
            obstacle_mask=torch.as_tensor(self._obstacle_mask),
            state=self.state,
        )
        frame = f"state={self.state}\n" + "\n".join(cells)
        if self.render_mode == "human":
            print(frame)
            return None
        return frame

    @classmethod
    def _render_sequence(cls, batch: SequenceBatch, batch_index: int = 0) -> str:
        """Render one sampled sequence as timestep-indexed 2D text grids."""

        if batch.observations.ndim != 5:
            raise ValueError(
                f"expected RGB observations rank 5, got {batch.observations.ndim}"
            )
        if batch.states.ndim != 2:
            raise ValueError(f"expected states rank 2, got {batch.states.ndim}")
        if batch.noise_masks.shape != batch.observations.shape[:-1]:
            raise ValueError(
                f"expected noise_masks shape {tuple(batch.observations.shape[:-1])}, "
                f"got {tuple(batch.noise_masks.shape)}"
            )
        if batch.obstacle_masks is None:
            raise ValueError("2D grid rendering requires obstacle_masks")
        if batch.obstacle_masks.shape != batch.noise_masks.shape:
            raise ValueError(
                f"expected obstacle_masks shape {tuple(batch.noise_masks.shape)}, "
                f"got {tuple(batch.obstacle_masks.shape)}"
            )

        batch_size, sequence_length, height, width, _ = batch.observations.shape
        if not 0 <= batch_index < batch_size:
            raise ValueError(f"batch_index must be in [0, {batch_size - 1}]")

        states = batch.states[batch_index].detach().long().cpu()
        noise_masks = batch.noise_masks[batch_index].detach().bool().cpu()
        obstacle_masks = batch.obstacle_masks[batch_index].detach().bool().cpu()
        rows: list[str] = []
        for step in range(sequence_length):
            if step > 0:
                rows.append("")
            rows.append(f"t={step:02d} state={int(states[step]):02d}")
            rows.extend(
                cls._render_grid(
                    noise_mask=noise_masks[step],
                    obstacle_mask=obstacle_masks[step],
                    state=int(states[step]),
                )
            )
        expected_max_state = height * width - 1
        if ((states < 0) | (states > expected_max_state)).any():
            raise ValueError(f"states must be in [0, {expected_max_state}]")
        return "\n".join(rows)

    @staticmethod
    def _render_grid(
        noise_mask: torch.Tensor,
        obstacle_mask: torch.Tensor,
        state: int,
    ) -> list[str]:
        if noise_mask.ndim != 2:
            raise ValueError(f"expected noise_mask rank 2, got {noise_mask.ndim}")
        if obstacle_mask.shape != noise_mask.shape:
            raise ValueError(
                f"expected obstacle_mask shape {tuple(noise_mask.shape)}, "
                f"got {tuple(obstacle_mask.shape)}"
            )
        height, width = noise_mask.shape
        if not 0 <= state < height * width:
            raise ValueError(f"state must be in [0, {height * width - 1}]")

        noise_mask = noise_mask.detach().bool().cpu()
        obstacle_mask = obstacle_mask.detach().bool().cpu()
        state_y, state_x = divmod(state, width)
        rows = []
        for y in range(height):
            cells = []
            for x in range(width):
                if y == state_y and x == state_x:
                    cells.append("g")
                elif bool(noise_mask[y, x]):
                    cells.append("g")
                elif bool(obstacle_mask[y, x]):
                    cells.append("X")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return rows

    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        start_state: int | torch.Tensor | None = None,
        virtual_border: GridVirtualBorder | None = None,
    ) -> SequenceBatch:
        """Sample a batch of trajectories as `(I_1:T, a_1:T-1)`."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if virtual_border is not None:
            self.config._validate_virtual_border(virtual_border, name="virtual_border")
        obstacle_mask = (
            self._make_obstacle_mask()
            if self.config.resample_obstacles_per_sequence
            else self._obstacle_mask.copy()
        )
        states = self._make_start_states(
            batch_size=batch_size,
            start_state=start_state,
            obstacle_mask=obstacle_mask,
            virtual_border=virtual_border,
        )
        observation, noise_mask = self._observe_with_noise(
            states,
            obstacle_mask=obstacle_mask,
        )
        observations = [observation]
        noise_masks = [noise_mask]
        obstacle_masks = [np.broadcast_to(obstacle_mask, noise_mask.shape).copy()]
        state_history = [states.copy()]
        actions = self._sample_sequence_actions(
            batch_size=batch_size,
            transition_count=sequence_length - 1,
        )
        actual_deltas = []
        noop_masks = []

        for step in range(sequence_length - 1):
            actions[:, step] = self._resample_virtual_border_actions(
                states=states,
                actions=actions[:, step],
                virtual_border=virtual_border,
            )
            noop_mask = self.np_random.random(batch_size) < self.config.p_noop
            actual_actions = np.where(
                noop_mask,
                GridWorld2DAction.STAY.value,
                actions[:, step],
            )
            previous_states = states
            states = self._next_states(
                states,
                actual_actions,
                obstacle_mask=obstacle_mask,
                virtual_border=virtual_border,
            )
            observation, noise_mask = self._observe_with_noise(
                states,
                obstacle_mask=obstacle_mask,
            )
            observations.append(observation)
            noise_masks.append(noise_mask)
            obstacle_masks.append(
                np.broadcast_to(obstacle_mask, noise_mask.shape).copy()
            )
            state_history.append(states.copy())
            actual_deltas.append(
                self._transition_deltas(
                    previous_states=previous_states,
                    next_states=states,
                    actual_actions=actual_actions,
                )
            )
            noop_masks.append(noop_mask)

        return SequenceBatch(
            observations=torch.as_tensor(np.stack(observations, axis=1)),
            states=torch.as_tensor(np.stack(state_history, axis=1), dtype=torch.long),
            noise_masks=torch.as_tensor(
                np.stack(noise_masks, axis=1), dtype=torch.bool
            ),
            actions=torch.as_tensor(actions, dtype=torch.long),
            actual_deltas=torch.as_tensor(
                np.stack(actual_deltas, axis=1), dtype=torch.long
            )
            if actual_deltas
            else torch.empty((batch_size, 0, 2), dtype=torch.long),
            noop_masks=torch.as_tensor(np.stack(noop_masks, axis=1), dtype=torch.bool)
            if noop_masks
            else torch.empty((batch_size, 0), dtype=torch.bool),
            obstacle_masks=torch.as_tensor(
                np.stack(obstacle_masks, axis=1),
                dtype=torch.bool,
            ),
        )

    def virtual_border_for_split(
        self,
        split: Literal["train", "validation"],
    ) -> GridVirtualBorder | None:
        """Return the configured sampling box for a data split."""

        if split == "train":
            return self.config.train_virtual_border
        if split == "validation":
            return self.config.validation_virtual_border
        raise ValueError(f"unsupported split: {split}")

    def _make_obstacle_mask(self) -> np.ndarray:
        mask = (
            self.np_random.random((self.config.height, self.config.width))
            < self.config.p_obstacle
        )
        if self.config.obstacles:
            flat = mask.reshape(-1)
            flat[np.array(self.config.obstacles, dtype=np.int64)] = True
        if mask.all():
            mask.reshape(-1)[0] = False
        return mask

    @staticmethod
    def _empty_positions_for_mask(
        obstacle_mask: np.ndarray,
        virtual_border: GridVirtualBorder | None = None,
    ) -> np.ndarray:
        empty_mask = ~obstacle_mask
        if virtual_border is not None:
            top, left, bottom, right = virtual_border
            border_mask = np.zeros_like(empty_mask, dtype=np.bool_)
            border_mask[top:bottom, left:right] = True
            empty_mask = empty_mask & border_mask
        return np.flatnonzero(empty_mask.reshape(-1))

    def _observe_with_noise(
        self,
        states: np.ndarray,
        obstacle_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        obstacle_mask = self._obstacle_mask if obstacle_mask is None else obstacle_mask
        batch_size = len(states)
        observations = np.ones(
            (
                batch_size,
                self.config.height,
                self.config.width,
                self.config.observation_channels,
            ),
            dtype=self.config.numpy_observation_dtype,
        )
        observations[:, obstacle_mask] = self.OBSTACLE_RGB

        state_y = states // self.config.width
        state_x = states % self.config.width
        observations[np.arange(batch_size), state_y, state_x] = self.GREEN_RGB

        candidate_mask = np.broadcast_to(
            ~obstacle_mask,
            (batch_size, self.config.height, self.config.width),
        ).copy()
        if self.config.noise_placement == GridNoisePlacement.EMPTY_AND_OBSTACLE:
            candidate_mask[:] = True
        candidate_mask[np.arange(batch_size), state_y, state_x] = False
        noise_masks = (
            self.np_random.random(candidate_mask.shape) < self.config.p_noise
        ) & candidate_mask
        observations[noise_masks] = self.GREEN_RGB
        return observations.astype(self.config.numpy_observation_dtype), noise_masks

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
        directions = self.np_random.choice(
            np.asarray(self.DIRECTION_ACTIONS, dtype=np.int64),
            size=batch_size,
        )
        for step in range(transition_count):
            if step > 0:
                turn_mask = (
                    self.np_random.random(batch_size) < self.config.p_action_turn
                )
                directions = np.where(
                    turn_mask,
                    self._change_direction_ids(directions),
                    directions,
                )
            stay_mask = self.np_random.random(batch_size) < self.config.p_action_stay
            actions[:, step] = np.where(
                stay_mask,
                GridWorld2DAction.STAY.value,
                directions,
            )
        return actions

    def _resample_virtual_border_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        virtual_border: GridVirtualBorder | None,
    ) -> np.ndarray:
        if virtual_border is None or self.config.virtual_border_pass_through == 0.0:
            return actions

        pass_through_mask = self._virtual_border_pass_through_mask(
            states=states,
            actions=actions,
            virtual_border=virtual_border,
        )
        resample_mask = pass_through_mask & (
            self.np_random.random(len(actions))
            < self.config.virtual_border_pass_through
        )
        if not resample_mask.any():
            return actions

        resampled_actions = actions.copy()
        for index in np.flatnonzero(resample_mask):
            candidates = self._actions_within_virtual_border(
                state=int(states[index]),
                virtual_border=virtual_border,
            )
            resampled_actions[index] = self.np_random.choice(candidates)
        return resampled_actions

    def _virtual_border_pass_through_mask(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        virtual_border: GridVirtualBorder,
    ) -> np.ndarray:
        top, left, bottom, right = virtual_border
        state_y = states // self.config.width
        state_x = states % self.config.width
        deltas = self._action_deltas(actions)
        candidate_y = state_y + deltas[:, 0]
        candidate_x = state_x + deltas[:, 1]
        return (
            (candidate_y < top)
            | (candidate_y >= bottom)
            | (candidate_x < left)
            | (candidate_x >= right)
        )

    def _actions_within_virtual_border(
        self,
        state: int,
        virtual_border: GridVirtualBorder,
    ) -> np.ndarray:
        candidate_actions = np.arange(self.config.num_actions, dtype=np.int64)
        states = np.full(len(candidate_actions), state, dtype=np.int64)
        valid_mask = ~self._virtual_border_pass_through_mask(
            states=states,
            actions=candidate_actions,
            virtual_border=virtual_border,
        )
        valid_actions = candidate_actions[valid_mask]
        if valid_actions.size == 0:
            raise ValueError("virtual border must allow at least one action")
        return valid_actions

    def _next_state(self, state: int, action: GridWorld2DAction) -> int:
        return int(self._next_states(np.array([state]), np.array([action.value]))[0])

    def _next_states(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        obstacle_mask: np.ndarray | None = None,
        virtual_border: GridVirtualBorder | None = None,
    ) -> np.ndarray:
        obstacle_mask = self._obstacle_mask if obstacle_mask is None else obstacle_mask
        state_y = states // self.config.width
        state_x = states % self.config.width
        deltas = self._action_deltas(actions)
        if virtual_border is None:
            top, left, bottom, right = 0, 0, self.config.height, self.config.width
        else:
            top, left, bottom, right = virtual_border
        next_y = np.clip(state_y + deltas[:, 0], top, bottom - 1)
        next_x = np.clip(state_x + deltas[:, 1], left, right - 1)
        blocked = obstacle_mask[next_y, next_x]
        next_y = np.where(blocked, state_y, next_y)
        next_x = np.where(blocked, state_x, next_x)
        return next_y * self.config.width + next_x

    def _action_deltas(self, actions: np.ndarray) -> np.ndarray:
        deltas = np.zeros((len(actions), 2), dtype=np.int64)
        for action, delta in self.ACTION_EFFECTS.items():
            deltas[actions == action.value] = delta
        return deltas

    def _transition_deltas(
        self,
        previous_states: np.ndarray,
        next_states: np.ndarray,
        actual_actions: np.ndarray,
    ) -> np.ndarray:
        del previous_states, next_states
        return self._action_deltas(actual_actions)

    def _step_actual_delta(
        self,
        previous_state: int,
        next_state: int,
        actual_action: GridWorld2DAction,
    ) -> tuple[int, int]:
        del previous_state, next_state
        return self.ACTION_EFFECTS[actual_action]

    def _change_direction_ids(self, actions: np.ndarray) -> np.ndarray:
        return change_direction_ids(
            rng=self.np_random,
            current_directions=actions,
            direction_ids=self.DIRECTION_ACTIONS,
        )

    def _decode_action(self, action: int) -> GridWorld2DAction:
        try:
            return GridWorld2DAction(int(action))
        except ValueError as error:
            raise ValueError("action must be one of {0, 1, 2, 3, 4}") from error

    def _make_start_state(
        self,
        start_state: Any,
        obstacle_mask: np.ndarray | None = None,
        virtual_border: GridVirtualBorder | None = None,
    ) -> int:
        obstacle_mask = self._obstacle_mask if obstacle_mask is None else obstacle_mask
        if start_state is None:
            empty_positions = self._empty_positions_for_mask(
                obstacle_mask,
                virtual_border=virtual_border,
            )
            if empty_positions.size == 0:
                raise ValueError("virtual border must contain at least one empty cell")
            return int(self.np_random.choice(empty_positions))
        state = int(start_state)
        if not 0 <= state < self.config.observation_size:
            raise ValueError(
                f"start_state must be in [0, {self.config.observation_size - 1}]"
            )
        if obstacle_mask.reshape(-1)[state]:
            raise ValueError("start_state must be an empty cell")
        if virtual_border is not None and not self._state_is_within_virtual_border(
            state,
            virtual_border,
        ):
            raise ValueError("start_state must be inside virtual_border")
        return state

    def _make_start_states(
        self,
        batch_size: int,
        start_state: int | torch.Tensor | None,
        obstacle_mask: np.ndarray | None = None,
        virtual_border: GridVirtualBorder | None = None,
    ) -> np.ndarray:
        obstacle_mask = self._obstacle_mask if obstacle_mask is None else obstacle_mask
        if start_state is None:
            empty_positions = self._empty_positions_for_mask(
                obstacle_mask,
                virtual_border=virtual_border,
            )
            if empty_positions.size == 0:
                raise ValueError("virtual border must contain at least one empty cell")
            return self.np_random.choice(empty_positions, size=batch_size)

        if isinstance(start_state, int):
            return np.full(
                batch_size,
                self._make_start_state(
                    start_state,
                    obstacle_mask=obstacle_mask,
                    virtual_border=virtual_border,
                ),
                dtype=np.int64,
            )

        states = start_state.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if states.shape != (batch_size,):
            raise ValueError(
                f"start_state shape {tuple(states.shape)} does not match "
                f"batch_size {batch_size}"
            )
        if ((states < 0) | (states >= self.config.observation_size)).any():
            raise ValueError(
                f"start_state values must be in [0, {self.config.observation_size - 1}]"
            )
        if obstacle_mask.reshape(-1)[states].any():
            raise ValueError("start_state values must be empty cells")
        if virtual_border is not None and not all(
            self._state_is_within_virtual_border(int(state), virtual_border)
            for state in states
        ):
            raise ValueError("start_state values must be inside virtual_border")
        return states

    def _state_is_within_virtual_border(
        self,
        state: int,
        virtual_border: GridVirtualBorder,
    ) -> bool:
        top, left, bottom, right = virtual_border
        state_y, state_x = divmod(state, self.config.width)
        return top <= state_y < bottom and left <= state_x < right
