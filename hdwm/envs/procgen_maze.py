"""Gymnasium-compatible procgen-maze environment with DFS-generated mazes."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding

from hdwm.config import GridNoisePlacement, GridWorld2DAction, ProcgenMazeConfig
from hdwm.envs.action_utils import change_direction_ids
from hdwm.envs.ring_world import SequenceBatch

GridVirtualBorder = tuple[int, int, int, int]


class ProcgenMazeEnv(gym.Env[np.ndarray, int]):
    """2D maze environment with agent and goal on a DFS-generated maze grid."""

    metadata = {"render_modes": ["ansi", "human"]}

    # One-hot channel indices for grid observations.
    CH_EMPTY = 0
    CH_WALL = 1
    CH_AGENT = 2
    CH_GOAL = 3
    CH_NOISE = 4

    # Legacy RGB constants for backward-compatible rendering.
    EMPTY_RGB = np.array([1.0, 1.0, 1.0])
    WALL_RGB = np.array([0.0, 0.0, 0.0])
    AGENT_RGB = np.array([0.0, 1.0, 0.0])
    GOAL_RGB = np.array([0.0, 0.0, 1.0])

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

    def __init__(self, config: ProcgenMazeConfig, seed: int | None = None) -> None:
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
        # Separate RNG for maze topology: topology_seed controls which maze layout,
        # independent of trajectory/episode randomness.
        maze_seed = seed if config.topology_seed is None else config.topology_seed
        self._maze_rng = np.random.default_rng(maze_seed)
        self._maze_mask = self._make_maze_mask()
        self._goal_position: int = self._sample_goal()
        self._empty_positions = self._empty_positions_for_mask(self._maze_mask)
        if self._empty_positions.size == 0:
            raise ValueError("maze must contain at least one empty cell")
        self._state: int | None = None
        self._last_observation: np.ndarray | None = None
        self._last_noise_mask: np.ndarray | None = None
        self._elapsed_steps = 0

    @property
    def state(self) -> int:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    @property
    def goal_position(self) -> int:
        return self._goal_position

    # ── Maze generation ──────────────────────────────────────────────────────

    def _make_maze_mask(self) -> np.ndarray:
        """Generate a perfect maze using randomized iterative DFS.

        Returns a boolean mask of shape (height, width) where True cells are
        walls and False cells are passages. The maze uses a cell-based layout
        where passage cells form a grid with walls separating them.
        """
        height = self.config.height
        width = self.config.width

        # Start with all walls.
        mask = np.ones((height, width), dtype=bool)

        # Mark passage cells: cells at (odd_row, odd_col) where both are < dim.
        passage_rows = list(range(1, height, 2))
        passage_cols = list(range(1, width, 2))
        for r in passage_rows:
            for c in passage_cols:
                mask[r, c] = False

        if not passage_rows or not passage_cols:
            return mask

        # Iterative randomized DFS to carve passages between passage cells.
        n_rows = len(passage_rows)
        n_cols = len(passage_cols)
        visited = np.zeros((n_rows, n_cols), dtype=bool)
        stack: list[tuple[int, int]] = [
            (self._maze_rng.integers(0, n_rows), self._maze_rng.integers(0, n_cols)),
        ]
        visited[stack[0]] = True

        # Direction offsets in (grid_row, grid_col) for wall cells between
        # two passage cells.
        neighbors = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            cr, cc = stack[-1]
            # Collect unvisited neighbouring passage cells.
            dirs = neighbors.copy()
            self._maze_rng.shuffle(dirs)
            carved = False
            for dr, dc in dirs:
                nr, nc = cr + dr // 2, cc + dc // 2
                if 0 <= nr < n_rows and 0 <= nc < n_cols and not visited[nr, nc]:
                    # Remove the wall between (cr, cc) and (nr, nc).
                    wall_r = passage_rows[cr] + dr // 2
                    wall_c = passage_cols[cc] + dc // 2
                    mask[wall_r, wall_c] = False
                    visited[nr, nc] = True
                    stack.append((nr, nc))
                    carved = True
                    break
            if not carved:
                stack.pop()

        # Apply fixed walls from config (overwrite as obstacles).
        for pos in self.config.walls:
            row, col = divmod(pos, width)
            mask[row, col] = True

        # Ensure at least one empty cell remains.
        if mask.all():
            mask[1, 1] = False

        return mask

    def _sample_goal(self) -> int:
        """Sample a random goal position from empty (passage) cells."""
        empty_positions = self._empty_positions_for_mask(self._maze_mask)
        return int(self._maze_rng.choice(empty_positions))

    @staticmethod
    def _empty_positions_for_mask(
        maze_mask: np.ndarray,
        virtual_border: GridVirtualBorder | None = None,
    ) -> np.ndarray:
        """Return flat indices of all non-wall cells, optionally within a border."""
        empty_mask = ~maze_mask
        if virtual_border is not None:
            top, left, bottom, right = virtual_border
            border_mask = np.zeros_like(empty_mask, dtype=bool)
            border_mask[top:bottom, left:right] = True
            empty_mask = empty_mask & border_mask
        return np.flatnonzero(empty_mask.reshape(-1))

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return (observation, info)."""
        super().reset(seed=seed)
        options = options or {}
        self._state = self._make_start_state(options.get("start_state"))
        self._elapsed_steps = 0
        observation, noise_mask = self._observe_with_noise(np.array([self.state]))
        self._last_observation = observation[0]
        self._last_noise_mask = noise_mask[0]
        return self._last_observation.copy(), {
            "state": self.state,
            "goal": self._goal_position,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action and return (obs, reward, terminated, truncated, info)."""
        action_meaning = self._decode_action(action)

        noop = bool(self.np_random.random() < self.config.p_noop)
        actual_action = GridWorld2DAction.STAY if noop else action_meaning
        self._state = self._next_state(self.state, actual_action)
        self._elapsed_steps += 1

        terminated = self.state == self._goal_position
        truncated = (
            self.config.max_episode_steps is not None
            and self._elapsed_steps >= self.config.max_episode_steps
        )
        reward = 1.0 if terminated else 0.0

        info = {
            "state": self.state,
            "goal": self._goal_position,
            "action": action_meaning.value,
            "action_name": action_meaning.name.lower(),
            "actual_action": actual_action.value,
            "actual_action_name": actual_action.name.lower(),
            "noop": noop,
            "reward": reward,
        }
        observation, noise_mask = self._observe_with_noise(np.array([self.state]))
        self._last_observation = observation[0]
        self._last_noise_mask = noise_mask[0]
        return self._last_observation.copy(), reward, terminated, truncated, info

    def render(
        self,
        batch: SequenceBatch | None = None,
        batch_index: int = 0,
    ) -> str | None:
        """Render the latest observation or one sampled sequence as ANSI text."""
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
            maze_mask=torch.as_tensor(self._maze_mask),
            state=self.state,
            goal=self._goal_position,
        )
        frame = f"state={self.state} goal={self._goal_position}\n" + "\n".join(cells)
        if self.render_mode == "human":
            print(frame)
            return None
        return frame

    # ── Sequence sampling ────────────────────────────────────────────────────

    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        start_state: int | torch.Tensor | None = None,
        virtual_border: GridVirtualBorder | None = None,
    ) -> SequenceBatch:
        """Sample a batch of trajectories as (I_1:T, a_1:T-1).

        If resample_maze_per_sequence is enabled, each call generates a fresh
        maze and goal.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if virtual_border is not None:
            self._validate_virtual_border(virtual_border, name="virtual_border")

        # Per-sequence maze resampling.
        maze_mask = (
            self._make_maze_mask()
            if self.config.resample_maze_per_sequence
            else self._maze_mask.copy()
        )
        goal_position = (
            self._sample_goal_for_mask(maze_mask, virtual_border)
            if self.config.resample_maze_per_sequence
            else self._goal_position
        )

        states = self._make_start_states(
            batch_size=batch_size,
            start_state=start_state,
            maze_mask=maze_mask,
            virtual_border=virtual_border,
            goal_position=goal_position,
        )
        observation, noise_mask = self._observe_with_noise(
            states,
            maze_mask=maze_mask,
            goal_position=goal_position,
        )
        observations = [observation]
        noise_masks = [noise_mask]
        maze_masks_batch = [np.broadcast_to(maze_mask, noise_mask.shape).copy()]
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
                maze_mask=maze_mask,
                virtual_border=virtual_border,
            )
            observation, noise_mask = self._observe_with_noise(
                states,
                maze_mask=maze_mask,
                goal_position=goal_position,
            )
            observations.append(observation)
            noise_masks.append(noise_mask)
            maze_masks_batch.append(np.broadcast_to(maze_mask, noise_mask.shape).copy())
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
                np.stack(maze_masks_batch, axis=1),
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

    # ── Observation helpers ──────────────────────────────────────────────────

    def _observe_with_noise(
        self,
        states: np.ndarray,
        maze_mask: np.ndarray | None = None,
        goal_position: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render grid observations with distractor noise.

        Returns (observations, noise_masks) where observations has shape
        (batch, H, W, C) with one-hot encoded grid cells.
        Channel order: EMPTY=0, WALL=1, AGENT=2, GOAL=3, NOISE=4.
        """
        maze_mask = self._maze_mask if maze_mask is None else maze_mask
        goal_position = self._goal_position if goal_position is None else goal_position
        batch_size = len(states)
        num_channels = self.config.observation_channels

        # Start with all-zeros, then set EMPTY channel to 1 everywhere.
        observations = np.zeros(
            (batch_size, self.config.height, self.config.width, num_channels),
            dtype=self.config.numpy_observation_dtype,
        )
        observations[..., self.CH_EMPTY] = 1.0

        # Paint walls: CH_WALL = 1, CH_EMPTY = 0.
        observations[:, maze_mask, self.CH_EMPTY] = 0.0
        observations[:, maze_mask, self.CH_WALL] = 1.0

        # Paint goal: CH_GOAL = 1, CH_EMPTY = 0.
        goal_y, goal_x = divmod(goal_position, self.config.width)
        observations[np.arange(batch_size), goal_y, goal_x, self.CH_EMPTY] = 0.0
        observations[np.arange(batch_size), goal_y, goal_x, self.CH_GOAL] = 1.0

        # Paint agent: CH_AGENT = 1, CH_EMPTY = 0.
        state_y = states // self.config.width
        state_x = states % self.config.width
        observations[np.arange(batch_size), state_y, state_x, self.CH_EMPTY] = 0.0
        observations[np.arange(batch_size), state_y, state_x, self.CH_AGENT] = 1.0

        # Candidate cells for noise: non-wall, non-agent, non-goal.
        candidate_mask = np.broadcast_to(
            ~maze_mask,
            (batch_size, self.config.height, self.config.width),
        ).copy()
        candidate_mask[np.arange(batch_size), state_y, state_x] = False
        candidate_mask[np.arange(batch_size), goal_y, goal_x] = False
        if self.config.noise_placement == GridNoisePlacement.EMPTY_AND_OBSTACLE:
            candidate_mask[:] = True
            candidate_mask[np.arange(batch_size), state_y, state_x] = False

        noise_masks = (
            self.np_random.random(candidate_mask.shape) < self.config.p_noise
        ) & candidate_mask
        # Paint noise: CH_NOISE = 1, clear other channels.
        observations[noise_masks, self.CH_EMPTY] = 0.0
        observations[noise_masks, self.CH_WALL] = 0.0
        observations[noise_masks, self.CH_NOISE] = 1.0

        return observations.astype(self.config.numpy_observation_dtype), noise_masks

    # ── Action helpers ───────────────────────────────────────────────────────

    def _sample_sequence_actions(
        self,
        batch_size: int,
        transition_count: int,
    ) -> np.ndarray:
        """Sample persistent-direction action sequences."""
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
        """Resample actions that would exit the virtual border."""
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

    # ── State transition ─────────────────────────────────────────────────────

    def _next_state(self, state: int, action: GridWorld2DAction) -> int:
        return int(self._next_states(np.array([state]), np.array([action.value]))[0])

    def _next_states(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        maze_mask: np.ndarray | None = None,
        virtual_border: GridVirtualBorder | None = None,
    ) -> np.ndarray:
        maze_mask = self._maze_mask if maze_mask is None else maze_mask
        state_y = states // self.config.width
        state_x = states % self.config.width
        deltas = self._action_deltas(actions)
        if virtual_border is None:
            top, left, bottom, right = 0, 0, self.config.height, self.config.width
        else:
            top, left, bottom, right = virtual_border
        next_y = np.clip(state_y + deltas[:, 0], top, bottom - 1)
        next_x = np.clip(state_x + deltas[:, 1], left, right - 1)
        blocked = maze_mask[next_y, next_x]
        next_y = np.where(blocked, state_y, next_y)
        next_x = np.where(blocked, state_x, next_x)
        return next_y * self.config.width + next_x

    # ── Start state helpers ──────────────────────────────────────────────────

    def _make_start_state(
        self,
        start_state: Any,
        maze_mask: np.ndarray | None = None,
        virtual_border: GridVirtualBorder | None = None,
        goal_position: int | None = None,
    ) -> int:
        maze_mask = self._maze_mask if maze_mask is None else maze_mask
        goal = goal_position if goal_position is not None else self._goal_position
        if start_state is None:
            empty_positions = self._empty_positions_for_mask(
                maze_mask,
                virtual_border=virtual_border,
            )
            # Exclude goal position from start states.
            empty_positions = empty_positions[empty_positions != goal]
            if empty_positions.size == 0:
                raise ValueError(
                    "virtual border must contain at least one empty non-goal cell"
                )
            return int(self.np_random.choice(empty_positions))
        state = int(start_state)
        if not 0 <= state < self.config.observation_size:
            raise ValueError(
                f"start_state must be in [0, {self.config.observation_size - 1}]"
            )
        if maze_mask.reshape(-1)[state]:
            raise ValueError("start_state must be an empty cell")
        if state == goal:
            raise ValueError("start_state must not be the goal position")
        if virtual_border is not None and not self._state_is_within_virtual_border(
            state, virtual_border
        ):
            raise ValueError("start_state must be inside virtual_border")
        return state

    def _make_start_states(
        self,
        batch_size: int,
        start_state: int | torch.Tensor | None,
        maze_mask: np.ndarray | None = None,
        virtual_border: GridVirtualBorder | None = None,
        goal_position: int | None = None,
    ) -> np.ndarray:
        maze_mask = self._maze_mask if maze_mask is None else maze_mask
        goal = goal_position if goal_position is not None else self._goal_position
        if start_state is None:
            empty_positions = self._empty_positions_for_mask(
                maze_mask,
                virtual_border=virtual_border,
            )
            empty_positions = empty_positions[empty_positions != goal]
            if empty_positions.size == 0:
                raise ValueError(
                    "virtual border must contain at least one empty non-goal cell"
                )
            return self.np_random.choice(empty_positions, size=batch_size)

        if isinstance(start_state, int):
            return np.full(
                batch_size,
                self._make_start_state(
                    start_state,
                    maze_mask=maze_mask,
                    virtual_border=virtual_border,
                    goal_position=goal,
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
        if maze_mask.reshape(-1)[states].any():
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

    @staticmethod
    def _validate_virtual_border(
        border: GridVirtualBorder,
        name: str = "virtual_border",
    ) -> None:
        if border is None:
            raise ValueError(f"{name} must be set")
        top, left, bottom, right = border
        if not (top < bottom and left < right):
            raise ValueError(f"{name} must satisfy top < bottom and left < right")

    def _sample_goal_for_mask(
        self,
        maze_mask: np.ndarray,
        virtual_border: GridVirtualBorder | None = None,
    ) -> int:
        empty_positions = self._empty_positions_for_mask(
            maze_mask,
            virtual_border=virtual_border,
        )
        if empty_positions.size == 0:
            raise ValueError("maze must contain at least one empty cell for goal")
        return int(self._maze_rng.choice(empty_positions))

    # ── Rendering helpers ────────────────────────────────────────────────────

    @staticmethod
    def _render_grid(
        noise_mask: torch.Tensor,
        maze_mask: torch.Tensor,
        state: int,
        goal: int,
    ) -> list[str]:
        """Render a single grid as a list of ANSI row strings."""
        if noise_mask.ndim != 2:
            raise ValueError(f"expected noise_mask rank 2, got {noise_mask.ndim}")
        if maze_mask.shape != noise_mask.shape:
            raise ValueError(
                f"expected maze_mask shape {tuple(noise_mask.shape)}, "
                f"got {tuple(maze_mask.shape)}"
            )
        height, width = noise_mask.shape
        if not 0 <= state < height * width:
            raise ValueError(f"state must be in [0, {height * width - 1}]")
        if not 0 <= goal < height * width:
            raise ValueError(f"goal must be in [0, {height * width - 1}]")

        noise_mask = noise_mask.detach().bool().cpu()
        maze_mask = maze_mask.detach().bool().cpu()
        state_y, state_x = divmod(state, width)
        goal_y, goal_x = divmod(goal, width)
        rows = []
        for y in range(height):
            cells = []
            for x in range(width):
                if y == state_y and x == state_x:
                    cells.append("AG")
                elif y == goal_y and x == goal_x:
                    cells.append("◆◆")
                elif bool(maze_mask[y, x]):
                    cells.append("██")
                elif bool(noise_mask[y, x]):
                    cells.append("+ ")
                else:
                    cells.append("  ")
            rows.append("".join(cells))
        return rows

    @classmethod
    def _render_sequence(cls, batch: SequenceBatch, batch_index: int = 0) -> str:
        """Render one sampled sequence as timestep-indexed 2D text grids."""
        if batch.observations.ndim != 5:
            raise ValueError(
                f"expected observations rank 5, got {batch.observations.ndim}"
            )
        if batch.states.ndim != 2:
            raise ValueError(f"expected states rank 2, got {batch.states.ndim}")
        if batch.noise_masks.shape != batch.observations.shape[:-1]:
            raise ValueError(
                f"expected noise_masks shape {tuple(batch.observations.shape[:-1])}, "
                f"got {tuple(batch.noise_masks.shape)}"
            )
        if batch.obstacle_masks is None:
            raise ValueError("maze rendering requires obstacle_masks")
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
                    maze_mask=obstacle_masks[step],
                    state=int(states[step]),
                    goal=0,  # Goal position not stored per-timestep in batch
                )
            )
        expected_max_state = height * width - 1
        if ((states < 0) | (states > expected_max_state)).any():
            raise ValueError(f"states must be in [0, {expected_max_state}]")
        return "\n".join(rows)
