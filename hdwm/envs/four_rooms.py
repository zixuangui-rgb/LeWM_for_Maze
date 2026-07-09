"""FourRooms environment: 4 rooms connected by narrow doorways."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding

from hdwm.config import FourRoomsConfig, GridNoisePlacement, GridWorld2DAction
from hdwm.envs.action_utils import change_direction_ids
from hdwm.envs.ring_world import SequenceBatch


class FourRoomsEnv(gym.Env[np.ndarray, int]):
    """2D grid divided into 4 rooms by cross-shaped walls with doorways.

    Layout (size=11, doorway_offset=3, doorway_size=1):
        +-------+---+-------+
        |       |   |       |
        |  R0   |   |  R1   |
        |       |   |       |
        +--- o ---+--- o ---+   o = doorway (1 cell)
        |       |   |       |
        |  R2   |   |  R3   |
        |       |   |       |
        +-------+---+-------+
    """

    metadata = {"render_modes": ["ansi", "human"]}

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

    def __init__(self, config: FourRoomsConfig, seed: int | None = None) -> None:
        self.config = config
        self.render_mode = config.render_mode
        self.action_space = spaces.Discrete(config.action_vocab_size)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(config.size, config.size, config.observation_channels),
            dtype=config.numpy_observation_dtype,
        )
        self._np_random, _ = seeding.np_random(seed)
        self._wall_mask = self._make_wall_mask()
        self._goal_position: int = self._sample_goal()
        self._empty_positions = self._compute_empty_positions()
        if self._empty_positions.size == 0:
            raise ValueError("four rooms must contain at least one empty cell")
        self._state: int | None = None
        self._last_observation: np.ndarray | None = None
        self._last_noise_mask: np.ndarray | None = None
        self._elapsed_steps = 0

    @property
    def state(self) -> int:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    # ── Wall layout ─────────────────────────────────────────────────────

    def _make_wall_mask(self) -> np.ndarray:
        """Build FourRooms wall layout with random doorway positions.

        When resample_per_sequence is True, doorway offsets are randomly
        sampled from [offset_min, offset_max] for each call.  When
        asymmetric is True, the interior cross is not centered.
        """
        s = self.config.size
        rng = self._np_random
        mask = np.zeros((s, s), dtype=bool)

        # Border walls
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        # Interior cross wall positions
        if self.config.asymmetric:
            mid_h = rng.integers(s // 3, 2 * s // 3 + 1)
            mid_v = rng.integers(s // 3, 2 * s // 3 + 1)
        else:
            mid_h = s // 2
            mid_v = s // 2

        mask[mid_h, :] = True  # horizontal wall
        mask[:, mid_v] = True  # vertical wall

        # Random doorway offsets
        d_size = self.config.doorway_size
        off_min = self.config.doorway_offset_min
        off_max = self.config.doorway_offset_max

        # Sample 4 doorway offsets independently
        offsets = rng.integers(off_min, off_max + 1, size=4)
        left_off, right_off = int(offsets[0]), int(offsets[1])
        top_off, bottom_off = int(offsets[2]), int(offsets[3])

        # Horizontal wall doorways (left and right of center)
        for dx in range(d_size):
            mask[mid_h, left_off + dx] = False
            mask[mid_h, s - 1 - right_off - dx] = False

        # Vertical wall doorways (top and bottom of center)
        for dy in range(d_size):
            mask[top_off + dy, mid_v] = False
            mask[s - 1 - bottom_off - dy, mid_v] = False

        return mask

    def _sample_goal(self) -> int:
        empty = self._compute_empty_positions()
        return int(self._np_random.choice(empty))

    def _compute_empty_positions(self) -> np.ndarray:
        return np.flatnonzero((~self._wall_mask).reshape(-1))

    # ── Observation ─────────────────────────────────────────────────────

    def _observe_with_noise(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_size = len(states)
        s = self.config.size

        # Start with white
        obs = np.ones((batch_size, s, s, 3), dtype=self.config.numpy_observation_dtype)
        # Walls: black
        obs[:, self._wall_mask] = self.WALL_RGB
        # Goal: blue
        gy, gx = divmod(self._goal_position, s)
        obs[:, gy, gx] = self.GOAL_RGB
        # Agent: green
        state_y = states // s
        state_x = states % s
        obs[np.arange(batch_size), state_y, state_x] = self.AGENT_RGB

        # Noise
        candidate_mask = np.broadcast_to(~self._wall_mask, (batch_size, s, s)).copy()
        candidate_mask[np.arange(batch_size), state_y, state_x] = False
        candidate_mask[np.arange(batch_size), gy, gx] = False
        if self.config.noise_placement == GridNoisePlacement.EMPTY_AND_OBSTACLE:
            candidate_mask[:] = True
            candidate_mask[np.arange(batch_size), state_y, state_x] = False

        noise_masks = (
            self._np_random.random(candidate_mask.shape) < self.config.p_noise
        ) & candidate_mask
        obs[noise_masks] = self.AGENT_RGB
        return obs.astype(self.config.numpy_observation_dtype), noise_masks

    # ── Gym API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self._goal_position = options.get("goal", self._goal_position)
        self._state = self._make_start_state(options.get("start_state"))
        self._elapsed_steps = 0
        obs, nm = self._observe_with_noise(np.array([self.state]))
        self._last_observation = obs[0]
        self._last_noise_mask = nm[0]
        return self._last_observation.copy(), {
            "state": self.state,
            "goal": self._goal_position,
        }

    def step(self, action: int):
        action_meaning = self._decode_action(action)
        noop = bool(self.np_random.random() < self.config.p_noop)
        actual = GridWorld2DAction.STAY if noop else action_meaning
        self._state = self._next_state(self.state, actual)
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
            "noop": noop,
            "reward": reward,
        }

        obs, nm = self._observe_with_noise(np.array([self.state]))
        self._last_observation = obs[0]
        self._last_noise_mask = nm[0]
        return self._last_observation.copy(), reward, terminated, truncated, info

    def render(self, batch: SequenceBatch | None = None, batch_index: int = 0):
        if batch is not None:
            return self._render_sequence_batch(batch, batch_index)
        if self._last_observation is None:
            raise RuntimeError("environment has not been reset")
        cells = self._render_grid(self.state)
        frame = f"state={self.state} goal={self._goal_position}\n" + "\n".join(cells)
        if self.render_mode == "human":
            print(frame)
            return None
        return frame

    def _render_grid(self, state: int) -> list[str]:
        s = self.config.size
        sy, sx = divmod(state, s)
        gy, gx = divmod(self._goal_position, s)
        rows = []
        for y in range(s):
            cells = []
            for x in range(s):
                if y == sy and x == sx:
                    cells.append("AG")
                elif y == gy and x == gx:
                    cells.append("◆◆")
                elif self._wall_mask[y, x]:
                    cells.append("██")
                else:
                    cells.append("  ")
            rows.append("".join(cells))
        return rows

    def _render_sequence_batch(self, batch: SequenceBatch, idx: int) -> str:
        if batch.obstacle_masks is None:
            raise ValueError("FourRooms rendering requires obstacle_masks")
        states = batch.states[idx].detach().long().cpu()
        rows = []
        for t in range(states.shape[0]):
            if t > 0:
                rows.append("")
            rows.append(f"t={t:02d} state={int(states[t]):02d}")
            rows.extend(self._render_grid(int(states[t])))
        return "\n".join(rows)

    # ── Sequence sampling ───────────────────────────────────────────────

    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        start_state=None,
        virtual_border=None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        wall_mask = (
            self._make_wall_mask()
            if self.config.resample_per_sequence
            else self._wall_mask.copy()
        )
        goal = self._sample_goal()

        states = self._make_start_states(batch_size, start_state)
        obs, nm = self._observe_with_noise(states)
        observations, noise_masks = [obs], [nm]
        wall_masks = [np.broadcast_to(wall_mask, nm.shape).copy()]
        state_history = [states.copy()]
        actions = self._sample_sequence_actions(batch_size, sequence_length - 1)
        actual_deltas, noop_masks = [], []

        for step in range(sequence_length - 1):
            noop_mask = self.np_random.random(batch_size) < self.config.p_noop
            actual_actions = np.where(
                noop_mask, GridWorld2DAction.STAY.value, actions[:, step]
            )
            states = self._next_states(states, actual_actions, wall_mask)
            obs, nm = self._observe_with_noise(states)
            observations.append(obs)
            noise_masks.append(nm)
            wall_masks.append(np.broadcast_to(wall_mask, nm.shape).copy())
            state_history.append(states.copy())
            actual_deltas.append(self._action_deltas(actual_actions))
            noop_masks.append(noop_mask)

        # HACK: temporarily set goal so _observe_with_noise renders it
        saved_goal = self._goal_position
        self._goal_position = goal
        # Re-render all observations with the correct goal
        for i in range(len(observations)):
            observations[i], _ = self._observe_with_noise(state_history[i])
        self._goal_position = saved_goal

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
                np.stack(wall_masks, axis=1), dtype=torch.bool
            ),
        )

    def _sample_sequence_actions(self, batch_size, transition_count):
        if transition_count <= 0:
            return np.empty((batch_size, 0), dtype=np.int64)
        actions = np.empty((batch_size, transition_count), dtype=np.int64)
        directions = self.np_random.choice(
            np.asarray(self.DIRECTION_ACTIONS, dtype=np.int64), size=batch_size
        )
        for step in range(transition_count):
            if step > 0:
                turn_mask = (
                    self.np_random.random(batch_size) < self.config.p_action_turn
                )
                directions = np.where(
                    turn_mask, self._change_direction_ids(directions), directions
                )
            stay_mask = self.np_random.random(batch_size) < self.config.p_action_stay
            actions[:, step] = np.where(
                stay_mask, GridWorld2DAction.STAY.value, directions
            )
        return actions

    # ── State transition ────────────────────────────────────────────────

    def _next_state(self, state: int, action: GridWorld2DAction) -> int:
        return int(self._next_states(np.array([state]), np.array([action.value]))[0])

    def _next_states(self, states, actions, wall_mask=None):
        if wall_mask is None:
            wall_mask = self._wall_mask
        s = self.config.size
        sy = states // s
        sx = states % s
        deltas = self._action_deltas(actions)
        ny = np.clip(sy + deltas[:, 0], 0, s - 1)
        nx = np.clip(sx + deltas[:, 1], 0, s - 1)
        blocked = wall_mask[ny, nx]
        ny = np.where(blocked, sy, ny)
        nx = np.where(blocked, sx, nx)
        return ny * s + nx

    def _action_deltas(self, actions):
        deltas = np.zeros((len(actions), 2), dtype=np.int64)
        for act, delta in self.ACTION_EFFECTS.items():
            deltas[actions == act.value] = delta
        return deltas

    def _change_direction_ids(self, actions):
        return change_direction_ids(
            rng=self.np_random,
            current_directions=actions,
            direction_ids=self.DIRECTION_ACTIONS,
        )

    def _decode_action(self, action: int) -> GridWorld2DAction:
        try:
            return GridWorld2DAction(int(action))
        except ValueError:
            raise ValueError("action must be one of {0,1,2,3,4}") from None

    def _make_start_state(self, start_state: Any) -> int:
        s = self.config.size
        if start_state is None:
            empty = self._compute_empty_positions()
            empty = empty[empty != self._goal_position]
            if empty.size == 0:
                raise ValueError("no empty non-goal cells")
            return int(self.np_random.choice(empty))
        st = int(start_state)
        if not 0 <= st < s * s:
            raise ValueError(f"start_state must be in [0, {s * s - 1}]")
        if self._wall_mask.reshape(-1)[st]:
            raise ValueError("start_state must be empty")
        return st

    def _make_start_states(self, batch_size, start_state):
        if start_state is None:
            empty = self._compute_empty_positions()
            empty = empty[empty != self._goal_position]
            return self.np_random.choice(empty, size=batch_size)
        if isinstance(start_state, int):
            return np.full(
                batch_size, self._make_start_state(start_state), dtype=np.int64
            )
        states = start_state.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if states.shape != (batch_size,):
            raise ValueError("start_state shape mismatch")
        return states


__all__ = ["FourRoomsEnv"]
