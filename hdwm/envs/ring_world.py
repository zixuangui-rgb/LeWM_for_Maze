"""Minimal Gymnasium-compatible noisy ring-world environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding

from hdwm.config import RingWorldAction, RingWorldConfig
from hdwm.envs.action_utils import change_direction_ids


@dataclass
class SequenceBatch:
    """A batch of sampled trajectories."""

    observations: torch.Tensor
    states: torch.Tensor
    noise_masks: torch.Tensor
    actions: torch.Tensor
    actual_deltas: torch.Tensor
    noop_masks: torch.Tensor
    obstacle_masks: torch.Tensor | None = None


class RingWorldEnv(gym.Env[np.ndarray, int]):
    """Noisy 1D ring-world from PLAN_v1.md.

    Gym API is single-environment. `sample_sequence` is a convenience helper for
    vectorized short trajectory batches used by the planned model training.
    """

    metadata = {"render_modes": ["ansi", "human"]}
    ACTION_EFFECTS = {
        RingWorldAction.LEFT: -1,
        RingWorldAction.STAY: 0,
        RingWorldAction.RIGHT: 1,
    }
    DIRECTION_ACTIONS = (RingWorldAction.LEFT.value, RingWorldAction.RIGHT.value)

    def __init__(self, config: RingWorldConfig, seed: int | None = None) -> None:
        self.config = config
        self.render_mode = config.render_mode
        self.action_space = spaces.Discrete(config.num_actions)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(config.length,),
            dtype=config.numpy_observation_dtype,
        )
        self._np_random, _ = seeding.np_random(seed)
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
        """Apply an action id through readable action meanings."""

        action_meaning = self._decode_action(action)

        noop = bool(self.np_random.random() < self.config.p_noop)
        actual_action = RingWorldAction.STAY if noop else action_meaning
        actual_delta = self.ACTION_EFFECTS[actual_action]
        self._state = (self.state + actual_delta) % self.config.length
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
            "actual_delta": actual_delta,
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

        cells = self._render_cells(
            observation=torch.as_tensor(self._last_observation),
            noise_mask=torch.as_tensor(self._last_noise_mask),
            state=self.state,
        )
        frame = f"state={self.state} | {' '.join(cells)}"
        if self.render_mode == "human":
            print(frame)
            return None
        return frame

    @classmethod
    def _render_sequence(cls, batch: SequenceBatch, batch_index: int = 0) -> str:
        """Render one sampled sequence as a timestep-by-position text matrix."""

        if batch.observations.ndim != 3:
            raise ValueError(
                f"expected observations rank 3, got {batch.observations.ndim}"
            )
        if batch.states.ndim != 2:
            raise ValueError(f"expected states rank 2, got {batch.states.ndim}")
        if batch.noise_masks.shape != batch.observations.shape:
            raise ValueError(
                f"expected noise_masks shape {tuple(batch.observations.shape)}, "
                f"got {tuple(batch.noise_masks.shape)}"
            )
        batch_size, sequence_length, observation_size = batch.observations.shape
        if not 0 <= batch_index < batch_size:
            raise ValueError(f"batch_index must be in [0, {batch_size - 1}]")

        observations = batch.observations[batch_index].detach().bool().cpu()
        noise_masks = batch.noise_masks[batch_index].detach().bool().cpu()
        states = batch.states[batch_index].detach().long().cpu()
        if states.shape != (sequence_length,):
            raise ValueError(
                f"expected states shape {(sequence_length,)}, got {tuple(states.shape)}"
            )
        if ((states < 0) | (states >= observation_size)).any():
            raise ValueError(f"batch states must be in [0, {observation_size - 1}]")

        header = "t  " + " ".join(
            f"{position:02d}" for position in range(observation_size)
        )
        rows = [header]
        for step in range(sequence_length):
            cells = cls._render_cells(
                observation=observations[step],
                noise_mask=noise_masks[step],
                state=int(states[step]),
            )
            rows.append(f"{step:02d} " + "  ".join(cells))
        return "\n".join(rows)

    @staticmethod
    def _render_cells(
        observation: torch.Tensor,
        noise_mask: torch.Tensor,
        state: int,
    ) -> list[str]:
        if observation.ndim != 1:
            raise ValueError(f"expected observation rank 1, got {observation.ndim}")
        if noise_mask.shape != observation.shape:
            raise ValueError(
                f"expected noise_mask shape {tuple(observation.shape)}, "
                f"got {tuple(noise_mask.shape)}"
            )
        observation = observation.detach().bool().cpu()
        noise_mask = noise_mask.detach().bool().cpu()
        if not 0 <= state < observation.shape[0]:
            raise ValueError(f"state must be in [0, {observation.shape[0] - 1}]")

        cells = []
        for position in range(observation.shape[0]):
            is_state = position == state
            is_noise = bool(noise_mask[position])
            is_lit = bool(observation[position])
            if is_state and is_noise:
                cells.append("*")
            elif is_state:
                cells.append("X")
            elif is_noise or is_lit:
                cells.append("+")
            else:
                cells.append(".")
        return cells

    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        start_state: int | torch.Tensor | None = None,
    ) -> SequenceBatch:
        """Sample a batch of trajectories as `(I_1:T, a_1:T-1)`."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        states = self._make_start_states(batch_size=batch_size, start_state=start_state)
        observation, noise_mask = self._observe_with_noise(states)
        observations = [observation]
        noise_masks = [noise_mask]
        state_history = [states.copy()]
        actions = self._sample_sequence_actions(
            batch_size=batch_size,
            transition_count=sequence_length - 1,
        )
        actual_deltas = []
        noop_masks = []

        for t in range(sequence_length - 1):
            # Vectorized no-op noise turns selected proposed actions into stay.
            noop_mask = self.np_random.random(batch_size) < self.config.p_noop
            actual_actions = np.where(
                noop_mask,
                RingWorldAction.STAY.value,
                actions[:, t],
            )
            actual_delta = self._action_deltas(actual_actions)
            states = (states + actual_delta) % self.config.length
            observation, noise_mask = self._observe_with_noise(states)
            observations.append(observation)
            noise_masks.append(noise_mask)
            state_history.append(states.copy())
            actual_deltas.append(actual_delta)
            noop_masks.append(noop_mask)

        return SequenceBatch(
            observations=torch.as_tensor(np.stack(observations, axis=1)),
            states=torch.as_tensor(np.stack(state_history, axis=1), dtype=torch.long),
            noise_masks=torch.as_tensor(
                np.stack(noise_masks, axis=1), dtype=torch.bool
            ),
            actions=torch.as_tensor(actions, dtype=torch.long),
            # sequence_length=1 has no transitions, but downstream code still expects
            # rank-2 tensors with a zero-width transition axis.
            actual_deltas=torch.as_tensor(
                np.stack(actual_deltas, axis=1), dtype=torch.long
            )
            if actual_deltas
            else torch.empty((batch_size, 0), dtype=torch.long),
            noop_masks=torch.as_tensor(np.stack(noop_masks, axis=1), dtype=torch.bool)
            if noop_masks
            else torch.empty((batch_size, 0), dtype=torch.bool),
        )

    def _observe(self, states: np.ndarray) -> np.ndarray:
        observations, _ = self._observe_with_noise(states)
        return observations

    def _observe_with_noise(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noise_masks = (
            self.np_random.random((len(states), self.config.length))
            < self.config.p_noise
        )
        # Observation noise may flip unrelated bits, but the true state is always
        # visible.
        observations = noise_masks.copy()
        observations[np.arange(len(states)), states] = True
        return observations.astype(self.config.numpy_observation_dtype), noise_masks

    def _sample_sequence_actions(
        self,
        batch_size: int,
        transition_count: int,
    ) -> np.ndarray:
        """Sample persistent actions so short sequences cover more of the ring."""

        if transition_count < 0:
            raise ValueError("transition_count must be non-negative")
        if transition_count == 0:
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
                    turn_mask,
                    self._change_direction_ids(directions),
                    directions,
                )
            stay_mask = self.np_random.random(batch_size) < self.config.p_action_stay
            actions[:, step] = np.where(
                stay_mask,
                RingWorldAction.STAY.value,
                directions,
            )
        return actions

    def _action_deltas(self, actions: np.ndarray) -> np.ndarray:
        deltas = np.zeros(len(actions), dtype=np.int64)
        for action, delta in self.ACTION_EFFECTS.items():
            deltas[actions == action.value] = delta
        return deltas

    def _change_direction_ids(self, actions: np.ndarray) -> np.ndarray:
        return change_direction_ids(
            rng=self.np_random,
            current_directions=actions,
            direction_ids=self.DIRECTION_ACTIONS,
        )

    def _decode_action(self, action: int) -> RingWorldAction:
        try:
            return RingWorldAction(int(action))
        except ValueError as error:
            raise ValueError("action must be one of {0, 1, 2}") from error

    def _make_start_state(self, start_state: Any) -> int:
        if start_state is None:
            return int(self.np_random.integers(0, self.config.length))
        state = int(start_state)
        if not 0 <= state < self.config.length:
            raise ValueError(f"start_state must be in [0, {self.config.length - 1}]")
        return state

    def _make_start_states(
        self,
        batch_size: int,
        start_state: int | torch.Tensor | None,
    ) -> np.ndarray:
        if start_state is None:
            return self.np_random.integers(0, self.config.length, size=batch_size)

        if isinstance(start_state, int):
            if not 0 <= start_state < self.config.length:
                raise ValueError(
                    f"start_state must be in [0, {self.config.length - 1}]"
                )
            return np.full(batch_size, start_state, dtype=np.int64)

        # Treat tensor start states as data for sampling; gradients are irrelevant here.
        states = start_state.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if states.shape != (batch_size,):
            raise ValueError(
                f"start_state shape {tuple(states.shape)} does not match "
                f"batch_size {batch_size}"
            )
        if ((states < 0) | (states >= self.config.length)).any():
            raise ValueError(
                f"start_state values must be in [0, {self.config.length - 1}]"
            )
        return states
