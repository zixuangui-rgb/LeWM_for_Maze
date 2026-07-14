"""Exact pooled-vector LeWM adapter with explicit rollout semantics."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from vector_jepa_planner_frontier.common import ComputeLedger
from vector_jepa_planner_frontier.schemas import RolloutSemantics


@dataclass(frozen=True)
class VectorContext:
    embeddings: torch.Tensor
    actions: torch.Tensor
    goal: torch.Tensor
    maze_size: int
    remaining_steps: int = 128

    def validate(self, history_size: int) -> None:
        if self.embeddings.ndim != 3 or self.embeddings.shape[0] != 1:
            raise ValueError("context embeddings must have shape [1, history, dim]")
        if self.actions.ndim != 2 or self.actions.shape[0] != 1:
            raise ValueError("context actions must have shape [1, history]")
        if self.goal.ndim != 3 or self.goal.shape[:2] != (1, 1):
            raise ValueError("goal embedding must have shape [1, 1, dim]")
        if self.embeddings.shape[1] != history_size:
            raise ValueError("context embedding history does not match planner")
        if self.actions.shape[1] != history_size:
            raise ValueError("context action history does not match planner")
        if self.goal.shape[-1] != self.embeddings.shape[-1]:
            raise ValueError("context and goal latent dimensions differ")
        if self.remaining_steps < 1:
            raise ValueError("context remaining_steps must be positive")


@dataclass(frozen=True)
class RolloutBatch:
    states: torch.Tensor
    terminal: torch.Tensor
    actions: torch.Tensor
    semantics: RolloutSemantics

    def validate(self) -> None:
        if self.states.ndim != 3:
            raise ValueError("rollout states must have shape [batch, horizon, dim]")
        if self.terminal.shape != self.states[:, -1].shape:
            raise ValueError("rollout terminal shape is inconsistent")
        if self.actions.shape != self.states.shape[:2]:
            raise ValueError("rollout actions and states must share batch/horizon")
        if not torch.isfinite(self.states).all():
            raise FloatingPointError("world-model rollout produced non-finite latents")


class VectorWorldModel:
    """Wrap the repository Unisize256 without changing its architecture."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        history_size: int = 3,
    ) -> None:
        if history_size < 2:
            raise ValueError("history_size must be at least two")
        self.model = model
        self.device = device
        self.history_size = int(history_size)

    def encode(self, observation: np.ndarray, maze_size: int) -> torch.Tensor:
        inputs = (
            torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        with torch.no_grad():
            encoded = self.model.encoder(inputs, int(maze_size))
            embedding, _ = self.model.embedding_projector(encoded)
        if embedding.shape[:2] != (1, 1) or not torch.isfinite(embedding).all():
            raise FloatingPointError("encoder/projector produced an invalid embedding")
        return embedding

    def initial_context(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        *,
        maze_size: int,
        context_action: int = 4,
        remaining_steps: int = 128,
    ) -> VectorContext:
        context = VectorContext(
            embeddings=start.repeat(1, self.history_size, 1),
            actions=torch.full(
                (1, self.history_size),
                int(context_action),
                dtype=torch.long,
                device=self.device,
            ),
            goal=goal,
            maze_size=int(maze_size),
            remaining_steps=int(remaining_steps),
        )
        context.validate(self.history_size)
        return context

    def advance_context(
        self,
        context: VectorContext,
        current: torch.Tensor,
        executed_action: int,
    ) -> VectorContext:
        context.validate(self.history_size)
        updated = VectorContext(
            embeddings=torch.cat([context.embeddings[:, 1:], current], dim=1),
            actions=torch.cat(
                [
                    context.actions[:, 1:],
                    torch.tensor(
                        [[int(executed_action)]],
                        dtype=torch.long,
                        device=self.device,
                    ),
                ],
                dim=1,
            ),
            goal=context.goal,
            maze_size=context.maze_size,
            remaining_steps=max(1, context.remaining_steps - 1),
        )
        updated.validate(self.history_size)
        return updated

    def rollout(
        self,
        context: VectorContext,
        candidate_actions: np.ndarray | torch.Tensor,
        *,
        semantics: RolloutSemantics,
        ledger: ComputeLedger | None = None,
        gradients: bool = False,
    ) -> RolloutBatch:
        context.validate(self.history_size)
        actions = torch.as_tensor(
            candidate_actions, dtype=torch.long, device=self.device
        )
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
            raise ValueError("candidate_actions must have shape [batch, horizon]")
        if bool(((actions < 0) | (actions >= 5)).any()):
            raise ValueError("candidate action is outside the model vocabulary [0, 4]")
        batch_size, horizon = (int(actions.shape[0]), int(actions.shape[1]))
        embeddings = context.embeddings.expand(batch_size, -1, -1).contiguous()
        action_history = (
            context.actions[:, : self.history_size - 1]
            .expand(batch_size, -1)
            .contiguous()
        )
        predicted_states: list[torch.Tensor] = []
        manager = nullcontext() if gradients else torch.no_grad()
        with manager:
            if semantics == RolloutSemantics.LEGACY_WARMUP_V1:
                for step in range(horizon):
                    proposed = actions[:, step : step + 1]
                    prediction = self.model.predictor(embeddings, action_history)
                    next_embedding = prediction[:, -1:]
                    predicted_states.append(next_embedding)
                    embeddings = torch.cat([embeddings[:, 1:], next_embedding], dim=1)
                    action_history = torch.cat([action_history[:, 1:], proposed], dim=1)
            else:
                # Context actions are stored with the action that produced each
                # frame in the same slot. The last H-1 entries therefore align
                # the H observed embeddings. Duplicate the current source into
                # the predictor's ignored final slot, append the candidate action,
                # then advance both histories with the newly predicted state.
                action_history = (
                    context.actions[:, 1:].expand(batch_size, -1).contiguous()
                )
                for step in range(horizon):
                    proposed = actions[:, step : step + 1]
                    predictor_embeddings = torch.cat(
                        [embeddings[:, 1:], embeddings[:, -1:]], dim=1
                    )
                    predictor_actions = torch.cat(
                        [action_history[:, 1:], proposed], dim=1
                    )
                    prediction = self.model.predictor(
                        predictor_embeddings,
                        predictor_actions,
                    )
                    next_embedding = prediction[:, -1:]
                    predicted_states.append(next_embedding)
                    embeddings = torch.cat([embeddings[:, 1:], next_embedding], dim=1)
                    action_history = predictor_actions
        states = torch.cat(predicted_states, dim=1)
        result = RolloutBatch(
            states=states,
            terminal=states[:, -1],
            actions=actions,
            semantics=semantics,
        )
        result.validate()
        if ledger is not None:
            ledger.record_plan(
                transitions=batch_size * horizon,
                batch_size=batch_size,
                calls=horizon,
            )
            ledger.candidate_sequences += batch_size
            ledger.duplicate_candidates += batch_size - int(
                torch.unique(actions, dim=0).shape[0]
            )
        return result

    def one_step_all_actions(
        self,
        context: VectorContext,
        *,
        action_vocab_size: int = 5,
        ledger: ComputeLedger | None = None,
    ) -> torch.Tensor:
        """Match final_closure's five-action corrected-v1 fallback exactly."""

        context.validate(self.history_size)
        if action_vocab_size != 5:
            raise ValueError("the source LeWM fallback uses a five-action vocabulary")
        embeddings = context.embeddings.expand(action_vocab_size, -1, -1)
        actions = context.actions[:, :-1].repeat(action_vocab_size, 1)
        actions[:, -1] = torch.arange(action_vocab_size, device=self.device)
        with torch.no_grad():
            predicted = self.model.predictor(embeddings, actions)[:, -1]
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("one-step fallback produced non-finite latents")
        if ledger is not None:
            ledger.record_assist(transitions=action_vocab_size)
        return predicted


__all__ = ["RolloutBatch", "VectorContext", "VectorWorldModel"]
