"""Offline oracle ladder for candidate coverage, selection, and dynamics diagnosis."""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from final_closure.common import (
    corrected_actions,
    next_state,
    observe_state,
    read_jsonl,
    sha256_file,
    summarize_rows,
    task_seed,
    validate_task_rows,
)
from final_closure.evaluate import run_episode
from vector_jepa_planner_frontier import ACTION_IDS, INVERSE_ACTION
from vector_jepa_planner_frontier.common import (
    ComputeLedger,
    analysis_spec_sha256,
    atomic_json_dump,
    hierarchical_seed,
    load_json,
    load_study_config,
    protocol_metadata,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import load_source_lewm
from vector_jepa_planner_frontier.schemas import RolloutSemantics
from vector_jepa_planner_frontier.world_model import VectorContext, VectorWorldModel

ORACLES = (
    "O0",
    "O1_PROP",
    "O2_SELECT",
    "O3_DYN",
    "O4_VALUE",
    "O5_JOIN",
    "O6_VALID_FUTURE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--oracle", choices=ORACLES, required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--search-seed", type=int, required=True)
    parser.add_argument(
        "--split-role", choices=("development", "validation"), required=True
    )
    parser.add_argument(
        "--action-selection",
        choices=("unmasked", "corrected_v1"),
        default="corrected_v1",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--diagnostic-limit", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


class OracleController:
    def __init__(
        self,
        world_model: VectorWorldModel,
        *,
        oracle: str,
        evaluation_seed: int,
        action_selection: str,
        horizon: int = 12,
        candidate_count: int = 64,
    ) -> None:
        if oracle not in ORACLES:
            raise ValueError("unknown oracle ladder rung")
        self.world_model = world_model
        self.oracle = oracle
        self.evaluation_seed = int(evaluation_seed)
        self.action_selection = action_selection
        self.horizon = int(horizon)
        self.candidate_count = int(candidate_count)
        self.context: VectorContext | None = None
        self.task_index = -1
        self.step = 0
        self.last_action: int | None = None

    def reset(self, env: Any, observation: np.ndarray, task_index: int) -> None:
        self.task_index = int(task_index)
        self.step = 0
        self.last_action = None
        size = int(env.config.width)
        source = self.world_model.encode(observation, size)
        goal = self.world_model.encode(
            observe_state(env, int(env._goal_position)), size
        )
        self.context = self.world_model.initial_context(
            source,
            goal,
            maze_size=size,
            context_action=int(env.config.action_vocab_size) - 1,
        )

    def _advance(self, observation: np.ndarray) -> None:
        if self.step == 0:
            return
        if self.context is None or self.last_action is None:
            raise RuntimeError("oracle controller context is missing")
        current = self.world_model.encode(observation, self.context.maze_size)
        self.context = self.world_model.advance_context(
            self.context, current, self.last_action
        )

    def _candidates(
        self,
        env: Any,
        state: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        action_values = np.asarray(ACTION_IDS, dtype=np.int64)
        draws = rng.random((self.candidate_count, self.horizon))
        candidates = action_values[np.floor(draws * len(action_values)).astype(int)]
        if self.oracle == "O6_VALID_FUTURE":
            for candidate_index in range(self.candidate_count):
                current = int(state)
                for step in range(self.horizon):
                    legal = [
                        action
                        for action in ACTION_IDS
                        if next_state(env, current, action) != current
                    ]
                    if not legal:
                        legal = list(ACTION_IDS)
                    slot = min(
                        int(draws[candidate_index, step] * len(legal)), len(legal) - 1
                    )
                    action = int(legal[slot])
                    candidates[candidate_index, step] = action
                    current = next_state(env, current, action)
        if self.oracle == "O1_PROP":
            goal = int(env._goal_position)
            distances = _distances(env, goal)
            current = int(state)
            for index in range(min(self.horizon, int(distances[current]))):
                distance = int(distances[current])
                optimal = [
                    action
                    for action in ACTION_IDS
                    if int(distances[next_state(env, current, action)]) == distance - 1
                ]
                action = min(optimal)
                candidates[0, index] = action
                current = next_state(env, current, action)
        return candidates

    def _true_terminals(
        self, env: Any, state: int, candidates: np.ndarray
    ) -> np.ndarray:
        terminal = np.empty(len(candidates), dtype=np.int64)
        for index, sequence in enumerate(candidates):
            current = int(state)
            for action in sequence.tolist():
                current = next_state(env, current, int(action))
            terminal[index] = current
        return terminal

    def _oracle_join_candidates(
        self,
        env: Any,
        state: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, ComputeLedger, int, dict[str, float]]:
        if self.context is None:
            raise RuntimeError("oracle join requires an initialized context")
        half = max(1, self.horizon // 2)
        per_side = self.candidate_count
        actions = np.asarray(ACTION_IDS, dtype=np.int64)
        forward_actions = rng.choice(actions, size=(per_side, half), replace=True)
        backward_actions = rng.choice(actions, size=(per_side, half), replace=True)
        ledger = ComputeLedger()
        self.world_model.rollout(
            self.context,
            forward_actions,
            semantics=RolloutSemantics.LEGACY_WARMUP_V1,
            ledger=ledger,
        )
        goal_context = VectorContext(
            embeddings=self.context.goal.repeat(1, self.world_model.history_size, 1),
            actions=torch.full_like(self.context.actions, 4),
            goal=self.context.embeddings[:, -1:],
            maze_size=self.context.maze_size,
        )
        self.world_model.rollout(
            goal_context,
            backward_actions,
            semantics=RolloutSemantics.LEGACY_WARMUP_V1,
            ledger=ledger,
        )
        forward_terminal = self._true_terminals(env, state, forward_actions)
        backward_terminal = self._true_terminals(
            env, int(env._goal_position), backward_actions
        )
        separations = np.empty((per_side, per_side), dtype=np.int64)
        for index, endpoint in enumerate(forward_terminal.tolist()):
            separations[index] = _distances(env, int(endpoint))[backward_terminal]
        flat_order = np.argsort(separations.reshape(-1), kind="stable")
        rerank_count = min(
            len(flat_order),
            (4 * 768 - ledger.plan_transitions) // self.horizon,
        )
        if rerank_count < 1:
            raise RuntimeError("oracle join has no budget for stitched reranking")
        stitched: list[np.ndarray] = []
        for flat in flat_order[:rerank_count].tolist():
            left, right = divmod(int(flat), per_side)
            inverse = np.asarray(
                [
                    INVERSE_ACTION[int(action)]
                    for action in backward_actions[right, ::-1]
                ],
                dtype=np.int64,
            )
            stitched.append(np.concatenate([forward_actions[left], inverse]))
        candidates = np.stack(stitched)
        rollout = self.world_model.rollout(
            self.context,
            candidates,
            semantics=RolloutSemantics.LEGACY_WARMUP_V1,
            ledger=ledger,
        )
        goal = self.context.goal.squeeze(1).expand_as(rollout.terminal)
        costs = (
            F.mse_loss(rollout.terminal, goal, reduction="none")
            .sum(dim=-1)
            .cpu()
            .numpy()
        )
        exact_pairs = int(np.count_nonzero(separations == 0))
        return (
            candidates,
            costs,
            ledger,
            int(2 * per_side * half + per_side * per_side),
            {
                "oracle_join_exact_coverage": float(exact_pairs > 0),
                "oracle_join_exact_pair_count": float(exact_pairs),
                "oracle_join_min_separation": float(separations.min()),
                "oracle_join_reranked_candidates": float(rerank_count),
            },
        )

    def _oracle_value_costs(
        self,
        env: Any,
        state: int,
        candidates: np.ndarray,
        predicted_terminal: torch.Tensor,
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        """Decode imagined terminals to real cells, then apply exact BFS value."""

        if self.context is None:
            raise RuntimeError("oracle value requires an initialized context")
        free_states = np.flatnonzero((~env._maze_mask).reshape(-1)).astype(np.int64)
        observations = np.stack(
            [observe_state(env, int(value)) for value in free_states.tolist()]
        )
        inputs = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.world_model.device,
        ).unsqueeze(1)
        with torch.no_grad():
            encoded = self.world_model.model.encoder(inputs, self.context.maze_size)
            state_embeddings, _ = self.world_model.model.embedding_projector(encoded)
            nearest = torch.cdist(
                predicted_terminal,
                state_embeddings[:, -1],
            ).argmin(dim=1)
        decoded_states = free_states[nearest.cpu().numpy()]
        distances = _distances(env, int(env._goal_position))
        true_terminals = self._true_terminals(env, state, candidates)
        return (
            distances[decoded_states].astype(np.float64),
            int(candidates.size + len(free_states)),
            {
                "oracle_value_decode_exact_rate": float(
                    np.mean(decoded_states == true_terminals)
                ),
                "oracle_value_decode_mismatch_rate": float(
                    np.mean(decoded_states != true_terminals)
                ),
            },
        )

    def choose(
        self,
        env: Any,
        observation: np.ndarray,
        state: int,
        previous: int | None,
    ) -> tuple[int, dict[str, float]]:
        self._advance(observation)
        if self.context is None:
            raise RuntimeError("oracle controller context is missing")
        seed = task_seed(self.evaluation_seed, self.task_index, self.step)
        rng = np.random.default_rng(seed)
        extra: dict[str, float] = {}
        if self.oracle == "O5_JOIN":
            candidates, costs, ledger, true_queries, extra = (
                self._oracle_join_candidates(env, state, rng)
            )
        else:
            candidates = self._candidates(env, state, rng)
            ledger = ComputeLedger()
            true_queries = 0
        if self.oracle == "O2_SELECT":
            terminal_states = self._true_terminals(env, state, candidates)
            distances = _distances(env, int(env._goal_position))
            costs = distances[terminal_states].astype(np.float64)
            true_queries = int(candidates.size)
        elif self.oracle == "O3_DYN":
            terminal_states = self._true_terminals(env, state, candidates)
            observations = np.stack(
                [observe_state(env, int(value)) for value in terminal_states]
            )
            inputs = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=self.world_model.device,
            ).unsqueeze(1)
            with torch.no_grad():
                encoded = self.world_model.model.encoder(inputs, self.context.maze_size)
                terminal, _ = self.world_model.model.embedding_projector(encoded)
            goal = self.context.goal.expand(len(candidates), -1, -1)
            costs = (
                F.mse_loss(terminal, goal, reduction="none")
                .sum(dim=(-1, -2))
                .cpu()
                .numpy()
            )
            true_queries = int(candidates.size)
        elif self.oracle == "O4_VALUE":
            rollout = self.world_model.rollout(
                self.context,
                candidates,
                semantics=RolloutSemantics.LEGACY_WARMUP_V1,
                ledger=ledger,
            )
            costs, true_queries, extra = self._oracle_value_costs(
                env,
                state,
                candidates,
                rollout.terminal,
            )
        elif self.oracle not in ("O3_DYN", "O5_JOIN"):
            rollout = self.world_model.rollout(
                self.context,
                candidates,
                semantics=RolloutSemantics.LEGACY_WARMUP_V1,
                ledger=ledger,
            )
            goal = self.context.goal.squeeze(1).expand_as(rollout.terminal)
            costs = (
                F.mse_loss(rollout.terminal, goal, reduction="none")
                .sum(dim=-1)
                .cpu()
                .numpy()
            )
        selected = int(np.argmin(costs))
        proposed = int(candidates[selected, 0])
        action = proposed
        assisted = False
        if self.action_selection == "corrected_v1":
            allowed = corrected_actions(env, state, previous)
            if proposed not in allowed:
                predicted = self.world_model.one_step_all_actions(
                    self.context, ledger=ledger
                )
                goal = self.context.goal.expand(5, -1, -1).squeeze(1)
                fallback = F.mse_loss(predicted, goal, reduction="none").sum(dim=-1)
                allowed_tensor = torch.tensor(
                    allowed, dtype=torch.long, device=self.world_model.device
                )
                action = int(allowed[int(fallback[allowed_tensor].argmin())])
                assisted = True
        self.last_action = action
        self.step += 1
        return action, {
            "proposed_invalid": float(next_state(env, state, proposed) == state),
            "proposed_backtrack": float(
                previous is not None and next_state(env, state, proposed) == previous
            ),
            "assisted_action": float(assisted),
            "plan_transitions": float(ledger.plan_transitions),
            "assist_transitions": float(ledger.assist_transitions),
            "oracle_environment_queries": float(true_queries),
            "oracle_best_cost": float(costs[selected]),
            **extra,
        }


def _distances(env: Any, goal: int) -> np.ndarray:
    from final_closure.common import bfs_distances_from

    return bfs_distances_from(env._maze_mask, goal, int(env.config.width))


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    if lock.get("status") != "locked":
        raise RuntimeError("oracle diagnostics require a completed protocol lock")
    if lock.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("config no longer matches the locked analysis specification")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside the locked matrix")
    if args.search_seed not in config.protocol.search_seeds:
        raise ValueError("search seed lies outside the locked matrix")
    if args.diagnostic_limit < 0:
        raise ValueError("diagnostic limit cannot be negative")
    output_path = resolve_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite oracle output: {output_path}")
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    device = resolve_device(args.device or config.device)
    set_seed(
        hierarchical_seed("oracle-ladder", args.backbone_seed, args.search_seed),
        deterministic=True,
    )
    model, _, source_path = load_source_lewm(
        config, lock, seed=args.backbone_seed, device=device
    )
    world_model = VectorWorldModel(model, device=device, history_size=3)
    controller = OracleController(
        world_model,
        oracle=args.oracle,
        evaluation_seed=hierarchical_seed(
            "paired-search-seed",
            config.protocol.evaluation_seed,
            args.search_seed,
        ),
        action_selection=args.action_selection,
    )
    manifest_path = resolve_path(getattr(config.paths, f"{args.split_role}_manifest"))
    lock_record = lock[f"{args.split_role}_manifest"]
    if sha256_file(manifest_path) != lock_record["sha256"]:
        raise ValueError(f"{args.split_role} manifest hash mismatch")
    entries = read_jsonl(manifest_path)
    if len(entries) != int(lock_record["count"]):
        raise ValueError(f"{args.split_role} manifest count mismatch")
    if args.diagnostic_limit:
        entries = entries[: args.diagnostic_limit]
    rows = [
        run_episode(
            entry,
            controller,
            task_index=index,
            max_steps=config.protocol.max_steps,
        )
        for index, entry in enumerate(entries)
    ]
    validate_task_rows(rows, len(entries))
    if not all(
        math.isfinite(float(value))
        for row in rows
        for value in row.get("auxiliary", {}).values()
    ):
        raise FloatingPointError("oracle ladder produced a non-finite metric")
    payload = {
        "metadata": protocol_metadata(
            config,
            lock,
            method=config.methods[0],
            seed=args.backbone_seed,
            search_seed=args.search_seed,
            device=device,
        ),
        "stage": "oracle_ladder_diagnostic",
        "oracle": args.oracle,
        "not_for_primary_table": True,
        "split_role": args.split_role,
        "action_selection": args.action_selection,
        "source_checkpoint_sha256": sha256_file(source_path),
        "summary": summarize_rows(rows, seen_max_size=config.protocol.seen_max_size),
        "tasks": rows,
    }
    atomic_json_dump(args.output, payload)


if __name__ == "__main__":
    main()
