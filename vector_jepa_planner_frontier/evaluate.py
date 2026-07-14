"""Paired unassisted/corrected-v1 evaluation for all planner variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from final_closure.common import (
    bfs_distances_from,
    corrected_actions,
    next_state,
    observe_state,
    read_jsonl,
    set_agent_state,
    sha256_file,
    summarize_rows,
    task_id,
    task_seed,
    validate_task_rows,
)
from spatial_jepa_planning.common import validate_manifest_entry
from vector_jepa_planner_frontier import ACTION_IDS, EXPERIMENT_FAMILY, FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    ComputeLedger,
    analysis_spec_sha256,
    atomic_json_dump,
    component_checkpoint_owner,
    component_checkpoint_path,
    hierarchical_seed,
    load_json,
    load_study_config,
    method_by_name,
    planner_seed_values,
    prepare_formal_outputs,
    protocol_metadata,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    training_spec_sha256,
    validate_compute_ledger,
    validate_finite_tree,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import load_source_lewm
from vector_jepa_planner_frontier.confirmation import (
    authorize_confirmatory_evaluation,
)
from vector_jepa_planner_frontier.effective_methods import resolve_effective_method
from vector_jepa_planner_frontier.heads import (
    ActionConsistencyVerifier,
    AutoregressiveProposal,
    CounterexampleRanker,
    DiscreteDenoisingProposal,
    DistributionalReachability,
    HeadConfig,
    StateJoinHead,
    VectorDTSHead,
    required_head_names,
)
from vector_jepa_planner_frontier.planners import (
    BasePlanner,
    CandidateBatch,
    CompositeScorer,
    PlannerResult,
    build_planner,
)
from vector_jepa_planner_frontier.proposals import (
    DenoisingSampler,
    LearnedAutoregressiveSampler,
    MixtureProposal,
    RetrievalBank,
    RetrievalProposal,
    UniformProposal,
)
from vector_jepa_planner_frontier.schemas import MethodConfig, ProposalKind, StudyConfig
from vector_jepa_planner_frontier.world_model import VectorContext, VectorWorldModel


class CandidateTraceSink:
    """Stream sampled candidate diagnostics and publish them atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        self._temporary = Path(temporary_name)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self.count = 0
        self._committed = False

    def write(self, value: dict[str, Any]) -> None:
        validate_finite_tree(value, label="candidate_trace")
        json.dump(
            value,
            self._stream,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._stream.write("\n")
        self.count += 1

    def commit(self) -> dict[str, Any]:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._temporary.replace(self.path)
        self._committed = True
        return {
            "path": str(self.path),
            "sha256": sha256_file(self.path),
            "decision_record_count": self.count,
            "format": "jsonl",
        }

    def abort(self) -> None:
        if not self._stream.closed:
            self._stream.close()
        if not self._committed:
            self._temporary.unlink(missing_ok=True)


def candidate_trace_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}.candidate_traces.jsonl")


def candidate_trace_selected(task_identifier: str, step: int, fraction: float) -> bool:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("candidate trace fraction must lie in (0, 1]")
    digest = hashlib.sha256(
        f"vector-jepa-candidate-trace-v1:{task_identifier}:{step}".encode()
    ).digest()
    draw = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    return draw < fraction


def exact_candidate_trace_keys(
    rows: list[dict[str, Any]], fraction: float
) -> tuple[set[tuple[str, int]], dict[str, Any]]:
    """Select the exact nearest integer fraction independently within each size."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("candidate trace fraction must lie in (0, 1]")
    by_size: defaultdict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for row in rows:
        identifier = str(row["task_id"])
        size = int(row["maze_size"])
        for trace in row["decision_traces"]:
            step = int(trace["step"])
            digest = hashlib.sha256(
                f"vector-jepa-exact-candidate-v1:{identifier}:{step}".encode()
            ).digest()
            by_size[size].append((int.from_bytes(digest[:8], "big"), identifier, step))
    selected: set[tuple[str, int]] = set()
    strata: dict[str, dict[str, float | int]] = {}
    for size, decisions in sorted(by_size.items()):
        count = len(decisions)
        selected_count = min(count, max(1, int(math.floor(count * fraction + 0.5))))
        chosen = sorted(decisions)[:selected_count]
        selected.update((identifier, step) for _, identifier, step in chosen)
        strata[str(size)] = {
            "decision_count": count,
            "selected_count": selected_count,
            "realized_fraction": selected_count / count,
        }
    if not selected:
        raise ValueError("formal evaluation produced no candidate-trace decisions")
    return selected, {
        "mode": "exact_stratified_bottom_hash_replay",
        "target_fraction": float(fraction),
        "rounding": "nearest_integer_minimum_one_per_nonempty_size",
        "hash_namespace": "vector-jepa-exact-candidate-v1",
        "strata": strata,
        "selected_count": len(selected),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _normalized_edit_distance(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> float:
    """Levenshtein distance normalized for variable-length search prefixes."""

    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_action in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_action in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_action != right_action),
                )
            )
        previous = current
    return float(previous[-1] / max(len(left), len(right)))


def _true_candidate_outcome(
    env: Any,
    *,
    root_state: int,
    goal_state: int,
    distances: np.ndarray,
    sequence: tuple[int, ...],
) -> dict[str, Any]:
    state = int(root_state)
    states = [state]
    distance_trace = [int(distances[state])]
    invalid_count = 0
    goal_reached = state == goal_state
    for action in sequence:
        successor = next_state(env, state, int(action))
        invalid_count += int(successor == state)
        state = successor
        states.append(state)
        distance_trace.append(int(distances[state]))
        goal_reached = goal_reached or state == goal_state
    short_cycle = any(
        states[index] == states[index - period]
        for index in range(2, len(states))
        for period in range(2, min(8, index) + 1)
    )
    root_distance = int(distance_trace[0])
    terminal_distance = int(distance_trace[-1])
    return {
        "terminal_state": int(state),
        "terminal_distance": terminal_distance,
        "true_progress": int(root_distance - terminal_distance),
        "invalid_count": int(invalid_count),
        "short_cycle": bool(short_cycle),
        "goal_reached": bool(goal_reached),
        "distance_trace": distance_trace,
    }


def analyze_candidate_result(
    env: Any,
    *,
    task_identifier: str,
    task_index: int,
    step: int,
    seed: int,
    root_state: int,
    result: PlannerResult,
    candidate_k: int,
    prefix_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Attach true maze outcomes after planning; none of these labels affect action."""

    goal_state = int(env._goal_position)
    distances = bfs_distances_from(env._maze_mask, goal_state, int(env.config.width))
    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    frequencies: Counter[tuple[int, ...]] = Counter()
    generated_count = 0
    for batch_index, batch in enumerate(result.candidate_batches):
        costs: list[float | None] = (
            [None] * len(batch.sequences)
            if batch.predicted_costs is None
            else [float(value) for value in batch.predicted_costs]
        )
        for candidate_index, (raw_sequence, raw_cost) in enumerate(
            zip(batch.sequences, costs, strict=True)
        ):
            generated_count += 1
            sequence = tuple(int(action) for action in raw_sequence.tolist())
            frequencies[sequence] += 1
            row = unique.get(sequence)
            if row is None or (
                raw_cost is not None
                and (
                    row["predicted_cost"] is None
                    or float(raw_cost) < float(row["predicted_cost"])
                )
            ):
                unique[sequence] = {
                    "sequence": sequence,
                    "predicted_cost": (
                        float(raw_cost) if raw_cost is not None else None
                    ),
                    "source": batch.source,
                    "batch_index": int(batch_index),
                    "candidate_index": int(candidate_index),
                    "generation_index": int(generated_count - 1),
                }
    selected_sequence = tuple(int(action) for action in result.sequence.tolist())
    if selected_sequence not in unique:
        unique[selected_sequence] = {
            "sequence": selected_sequence,
            "predicted_cost": float(result.cost),
            "source": "selected_result",
            "batch_index": -1,
            "candidate_index": -1,
            "generation_index": int(generated_count),
        }
    elif unique[selected_sequence]["predicted_cost"] is None:
        unique[selected_sequence]["predicted_cost"] = float(result.cost)
    candidates = sorted(unique.values(), key=lambda row: int(row["generation_index"]))
    for row in candidates:
        row["truth"] = _true_candidate_outcome(
            env,
            root_state=root_state,
            goal_state=goal_state,
            distances=distances,
            sequence=row["sequence"],
        )
    first_k = candidates[:candidate_k]
    root_distance = int(distances[root_state])
    optimal_actions = {
        action
        for action in ACTION_IDS
        if int(distances[next_state(env, root_state, action)]) == root_distance - 1
    }
    selected = unique[selected_sequence]
    oracle = min(
        candidates,
        key=lambda row: (
            int(row["truth"]["terminal_distance"]),
            int(row["truth"]["invalid_count"]),
            bool(row["truth"]["short_cycle"]),
            tuple(row["sequence"]),
        ),
    )
    scored_candidates = [row for row in candidates if row["predicted_cost"] is not None]
    predicted = np.asarray(
        [float(row["predicted_cost"]) for row in scored_candidates],
        dtype=np.float64,
    )
    true_distance = np.asarray(
        [float(row["truth"]["terminal_distance"]) for row in scored_candidates],
        dtype=np.float64,
    )
    prefix_coverage = {
        str(length): any(
            len(row["truth"]["distance_trace"]) > length
            and all(
                row["truth"]["distance_trace"][index + 1]
                == row["truth"]["distance_trace"][index] - 1
                for index in range(length)
            )
            for row in first_k
        )
        for length in prefix_lengths
    }
    selected_truth = selected["truth"]
    progress_candidate_available = any(
        int(row["truth"]["true_progress"]) > 0 for row in candidates
    )
    false_optimistic = bool(
        int(selected_truth["invalid_count"]) > 0
        or bool(selected_truth["short_cycle"])
        or (int(selected_truth["true_progress"]) <= 0 and progress_candidate_available)
    )
    first_k_sequences = [tuple(row["sequence"]) for row in first_k]
    pairwise_distances = [
        _normalized_edit_distance(left, right)
        for left_index, left in enumerate(first_k_sequences)
        for right in first_k_sequences[left_index + 1 :]
    ]
    generated_probabilities = np.asarray(
        [count / max(generated_count, 1) for count in frequencies.values()],
        dtype=np.float64,
    )
    frequency_ess = float(
        1.0 / np.square(generated_probabilities).sum()
        if generated_probabilities.size
        else 0.0
    )
    entropy_ess = float(
        np.exp(
            -np.sum(
                generated_probabilities
                * np.log(np.clip(generated_probabilities, 1e-12, 1.0))
            )
        )
        if generated_probabilities.size
        else 0.0
    )
    metrics = {
        "candidate_k": int(candidate_k),
        "effective_k": int(len(first_k)),
        "generated_candidate_count": int(generated_count),
        "unique_candidate_count": int(len(candidates)),
        "first_action_coverage_at_k": bool(
            any(int(row["sequence"][0]) in optimal_actions for row in first_k)
        ),
        "prefix_coverage_at_k": prefix_coverage,
        "goal_reaching_coverage_at_k": bool(
            any(bool(row["truth"]["goal_reached"]) for row in first_k)
        ),
        "selection_accuracy": bool(
            int(selected_truth["terminal_distance"])
            == int(oracle["truth"]["terminal_distance"])
        ),
        "selection_regret": int(selected_truth["terminal_distance"])
        - int(oracle["truth"]["terminal_distance"]),
        "false_optimistic": false_optimistic,
        "selected_invalid": bool(int(selected_truth["invalid_count"]) > 0),
        "selected_short_cycle": bool(selected_truth["short_cycle"]),
        "selected_no_progress": bool(int(selected_truth["true_progress"]) <= 0),
        "selected_true_progress": int(selected_truth["true_progress"]),
        "oracle_best_progress": int(oracle["truth"]["true_progress"]),
        "unique_route_ratio": float(len(candidates) / max(generated_count, 1)),
        "pairwise_normalized_edit_distance": float(np.mean(pairwise_distances))
        if pairwise_distances
        else 0.0,
        "frequency_effective_sample_size": frequency_ess,
        "entropy_effective_sample_size": entropy_ess,
        "predicted_true_distance_spearman": _rank_correlation(predicted, true_distance),
    }
    stored_keys = {tuple(row["sequence"]) for row in first_k}
    stored_keys.update((selected_sequence, tuple(oracle["sequence"])))
    stored = [row for row in candidates if tuple(row["sequence"]) in stored_keys]
    serializable_candidates = [
        {
            **{key: value for key, value in row.items() if key != "truth"},
            "sequence": list(row["sequence"]),
            "truth": row["truth"],
            "selected": tuple(row["sequence"]) == selected_sequence,
            "oracle_best": tuple(row["sequence"]) == tuple(oracle["sequence"]),
        }
        for row in stored
    ]
    return {
        "schema": "vector-jepa-candidate-trace-v1",
        "analysis_only_no_action_influence": True,
        "task_id": task_identifier,
        "task_index": int(task_index),
        "step": int(step),
        "maze_size": int(env.config.width),
        "planner_seed": int(seed),
        "root_state": int(root_state),
        "goal_state": goal_state,
        "root_distance": root_distance,
        "optimal_first_actions": sorted(optimal_actions),
        "metrics": metrics,
        "stored_candidate_count": len(serializable_candidates),
        "candidates": serializable_candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--search-seed", type=int, required=True)
    parser.add_argument(
        "--split-role",
        choices=("development", "validation", "confirmatory"),
        required=True,
    )
    parser.add_argument(
        "--action-selection", choices=("unmasked", "corrected_v1"), required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--component-checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--diagnostic-limit", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", default="")
    return parser.parse_args()


class FrontierController:
    def __init__(
        self,
        world_model: VectorWorldModel,
        planner: BasePlanner,
        *,
        evaluation_seed: int,
        action_selection: str,
        max_steps: int = 128,
        candidate_trace_sink: CandidateTraceSink | None = None,
        candidate_trace_fraction: float = 0.1,
        candidate_trace_k: int = 64,
        candidate_prefix_lengths: tuple[int, ...] = (2, 4, 8),
        candidate_trace_keys: set[tuple[str, int]] | None = None,
    ) -> None:
        if action_selection not in ("unmasked", "corrected_v1"):
            raise ValueError("unknown action-selection protocol")
        self.world_model = world_model
        self.planner = planner
        self.evaluation_seed = int(evaluation_seed)
        self.action_selection = action_selection
        self.max_steps = int(max_steps)
        self.candidate_trace_sink = candidate_trace_sink
        self.candidate_trace_fraction = float(candidate_trace_fraction)
        self.candidate_trace_k = int(candidate_trace_k)
        self.candidate_prefix_lengths = tuple(candidate_prefix_lengths)
        self.candidate_trace_keys = candidate_trace_keys
        self.context: VectorContext | None = None
        self.task_index = -1
        self.task_identifier = ""
        self.step = 0
        self.last_action: int | None = None
        self.decision_traces: list[dict[str, Any]] = []

    def reset(
        self,
        env: Any,
        observation: np.ndarray,
        task_index: int,
        task_identifier: str | None = None,
    ) -> None:
        self.task_index = int(task_index)
        self.task_identifier = task_identifier or f"task-index-{task_index}"
        self.step = 0
        self.last_action = None
        self.decision_traces = []
        maze_size = int(env.config.width)
        start = self.world_model.encode(observation, maze_size)
        goal_observation = observe_state(env, int(env._goal_position))
        goal = self.world_model.encode(goal_observation, maze_size)
        self.context = self.world_model.initial_context(
            start,
            goal,
            maze_size=maze_size,
            context_action=int(env.config.action_vocab_size) - 1,
            remaining_steps=self.max_steps,
        )
        self.planner.reset()

    def _advance_context(self, observation: np.ndarray) -> None:
        if self.step == 0:
            return
        if self.context is None or self.last_action is None:
            raise RuntimeError("controller context was not initialized")
        current = self.world_model.encode(observation, self.context.maze_size)
        self.context = self.world_model.advance_context(
            self.context, current, self.last_action
        )

    def _corrected_one_step(
        self,
        env: Any,
        state: int,
        previous: int | None,
        ledger: ComputeLedger,
    ) -> int:
        if self.context is None:
            raise RuntimeError("controller context was not initialized")
        allowed = corrected_actions(env, state, previous)
        if not allowed:
            raise RuntimeError("a free maze state must have one moving action")
        predicted = self.world_model.one_step_all_actions(
            self.context, action_vocab_size=5, ledger=ledger
        )
        goal = self.context.goal.expand(5, -1, -1).squeeze(1)
        scores = F.mse_loss(predicted, goal, reduction="none").sum(dim=-1)
        allowed_tensor = torch.tensor(
            allowed, dtype=torch.long, device=self.world_model.device
        )
        return int(allowed[int(scores[allowed_tensor].argmin())])

    def choose(
        self,
        env: Any,
        observation: np.ndarray,
        state: int,
        previous: int | None,
    ) -> tuple[int, dict[str, float]]:
        self._advance_context(observation)
        if self.context is None:
            raise RuntimeError("controller context was not initialized")
        self.planner.observe_real_state(self.context.embeddings[:, -1])
        started = time.perf_counter()
        planner_seed = task_seed(self.evaluation_seed, self.task_index, self.step)
        result = self.planner.plan(
            self.context,
            seed=planner_seed,
        )
        candidate_metrics: dict[str, Any] | None = None
        trace_selected = (
            (self.task_identifier, self.step) in self.candidate_trace_keys
            if self.candidate_trace_keys is not None
            else candidate_trace_selected(
                self.task_identifier,
                self.step,
                self.candidate_trace_fraction,
            )
        )
        if self.candidate_trace_sink is not None and trace_selected:
            diagnostic_ledger = ComputeLedger()
            rescored_batches = []
            for batch in result.candidate_batches:
                costs = batch.predicted_costs
                source = batch.source
                if costs is None:
                    costs, rescore_ledger = self.planner.diagnostic_costs(
                        self.context, batch.sequences
                    )
                    diagnostic_ledger.merge(rescore_ledger)
                    source = f"{source}:post_decision_rescore"
                rescored_batches.append(
                    CandidateBatch(
                        sequences=batch.sequences,
                        predicted_costs=costs,
                        source=source,
                    )
                )
            result_for_analysis = PlannerResult(
                sequence=result.sequence,
                cost=result.cost,
                ledger=result.ledger,
                diagnostics=result.diagnostics,
                candidate_batches=tuple(rescored_batches),
            )
            candidate_record = analyze_candidate_result(
                env,
                task_identifier=self.task_identifier,
                task_index=self.task_index,
                step=self.step,
                seed=planner_seed,
                root_state=state,
                result=result_for_analysis,
                candidate_k=self.candidate_trace_k,
                prefix_lengths=self.candidate_prefix_lengths,
            )
            candidate_record["diagnostic_rescore_compute"] = diagnostic_ledger.to_dict()
            candidate_record["diagnostic_rescore_excluded_from_planner_budget"] = True
            self.candidate_trace_sink.write(candidate_record)
            candidate_metrics = candidate_record["metrics"]
        proposed = int(result.sequence[0])
        action = proposed
        assisted = False
        assistance_reason = 0
        if self.action_selection == "corrected_v1":
            allowed = corrected_actions(env, state, previous)
            if proposed not in allowed:
                proposed_state = next_state(env, state, proposed)
                if proposed_state == state:
                    assistance_reason = 1
                elif previous is not None and proposed_state == previous:
                    assistance_reason = 2
                else:
                    assistance_reason = 4
                action = self._corrected_one_step(env, state, previous, result.ledger)
                assisted = True
        elapsed = time.perf_counter() - started
        candidate = next_state(env, state, proposed)
        validate_compute_ledger(result.ledger.to_dict())
        trace = {
            "step": self.step,
            "seed": planner_seed,
            "proposed_action": proposed,
            "executed_action": action,
            "assisted": assisted,
            "assistance_reason": assistance_reason,
            "best_cost": result.cost,
            "sequence": result.sequence.tolist(),
            "compute": result.ledger.to_dict(),
            "planner_diagnostics": result.diagnostics,
            "candidate_trace_recorded": candidate_metrics is not None,
            "candidate_trace_metrics": candidate_metrics,
        }
        validate_finite_tree(trace)
        self.decision_traces.append(trace)
        self.last_action = action
        self.step += 1
        ledger = result.ledger
        return action, {
            "decision_seconds": float(elapsed),
            "proposed_invalid": float(candidate == state),
            "proposed_backtrack": float(previous is not None and candidate == previous),
            "assisted_action": float(assisted),
            "assistance_reason_1": float(assistance_reason == 1),
            "assistance_reason_2": float(assistance_reason == 2),
            "assistance_reason_4": float(assistance_reason == 4),
            "plan_transitions": float(ledger.plan_transitions),
            "assist_transitions": float(ledger.assist_transitions),
            "total_transitions": float(ledger.total_transitions),
            "planner_forward_calls": float(ledger.planner_forward_calls),
            "assist_forward_calls": float(ledger.assist_forward_calls),
            "planner_max_batch": float(ledger.planner_max_batch),
            "node_expansions": float(ledger.node_expansions),
            "candidate_sequences": float(ledger.candidate_sequences),
            "duplicate_candidates": float(ledger.duplicate_candidates),
            "verifier_forward_calls": float(ledger.verifier_forward_calls),
            "reachability_forward_calls": float(ledger.reachability_forward_calls),
            "ranker_forward_calls": float(ledger.ranker_forward_calls),
            "proposal_forward_calls": float(ledger.proposal_forward_calls),
            "join_forward_calls": float(ledger.join_forward_calls),
            "dts_forward_calls": float(ledger.dts_forward_calls),
            "best_planner_cost": float(result.cost),
        }


def _path_revisit_metrics(path: list[int]) -> dict[str, Any]:
    if not path:
        raise ValueError("path diagnostics require the initial state")
    visits = Counter(path)
    executed_steps = len(path) - 1
    step_denominator = max(executed_steps, 1)
    two_cycle_count = sum(
        path[index] == path[index - 2] for index in range(2, len(path))
    )
    short_cycle_periods = [
        period
        for index in range(2, len(path))
        for period in range(2, min(8, index) + 1)
        if path[index] == path[index - period]
    ]
    metrics = {
        "repeat_states": int(sum(max(count - 1, 0) for count in visits.values())),
        "revisit_rate": float(
            sum(max(count - 1, 0) for count in visits.values()) / step_denominator
        ),
        "unique_state_ratio": float(max(len(visits) - 1, 0) / step_denominator),
        "two_cycle_rate": float(two_cycle_count / max(executed_steps - 1, 1)),
        "short_cycle_event": bool(short_cycle_periods),
        "short_cycle_periods": sorted(set(short_cycle_periods)),
        "max_state_visits": int(max(visits.values())),
        "loop_or_cycle": bool(max(visits.values()) >= 4),
    }
    for name in ("revisit_rate", "unique_state_ratio", "two_cycle_rate"):
        if not 0.0 <= float(metrics[name]) <= 1.0:
            raise AssertionError(f"path metric outside [0,1]: {name}")
    return metrics


def run_episode(
    entry: dict[str, Any],
    controller: FrontierController,
    *,
    task_index: int,
    max_steps: int,
) -> dict[str, Any]:
    env = validate_manifest_entry(entry)
    start = int(entry["start_cell"])
    goal = int(entry["goal_cell"])
    optimal_length = int(entry["bfs_path_length"])
    observation = set_agent_state(env, start)
    started = time.perf_counter()
    controller.reset(
        env,
        observation,
        task_index,
        task_identifier=task_id(entry),
    )
    state = start
    previous: int | None = None
    path = [state]
    invalid_actions = 0
    auxiliary: defaultdict[str, float] = defaultdict(float)
    for _ in range(max_steps):
        if state == goal:
            break
        action, metrics = controller.choose(env, observation, state, previous)
        if action not in ACTION_IDS:
            raise ValueError(f"controller selected out-of-protocol action {action}")
        for name, value in metrics.items():
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"non-finite episode metric: {name}")
            auxiliary[name] += float(value)
        old_state = state
        observation, _, _, _, info = env.step(action)
        state = int(info["state"])
        previous = old_state
        invalid_actions += int(state == old_state)
        path.append(state)
    success = state == goal
    distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
    structure = maze_structure_metrics(env)
    path_length = len(path) - 1
    path_metrics = _path_revisit_metrics(path)
    decision_count = max(path_length, 1)
    shortest_path_bin = (
        "le16"
        if optimal_length <= 16
        else "17_32"
        if optimal_length <= 32
        else "33_64"
        if optimal_length <= 64
        else "65_128"
        if optimal_length <= 128
        else "gt128"
    )
    dead_end_states = {
        int(cell)
        for cell in np.flatnonzero((~env._maze_mask).reshape(-1)).tolist()
        if sum(next_state(env, int(cell), action) != int(cell) for action in ACTION_IDS)
        == 1
        and int(cell) != goal
    }
    dead_end_entries = [
        index
        for index in range(1, len(path))
        if path[index] in dead_end_states and path[index - 1] != path[index]
    ]
    dead_end_recoveries = sum(
        any(distances[later] < distances[path[index]] for later in path[index + 1 :])
        for index in dead_end_entries
    )
    row = {
        "task_id": task_id(entry),
        "maze_size": int(entry["maze_size"]),
        "topology_seed": int(entry["topology_seed"]),
        "start_cell": start,
        "goal_cell": goal,
        "optimal_length": optimal_length,
        "success": bool(success),
        "path_length": path_length,
        "spl": float(optimal_length / max(optimal_length, path_length))
        if success
        else 0.0,
        "invalid_actions": int(invalid_actions),
        **path_metrics,
        "final_bfs_distance": int(distances[state]),
        "shortest_path_bin": shortest_path_bin,
        **structure,
        "decision_count": path_length,
        "assistance_rate": float(auxiliary["assisted_action"] / decision_count),
        "invalid_correction_rate": float(
            auxiliary["assistance_reason_1"] / decision_count
        ),
        "backtrack_correction_rate": float(
            auxiliary["assistance_reason_2"] / decision_count
        ),
        "dead_end_entries": len(dead_end_entries),
        "dead_end_recoveries": int(dead_end_recoveries),
        "dead_end_recovery_rate": float(
            dead_end_recoveries / max(len(dead_end_entries), 1)
        ),
        "episode_seconds": float(time.perf_counter() - started),
        "auxiliary": dict(auxiliary),
        "decision_traces": list(controller.decision_traces),
    }
    validate_task_rows([row], 1)
    validate_finite_tree(row)
    return row


def validate_candidate_replay(
    official: list[dict[str, Any]], replayed: list[dict[str, Any]]
) -> None:
    if len(official) != len(replayed):
        raise ValueError("candidate replay changed the task count")
    for reference, replay in zip(official, replayed, strict=True):
        for key in ("task_id", "success", "path_length", "final_bfs_distance"):
            if replay.get(key) != reference.get(key):
                raise ValueError(f"candidate replay diverged at {key}")
        reference_traces = reference["decision_traces"]
        replay_traces = replay["decision_traces"]
        if len(reference_traces) != len(replay_traces):
            raise ValueError("candidate replay changed the decision count")
        for left, right in zip(reference_traces, replay_traces, strict=True):
            for key in (
                "step",
                "seed",
                "proposed_action",
                "executed_action",
                "assistance_reason",
                "sequence",
            ):
                if left.get(key) != right.get(key):
                    raise ValueError(f"candidate replay diverged at decision {key}")


def maze_structure_metrics(env: Any) -> dict[str, float | int]:
    free = np.flatnonzero((~env._maze_mask).reshape(-1)).astype(np.int64)
    adjacency = {
        int(state): {
            next_state(env, int(state), action)
            for action in ACTION_IDS
            if next_state(env, int(state), action) != int(state)
        }
        for state in free.tolist()
    }
    degrees = {state: len(neighbors) for state, neighbors in adjacency.items()}
    dead_ends = sum(degree == 1 for degree in degrees.values())
    junctions = sum(degree >= 3 for degree in degrees.values())
    visited_edges: set[tuple[int, int]] = set()
    corridor_lengths: list[int] = []
    anchors = [state for state, degree in degrees.items() if degree != 2]
    for anchor in anchors:
        for neighbor in adjacency[anchor]:
            edge = tuple(sorted((anchor, neighbor)))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            length = 1
            previous, current = anchor, neighbor
            while degrees[current] == 2:
                following = next(
                    value for value in adjacency[current] if value != previous
                )
                next_edge = tuple(sorted((current, following)))
                if next_edge in visited_edges:
                    break
                visited_edges.add(next_edge)
                previous, current = current, following
                length += 1
            corridor_lengths.append(length)
    return {
        "free_cell_count": int(len(free)),
        "dead_end_density": float(dead_ends / max(len(free), 1)),
        "junction_count": int(junctions),
        "mean_corridor_length": float(np.mean(corridor_lengths))
        if corridor_lengths
        else 0.0,
    }


def _load_component_checkpoint(
    path: Path,
    *,
    config: StudyConfig,
    lock: dict[str, Any],
    method: MethodConfig,
    backbone_seed: int,
    planner_seed: int,
    device: torch.device,
    source_model: torch.nn.Module,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_method = resolve_effective_method(
        config, lock, component_checkpoint_owner(config, method)
    )
    if value.get("experiment_family") != EXPERIMENT_FAMILY:
        raise ValueError("component checkpoint belongs to another experiment")
    if int(value.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported component-checkpoint format")
    allowed_stages = {"component_calibration", "counterexample_training_round"}
    if value.get("stage") not in allowed_stages:
        raise ValueError("formal evaluation requires a calibrated component checkpoint")
    if (
        value.get("method_name") != checkpoint_method.name
        or int(value.get("backbone_seed", -1)) != backbone_seed
        or int(value.get("planner_seed", -1)) != planner_seed
    ):
        raise ValueError("component checkpoint method/seed mismatch")
    if value.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("component checkpoint analysis-spec mismatch")
    if value.get("training_spec_sha256") != training_spec_sha256(
        config,
        lock,
        method=checkpoint_method,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    ):
        raise ValueError("component checkpoint training-spec mismatch")
    checkpoint_protocol = value.get("protocol", {})
    if checkpoint_protocol.get("git_dirty") is not False:
        raise ValueError("formal evaluation rejects a dirty component checkpoint")
    if checkpoint_protocol.get("code_fingerprint") != lock["code_fingerprint"]:
        raise ValueError("component checkpoint code fingerprint mismatch")
    head_config = HeadConfig.from_dict(value["head_config"])
    constructors: dict[str, type[torch.nn.Module]] = {
        "verifier": ActionConsistencyVerifier,
        "reachability": DistributionalReachability,
        "join": StateJoinHead,
        "autoregressive_proposal": AutoregressiveProposal,
        "denoising_proposal": DiscreteDenoisingProposal,
        "dts": VectorDTSHead,
        "ranker": CounterexampleRanker,
    }
    modules: dict[str, torch.nn.Module] = {}
    for name, state_dict in value.get("head_state_dicts", {}).items():
        if name not in constructors:
            raise ValueError(f"unknown component head in checkpoint: {name}")
        module = constructors[name](head_config).to(device)
        module.load_state_dict(state_dict, strict=True)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
        modules[name] = module
    expected_heads = required_head_names(method)
    if set(modules) != expected_heads:
        raise ValueError(
            "component checkpoint head set mismatch: "
            f"expected={sorted(expected_heads)} actual={sorted(modules)}"
        )
    if method.track == "J":
        if value.get("model_state_dict") is None:
            raise ValueError("Track J checkpoint does not contain an adapted backbone")
        source_model.load_state_dict(value["model_state_dict"], strict=True)
        source_model.eval()
        for parameter in source_model.parameters():
            parameter.requires_grad = False
    elif value.get("model_state_dict") is not None:
        raise ValueError("Track F checkpoint unexpectedly contains backbone parameters")
    return modules, value


def build_controller(
    config: StudyConfig,
    lock: dict[str, Any],
    method: MethodConfig,
    *,
    seed: int,
    planner_seed: int = 0,
    search_seed: int | None = None,
    device: torch.device,
    action_selection: str,
    component_checkpoint: Path | None,
    candidate_trace_sink: CandidateTraceSink | None = None,
    candidate_trace_keys: set[tuple[str, int]] | None = None,
) -> tuple[FrontierController, dict[str, Any]]:
    model, source_checkpoint, source_path = load_source_lewm(
        config, lock, seed=seed, device=device
    )
    modules: dict[str, torch.nn.Module] = {}
    component_data: dict[str, Any] | None = None
    if method.component_checkpoint_required:
        if component_checkpoint is None or not component_checkpoint.exists():
            raise FileNotFoundError("method requires a component checkpoint")
        modules, component_data = _load_component_checkpoint(
            component_checkpoint,
            config=config,
            lock=lock,
            method=method,
            backbone_seed=seed,
            planner_seed=planner_seed,
            device=device,
            source_model=model,
        )
        if component_data.get("source_checkpoint_sha256") != sha256_file(source_path):
            raise ValueError("component and evaluation source checkpoints differ")
    world_model = VectorWorldModel(
        model, device=device, history_size=method.planner.history_size
    )
    validation_metrics = (component_data or {}).get("validation_metrics", {})
    retrieval_provenance: dict[str, Any] | None = None
    scorer = CompositeScorer(
        method.scorer,
        verifier=modules.get("verifier"),
        reachability=modules.get("reachability"),
        ranker=modules.get("ranker"),
        shuffle_candidate_association=(
            method.control.predictor_association == "candidate_shuffle"
        ),
    )
    proposal: Any = UniformProposal()
    if method.proposal.kind != ProposalKind.UNIFORM:
        retrieval = None
        learned = None
        if method.proposal.retrieval_weight > 0.0:
            bank_path = resolve_path(
                component_data["retrieval_bank_path"]
                if component_data is not None
                else config.paths.retrieval_bank_template.format(
                    method=method.name,
                    backbone_seed=seed,
                    planner_seed=planner_seed,
                )
            )
            bank = RetrievalBank.load(bank_path)
            train_task_hashes = {
                str(entry["task_hash"])
                for entry in read_jsonl(resolve_path(config.paths.train_manifest))
            }
            if not set(bank.task_hashes) <= train_task_hashes:
                raise ValueError("retrieval bank contains a non-training task hash")
            if component_data is not None and bank.fingerprint != (
                validation_metrics.get("retrieval_bank_fingerprint")
            ):
                raise ValueError("retrieval bank no longer matches calibration")
            retrieval_provenance = {
                "path": str(bank_path),
                "sha256": sha256_file(bank_path),
                "fingerprint": bank.fingerprint,
                "task_count": len(bank.task_hashes),
            }
            retrieval = RetrievalProposal(bank, top_k=method.proposal.retrieval_top_k)
        if method.proposal.learned_weight > 0.0:
            if "autoregressive_proposal" in modules:
                learned = LearnedAutoregressiveSampler(
                    modules["autoregressive_proposal"],
                    temperature=method.proposal.temperature,
                )
            elif "denoising_proposal" in modules:
                learned = DenoisingSampler(
                    modules["denoising_proposal"],
                    steps=method.proposal.denoising_steps,
                    temperature=method.proposal.temperature,
                )
            else:
                raise ValueError(
                    "learned proposal weight is positive but no model exists"
                )
        proposal = MixtureProposal(
            method.proposal, retrieval=retrieval, learned=learned
        )
    calibrated_memory = method.memory
    if method.memory.enabled:
        if "join_threshold" not in validation_metrics:
            raise ValueError("memory method has no calibrated join threshold")
        calibrated_memory = method.memory.model_copy(
            update={"join_threshold": float(validation_metrics["join_threshold"])}
        )
    planner = build_planner(
        world_model,
        method.planner,
        scorer,
        proposal=proposal,
        memory_config=calibrated_memory,
        join_head=modules.get("join"),
        join_precision=validation_metrics.get("join_precision"),
        dts_head=modules.get("dts"),
        proposal_only=method.control.proposal_execution == "proposal_only",
        dts_expansion=method.control.dts_expansion,
    )
    controller = FrontierController(
        world_model,
        planner,
        evaluation_seed=(
            config.protocol.evaluation_seed
            if search_seed is None
            else hierarchical_seed(
                "paired-search-seed",
                config.protocol.evaluation_seed,
                search_seed,
            )
        ),
        action_selection=action_selection,
        max_steps=config.protocol.max_steps,
        candidate_trace_sink=candidate_trace_sink,
        candidate_trace_fraction=config.analysis.candidate_trace_fraction,
        candidate_trace_k=config.analysis.candidate_trace_k,
        candidate_prefix_lengths=config.analysis.candidate_prefix_lengths,
        candidate_trace_keys=candidate_trace_keys,
    )
    provenance = {
        "backbone_seed": int(seed),
        "planner_seed": (
            int(planner_seed) if method.component_checkpoint_required else None
        ),
        "search_seed": int(search_seed) if search_seed is not None else None,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "source_training_seed": int(source_checkpoint["training_seed"]),
        "component_checkpoint": str(component_checkpoint)
        if component_checkpoint is not None
        else None,
        "component_checkpoint_sha256": sha256_file(component_checkpoint)
        if component_checkpoint is not None
        else None,
        "component_checkpoint_owner": (
            resolve_effective_method(
                config, lock, component_checkpoint_owner(config, method)
            ).name
            if component_checkpoint is not None
            else None
        ),
        "retrieval_bank": retrieval_provenance,
        "backbone_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "planner_parameter_count": int(
            sum(
                parameter.numel()
                for module in modules.values()
                for parameter in module.parameters()
            )
        ),
    }
    provenance["total_system_parameter_count"] = (
        provenance["backbone_parameter_count"] + provenance["planner_parameter_count"]
    )
    return controller, provenance


def run_evaluation(args: argparse.Namespace) -> None:
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    if lock.get("status") != "locked":
        raise RuntimeError("formal evaluation requires a completed protocol lock")
    if lock.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("config no longer matches the locked analysis specification")
    base_method = method_by_name(config, args.method)
    method = resolve_effective_method(config, lock, base_method)
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside the locked matrix")
    if args.planner_seed not in planner_seed_values(config, method):
        raise ValueError("planner seed lies outside this method's locked matrix")
    if args.search_seed not in config.protocol.search_seeds:
        raise ValueError("search seed lies outside the locked matrix")
    if args.split_role == "confirmatory" and not method.confirmatory_eligible:
        raise ValueError("method was not frozen as confirmatory eligible")
    if args.diagnostic_limit < 0:
        raise ValueError("diagnostic limit cannot be negative")
    if args.component_checkpoint and args.diagnostic_limit == 0:
        raise ValueError(
            "formal evaluation cannot override the locked component checkpoint"
        )
    if args.diagnostic_limit and args.split_role == "confirmatory":
        raise ValueError("confirmatory evaluation cannot use a task subset")
    formal_output = resolve_path(
        config.paths.result_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            search_seed=args.search_seed,
            split=args.split_role,
            action_selection=args.action_selection,
        )
    )
    if args.diagnostic_limit and resolve_path(args.output) == formal_output:
        raise ValueError("subset diagnostics cannot use the formal result path")
    if args.allow_dirty_worktree and args.diagnostic_limit == 0:
        raise ValueError("dirty worktrees are allowed only for subset diagnostics")
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    output_path = resolve_path(args.output)
    component_path = (
        resolve_path(args.component_checkpoint)
        if args.component_checkpoint
        else component_checkpoint_path(
            config,
            method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    opaque_run_id: str | None = None
    if args.split_role == "confirmatory":
        opaque_run_id = authorize_confirmatory_evaluation(
            config,
            lock,
            method=method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            search_seed=args.search_seed,
            action_selection=args.action_selection,
            output=output_path,
            component_checkpoint=component_path,
        )
        requested_run_id = getattr(args, "opaque_run_id", None)
        if requested_run_id is not None and requested_run_id != opaque_run_id:
            raise PermissionError("opaque run id does not match its private mapping")
    candidate_output = candidate_trace_path(output_path)
    rerun = prepare_formal_outputs(
        [output_path, candidate_output],
        overwrite=args.overwrite,
        rerun_reason=args.rerun_reason,
    )
    device = resolve_device(args.device or config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    set_seed(
        hierarchical_seed(
            "planner-evaluation",
            args.backbone_seed,
            args.planner_seed,
            args.search_seed,
        ),
        deterministic=True,
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
    controller, provenance = build_controller(
        config,
        lock,
        method,
        seed=args.backbone_seed,
        planner_seed=args.planner_seed,
        search_seed=args.search_seed,
        device=device,
        action_selection=args.action_selection,
        component_checkpoint=component_path,
    )
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
    selected_trace_keys, sampling_record = exact_candidate_trace_keys(
        rows, config.analysis.candidate_trace_fraction
    )
    trace_sink = CandidateTraceSink(candidate_output)
    replay_started = time.perf_counter()
    try:
        replay_controller, replay_provenance = build_controller(
            config,
            lock,
            method,
            seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            search_seed=args.search_seed,
            device=device,
            action_selection=args.action_selection,
            component_checkpoint=component_path,
            candidate_trace_sink=trace_sink,
            candidate_trace_keys=selected_trace_keys,
        )
        replay_rows = [
            run_episode(
                entry,
                replay_controller,
                task_index=index,
                max_steps=config.protocol.max_steps,
            )
            for index, entry in enumerate(entries)
        ]
        validate_candidate_replay(rows, replay_rows)
        if (
            replay_provenance["source_checkpoint_sha256"]
            != provenance["source_checkpoint_sha256"]
            or replay_provenance["component_checkpoint_sha256"]
            != provenance["component_checkpoint_sha256"]
        ):
            raise ValueError("candidate replay loaded different checkpoints")
        if trace_sink.count != len(selected_trace_keys):
            raise ValueError("candidate replay did not emit every selected decision")
        candidate_trace_artifact = trace_sink.commit()
    except BaseException:
        trace_sink.abort()
        raise
    candidate_trace_artifact.update(
        {
            "sampling": sampling_record,
            "replay_verified": True,
            "replay_wall_seconds": float(time.perf_counter() - replay_started),
            "replay_compute_excluded_from_planner_budget": True,
        }
    )
    payload = {
        "metadata": protocol_metadata(
            config,
            lock,
            method=method,
            seed=args.backbone_seed,
            planner_seed=(
                args.planner_seed if method.component_checkpoint_required else None
            ),
            search_seed=args.search_seed,
            device=device,
        ),
        "stage": "planner_evaluation",
        "opaque_run_id": opaque_run_id,
        "split_role": args.split_role,
        "action_selection": args.action_selection,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "count": len(entries),
        },
        "provenance": provenance,
        "candidate_traces": candidate_trace_artifact,
        "rerun": rerun,
        "summary": summarize_rows(rows, seen_max_size=config.protocol.seen_max_size),
        "resources": {
            "peak_accelerator_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "evaluation_wall_seconds": float(
                sum(float(row["episode_seconds"]) for row in rows)
            ),
            "candidate_replay_wall_seconds": candidate_trace_artifact[
                "replay_wall_seconds"
            ],
        },
        "tasks": rows,
    }
    validate_finite_tree(payload)
    atomic_json_dump(output_path, payload)


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
