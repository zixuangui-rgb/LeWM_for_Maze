#!/usr/bin/env python3
"""Evaluate fixed baselines on the exact development or confirmatory tasks."""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from final_closure.common import (
    ACTION_IDS,
    ACTION_TO_SLOT,
    RERUN_REASONS,
    atomic_json_dump,
    baseline_config,
    bfs_distances_from,
    corrected_actions,
    estimate_forward_macs,
    load_checkpoint,
    load_config,
    next_state,
    observe_state,
    pad_bc_observation,
    prepare_rerun,
    protocol_metadata,
    read_jsonl,
    require_clean_worktree,
    require_new_output,
    require_study_open,
    resolve_device,
    set_agent_state,
    set_seed,
    sha256_file,
    summarize_rows,
    task_id,
    task_seed,
    validate_manifest_entry,
)
from final_closure.models import (
    BCPolicyConfig,
    DeepCNNPolicy,
    deserialize_lewm_config,
)
from hdwm.planning import cem_plan
from scripts.train.train_dim256 import Unisize256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="final_closure/configs/default.json")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split-role", choices=("development", "confirmatory"), required=True
    )
    parser.add_argument(
        "--action-selection", choices=("unmasked", "corrected"), required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diagnostic-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", choices=RERUN_REASONS, default="")
    return parser.parse_args()


class BCController:
    def __init__(
        self,
        model: DeepCNNPolicy,
        *,
        device: torch.device,
        canvas_size: int,
        action_selection: str,
    ) -> None:
        self.model = model
        self.device = device
        self.canvas_size = canvas_size
        self.action_selection = action_selection

    def reset(self, env: Any, observation: np.ndarray, task_index: int) -> None:
        del env, observation, task_index

    def choose(
        self,
        env: Any,
        observation: np.ndarray,
        state: int,
        previous: int | None,
    ) -> tuple[int, dict[str, float]]:
        inputs = pad_bc_observation(observation, self.canvas_size).unsqueeze(0)
        started = time.perf_counter()
        with torch.no_grad():
            logits = self.model(inputs.to(self.device)).squeeze(0)
        elapsed = time.perf_counter() - started
        if not torch.isfinite(logits).all():
            raise FloatingPointError("BC produced non-finite logits")
        proposed_slot = int(logits.argmax())
        proposed = int(ACTION_IDS[proposed_slot])
        action = proposed
        assisted = False
        if self.action_selection == "corrected":
            allowed = corrected_actions(env, state, previous)
            if allowed and proposed not in allowed:
                allowed_slots = torch.tensor(
                    [ACTION_TO_SLOT[value] for value in allowed],
                    dtype=torch.long,
                    device=logits.device,
                )
                action = int(allowed[int(logits[allowed_slots].argmax())])
                assisted = True
        candidate = next_state(env, state, proposed)
        return action, {
            "decision_seconds": float(elapsed),
            "proposed_invalid": float(candidate == state),
            "proposed_backtrack": float(previous is not None and candidate == previous),
            "assisted_action": float(assisted),
            "policy_forward_calls": 1.0,
        }


class LeWMCEMController:
    def __init__(
        self,
        model: Unisize256,
        planner_config: dict[str, Any],
        *,
        device: torch.device,
        evaluation_seed: int,
        action_selection: str,
    ) -> None:
        self.model = model
        self.config = planner_config
        self.device = device
        self.evaluation_seed = evaluation_seed
        self.action_selection = action_selection
        if self.config["context_action_initialization"] != "repeat_action_id_4":
            raise ValueError("unsupported locked LeWM context initialization")
        if self.config["cem_seed_schedule"] != "historical_eval_seed_task_step":
            raise ValueError("unsupported locked CEM seed schedule")
        self.context_embedding: torch.Tensor | None = None
        self.context_actions: torch.Tensor | None = None
        self.goal_embedding: torch.Tensor | None = None
        self.last_action: int | None = None
        self.step = 0
        self.task_index = -1
        self.maze_size = -1

    def encode(self, observation: np.ndarray) -> torch.Tensor:
        inputs = (
            torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        with torch.no_grad():
            encoded = self.model.encoder(inputs, self.maze_size)
            embedding, _ = self.model.embedding_projector(encoded)
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("LeWM encoder produced non-finite embeddings")
        return embedding

    @staticmethod
    def score(terminal: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(terminal, goal, reduction="none").sum(dim=-1)

    def reset(self, env: Any, observation: np.ndarray, task_index: int) -> None:
        self.maze_size = int(env.config.width)
        self.task_index = int(task_index)
        self.step = 0
        self.last_action = None
        start_embedding = self.encode(observation)
        history = int(self.config["history_size"])
        self.context_embedding = start_embedding.repeat(1, history, 1)
        self.context_actions = torch.full(
            (1, history),
            env.config.action_vocab_size - 1,
            dtype=torch.long,
            device=self.device,
        )
        goal_observation = observe_state(env, int(env._goal_position))
        self.goal_embedding = self.encode(goal_observation)

    def _advance_context(self, observation: np.ndarray) -> None:
        if self.step == 0:
            return
        if self.last_action is None:
            raise RuntimeError("LeWM context has no previous action")
        assert self.context_embedding is not None
        assert self.context_actions is not None
        current = self.encode(observation)
        self.context_embedding = torch.cat(
            [self.context_embedding[:, 1:], current], dim=1
        )
        action = torch.tensor(
            [[self.last_action]], dtype=torch.long, device=self.device
        )
        self.context_actions = torch.cat([self.context_actions[:, 1:], action], dim=1)

    def _corrected_one_step(
        self,
        env: Any,
        state: int,
        previous: int | None,
    ) -> int:
        allowed = corrected_actions(env, state, previous)
        if not allowed:
            raise RuntimeError("a free maze state must have at least one moving action")
        assert self.context_embedding is not None
        assert self.context_actions is not None
        assert self.goal_embedding is not None
        action_count = int(env.config.action_vocab_size)
        embedding = self.context_embedding.expand(action_count, -1, -1)
        actions = self.context_actions[:, :-1].repeat(action_count, 1)
        actions[:, -1] = torch.arange(action_count, device=self.device)
        with torch.no_grad():
            predicted = self.model.predictor(embedding, actions)[:, -1]
            goal = self.goal_embedding.expand(action_count, -1, -1).squeeze(1)
            scores = self.score(predicted, goal)
        if not torch.isfinite(scores).all():
            raise FloatingPointError(
                "LeWM one-step correction produced non-finite scores"
            )
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=self.device)
        return int(allowed[int(scores[allowed_tensor].argmin())])

    def choose(
        self,
        env: Any,
        observation: np.ndarray,
        state: int,
        previous: int | None,
    ) -> tuple[int, dict[str, float]]:
        self._advance_context(observation)
        assert self.context_embedding is not None
        assert self.context_actions is not None
        assert self.goal_embedding is not None
        started = time.perf_counter()
        best_sequence, best_cost, _ = cem_plan(
            self.model,
            self.context_embedding,
            self.context_actions,
            self.goal_embedding,
            horizon=int(self.config["horizon"]),
            history_size=int(self.config["history_size"]),
            num_candidates=int(self.config["num_candidates"]),
            num_elites=int(self.config["num_elites"]),
            cem_iters=int(self.config["cem_iters"]),
            momentum=float(self.config["momentum"]),
            num_actions=int(env.config.action_vocab_size),
            device=self.device,
            seed=task_seed(self.evaluation_seed, self.task_index, self.step),
            score_fn=self.score,
            allowed_actions=np.asarray(self.config["allowed_actions"], np.int64),
        )
        if not math.isfinite(float(best_cost)):
            raise FloatingPointError("CEM produced a non-finite best cost")
        proposed = int(best_sequence[0])
        action = proposed
        assisted = False
        fallback_predictions = 0
        if self.action_selection == "corrected":
            allowed = corrected_actions(env, state, previous)
            if proposed not in allowed:
                action = self._corrected_one_step(env, state, previous)
                assisted = True
                fallback_predictions = int(env.config.action_vocab_size)
        elapsed = time.perf_counter() - started
        candidate = next_state(env, state, proposed)
        self.last_action = action
        self.step += 1
        rollout_predictions = (
            int(self.config["num_candidates"])
            * int(self.config["horizon"])
            * int(self.config["cem_iters"])
        )
        return action, {
            "decision_seconds": float(elapsed),
            "proposed_invalid": float(candidate == state),
            "proposed_backtrack": float(previous is not None and candidate == previous),
            "assisted_action": float(assisted),
            "cem_calls": 1.0,
            "cem_candidate_transition_predictions": float(rollout_predictions),
            "fallback_transition_predictions": float(fallback_predictions),
            "best_cem_cost": float(best_cost),
        }


def load_model(
    baseline: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    if baseline["kind"] == "bc":
        model = DeepCNNPolicy(BCPolicyConfig.from_dict(checkpoint["model_config"])).to(
            device
        )
        model.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    elif baseline["kind"] == "lewm_l2_cem":
        model_config = deserialize_lewm_config(checkpoint["model_config"])
        model = Unisize256(
            model_config, max_size=int(baseline["train"]["max_size_embedding"])
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        raise ValueError(f"unsupported baseline kind: {baseline['kind']}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def run_episode(
    entry: dict[str, Any],
    controller: BCController | LeWMCEMController,
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
    controller.reset(env, observation, task_index)
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
    visits = Counter(path)
    path_length = len(path) - 1
    return {
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
        "repeat_states": int(sum(max(count - 1, 0) for count in visits.values())),
        "max_state_visits": int(max(visits.values())),
        "loop_or_cycle": bool(max(visits.values()) >= 4),
        "final_bfs_distance": int(distances[state]),
        "episode_seconds": float(time.perf_counter() - started),
        "auxiliary": dict(auxiliary),
    }


def aggregate_compute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for row in rows for name in row.get("auxiliary", {})})
    return {
        "task_count": len(rows),
        "decision_count": int(sum(int(row["path_length"]) for row in rows)),
        "wallclock_seconds": float(sum(float(row["episode_seconds"]) for row in rows)),
        "auxiliary_totals": {
            name: float(
                sum(float(row.get("auxiliary", {}).get(name, 0.0)) for row in rows)
            )
            for name in names
        },
    }


def main() -> None:
    args = parse_args()
    config, lock = load_config(args.config)
    baseline = baseline_config(config, args.baseline)
    if args.training_seed not in [int(value) for value in config["seeds"]]:
        raise ValueError("training seed is outside the locked matrix")
    if args.diagnostic_limit < 0:
        raise ValueError("diagnostic task limit must be non-negative")
    if args.diagnostic_limit > 0 and not args.diagnostic:
        raise ValueError("a task limit requires --diagnostic")
    expected_primary = str(config["protocol"]["primary_action_selection"])
    allowed_diagnostics = set(config["protocol"]["diagnostic_action_selections"])
    if (
        args.action_selection != expected_primary
        and args.action_selection not in allowed_diagnostics
    ):
        raise ValueError("action-selection mode is outside the protocol lock")
    if not args.diagnostic:
        require_study_open(config)
    require_clean_worktree(args.allow_dirty_worktree or args.diagnostic)
    rerun = prepare_rerun(
        [args.output], overwrite=args.overwrite, reason=args.rerun_reason
    )
    require_new_output(args.output, args.overwrite)
    role_key = f"{args.split_role}_manifest"
    manifest = config["paths"][role_key]
    if sha256_file(manifest) != lock[role_key]["sha256"]:
        raise ValueError(f"{role_key} hash mismatch")
    entries = read_jsonl(manifest)
    expected_count = int(lock[role_key]["count"])
    if len(entries) != expected_count:
        raise ValueError(f"{role_key} count mismatch")
    if args.diagnostic_limit > 0:
        entries = entries[: args.diagnostic_limit]
    formal = not args.diagnostic
    checkpoint = load_checkpoint(
        args.checkpoint,
        config=config,
        lock=lock,
        name=baseline["name"],
        seed=args.training_seed,
        strict_provenance=formal,
    )
    if formal and checkpoint.get("formal_run") is not True:
        raise ValueError("formal evaluation rejects a diagnostic checkpoint")
    set_seed(int(config["protocol"]["evaluation_seed"]), deterministic=True)
    device = resolve_device(args.device or config["device"])
    model = load_model(baseline, checkpoint, device)
    if baseline["kind"] == "bc":
        controller: BCController | LeWMCEMController = BCController(
            model,
            device=device,
            canvas_size=int(baseline["train"]["train_canvas_size"]),
            action_selection=args.action_selection,
        )
    else:
        controller = LeWMCEMController(
            model,
            baseline["planner"],
            device=device,
            evaluation_seed=int(config["protocol"]["evaluation_seed"]),
            action_selection=args.action_selection,
        )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, entry in enumerate(entries):
        rows.append(
            run_episode(
                entry,
                controller,
                task_index=index,
                max_steps=int(config["protocol"]["max_steps"]),
            )
        )
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            current_sr = float(np.mean([float(row["success"]) for row in rows]))
            print(
                f"{baseline['name']} {args.split_role} {args.action_selection} "
                f"{index + 1:>4d}/{len(entries)} SR={current_sr:.4f}",
                flush=True,
            )
    compute = aggregate_compute(rows)
    if baseline["kind"] == "bc":
        canvas = int(baseline["train"]["train_canvas_size"])
        compute["forward_macs_by_maze_size"] = {
            str(size): estimate_forward_macs(
                model,
                torch.zeros(
                    (1, 5, max(size, canvas), max(size, canvas)),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            for size in sorted({int(entry["maze_size"]) for entry in entries})
        }
    metadata = protocol_metadata(
        config,
        lock,
        seed=int(config["protocol"]["evaluation_seed"]),
        device=device,
    )
    metadata.update(
        {
            "baseline_name": baseline["name"],
            "baseline_kind": baseline["kind"],
            "training_seed": int(args.training_seed),
            "split_role": args.split_role,
            "evaluated_manifest_sha256": sha256_file(manifest),
            "task_count": len(entries),
            "expected_full_task_count": expected_count,
            "action_selection": args.action_selection,
            "oracle_action_assistance": args.action_selection == "corrected",
            "formal_evaluation": formal,
            "rerun": rerun,
            "comparable_to_primary": bool(
                formal
                and args.split_role == "confirmatory"
                and args.action_selection == expected_primary
                and len(entries) == int(config["protocol"]["full_eval_count"])
            ),
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "training_git_commit": checkpoint["protocol"]["git_commit"],
            "training_git_dirty": checkpoint["protocol"]["git_dirty"],
            "training_code_fingerprint": checkpoint["protocol"]["code_fingerprint"],
            "training_runtime": checkpoint["protocol"]["runtime"],
            "parameter_count": int(checkpoint["parameter_count"]),
            "evaluation_elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    payload = {
        "metadata": metadata,
        "results": {
            "navigation": summarize_rows(
                rows,
                seen_max_size=int(config["protocol"]["seen_max_size"]),
                max_steps=int(config["protocol"]["max_steps"]),
            ),
            "task_rows": rows,
            "compute": compute,
        },
    }
    atomic_json_dump(args.output, payload)
    overall = payload["results"]["navigation"]["overall"]
    print(f"saved {args.output}: SR={overall['sr']:.4f}, SPL={overall['spl']:.4f}")


if __name__ == "__main__":
    main()
