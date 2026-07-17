"""Evaluate one locked method with task-level corrected and unmasked outputs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from distance_head_study import MODEL_ACTION_VOCAB_SIZE, PROTOCOL_ID
from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    head_checkpoint_path,
    load_json,
    load_study_config,
    read_jsonl,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    sha256_file,
    source_backbone_path,
    validate_backbone_protocol_binding,
)
from distance_head_study.data import (
    load_backbone_checkpoint,
    validate_recorded_cache_binding,
)
from distance_head_study.gates import (
    load_signed_artifact,
    require_evaluation_gate,
    require_seed_released,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.models import DistanceHeadModel, build_distance_head
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.schemas import CostKind, HeadSpec, PlannerKind, ResolvedMethod
from final_closure.common import corrected_actions, task_seed
from hdwm.planning import cem_plan
from spatial_jepa_planning.common import (
    ACTION_IDS,
    bfs_distances_from,
    next_state,
    observe_state,
    set_agent_state,
    summarize_rows,
    task_id,
    validate_manifest_entry,
)
from vector_jepa_planner_frontier.common import ComputeLedger
from vector_jepa_planner_frontier.planners import (
    BasePlanner,
    ScoreBatch,
    build_planner,
)
from vector_jepa_planner_frontier.schemas import (
    BudgetConfig,
    RolloutSemantics,
)
from vector_jepa_planner_frontier.schemas import (
    PlannerConfig as FrontierPlannerConfig,
)
from vector_jepa_planner_frontier.schemas import (
    PlannerKind as FrontierPlannerKind,
)
from vector_jepa_planner_frontier.world_model import (
    RolloutBatch,
    VectorContext,
    VectorWorldModel,
)

RESULT_SCHEMA = "distance-head-task-results-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--split-role",
        choices=("screen", "select", "confirm", "stress", "legacy"),
        required=True,
    )
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--head-seed", type=int, default=0)
    parser.add_argument(
        "--action-protocol", choices=("corrected_v1", "unmasked"), required=True
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--diagnostic-limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def _require_method_evaluation_gate(
    config: Any,
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
) -> dict[str, Any]:
    """Gate formal and limited evaluations identically."""

    gate = require_evaluation_gate(
        config,
        split_role=split_role,
        method=method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    if gate is not None:
        return gate
    return require_seed_released(
        config,
        backbone_seed=backbone_seed,
        head_seed=0 if method == "b_l2_cem" else head_seed,
    )


def _signed_decision_selected(
    path: Path, protocol_lock: dict[str, Any] | None = None
) -> tuple[str, str]:
    payload = load_signed_artifact(
        path,
        signature_field="decision_sha256",
        expected_protocol_id=PROTOCOL_ID,
        verify_hash_fields=("input_hashes",),
    )
    if protocol_lock is not None and (
        payload.get("analysis_spec_sha256") != protocol_lock["analysis_spec_sha256"]
        or payload.get("protocol_lock_sha256") != protocol_lock["protocol_lock_sha256"]
    ):
        raise ValueError(f"decision uses another protocol lock: {path}")
    selected = payload.get("selected_method")
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"decision has no selected method: {path}")
    return selected, str(payload["decision_sha256"])


def _checkpoint_owner(
    config: Any,
    method: ResolvedMethod,
    *,
    protocol_lock: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    if method.head is None:
        return None, None
    if method.checkpoint_owner:
        return method.checkpoint_owner, None
    if method.reuse_parent_checkpoint:
        assert method.checkpoint_decision_alias is not None
        return _signed_decision_selected(
            resolve_path(config.paths.decision_root)
            / f"{method.checkpoint_decision_alias}.json",
            protocol_lock,
        )
    return method.name, None


def _load_models(
    config: Any,
    method: ResolvedMethod,
    *,
    backbone_seed: int,
    head_seed: int,
    device: torch.device,
    expected_analysis_spec_sha256: str | None = None,
    expected_protocol_lock_sha256: str | None = None,
) -> tuple[VectorWorldModel, DistanceHeadModel | None, dict[str, Any]]:
    if (expected_analysis_spec_sha256 is None) != (
        expected_protocol_lock_sha256 is None
    ):
        raise ValueError("analysis and protocol lock expectations must be paired")
    backbone_path = source_backbone_path(config, backbone_seed)
    model, backbone_payload = load_backbone_checkpoint(
        backbone_path, device, freeze=True
    )
    if expected_protocol_lock_sha256 is not None:
        validate_backbone_protocol_binding(
            config,
            backbone_payload,
            backbone_seed=backbone_seed,
            protocol_lock={
                "analysis_spec_sha256": expected_analysis_spec_sha256,
                "protocol_lock_sha256": expected_protocol_lock_sha256,
            },
        )
    active_lock = (
        {
            "analysis_spec_sha256": expected_analysis_spec_sha256,
            "protocol_lock_sha256": expected_protocol_lock_sha256,
        }
        if expected_protocol_lock_sha256 is not None
        else None
    )
    owner, decision_hash = _checkpoint_owner(config, method, protocol_lock=active_lock)
    provenance: dict[str, Any] = {
        "backbone_path": backbone_path.as_posix(),
        "backbone_sha256": sha256_file(backbone_path),
        "checkpoint_owner": owner,
        "checkpoint_owner_decision_sha256": decision_hash,
        "joint_model_state_loaded": False,
    }
    head: DistanceHeadModel | None = None
    if owner is not None:
        owner_method, owner_method_hash, _ = load_and_resolve_method(
            config.paths.method_catalog,
            owner,
            decision_root=config.paths.decision_root,
            protocol_lock=active_lock,
        )
        checkpoint_path = head_checkpoint_path(
            config,
            method=owner,
            backbone_seed=backbone_seed,
            head_seed=head_seed,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if (
            checkpoint.get("protocol_id") != config.protocol_id
            or checkpoint.get("formal_run") is not True
            or checkpoint.get("checkpoint_selection") != "final_step"
            or int(checkpoint.get("final_step", -1)) != config.training.steps
        ):
            raise ValueError("formal evaluation accepts final-step heads only")
        if (
            expected_analysis_spec_sha256 is not None
            and checkpoint.get("analysis_spec_sha256") != expected_analysis_spec_sha256
        ):
            raise ValueError("DistanceHead checkpoint uses another analysis lock")
        if (
            expected_protocol_lock_sha256 is not None
            and checkpoint.get("protocol_lock_sha256") != expected_protocol_lock_sha256
        ):
            raise ValueError("DistanceHead checkpoint uses another protocol lock")
        if checkpoint.get("backbone_sha256") != sha256_file(backbone_path):
            raise ValueError("DistanceHead checkpoint uses another backbone")
        if checkpoint.get("method", {}).get("name") != owner:
            raise ValueError("DistanceHead checkpoint owner metadata mismatch")
        if checkpoint.get("method_sha256") != owner_method_hash:
            raise ValueError("DistanceHead checkpoint method hash mismatch")
        if method.head is None or checkpoint.get("head_spec") != method.head.model_dump(
            mode="json"
        ):
            raise ValueError("DistanceHead checkpoint spec differs from the method")
        if owner_method.head is None or checkpoint.get(
            "head_spec"
        ) != owner_method.head.model_dump(mode="json"):
            raise ValueError("DistanceHead checkpoint spec differs from its owner")
        bank = checkpoint.get("candidate_bank", {})
        bank_path = bank.get("path")
        if not isinstance(bank_path, str) or sha256_file(bank_path) != bank.get(
            "sha256"
        ):
            raise ValueError("DistanceHead checkpoint candidate bank changed")
        cache_bindings = checkpoint.get("cache_bindings")
        if not isinstance(cache_bindings, dict) or set(cache_bindings) != {
            "train",
            "cal",
        }:
            raise ValueError("DistanceHead checkpoint cache bindings are incomplete")
        if active_lock is None:
            raise ValueError(
                "formal DistanceHead loading requires an active protocol lock"
            )
        for split_role, binding in cache_bindings.items():
            if not isinstance(binding, dict):
                raise ValueError("DistanceHead checkpoint cache binding is malformed")
            validate_recorded_cache_binding(
                binding,
                split_role=split_role,
                backbone_seed=backbone_seed,
                protocol_lock=active_lock,
            )
        initialization = checkpoint.get("initialization", {})
        parent_path = initialization.get("parent_checkpoint_path")
        if parent_path is not None and sha256_file(parent_path) != initialization.get(
            "parent_checkpoint_sha256"
        ):
            raise ValueError("DistanceHead initialization parent changed")
        spec = HeadSpec.model_validate(checkpoint["head_spec"])
        head = build_distance_head(spec).to(device)
        head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad = False
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            provenance["joint_model_state_loaded"] = True
        if provenance["joint_model_state_loaded"] != (
            owner_method.training_scope.value != "frozen"
        ):
            raise ValueError("checkpoint model-state scope differs from its method")
        provenance.update(
            {
                "head_checkpoint_path": checkpoint_path.as_posix(),
                "head_checkpoint_sha256": sha256_file(checkpoint_path),
                "head_training_spec_sha256": checkpoint["training_spec_sha256"],
            }
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return (
        VectorWorldModel(
            model, device=device, history_size=config.planner.history_size
        ),
        head,
        provenance,
    )


def _head_output(
    head: DistanceHeadModel,
    source: torch.Tensor,
    goal: torch.Tensor,
    *,
    horizon: int,
    predicted_domain: bool = True,
) -> Any:
    horizon_tensor = (
        torch.full((source.shape[0],), float(horizon), device=source.device)
        if head.spec.horizon_conditioned
        else None
    )
    return head(
        source,
        goal,
        horizon=horizon_tensor,
        predicted_domain=predicted_domain,
    )


class DistancePlanningScorer:
    def __init__(self, method: ResolvedMethod, head: DistanceHeadModel | None) -> None:
        self.method = method
        self.head = head
        self._real_state_memory: list[torch.Tensor] = []

    def observe_real_state(self, latent: torch.Tensor) -> None:
        if latent.ndim != 3 or latent.shape[:2] != (1, 1):
            raise ValueError("real-state memory expects one [1,1,dim] latent")
        self._real_state_memory.append(latent.detach().reshape(-1).cpu())

    @staticmethod
    def _standardize(value: torch.Tensor) -> torch.Tensor:
        deviation = value.std(unbiased=False).clamp_min(1e-6)
        return (value - value.mean()) / deviation

    def terminal_cost(
        self,
        terminal: torch.Tensor,
        goal: torch.Tensor,
        *,
        horizon: int,
        predicted_domain: bool = True,
    ) -> torch.Tensor:
        if self.method.planner.cost == CostKind.LATENT_L2:
            return F.mse_loss(terminal, goal, reduction="none").sum(dim=-1)
        if self.head is None:
            raise ValueError("learned planner cost requires a DistanceHead")
        return _head_output(
            head=self.head,
            source=terminal,
            goal=goal,
            horizon=horizon,
            predicted_domain=predicted_domain,
        ).score

    def __call__(
        self,
        context: VectorContext,
        rollout: RolloutBatch,
        *,
        remaining_budget: int,
        ledger: ComputeLedger,
    ) -> ScoreBatch:
        del ledger
        batch_size, horizon, latent_dim = rollout.states.shape
        goal = context.goal.squeeze(1).expand(batch_size, -1)
        terminal = self.terminal_cost(rollout.terminal, goal, horizon=horizon)
        latent_l2 = F.mse_loss(rollout.terminal, goal, reduction="none").sum(dim=-1)
        components = {"terminal": terminal, "latent_l2": latent_l2}
        cost_kind = self.method.planner.cost
        if cost_kind in (CostKind.TERMINAL_DISTANCE, CostKind.LATENT_L2):
            total = terminal
        elif cost_kind == CostKind.PATH_INTEGRATED:
            if self.head is None:
                raise ValueError("path cost requires a DistanceHead")
            expanded_goal = goal[:, None, :].expand(-1, horizon, -1)
            path = (
                _head_output(
                    self.head,
                    rollout.states.reshape(-1, latent_dim),
                    expanded_goal.reshape(-1, latent_dim),
                    horizon=horizon,
                )
                .score.reshape(batch_size, horizon)
                .mean(dim=1)
            )
            components["path"] = path
            total = terminal + self.method.planner.path_weight * path
        elif cost_kind == CostKind.HYBRID:
            total = self._standardize(
                terminal
            ) + self.method.planner.latent_l2_weight * self._standardize(latent_l2)
        elif cost_kind == CostKind.REACHABILITY:
            if self.head is None:
                raise ValueError("reachability cost requires a DistanceHead")
            budgets = np.asarray(self.head.spec.reachability_budgets)
            effective_remaining = remaining_budget + int(
                rollout.semantics == RolloutSemantics.LEGACY_WARMUP_V1
            )
            index = int(np.searchsorted(budgets, effective_remaining, side="left"))
            index = min(index, len(budgets) - 1)
            matched_budget = int(budgets[index])
            output = _head_output(
                self.head, rollout.terminal, goal, horizon=matched_budget
            )
            if output.reachability_logits is None:
                raise ValueError("reachability planner requires reachability logits")
            reachability = -torch.log(
                torch.sigmoid(output.reachability_logits[:, index]).clamp_min(1e-8)
            )
            components["reachability"] = reachability
            total = self._standardize(
                terminal
            ) + self.method.planner.reachability_weight * self._standardize(
                reachability
            )
        elif cost_kind == CostKind.RISK_LOOP:
            if self.head is None:
                raise ValueError("risk cost requires a DistanceHead")
            output = _head_output(self.head, rollout.terminal, goal, horizon=horizon)
            if output.log_variance is None:
                raise ValueError("risk cost requires uncertainty output")
            uncertainty = torch.exp(0.5 * output.log_variance)
            pairwise = torch.cdist(rollout.states, rollout.states)
            index = torch.arange(horizon, device=pairwise.device)
            nearby = (index[:, None] - index[None, :]).abs() <= 1
            pairwise = pairwise.masked_fill(nearby[None, :, :], float("inf"))
            minimum_revisit_distance = pairwise.flatten(1).min(dim=1).values
            internal_loop_risk = torch.exp(-minimum_revisit_distance.clamp(max=50.0))
            if self._real_state_memory:
                real_states = torch.stack(self._real_state_memory).to(rollout.states)
                real_distance = (
                    torch.cdist(
                        rollout.states,
                        real_states[None].expand(batch_size, -1, -1),
                    )
                    .flatten(1)
                    .min(dim=1)
                    .values
                )
                real_loop_risk = torch.exp(-real_distance.clamp(max=50.0))
                loop_risk = torch.maximum(internal_loop_risk, real_loop_risk)
            else:
                real_loop_risk = torch.zeros_like(internal_loop_risk)
                loop_risk = internal_loop_risk
            components["uncertainty"] = uncertainty
            components["internal_loop_risk"] = internal_loop_risk
            components["real_loop_risk"] = real_loop_risk
            components["loop_risk"] = loop_risk
            total = (
                self._standardize(terminal)
                + self.method.planner.uncertainty_weight
                * self._standardize(uncertainty)
                + self.method.planner.loop_weight * self._standardize(loop_risk)
            )
        else:
            raise ValueError(f"unsupported planner cost: {cost_kind}")
        result = ScoreBatch(total=total, components=components)
        result.validate(batch_size)
        return result


def _frontier_planner(
    world_model: VectorWorldModel,
    method: ResolvedMethod,
    scorer: DistancePlanningScorer,
    config: Any,
) -> BasePlanner:
    if method.planner.kind in (
        PlannerKind.MODEL_FREE_GREEDY,
        PlannerKind.PREDICTOR_GREEDY,
    ):
        raise ValueError("greedy methods do not instantiate a search planner")
    planner = FrontierPlannerConfig(
        kind=FrontierPlannerKind(method.planner.kind.value),
        rollout_semantics=RolloutSemantics.LEGACY_WARMUP_V1,
        horizon=config.planner.horizon,
        history_size=config.planner.history_size,
        num_candidates=config.planner.num_candidates,
        num_elites=config.planner.num_elites,
        cem_iters=config.planner.cem_iters,
        momentum=config.planner.momentum,
        budget=BudgetConfig(reference_transitions=config.planner.reference_transitions),
    )
    return build_planner(world_model, planner, scorer)


def _exact_cem(
    world_model: VectorWorldModel,
    context: VectorContext,
    scorer: DistancePlanningScorer,
    config: Any,
    *,
    seed: int,
) -> tuple[np.ndarray, float, ComputeLedger]:
    def score_fn(terminal: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return scorer.terminal_cost(terminal, goal, horizon=config.planner.horizon)

    sequence, cost, _ = cem_plan(
        world_model.model,
        context.embeddings,
        context.actions,
        context.goal,
        horizon=config.planner.horizon,
        history_size=config.planner.history_size,
        num_candidates=config.planner.num_candidates,
        num_elites=config.planner.num_elites,
        cem_iters=config.planner.cem_iters,
        momentum=config.planner.momentum,
        num_actions=MODEL_ACTION_VOCAB_SIZE,
        device=world_model.device,
        seed=seed,
        score_fn=score_fn,
        allowed_actions=np.asarray(ACTION_IDS, dtype=np.int64),
    )
    return (
        sequence,
        float(cost),
        ComputeLedger(
            plan_transitions=config.planner.num_candidates
            * config.planner.horizon
            * config.planner.cem_iters,
            planner_forward_calls=config.planner.horizon * config.planner.cem_iters,
            candidate_sequences=config.planner.num_candidates
            * config.planner.cem_iters,
        ),
    )


def _greedy_proposal(
    world_model: VectorWorldModel,
    head: DistanceHeadModel | None,
    method: ResolvedMethod,
    context: VectorContext,
    env: Any,
    state: int,
) -> tuple[int, float, ComputeLedger]:
    actions = torch.tensor(ACTION_IDS, device=world_model.device)
    if method.planner.kind == PlannerKind.MODEL_FREE_GREEDY:
        targets = [next_state(env, state, int(action)) for action in ACTION_IDS]
        observations = np.stack([observe_state(env, target) for target in targets])
        latents = torch.cat(
            [
                world_model.encode(observation, context.maze_size)
                for observation in observations
            ],
            dim=0,
        ).squeeze(1)
        ledger = ComputeLedger()
    else:
        predicted = world_model.one_step_all_actions(context)
        if predicted.shape[0] != MODEL_ACTION_VOCAB_SIZE:
            raise ValueError("predictor returned an unexpected action vocabulary")
        latents = predicted.index_select(0, actions)
        ledger = ComputeLedger(
            plan_transitions=MODEL_ACTION_VOCAB_SIZE,
            planner_forward_calls=1,
        )
    goal = context.goal.squeeze(1).expand(len(ACTION_IDS), -1)
    scorer = DistancePlanningScorer(method, head)
    costs = scorer.terminal_cost(
        latents,
        goal,
        horizon=1,
        predicted_domain=method.planner.kind != PlannerKind.MODEL_FREE_GREEDY,
    )
    index = int(costs.argmin())
    return int(ACTION_IDS[index]), float(costs[index]), ledger


def _oracle_proposal(
    world_model: VectorWorldModel,
    head: DistanceHeadModel | None,
    method: ResolvedMethod,
    context: VectorContext,
    env: Any,
    state: int,
    *,
    seed: int,
    config: Any,
) -> tuple[int, float, ComputeLedger]:
    distances = bfs_distances_from(
        env._maze_mask, int(env._goal_position), int(env.config.width)
    )
    if method.name == "o_bfs1":
        costs = [
            int(distances[next_state(env, state, action)]) for action in ACTION_IDS
        ]
        index = int(np.argmin(costs))
        return int(ACTION_IDS[index]), float(costs[index]), ComputeLedger()
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(ACTION_IDS), 1.0 / len(ACTION_IDS))
    sequences = np.stack(
        [
            rng.choice(ACTION_IDS, size=config.planner.num_candidates, p=probabilities)
            for _ in range(config.planner.horizon)
        ],
        axis=1,
    )
    endpoints = []
    for sequence in sequences:
        cursor = state
        for action in sequence[: max(config.planner.horizon - 1, 0)]:
            cursor = next_state(env, cursor, int(action))
        endpoints.append(cursor)
    if method.name == "o_score_true_bfs":
        costs = np.asarray(
            [distances[endpoint] for endpoint in endpoints], dtype=np.float64
        )
    elif method.name == "o_dyn_true_rollout":
        if head is None:
            raise ValueError("true-rollout learned oracle needs a DistanceHead")
        latents = torch.cat(
            [
                world_model.encode(observe_state(env, endpoint), context.maze_size)
                for endpoint in endpoints
            ],
            dim=0,
        ).squeeze(1)
        goal = context.goal.squeeze(1).expand(len(endpoints), -1)
        costs = (
            _head_output(
                head,
                latents,
                goal,
                horizon=config.planner.horizon,
                predicted_domain=False,
            )
            .score.detach()
            .cpu()
            .numpy()
        )
    else:
        raise ValueError(f"unsupported oracle method: {method.name}")
    best = int(np.argmin(costs))
    return int(sequences[best, 0]), float(costs[best]), ComputeLedger()


def _fallback_action(
    world_model: VectorWorldModel,
    head: DistanceHeadModel | None,
    method: ResolvedMethod,
    context: VectorContext,
    allowed: list[int],
) -> tuple[int, int]:
    predicted = world_model.one_step_all_actions(context)
    if predicted.shape[0] != MODEL_ACTION_VOCAB_SIZE:
        raise ValueError("fallback predictor returned an unexpected action vocabulary")
    goal = context.goal.squeeze(1).expand(MODEL_ACTION_VOCAB_SIZE, -1)
    if method.name == "o_bfs1":
        raise RuntimeError("BFS oracle fallback must be selected before this function")
    scorer = DistancePlanningScorer(method, head)
    costs = scorer.terminal_cost(predicted, goal, horizon=1)
    indices = torch.tensor(allowed, dtype=torch.long, device=world_model.device)
    return (
        int(allowed[int(costs.index_select(0, indices).argmin())]),
        MODEL_ACTION_VOCAB_SIZE,
    )


def run_episode(
    entry: dict[str, Any],
    world_model: VectorWorldModel,
    head: DistanceHeadModel | None,
    method: ResolvedMethod,
    config: Any,
    *,
    task_index: int,
    action_protocol: str,
) -> dict[str, Any]:
    env = validate_manifest_entry(entry)
    state = int(entry["start_cell"])
    goal_state = int(entry["goal_cell"])
    observation = set_agent_state(env, state)
    start_latent = world_model.encode(observation, int(entry["maze_size"]))
    goal_latent = world_model.encode(
        observe_state(env, goal_state), int(entry["maze_size"])
    )
    context = world_model.initial_context(
        start_latent,
        goal_latent,
        maze_size=int(entry["maze_size"]),
        context_action=4,
        remaining_steps=config.planner.max_steps,
    )
    scorer = DistancePlanningScorer(method, head)
    scorer.observe_real_state(start_latent)
    use_exact = (
        method.planner.kind == PlannerKind.CATEGORICAL_CEM
        and method.planner.cost in (CostKind.LATENT_L2, CostKind.TERMINAL_DISTANCE)
        and method.role != "oracle"
    )
    planner = (
        None
        if use_exact
        or method.role == "oracle"
        or method.planner.kind
        in (PlannerKind.MODEL_FREE_GREEDY, PlannerKind.PREDICTOR_GREEDY)
        else _frontier_planner(world_model, method, scorer, config)
    )
    if planner is not None:
        planner.reset()
        planner.observe_real_state(start_latent)
    previous: int | None = None
    path = [state]
    invalid_actions = 0
    proposed_invalid = 0
    proposed_backtrack = 0
    assistance_count = 0
    plan_transitions = 0
    fallback_transitions = 0
    best_costs: list[float] = []
    started = time.perf_counter()
    for step in range(config.planner.max_steps):
        if state == goal_state:
            break
        seed = task_seed(config.seeds.run_order_seed, task_index, step)
        if method.role == "oracle":
            proposed, cost, ledger = _oracle_proposal(
                world_model,
                head,
                method,
                context,
                env,
                state,
                seed=seed,
                config=config,
            )
        elif method.planner.kind in (
            PlannerKind.MODEL_FREE_GREEDY,
            PlannerKind.PREDICTOR_GREEDY,
        ):
            proposed, cost, ledger = _greedy_proposal(
                world_model, head, method, context, env, state
            )
        elif use_exact:
            sequence, cost, ledger = _exact_cem(
                world_model, context, scorer, config, seed=seed
            )
            proposed = int(sequence[0])
        else:
            assert planner is not None
            result = planner.plan(context, seed=seed)
            proposed = int(result.sequence[0])
            cost = float(result.cost)
            ledger = result.ledger
        best_costs.append(cost)
        candidate = next_state(env, state, proposed)
        proposed_invalid += int(candidate == state)
        proposed_backtrack += int(previous is not None and candidate == previous)
        action = proposed
        if action_protocol == "corrected_v1":
            allowed = corrected_actions(env, state, previous)
            if proposed not in allowed:
                if method.uses_test_bfs:
                    distances = bfs_distances_from(
                        env._maze_mask, goal_state, int(env.config.width)
                    )
                    action = min(
                        allowed,
                        key=lambda value: int(distances[next_state(env, state, value)]),
                    )
                else:
                    action, fallback = _fallback_action(
                        world_model, head, method, context, allowed
                    )
                    fallback_transitions += fallback
                assistance_count += 1
        old_state = state
        observation, _, _, _, info = env.step(action)
        state = int(info["state"])
        invalid_actions += int(state == old_state)
        previous = old_state
        path.append(state)
        current = world_model.encode(observation, int(entry["maze_size"]))
        scorer.observe_real_state(current)
        context = world_model.advance_context(context, current, action)
        if planner is not None:
            planner.observe_real_state(current)
        plan_transitions += int(ledger.plan_transitions)
    distances = bfs_distances_from(env._maze_mask, goal_state, int(env.config.width))
    visits = Counter(path)
    success = state == goal_state
    path_length = len(path) - 1
    initial_distance = int(entry["bfs_path_length"])
    final_distance = int(distances[state])
    loop = max(visits.values()) >= 4
    if success:
        failure_mode = "success"
    elif initial_distance > config.planner.max_steps:
        failure_mode = "step_cap_ineligible"
    elif invalid_actions > path_length / 4:
        failure_mode = "invalid_action"
    elif loop:
        failure_mode = "loop_or_cycle"
    elif final_distance >= initial_distance:
        failure_mode = "insufficient_progress"
    else:
        failure_mode = "timeout_inefficient"
    return {
        "task_id": task_id(entry),
        "maze_size": int(entry["maze_size"]),
        "topology_seed": int(entry["topology_seed"]),
        "start_cell": int(entry["start_cell"]),
        "goal_cell": goal_state,
        "optimal_length": initial_distance,
        "success": bool(success),
        "path_length": path_length,
        "spl": float(initial_distance / max(initial_distance, path_length))
        if success
        else 0.0,
        "invalid_actions": int(invalid_actions),
        "repeat_states": int(sum(max(count - 1, 0) for count in visits.values())),
        "max_state_visits": int(max(visits.values())),
        "loop_or_cycle": bool(loop),
        "final_bfs_distance": final_distance,
        "proposed_invalid": int(proposed_invalid),
        "proposed_backtrack": int(proposed_backtrack),
        "assistance_count": int(assistance_count),
        "assistance_rate": float(assistance_count / max(path_length, 1)),
        "plan_transitions": int(plan_transitions),
        "fallback_transitions": int(fallback_transitions),
        "mean_best_cost": float(np.mean(best_costs)) if best_costs else 0.0,
        "failure_mode": failure_mode,
        "episode_seconds": float(time.perf_counter() - started),
    }


def _run_directory(config: Any, args: argparse.Namespace) -> Path:
    base = resolve_path(
        config.paths.result_template.format(
            split_role=args.split_role,
            method=args.method,
            backbone_seed=args.backbone_seed,
            head_seed=args.head_seed,
            action_protocol=args.action_protocol,
        )
    )
    if args.diagnostic_limit:
        base = resolve_path(
            "distance_head_study_runs/smoke/evaluation/"
            f"{args.split_role}/{args.method}/backbone{args.backbone_seed}_"
            f"head{args.head_seed}/{args.action_protocol}_"
            f"limit{args.diagnostic_limit}"
        )
    if args.num_shards > 1:
        return base / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    return base


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid evaluation shard specification")
    if args.diagnostic_limit < 0:
        raise ValueError("diagnostic limit must be non-negative")
    diagnostic = args.diagnostic_limit > 0
    if diagnostic and not args.allow_dirty_worktree:
        raise ValueError("diagnostic evaluation requires the explicit dirty flag")
    config = load_study_config(args.config)
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    lock = verify_protocol_lock(config)
    method, method_hash, decision_hashes = load_and_resolve_method(
        config.paths.method_catalog,
        args.method,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    gate = _require_method_evaluation_gate(
        config,
        split_role=args.split_role,
        method=method.name,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )
    if args.split_role == "confirm" and not method.confirmatory_eligible:
        raise ValueError("diagnostic/oracle method cannot enter confirmatory ranking")
    manifest_path = getattr(config.paths, f"{args.split_role}_manifest")
    entries = read_jsonl(manifest_path)
    indexed = [
        (index, entry)
        for index, entry in enumerate(entries)
        if index % args.num_shards == args.shard_index
    ]
    if args.diagnostic_limit:
        indexed = indexed[: args.diagnostic_limit]
    device = resolve_device(args.device or config.device)
    world_model, head, checkpoint = _load_models(
        config,
        method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
        device=device,
        expected_analysis_spec_sha256=lock["analysis_spec_sha256"],
        expected_protocol_lock_sha256=lock["protocol_lock_sha256"],
    )
    if args.split_role == "confirm":
        if gate is None:
            raise RuntimeError("confirmation evaluation did not receive its open gate")
        if gate.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]:
            raise ValueError("confirmation gate uses another analysis lock")
        if gate.get("confirm_manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("confirmation gate manifest hash differs")
        locked_hashes = gate["locked_checkpoint_hashes"]
        for path_key, hash_key in (
            ("backbone_path", "backbone_sha256"),
            ("head_checkpoint_path", "head_checkpoint_sha256"),
        ):
            path = checkpoint.get(path_key)
            if path is not None and locked_hashes.get(path) != checkpoint.get(hash_key):
                raise ValueError("evaluation checkpoint differs from confirmation seal")
    output = _run_directory(config, args)
    metadata_path = output / "metadata.json"
    rows_path = output / "rows.jsonl"
    summary_path = output / "summary.json"
    run_spec = {
        "schema": RESULT_SCHEMA,
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "method": method.model_dump(mode="json"),
        "method_sha256": method_hash,
        "decision_sha256s": list(decision_hashes),
        "split_role": args.split_role,
        "manifest_path": resolve_path(manifest_path).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "backbone_seed": args.backbone_seed,
        "head_seed": args.head_seed,
        "action_protocol": args.action_protocol,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "diagnostic_limit": args.diagnostic_limit,
        "checkpoint": checkpoint,
    }
    run_spec["run_spec_sha256"] = canonical_json_sha256(run_spec)
    if args.resume and output.exists():
        if not metadata_path.exists() or load_json(metadata_path) != run_spec:
            raise ValueError("resume metadata differs from requested evaluation")
        existing = read_jsonl(rows_path) if rows_path.exists() else []
    elif args.resume:
        output.mkdir(parents=True)
        atomic_json_dump(metadata_path, run_spec)
        existing = []
    else:
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite evaluation directory: {output}"
            )
        output.mkdir(parents=True)
        atomic_json_dump(metadata_path, run_spec)
        existing = []
    completed = {str(row["task_id"]) for row in existing}
    if len(completed) != len(existing):
        raise ValueError("resume rows contain duplicate task IDs")
    mode = "a" if rows_path.exists() else "w"
    with open(rows_path, mode, encoding="utf-8") as stream:
        for local_index, (global_index, entry) in enumerate(indexed):
            identifier = task_id(entry)
            if identifier in completed:
                continue
            row = run_episode(
                entry,
                world_model,
                head,
                method,
                config,
                task_index=global_index,
                action_protocol=args.action_protocol,
            )
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            if (local_index + 1) % 25 == 0:
                print(f"{local_index + 1}/{len(indexed)}", flush=True)
    rows = read_jsonl(rows_path)
    if len(rows) != len(indexed):
        raise ValueError("evaluation row count differs from assigned shard")
    identifiers = [str(row["task_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evaluation rows contain duplicate tasks")
    summary = summarize_rows(rows, seen_max_size=21, max_steps=config.planner.max_steps)
    summary["failure_modes"] = dict(
        sorted(Counter(row["failure_mode"] for row in rows).items())
    )
    summary["mean_assistance_rate"] = float(
        np.mean([row["assistance_rate"] for row in rows])
    )
    summary["run_spec_sha256"] = run_spec["run_spec_sha256"]
    atomic_json_dump(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
