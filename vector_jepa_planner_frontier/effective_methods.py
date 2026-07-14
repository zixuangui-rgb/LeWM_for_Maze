"""Derive result-dependent methods from preregistered immutable decisions."""

from __future__ import annotations

from typing import Any

from final_closure.common import sha256_file
from spatial_jepa_planning.common import canonical_json_sha256
from vector_jepa_planner_frontier.common import method_by_name, resolve_path
from vector_jepa_planner_frontier.schemas import (
    MethodConfig,
    PlannerConfig,
    ProposalKind,
    RolloutSemantics,
)

COMPONENT_ORDER = ("verifier", "reachability", "proposal", "memory")
RADICAL_METHODS = {
    "vector_dts": "p4_vector_dts",
    "bidirectional": "p4_bidirectional",
    "denoising": "p4_denoising_icem",
}


def p3_cell_for_components(components: tuple[str, ...] | list[str]) -> str:
    selected = set(components)
    if not selected <= set(COMPONENT_ORDER):
        raise ValueError("P5 selected an unknown factorial component")
    bits = {
        "v": int("verifier" in selected),
        "r": int("reachability" in selected),
        "p": int("proposal" in selected),
        "m": int("memory" in selected),
    }
    return "p3_factorial_" + "".join(
        f"{label}{bits[label]}" for label in ("v", "r", "p", "m")
    )


def _decision_sha256(config: Any, attribute: str) -> str:
    return sha256_file(resolve_path(getattr(config.paths, attribute)))


def _with_decisions(method: MethodConfig, *digests: str) -> MethodConfig:
    unique = tuple(dict.fromkeys((*method.effective_decision_sha256s, *digests)))
    return method.model_copy(update={"effective_decision_sha256s": unique})


def derive_p3_method(
    config: Any,
    method: MethodConfig,
    selection: dict[str, Any],
) -> MethodConfig:
    if method.stage != "P3":
        raise ValueError("P2 planner derivation is valid only for P3")
    planner = PlannerConfig.model_validate(selection["selected_4x_planner"])
    if planner.budget.multiplier != 4.0:
        raise ValueError("P3 must inherit the selected planner at 4x")
    derived = method.model_copy(update={"planner": planner})
    return _with_decisions(derived, _decision_sha256(config, "p2_selection"))


def derive_p5_method(
    config: Any,
    method: MethodConfig,
    p2_selection: dict[str, Any],
    p5_selection: dict[str, Any],
) -> MethodConfig:
    selected_components = tuple(p5_selection["selected_components"])
    cell = method_by_name(config, p3_cell_for_components(selected_components))
    cell = derive_p3_method(config, cell, p2_selection)
    radical_name = p5_selection.get("selected_radical")
    radical = (
        method_by_name(config, RADICAL_METHODS[str(radical_name)])
        if radical_name is not None
        else None
    )
    planner = radical.planner if radical is not None else cell.planner
    proposal = radical.proposal if radical_name == "denoising" else cell.proposal
    control = radical.control if radical is not None else cell.control
    derived = method.model_copy(
        update={
            "planner": planner,
            "scorer": cell.scorer,
            "proposal": proposal,
            "memory": cell.memory,
            "control": control,
            "initialization_parent": cell.name,
            "reuse_component_from": None,
            "trainable_components": (),
        }
    )
    if (
        radical_name == "denoising"
        and derived.proposal.kind != ProposalKind.DISCRETE_DENOISING
    ):
        raise AssertionError("denoising radical did not replace the proposal")
    return _with_decisions(
        derived,
        _decision_sha256(config, "p2_selection"),
        _decision_sha256(config, "p5_advancement"),
    )


def derive_p6_or_p7_method(
    config: Any,
    method: MethodConfig,
    p2_selection: dict[str, Any],
    p5_selection: dict[str, Any],
) -> MethodConfig:
    p5_base = method_by_name(config, "p5_track_f_all_hard_memory")
    p5 = derive_p5_method(config, p5_base, p2_selection, p5_selection)
    if method.stage == "P6":
        scorer = p5.scorer.model_copy(
            update={
                "counterexample_ranker_weight": (
                    method.scorer.counterexample_ranker_weight
                )
            }
        )
        derived = method.model_copy(
            update={
                "planner": p5.planner,
                "scorer": scorer,
                "proposal": p5.proposal,
                "memory": p5.memory,
            }
        )
        return _with_decisions(derived, *p5.effective_decision_sha256s)
    if method.stage != "P7":
        raise ValueError("P5 architecture inheritance is valid only for P6/P7")
    p6_base = method_by_name(config, "p6_track_f_counterexample_ranked")
    p6 = derive_p6_or_p7_method(config, p6_base, p2_selection, p5_selection)
    planner = p6.planner.model_copy(
        update={"rollout_semantics": RolloutSemantics.ACTION_ALIGNED_V2}
    )
    derived = method.model_copy(
        update={
            "planner": planner,
            "scorer": p6.scorer,
            "proposal": p6.proposal,
            "memory": p6.memory,
            "control": p6.control,
        }
    )
    return _with_decisions(derived, *p6.effective_decision_sha256s)


def resolve_effective_method(
    config: Any,
    lock: dict[str, Any],
    method: MethodConfig | str,
) -> MethodConfig:
    """Resolve one method and fail closed on every required decision artifact."""

    base = method_by_name(config, method) if isinstance(method, str) else method
    if base.adaptive_role == "static":
        return base
    from vector_jepa_planner_frontier.stage_gates import (
        validate_p2_selection,
        validate_p5_advancement,
    )

    p2 = validate_p2_selection(config, lock)
    if base.adaptive_role == "p2_selected_planner":
        return derive_p3_method(config, base, p2)
    p5 = validate_p5_advancement(config, lock)
    if base.stage == "P5":
        return derive_p5_method(config, base, p2, p5)
    if base.stage in {"P6", "P7"}:
        return derive_p6_or_p7_method(config, base, p2, p5)
    if base.stage == "P8":
        from vector_jepa_planner_frontier.stage_gates import validate_p7_selection

        p7 = validate_p7_selection(config, lock)
        source_name = str(base.reuse_component_from)
        if source_name == "p7_track_j_joint_all":
            selected = p7.get("selected_track_j")
            if selected is None:
                raise RuntimeError(
                    "Track J failed selection; its P8 aliases are closed"
                )
            source_name = str(selected)
        source = resolve_effective_method(config, lock, source_name)
        planner = source.planner.model_copy(update={"budget": base.planner.budget})
        derived = base.model_copy(
            update={
                "track": source.track,
                "planner": planner,
                "scorer": source.scorer,
                "proposal": source.proposal,
                "memory": source.memory,
                "control": source.control,
                "reuse_component_from": source.name,
                "joint_hyperparameters": None,
            }
        )
        return _with_decisions(
            derived,
            *source.effective_decision_sha256s,
            _decision_sha256(config, "p7_selection"),
        )
    raise ValueError(f"unsupported adaptive method role: {base.name}")


def effective_method_sha256(method: MethodConfig) -> str:
    return canonical_json_sha256(method.model_dump(mode="json"))


__all__ = [
    "COMPONENT_ORDER",
    "RADICAL_METHODS",
    "derive_p3_method",
    "derive_p5_method",
    "effective_method_sha256",
    "p3_cell_for_components",
    "resolve_effective_method",
]
