"""Preregistered method families and negative-route finalist strata."""

from __future__ import annotations

MAIN_CANDIDATES = (
    "a1_log",
    "a2_distance_balanced",
    "a3_full_horizon",
    "b1_listwise",
    "b2_bellman",
    "b3_multistep",
    "b5_local_structural",
    "c1_predicted_listwise",
    "c2_dual_calibration",
    "d2_trm_full",
    "d4_reachability",
)

NEGATIVE_CLOSURE_CANDIDATES = (
    *MAIN_CANDIDATES,
    "r_loss_mae",
    "r_output_ordinal",
    "r_output_distribution",
    "r_pairwise",
    "r_delta",
    "r_eikonal",
    "r_quasimetric",
    "r_successor_contrastive",
    "r_arch_asymmetric",
    "r_arch_hierarchical_budget",
    "r_uncertainty",
    "j0_dist_predictor",
    "j1_dist_projector",
    "j2_dist_full",
    "j3_rcaux_reach",
    "p_path_integrated",
    "p_hybrid_l2",
    "p_reachability",
    "p_risk_loop",
    "p_icem",
    "p_beam",
    "p_best_first",
)

# Additional controls plus every reserve result required before a broad negative claim.
NEGATIVE_CLOSURE_REQUIRED_RUNS = (
    "d3_trm_shuffle",
    "r_loss_mae",
    "r_output_ordinal",
    "r_output_distribution",
    "r_pairwise",
    "r_delta",
    "r_eikonal",
    "r_quasimetric",
    "r_successor_contrastive",
    "r_arch_asymmetric",
    "r_arch_hierarchical_budget",
    "r_uncertainty",
    "j0_cont_predictor",
    "j0_dist_predictor",
    "j1_cont_projector",
    "j1_dist_projector",
    "j2_cont_full",
    "j2_dist_full",
    "j3_rcaux_reach",
    "p_path_integrated",
    "p_hybrid_l2",
    "p_reachability",
    "p_risk_loop",
    "p_icem",
    "p_beam",
    "p_best_first",
)

MAIN_MECHANISM_FAMILIES = {
    "a1_log": "target_parameterization",
    "a2_distance_balanced": "sampling_curriculum",
    "a3_full_horizon": "sampling_curriculum",
    "b1_listwise": "local_action_ordering",
    "b2_bellman": "bellman_structure",
    "b3_multistep": "multistep_metric_structure",
    "b5_local_structural": "local_structural_factorial",
    "c1_predicted_listwise": "predicted_latent_alignment",
    "c2_dual_calibration": "predicted_domain_calibration",
    "d2_trm_full": "trajectory_ordering",
    "d4_reachability": "budgeted_reachability",
    "r_loss_mae": "regression_robustness",
    "r_output_ordinal": "ordinal_output",
    "r_output_distribution": "distributional_output",
    "r_pairwise": "pairwise_local_ordering",
    "r_delta": "distance_delta",
    "r_eikonal": "eikonal_structure",
    "r_quasimetric": "directed_quasimetric",
    "r_successor_contrastive": "successor_metric",
    "r_arch_asymmetric": "asymmetric_head_architecture",
    "r_arch_hierarchical_budget": "hierarchical_head_architecture",
    "r_uncertainty": "heteroscedastic_uncertainty",
    "j0_dist_predictor": "joint_predictor",
    "j1_dist_projector": "joint_projector_predictor",
    "j2_dist_full": "joint_full_backbone",
    "j3_rcaux_reach": "joint_reachability_auxiliary",
    "p_path_integrated": "path_integrated_cost",
    "p_hybrid_l2": "hybrid_latent_cost",
    "p_reachability": "reachability_cost",
    "p_risk_loop": "risk_and_loop_cost",
    "p_icem": "icem_search",
    "p_beam": "beam_search",
    "p_best_first": "best_first_search",
}

NEGATIVE_ROUTE_GROUPS = {
    method: (
        "system_or_planner"
        if method.startswith(("j", "p", "r_arch"))
        else "frozen_scorer"
    )
    for method in NEGATIVE_CLOSURE_CANDIDATES
}


def mechanism_family(method: str) -> str:
    try:
        return MAIN_MECHANISM_FAMILIES[method]
    except KeyError as error:
        raise ValueError(f"method has no preregistered mechanism: {method}") from error


def negative_route_group(method: str) -> str:
    try:
        return NEGATIVE_ROUTE_GROUPS[method]
    except KeyError as error:
        raise ValueError(f"method has no negative-route group: {method}") from error


def strongest_negative_pair(ranked_methods: list[str]) -> tuple[str, str]:
    selected = []
    for group in ("frozen_scorer", "system_or_planner"):
        match = next(
            (
                method
                for method in ranked_methods
                if negative_route_group(method) == group
            ),
            None,
        )
        if match is None:
            raise ValueError(f"closure ranking has no finalist in group {group}")
        selected.append(match)
    selected.sort(key=ranked_methods.index)
    return selected[0], selected[1]


__all__ = [
    "MAIN_CANDIDATES",
    "MAIN_MECHANISM_FAMILIES",
    "NEGATIVE_CLOSURE_CANDIDATES",
    "NEGATIVE_CLOSURE_REQUIRED_RUNS",
    "NEGATIVE_ROUTE_GROUPS",
    "mechanism_family",
    "negative_route_group",
    "strongest_negative_pair",
]
