"""Fixed post-confirmatory baselines and paper-closure analysis."""

EXPERIMENT_FAMILY = "maze_jepa_final_closure"
FORMAT_VERSION = 1
PROTOCOL_ID = "maze-jepa-paper-closure-v1"
TABLE_FILENAMES = (
    "primary_results.csv",
    "per_seed_results.csv",
    "size_generalization.csv",
    "path_length_generalization.csv",
    "paired_effects.csv",
    "assistance_effects.csv",
    "development_alignment.csv",
    "spatial_k_curves.csv",
    "compute_summary.csv",
)
FIGURE_FILENAMES = (
    "primary_results.png",
    "generalization_by_size.png",
    "generalization_by_path_length.png",
    "spatial_iteration_curve.png",
    "spatial_compute_curve.png",
)

__all__ = [
    "EXPERIMENT_FAMILY",
    "FIGURE_FILENAMES",
    "FORMAT_VERSION",
    "PROTOCOL_ID",
    "TABLE_FILENAMES",
]
