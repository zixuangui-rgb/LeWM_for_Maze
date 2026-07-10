#!/usr/bin/env python3
"""Fast CPU integration smoke test for the complete experiment package."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdwm.losses import SIGReg
from spatial_jepa_planning import EXPERIMENT_FAMILY, FORMAT_VERSION
from spatial_jepa_planning.common import (
    ManifestSampler,
    build_map_targets,
    load_planner_checkpoint,
    load_representation_checkpoint,
    planner_features,
    read_jsonl,
    sample_map_batch,
    sample_sequence_batch,
    save_checkpoint,
    set_seed,
)
from spatial_jepa_planning.losses import (
    PlannerLossWeights,
    RepresentationLossWeights,
    planner_loss,
)
from spatial_jepa_planning.models import (
    OracleValueIteration,
    PlannerConfig,
    SpatialRepresentation,
    SpatialRepresentationConfig,
    build_planner,
    make_ema_target,
)
from spatial_jepa_planning.train import compute_representation_loss


def main() -> None:
    set_seed(123, deterministic=True)
    device = torch.device("cpu")
    entries = read_jsonl("data/splits/unisize_train_manifest.jsonl")
    sampler = ManifestSampler(entries)
    import numpy as np

    rng = np.random.default_rng(123)
    selected = sampler.sample(rng, batch_size=2, size=9)
    map_observations, map_targets = sample_map_batch(selected, rng, device)
    sequence_observations, sequence_actions, sequence_valid = sample_sequence_batch(
        selected,
        rng=rng,
        device=device,
        sequence_length=3,
        trajectories_per_map=1,
    )
    assert map_observations.shape == (2, 9, 9, 5)
    assert map_targets["valid_action_mask"].shape == (2, 4, 9, 9)
    assert sequence_observations.shape == (2, 3, 9, 9, 5)
    assert sequence_actions.shape == (2, 2)

    representation_config = SpatialRepresentationConfig(
        spatial_dim=16,
        planning_dim=16,
        encoder_blocks=1,
        predictor_blocks=1,
    )
    representation = SpatialRepresentation(representation_config)
    target = make_ema_target(representation)
    representation_total, representation_metrics = compute_representation_loss(
        representation,
        target,
        sequence_observations,
        sequence_actions,
        sequence_valid,
        SIGReg(knots=5, num_proj=16),
        RepresentationLossWeights(),
        sigreg_max_tokens=128,
    )
    assert torch.isfinite(representation_total)
    assert set(representation_metrics) >= {"prediction", "sigreg", "map_wall"}

    planner_config = PlannerConfig(
        input_channels=representation_config.planning_dim,
        hidden_dim=16,
        planner_type="iterative",
        depth=4,
    )
    planner = build_planner(planner_config)
    features = planner_features(map_observations, "spatial_jepa", representation)
    outputs = planner(features, iterations=4, deep_supervision_every=2)
    planning_total, planning_metrics = planner_loss(
        outputs,
        map_targets,
        PlannerLossWeights(),
        distance_scale=128.0,
    )
    assert torch.isfinite(planning_total)
    assert outputs[-1].value.shape == (2, 9, 9)
    assert outputs[-1].policy_logits.shape == (2, 4, 9, 9)
    (representation_total + planning_total).backward()
    assert any(parameter.grad is not None for parameter in representation.parameters())
    assert any(parameter.grad is not None for parameter in planner.parameters())

    raw_config = PlannerConfig(
        input_channels=5,
        hidden_dim=16,
        planner_type="feedforward",
        depth=2,
    )
    raw_planner = build_planner(raw_config)
    raw_outputs = raw_planner(planner_features(map_observations, "raw", None))
    assert raw_outputs[-1].iterations == 2

    env = __import__(
        "spatial_jepa_planning.common", fromlist=["validate_manifest_entry"]
    )
    validated_env = env.validate_manifest_entry(selected[0])
    target_map = build_map_targets(validated_env, device)
    oracle = OracleValueIteration()
    oracle_output = oracle(
        ~target_map["free_mask"].unsqueeze(0),
        target_map["goal_mask"].bool().unsqueeze(0),
        iterations=9 * 9,
    )
    free = target_map["free_mask"]
    assert torch.allclose(
        oracle_output.value[0][free],
        target_map["distance"][free],
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        representation_path = root / "representation.pt"
        save_checkpoint(
            representation_path,
            {
                "experiment_family": EXPERIMENT_FAMILY,
                "format_version": FORMAT_VERSION,
                "stage": "representation",
                "representation_config": representation_config.to_dict(),
                "representation_state_dict": representation.state_dict(),
            },
        )
        loaded_representation, _ = load_representation_checkpoint(
            representation_path, device
        )
        assert loaded_representation.config == representation_config

        planner_path = root / "planner.pt"
        save_checkpoint(
            planner_path,
            {
                "experiment_family": EXPERIMENT_FAMILY,
                "format_version": FORMAT_VERSION,
                "stage": "planner",
                "input_mode": "spatial_jepa",
                "representation_config": representation_config.to_dict(),
                "representation_state_dict": representation.state_dict(),
                "planner_config": planner_config.to_dict(),
                "planner_state_dict": planner.state_dict(),
            },
        )
        loaded_planner, loaded_representation, _ = load_planner_checkpoint(
            planner_path, device
        )
        assert loaded_planner is not None
        assert loaded_representation is not None

    print("spatial_jepa_planning smoke test passed")


if __name__ == "__main__":
    main()
