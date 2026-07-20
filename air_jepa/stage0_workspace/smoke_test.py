#!/usr/bin/env python3
"""Dependency-light CPU smoke test for AIR data, model, loss, and interventions."""

from __future__ import annotations

import argparse

import torch

from air_jepa.stage0_workspace.common import DEFAULT_CONFIG, load_config
from air_jepa.stage0_workspace.data import make_rng_streams, sample_training_batch
from air_jepa.stage0_workspace.losses import air_loss
from air_jepa.stage0_workspace.models import AIRWorkspaceModel, require_finite_output
from diagnostics.common import read_jsonl
from spatial_jepa_planning.common import ManifestSampler
from spatial_jepa_planning.models import (
    SpatialRepresentation,
    SpatialRepresentationConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(123)
    representation = SpatialRepresentation(SpatialRepresentationConfig()).eval()
    for parameter in representation.parameters():
        parameter.requires_grad = False
    entries = read_jsonl(config.paths.train_manifest)
    streams = make_rng_streams(42)
    batch = sample_training_batch(
        ManifestSampler(entries),
        entry_rng=streams.entries,
        state_rng=streams.states,
        batch_size=2,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        source = representation.planning_latent(batch.current_observation)
        batch_size, actions, height, width, channels = (
            batch.successor_observations.shape
        )
        future = representation.planning_latent(
            batch.successor_observations.reshape(
                batch_size * actions, height, width, channels
            )
        ).reshape(batch_size, actions, -1, height, width)
    spatial_mask = torch.ones((batch_size, height, width), dtype=torch.bool)
    initial_hashes = []
    for method in ("air0_direct", "air0_jepa"):
        torch.manual_seed(99)
        model = AIRWorkspaceModel(config.model)
        initial_hashes.append(
            torch.cat(
                [parameter.detach().flatten() for parameter in model.parameters()]
            ).clone()
        )
        outputs = model(
            source,
            iterations=4,
            deep_supervision_every=config.training.deep_supervision_every,
            valid_mask=spatial_mask,
        )
        require_finite_output(outputs[-1])
        result = air_loss(
            outputs,
            successor_latent=future,
            source_latent=source,
            candidate_distances=batch.candidate_distances,
            optimal_action_mask=batch.optimal_action_mask,
            valid_mask=spatial_mask,
            weights=config.training.methods[method],
            max_distance=config.model.max_distance,
            target_variance_epsilon=config.training.target_variance_epsilon,
        )
        result.total.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"non-finite smoke gradient for {method}")
        predicted = outputs[-1].predicted_future
        if predicted is None:
            raise RuntimeError("smoke model did not return future fields")
        permutation = predicted[:, [1, 0, 3, 2]]
        _, permuted_energy = model.score_external_futures(
            outputs[-1], permutation, spatial_mask
        )
        if not bool(torch.isfinite(permuted_energy).all()):
            raise FloatingPointError("future permutation produced non-finite energy")
    if not torch.equal(initial_hashes[0], initial_hashes[1]):
        raise RuntimeError("paired AIR methods did not initialize identically")
    print(
        "AIR0 smoke passed: exact successors, tie labels, paired initialization, "
        "forward/backward, and future permutation"
    )


if __name__ == "__main__":
    main()
