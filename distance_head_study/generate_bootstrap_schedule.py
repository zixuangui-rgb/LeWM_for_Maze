"""Generate the preregistered stateless crossed-bootstrap seed schedule."""

from __future__ import annotations

import argparse

import numpy as np

from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    load_study_config,
    resolve_path,
)


def build_schedule(base_seed: int, count: int) -> dict[str, object]:
    if count < 10_000:
        raise ValueError("confirmatory bootstrap requires at least 10,000 replicates")
    sequence = np.random.SeedSequence(base_seed)
    children = sequence.spawn(count)
    seeds = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
    payload: dict[str, object] = {
        "schema": "distance-head-bootstrap-schedule-v1",
        "base_seed": int(base_seed),
        "replicates": int(count),
        "replicate_seeds": seeds,
        "generation": "numpy.SeedSequence.spawn_then_uint64",
    }
    payload["schedule_sha256"] = canonical_json_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    args = parser.parse_args()
    config = load_study_config(args.config)
    output = resolve_path(config.paths.bootstrap_schedule)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap schedule: {output}")
    atomic_json_dump(
        output,
        build_schedule(config.seeds.bootstrap_seed, config.analysis.bootstrap_samples),
    )
    print(output)


if __name__ == "__main__":
    main()
