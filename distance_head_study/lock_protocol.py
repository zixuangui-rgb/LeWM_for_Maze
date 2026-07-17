"""Create the immutable DistanceHead protocol lock after full regeneration."""

from __future__ import annotations

import argparse

from distance_head_study.common import atomic_json_dump, load_study_config, resolve_path
from distance_head_study.protocol import build_protocol_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    args = parser.parse_args()
    config = load_study_config(args.config)
    output = resolve_path(config.paths.protocol_lock)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protocol lock: {output}")
    atomic_json_dump(output, build_protocol_lock(config, regenerate=True))
    print(output)


if __name__ == "__main__":
    main()
