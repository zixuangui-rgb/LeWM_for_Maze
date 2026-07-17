"""Recompute the full protocol contract and fail on any drift or leakage."""

from __future__ import annotations

import argparse

from distance_head_study.common import atomic_json_dump, load_study_config, resolve_path
from distance_head_study.protocol import verify_protocol_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--regenerate-manifests", action="store_true")
    parser.add_argument(
        "--output", default="distance_head_study_runs/audits/protocol_audit.json"
    )
    args = parser.parse_args()
    config = load_study_config(args.config)
    lock = verify_protocol_lock(config, regenerate=args.regenerate_manifests)
    output = resolve_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protocol audit: {output}")
    atomic_json_dump(
        output,
        {
            "schema": "distance-head-protocol-audit-v1",
            "protocol_id": config.protocol_id,
            "protocol_lock_sha256": lock["protocol_lock_sha256"],
            "regenerated_manifests": bool(args.regenerate_manifests),
            "status": "pass",
        },
    )
    print(output)


if __name__ == "__main__":
    main()
