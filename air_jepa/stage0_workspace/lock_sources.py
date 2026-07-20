#!/usr/bin/env python3
"""Create or verify the server-local immutable source checkpoint lock."""

from __future__ import annotations

import argparse

from air_jepa.stage0_workspace.checkpoints import (
    build_source_lock_payload,
    verify_source_lock,
)
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    load_config,
    prepare_new_output,
    require_clean_worktree,
)
from air_jepa.stage0_workspace.protocol import (
    verify_package_lock,
    verify_protocol_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_clean_worktree(allow_dirty=False)
    config = load_config(args.config)
    verify_protocol_lock(config)
    verify_package_lock(config)
    if args.check:
        verify_source_lock(config, deep=True)
        print(f"verified={config.paths.source_lock}")
        return
    output = prepare_new_output(config.paths.source_lock)
    atomic_json_dump(output, build_source_lock_payload(config))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
