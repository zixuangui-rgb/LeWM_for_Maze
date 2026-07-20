"""Create the immutable outer lock after code, configs, and docs are frozen."""

from __future__ import annotations

import argparse

from a1_quick_validation.common import (
    DEFAULT_PACKAGE_LOCK,
    DEFAULT_PROFILE,
    atomic_json_dump,
    prepare_immutable,
)
from a1_quick_validation.profile import build_package_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--output", default=str(DEFAULT_PACKAGE_LOCK))
    args = parser.parse_args()
    output = prepare_immutable(args.output)
    atomic_json_dump(output, build_package_lock(args.profile))
    print(output)


if __name__ == "__main__":
    main()
