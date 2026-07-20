"""Locked command gateway for every formal quick-validation operation."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

from a1_quick_validation import ALL_METHODS, NEW_METHODS, PROMOTABLE_METHODS
from a1_quick_validation.audit import run_audit
from a1_quick_validation.cache_bridge import rebind_cache, validate_quick_cache
from a1_quick_validation.checkpoint_bridge import import_reference_checkpoint
from a1_quick_validation.common import (
    DEFAULT_PROFILE,
    require_clean_worktree,
    resolve_path,
)
from a1_quick_validation.evidence import (
    load_validated_diagnostic,
    validate_candidate_bank,
    validate_quick_checkpoint,
    validate_result_cell,
)
from a1_quick_validation.profile import verify_package_lock
from a1_quick_validation.release_seed_tier import (
    release_seed_tier,
    validate_seed_release,
)
from a1_quick_validation.selection import (
    assess_q3,
    diagnostic_path,
    load_q1_shortlist,
    load_q2_winner,
    select_q1,
    select_q2,
)
from distance_head_study import candidates as base_candidates
from distance_head_study import diagnose as base_diagnose
from distance_head_study import evaluate as base_evaluate
from distance_head_study import train_head as base_train
from distance_head_study.candidates import candidate_bank_path
from distance_head_study.common import (
    head_checkpoint_path,
    load_study_config,
)
from distance_head_study.results import result_directory


@contextlib.contextmanager
def _argv(module: ModuleType, values: list[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = [module.__name__, *values]
    try:
        yield
    finally:
        sys.argv = previous


def _invoke(module: ModuleType, values: list[str]) -> None:
    with _argv(module, values):
        module.main()


def _load_context(profile_path: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    profile, _, quick_lock = verify_package_lock(profile_path)
    require_clean_worktree()
    config = load_study_config(profile.paths.quick_config)
    return profile, config, quick_lock


def _load_shortlisted_new_methods(
    config: Any, quick_lock: dict[str, Any]
) -> tuple[str, ...]:
    shortlist = load_q1_shortlist(config, quick_lock)
    return tuple(str(value) for value in shortlist["new_methods"])


def _load_winner(config: Any, quick_lock: dict[str, Any]) -> str:
    payload = load_q2_winner(config, quick_lock)
    winner = payload.get("selected_method")
    if winner not in PROMOTABLE_METHODS:
        raise RuntimeError("there is no promotable Q2 winner")
    return str(winner)


def _validate_train_request(
    config: Any,
    quick_lock: dict[str, Any],
    *,
    method: str,
    backbone_seed: int,
    head_seed: int,
) -> None:
    if method not in NEW_METHODS:
        raise ValueError("reference methods must be imported, not retrained here")
    if backbone_seed != 42 or head_seed not in (0, 1):
        raise ValueError("quick training is locked to backbone 42 and head seeds 0/1")
    if head_seed == 1 and method not in _load_shortlisted_new_methods(
        config, quick_lock
    ):
        raise ValueError("head seed 1 is restricted to Q1-promoted methods")


def _allowed_eval_methods(
    config: Any, quick_lock: dict[str, Any], split_role: str
) -> tuple[str, ...]:
    if split_role == "screen":
        return ALL_METHODS
    if split_role == "select":
        return (
            "b_dh_cem",
            "a1_log",
            *_load_shortlisted_new_methods(config, quick_lock),
        )
    if split_role == "legacy":
        return ("b_l2_cem", "b_dh_cem", "a1_log", _load_winner(config, quick_lock))
    raise ValueError(f"quick profile forbids split role {split_role}")


def _validate_eval_request(
    config: Any,
    quick_lock: dict[str, Any],
    *,
    method: str,
    split_role: str,
    backbone_seed: int,
    head_seed: int,
    action_protocol: str,
    diagnostic: bool,
) -> None:
    if method not in _allowed_eval_methods(config, quick_lock, split_role):
        raise ValueError(f"method is outside the locked {split_role} matrix")
    if backbone_seed != 42:
        raise ValueError("quick evaluation is locked to backbone 42")
    if split_role == "screen":
        expected_heads, expected_actions = (0,), ("corrected_v1",)
    elif split_role == "select":
        expected_heads, expected_actions = (0, 1), ("corrected_v1", "unmasked")
    else:
        expected_heads, expected_actions = (0,), ("corrected_v1", "unmasked")
    if head_seed not in expected_heads:
        raise ValueError("head seed is outside the locked phase matrix")
    if action_protocol not in expected_actions:
        raise ValueError("action protocol is outside the locked phase matrix")
    if diagnostic and split_role == "legacy":
        raise ValueError("legacy full-900 has no post-selection BFS diagnostics")


@contextlib.contextmanager
def _redirect_training_state(run_root: str | Path) -> Iterator[None]:
    original_state = base_train._training_state_path
    original_smoke = base_train._diagnostic_checkpoint_path

    def state_path(
        method: str,
        backbone_seed: int,
        head_seed: int,
        *,
        diagnostic_steps: int = 0,
    ) -> Path:
        prefix = "smoke/train_state" if diagnostic_steps else "train_state"
        suffix = f"_steps{diagnostic_steps}" if diagnostic_steps else ""
        return resolve_path(run_root) / (
            f"{prefix}/{method}/backbone{backbone_seed}_head{head_seed}{suffix}.pt"
        )

    def smoke_path(method: str, backbone_seed: int, head_seed: int, steps: int) -> Path:
        return resolve_path(run_root) / (
            "smoke/checkpoints/heads/"
            f"{method}/backbone{backbone_seed}_head{head_seed}_steps{steps}.pt"
        )

    base_train._training_state_path = state_path
    base_train._diagnostic_checkpoint_path = smoke_path
    try:
        yield
    finally:
        base_train._training_state_path = original_state
        base_train._diagnostic_checkpoint_path = original_smoke


@contextlib.contextmanager
def _redirect_diagnostics(run_root: str | Path) -> Iterator[None]:
    original = base_diagnose.resolve_path

    def redirected(path: str | Path) -> Path:
        text = str(path)
        prefix = "distance_head_study_runs/diagnostics/"
        smoke_prefix = "distance_head_study_runs/smoke/diagnostics/"
        if text.startswith(prefix):
            return resolve_path(run_root) / "diagnostics" / text.removeprefix(prefix)
        if text.startswith(smoke_prefix):
            return (
                resolve_path(run_root)
                / "smoke/diagnostics"
                / text.removeprefix(smoke_prefix)
            )
        return original(path)

    base_diagnose.resolve_path = redirected
    try:
        yield
    finally:
        base_diagnose.resolve_path = original


def _checkpoint_complete(
    profile_path: str | Path,
    path: Path,
    *,
    method: str,
    backbone_seed: int,
    head_seed: int,
) -> bool:
    if not path.exists():
        return False
    validate_quick_checkpoint(
        profile_path,
        method_name=method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    return True


def _diagnostic_complete(
    config: Any,
    path: Path,
    quick_lock: dict[str, Any],
    *,
    method: str,
    split_role: str,
    backbone_seed: int,
    head_seed: int,
) -> bool:
    if not path.exists():
        return False
    load_validated_diagnostic(
        config,
        quick_lock,
        path,
        split_role=split_role,
        method_name=method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    return True


def _run_train(
    args: argparse.Namespace, profile: Any, config: Any, quick_lock: dict[str, Any]
) -> None:
    _validate_train_request(
        config,
        quick_lock,
        method=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )
    for role in ("train", "cal"):
        validate_quick_cache(
            profile_path=args.profile,
            split_role=role,
            backbone_seed=args.backbone_seed,
        )
    output = head_checkpoint_path(
        config,
        method=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )
    validate_candidate_bank(config, quick_lock, backbone_seed=args.backbone_seed)
    if _checkpoint_complete(
        args.profile,
        output,
        method=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    ):
        print(output)
        return
    values = [
        "--config",
        str(profile.paths.quick_config),
        "--method",
        args.method,
        "--backbone-seed",
        str(args.backbone_seed),
        "--head-seed",
        str(args.head_seed),
        "--device",
        args.device,
        "--resume",
    ]
    with _redirect_training_state(profile.paths.run_root):
        _invoke(base_train, values)
    validate_quick_checkpoint(
        args.profile,
        method_name=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )


def _run_diagnose(
    args: argparse.Namespace, profile: Any, config: Any, quick_lock: dict[str, Any]
) -> None:
    _validate_eval_request(
        config,
        quick_lock,
        method=args.method,
        split_role=args.split_role,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
        action_protocol="corrected_v1",
        diagnostic=True,
    )
    validate_quick_cache(
        profile_path=args.profile,
        split_role=args.split_role,
        backbone_seed=args.backbone_seed,
    )
    validate_candidate_bank(config, quick_lock, backbone_seed=args.backbone_seed)
    validate_quick_checkpoint(
        args.profile,
        method_name=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )
    output = diagnostic_path(
        profile.paths.run_root,
        split_role=args.split_role,
        method=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )
    if _diagnostic_complete(
        config,
        output,
        quick_lock,
        method=args.method,
        split_role=args.split_role,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    ):
        print(output)
        return
    values = [
        "--config",
        str(profile.paths.quick_config),
        "--method",
        args.method,
        "--split-role",
        args.split_role,
        "--backbone-seed",
        str(args.backbone_seed),
        "--head-seed",
        str(args.head_seed),
        "--device",
        args.device,
    ]
    with _redirect_diagnostics(profile.paths.run_root):
        _invoke(base_diagnose, values)
    load_validated_diagnostic(
        config,
        quick_lock,
        output,
        split_role=args.split_role,
        method_name=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )


def _run_evaluate(
    args: argparse.Namespace, profile: Any, config: Any, quick_lock: dict[str, Any]
) -> None:
    _validate_eval_request(
        config,
        quick_lock,
        method=args.method,
        split_role=args.split_role,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
        action_protocol=args.action_protocol,
        diagnostic=False,
    )
    validate_quick_checkpoint(
        args.profile,
        method_name=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )
    output = result_directory(
        config,
        split_role=args.split_role,
        method=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
        action_protocol=args.action_protocol,
    )
    if output.exists():
        try:
            validate_result_cell(
                config,
                quick_lock,
                split_role=args.split_role,
                method_name=args.method,
                backbone_seed=args.backbone_seed,
                head_seed=args.head_seed,
                action_protocol=args.action_protocol,
            )
            print(output / "summary.json")
            return
        except FileNotFoundError:
            pass
    values = [
        "--config",
        str(profile.paths.quick_config),
        "--method",
        args.method,
        "--split-role",
        args.split_role,
        "--backbone-seed",
        str(args.backbone_seed),
        "--head-seed",
        str(args.head_seed),
        "--action-protocol",
        args.action_protocol,
        "--device",
        args.device,
        "--resume",
    ]
    _invoke(base_evaluate, values)
    validate_result_cell(
        config,
        quick_lock,
        split_role=args.split_role,
        method_name=args.method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
        action_protocol=args.action_protocol,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    cache = subparsers.add_parser("rebind-cache")
    cache.add_argument(
        "--split-role", choices=("train", "cal", "screen", "select"), required=True
    )
    cache.add_argument("--backbone-seed", type=int, default=42)
    bank = subparsers.add_parser("candidate-bank")
    bank.add_argument("--backbone-seed", type=int, default=42)
    release = subparsers.add_parser("release-seeds")
    release.add_argument("--tier", choices=("seed1", "seed3"), required=True)
    imported = subparsers.add_parser("import-reference")
    imported.add_argument("--method", choices=("b_dh_cem", "a1_log"), required=True)
    imported.add_argument("--backbone-seed", type=int, default=42)
    imported.add_argument("--head-seed", type=int, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--method", choices=NEW_METHODS, required=True)
    train.add_argument("--backbone-seed", type=int, default=42)
    train.add_argument("--head-seed", type=int, required=True)
    train.add_argument("--device", required=True)
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--method", choices=ALL_METHODS, required=True)
    diagnose.add_argument("--split-role", choices=("screen", "select"), required=True)
    diagnose.add_argument("--backbone-seed", type=int, default=42)
    diagnose.add_argument("--head-seed", type=int, required=True)
    diagnose.add_argument("--device", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--method", choices=ALL_METHODS, required=True)
    evaluate.add_argument(
        "--split-role", choices=("screen", "select", "legacy"), required=True
    )
    evaluate.add_argument("--backbone-seed", type=int, default=42)
    evaluate.add_argument("--head-seed", type=int, required=True)
    evaluate.add_argument(
        "--action-protocol", choices=("corrected_v1", "unmasked"), required=True
    )
    evaluate.add_argument("--device", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--phase", choices=("q1", "q2", "q3"), required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile, config, quick_lock = _load_context(args.profile)
    if args.command == "audit":
        print(run_audit(args.profile)["audit_sha256"])
    elif args.command == "rebind-cache":
        destination = resolve_path(
            config.paths.cache_index_template.format(
                split_role=args.split_role, backbone_seed=args.backbone_seed
            )
        )
        if destination.exists():
            validate_quick_cache(
                profile_path=args.profile,
                split_role=args.split_role,
                backbone_seed=args.backbone_seed,
            )
            print(destination)
        else:
            written = rebind_cache(
                profile_path=args.profile,
                split_role=args.split_role,
                backbone_seed=args.backbone_seed,
            )
            validate_quick_cache(
                profile_path=args.profile,
                split_role=args.split_role,
                backbone_seed=args.backbone_seed,
            )
            print(written)
    elif args.command == "candidate-bank":
        path = candidate_bank_path(
            config, split_role="train", backbone_seed=args.backbone_seed
        )
        if path.exists():
            validate_candidate_bank(
                config,
                quick_lock,
                backbone_seed=args.backbone_seed,
                path=path,
            )
            print(path)
        else:
            _invoke(
                base_candidates,
                [
                    "--config",
                    str(profile.paths.quick_config),
                    "--split-role",
                    "train",
                    "--backbone-seed",
                    str(args.backbone_seed),
                ],
            )
            validate_candidate_bank(
                config,
                quick_lock,
                backbone_seed=args.backbone_seed,
                path=path,
            )
    elif args.command == "release-seeds":
        path = resolve_path(config.paths.seed_release_root) / f"{args.tier}.json"
        if path.exists():
            validate_seed_release(args.profile, tier=args.tier)
            print(path)
        else:
            written = release_seed_tier(args.profile, tier=args.tier)
            validate_seed_release(args.profile, tier=args.tier)
            print(written)
    elif args.command == "import-reference":
        for split_role in ("train", "cal"):
            validate_quick_cache(
                profile_path=args.profile,
                split_role=split_role,
                backbone_seed=args.backbone_seed,
            )
        validate_candidate_bank(config, quick_lock, backbone_seed=args.backbone_seed)
        output = head_checkpoint_path(
            config,
            method=args.method,
            backbone_seed=args.backbone_seed,
            head_seed=args.head_seed,
        )
        if _checkpoint_complete(
            args.profile,
            output,
            method=args.method,
            backbone_seed=args.backbone_seed,
            head_seed=args.head_seed,
        ):
            print(output)
        else:
            written = import_reference_checkpoint(
                profile_path=args.profile,
                method_name=args.method,
                backbone_seed=args.backbone_seed,
                head_seed=args.head_seed,
            )
            validate_quick_checkpoint(
                args.profile,
                method_name=args.method,
                backbone_seed=args.backbone_seed,
                head_seed=args.head_seed,
            )
            print(written)
    elif args.command == "train":
        _run_train(args, profile, config, quick_lock)
    elif args.command == "diagnose":
        _run_diagnose(args, profile, config, quick_lock)
    elif args.command == "evaluate":
        _run_evaluate(args, profile, config, quick_lock)
    elif args.phase == "q1":
        print(select_q1(args.profile))
    elif args.phase == "q2":
        print(select_q2(args.profile))
    else:
        print(assess_q3(args.profile))


if __name__ == "__main__":
    main()
