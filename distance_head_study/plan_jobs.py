"""Materialize an engineer-readable, dependency-checked experiment job DAG."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from distance_head_study.candidates import (
    candidate_bank_path,
    load_candidate_bank,
)
from distance_head_study.common import (
    atomic_json_dump,
    atomic_text_dump,
    canonical_json_sha256,
    head_checkpoint_path,
    load_study_config,
    require_clean_worktree,
    resolve_path,
    sha256_file,
    source_backbone_path,
    validate_backbone_protocol_binding,
)
from distance_head_study.data import (
    ShardedGoalDataset,
    cache_index_path,
    validate_cache_binding,
    validate_recorded_cache_binding,
)
from distance_head_study.evaluate import _checkpoint_owner
from distance_head_study.gates import (
    load_confirmation_selection,
    load_signed_artifact,
    seed_release_path,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.results import load_complete_rows, result_directory
from distance_head_study.taxonomy import NEGATIVE_CLOSURE_REQUIRED_RUNS


@dataclass(frozen=True)
class Job:
    job_id: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    outputs: tuple[str, ...]
    gpu_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "command": list(self.command),
            "dependencies": list(self.dependencies),
            "outputs": list(self.outputs),
            "gpu_required": self.gpu_required,
        }


def _method_owner(
    config: Any, lock: dict[str, Any], method_name: str
) -> tuple[str | None, bool]:
    method, _, _ = load_and_resolve_method(
        config.paths.method_catalog,
        method_name,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    owner, _ = _checkpoint_owner(config, method, protocol_lock=lock)
    if owner is None:
        return None, False
    return owner, bool(
        method.name == owner and method.trainable and not method.reuse_parent_checkpoint
    )


def _add(jobs: dict[str, Job], job: Job) -> None:
    existing = jobs.get(job.job_id)
    if existing is not None and existing != job:
        raise ValueError(f"job ID collision with different specs: {job.job_id}")
    jobs[job.job_id] = job


def _validate_job_graph(jobs: dict[str, Job]) -> None:
    if not jobs:
        raise ValueError("job plan is empty")
    known = set(jobs)
    outputs: dict[str, str] = {}
    for job in jobs.values():
        missing = set(job.dependencies) - known
        if missing:
            raise ValueError(
                f"job {job.job_id} has missing dependencies: {sorted(missing)}"
            )
        if job.job_id in job.dependencies:
            raise ValueError(f"job {job.job_id} depends on itself")
        if len(job.command) < 3 or job.command[:2] != ("python", "-m"):
            raise ValueError(f"job {job.job_id} has a malformed Python command")
        if not job.command[2].startswith("distance_head_study."):
            raise ValueError(f"job {job.job_id} escapes the study package")
        if not job.outputs:
            raise ValueError(f"job {job.job_id} declares no completion output")
        for output in job.outputs:
            previous = outputs.setdefault(output, job.job_id)
            if previous != job.job_id:
                raise ValueError(
                    f"jobs {previous} and {job.job_id} share output {output}"
                )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(job_id: str) -> None:
        if job_id in visiting:
            raise ValueError(f"job dependency cycle reaches {job_id}")
        if job_id in visited:
            return
        visiting.add(job_id)
        for dependency in jobs[job_id].dependencies:
            visit(dependency)
        visiting.remove(job_id)
        visited.add(job_id)

    for job_id in jobs:
        visit(job_id)


def job_plan_metadata_path(path: str | Path) -> Path:
    return Path(f"{resolve_path(path)}.metadata.json")


def _existing_cache(
    config: Any,
    lock: dict[str, Any],
    *,
    split_role: str,
    backbone_seed: int,
) -> bool:
    path = cache_index_path(config, split_role=split_role, backbone_seed=backbone_seed)
    if not path.exists():
        return False
    dataset = ShardedGoalDataset(path)
    validate_cache_binding(
        dataset,
        config,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=lock,
    )
    return True


def _existing_candidate_bank(
    config: Any,
    lock: dict[str, Any],
    *,
    backbone_seed: int,
) -> bool:
    path = candidate_bank_path(config, split_role="train", backbone_seed=backbone_seed)
    if not path.exists():
        return False
    metadata, actions = load_candidate_bank(path)
    expected = {
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "split_role": "train",
        "backbone_seed": backbone_seed,
        "set_count": config.training.candidate_sets_per_backbone,
        "candidate_count": config.training.trajectory_candidates,
        "horizon": config.planner.horizon,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"existing candidate bank metadata differs: {path}")
    if tuple(actions.shape) != (
        config.training.candidate_sets_per_backbone,
        config.training.trajectory_candidates,
        config.planner.horizon,
    ):
        raise ValueError(f"existing candidate bank shape differs: {path}")
    return True


def _existing_head(
    config: Any,
    lock: dict[str, Any],
    *,
    owner: str,
    backbone_seed: int,
    head_seed: int,
) -> bool:
    path = head_checkpoint_path(
        config,
        method=owner,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    if not path.exists():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    backbone = source_backbone_path(config, backbone_seed)
    owner_method, owner_hash, _ = load_and_resolve_method(
        config.paths.method_catalog,
        owner,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    expected = (
        payload.get("formal_run") is True
        and payload.get("protocol_id") == config.protocol_id
        and payload.get("checkpoint_selection") == "final_step"
        and int(payload.get("final_step", -1)) == config.training.steps
        and payload.get("analysis_spec_sha256") == lock["analysis_spec_sha256"]
        and payload.get("protocol_lock_sha256") == lock["protocol_lock_sha256"]
        and payload.get("backbone_sha256") == sha256_file(backbone)
        and payload.get("method", {}).get("name") == owner
        and payload.get("method_sha256") == owner_hash
        and payload.get("head_spec") == owner_method.head.model_dump(mode="json")
        and int(payload.get("backbone_seed", -1)) == backbone_seed
        and int(payload.get("head_seed", -1)) == head_seed
    )
    if not expected:
        raise ValueError(f"existing head checkpoint provenance differs: {path}")
    bank = payload.get("candidate_bank", {})
    bank_path = bank.get("path")
    if not isinstance(bank_path, str) or sha256_file(bank_path) != bank.get("sha256"):
        raise ValueError(f"existing head candidate bank changed: {path}")
    cache_bindings = payload.get("cache_bindings")
    if not isinstance(cache_bindings, dict) or set(cache_bindings) != {"train", "cal"}:
        raise ValueError(f"existing head cache bindings are incomplete: {path}")
    for split_role, binding in cache_bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"existing head cache binding is malformed: {path}")
        validate_recorded_cache_binding(
            binding,
            split_role=split_role,
            backbone_seed=backbone_seed,
            protocol_lock=lock,
        )
    initialization = payload.get("initialization", {})
    parent_path = initialization.get("parent_checkpoint_path")
    if parent_path is not None and sha256_file(parent_path) != initialization.get(
        "parent_checkpoint_sha256"
    ):
        raise ValueError(f"existing head initialization parent changed: {path}")
    return True


def _existing_diagnostic(
    config: Any,
    lock: dict[str, Any],
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
) -> bool:
    path = resolve_path(
        f"distance_head_study_runs/diagnostics/{split_role}/{method}/"
        f"backbone{backbone_seed}_head{head_seed}.json"
    )
    if not path.exists():
        return False
    payload = load_signed_artifact(
        path,
        signature_field="diagnostic_sha256",
        expected_protocol_id=config.protocol_id,
    )
    _, method_hash, decision_hashes = load_and_resolve_method(
        config.paths.method_catalog,
        method,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    expected = (
        payload.get("analysis_spec_sha256") == lock["analysis_spec_sha256"]
        and payload.get("protocol_lock_sha256") == lock["protocol_lock_sha256"]
        and payload.get("split_role") == split_role
        and payload.get("method") == method
        and payload.get("method_sha256") == method_hash
        and payload.get("decision_sha256s") == list(decision_hashes)
        and int(payload.get("backbone_seed", -1)) == backbone_seed
        and int(payload.get("head_seed", -1)) == head_seed
        and int(payload.get("sample_count", -1))
        == config.analysis.diagnostic_batches * config.training.effective_batch_size
        and int(payload.get("cache_binding", {}).get("diagnostic_limit", -1)) == 0
    )
    if not expected:
        raise ValueError(f"existing diagnostic provenance differs: {path}")
    checkpoint = payload.get("checkpoint", {})
    for path_key, hash_key in (
        ("backbone_path", "backbone_sha256"),
        ("head_checkpoint_path", "head_checkpoint_sha256"),
    ):
        checkpoint_path = checkpoint.get(path_key)
        if checkpoint_path is not None and sha256_file(
            checkpoint_path
        ) != checkpoint.get(hash_key):
            raise ValueError(f"existing diagnostic checkpoint changed: {path}")
    bank = payload.get("candidate_bank", {})
    bank_path = bank.get("path")
    if not isinstance(bank_path, str) or sha256_file(bank_path) != bank.get("sha256"):
        raise ValueError(f"existing diagnostic candidate bank changed: {path}")
    cache_binding = payload.get("cache_binding")
    if not isinstance(cache_binding, dict):
        raise ValueError(f"existing diagnostic cache binding is malformed: {path}")
    validate_recorded_cache_binding(
        cache_binding,
        split_role=split_role,
        backbone_seed=backbone_seed,
        protocol_lock=lock,
    )
    return True


def _existing_result(
    config: Any,
    lock: dict[str, Any],
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
    action_protocol: str,
) -> bool:
    directory = result_directory(
        config,
        split_role=split_role,
        method=method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
        action_protocol=action_protocol,
    )
    if not directory.exists():
        return False
    metadata, rows = load_complete_rows(directory)
    resolved_method, method_hash, decision_hashes = load_and_resolve_method(
        config.paths.method_catalog,
        method,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    expected_manifest = lock["analysis_spec"]["manifests"][split_role]
    expected = (
        metadata.get("analysis_spec_sha256") == lock["analysis_spec_sha256"]
        and metadata.get("protocol_lock_sha256") == lock["protocol_lock_sha256"]
        and metadata.get("split_role") == split_role
        and metadata.get("method", {}).get("name") == method
        and metadata.get("method") == resolved_method.model_dump(mode="json")
        and metadata.get("method_sha256") == method_hash
        and metadata.get("decision_sha256s") == list(decision_hashes)
        and int(metadata.get("backbone_seed", -1)) == backbone_seed
        and int(metadata.get("head_seed", -1)) == head_seed
        and metadata.get("action_protocol") == action_protocol
        and metadata.get("manifest_sha256") == expected_manifest["sha256"]
        and int(metadata.get("diagnostic_limit", -1)) == 0
        and int(metadata.get("num_shards", -1)) == 1
        and len(rows) == int(expected_manifest["count"])
    )
    if not expected:
        raise ValueError(f"existing result provenance differs: {directory}")
    checkpoint = metadata.get("checkpoint", {})
    for path_key, hash_key in (
        ("backbone_path", "backbone_sha256"),
        ("head_checkpoint_path", "head_checkpoint_sha256"),
    ):
        if path_key in checkpoint and sha256_file(
            checkpoint[path_key]
        ) != checkpoint.get(hash_key):
            raise ValueError(f"existing result checkpoint changed: {directory}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--phase", choices=("seed1", "seed3", "seed10"), required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    lock = verify_protocol_lock(config)
    release = load_signed_artifact(
        seed_release_path(config, args.phase),
        signature_field="release_sha256",
        expected_protocol_id=config.protocol_id,
        verify_hash_fields=("prerequisite_hashes",) if args.phase != "seed1" else (),
    )
    if (
        release.get("tier") != args.phase
        or release.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
        or release.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
    ):
        raise ValueError("seed release differs from the requested locked phase")
    backbones = tuple(int(value) for value in release["backbone_seeds"])
    heads = tuple(int(value) for value in release["head_seeds"])
    if args.phase == "seed1":
        methods = (
            NEGATIVE_CLOSURE_REQUIRED_RUNS
            if args.methods.strip() == "@negative_closure"
            else tuple(item.strip() for item in args.methods.split(",") if item.strip())
        )
        if not methods:
            raise ValueError("Seed-1 job plan requires an explicit method block")
        split_role = "screen"
    elif args.phase == "seed3":
        shortlist = load_signed_artifact(
            config.paths.shortlist_lock,
            signature_field="shortlist_sha256",
            expected_protocol_id=config.protocol_id,
            verify_hash_fields=("input_hashes",),
        )
        if (
            shortlist.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
            or shortlist.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
        ):
            raise ValueError("shortlist uses another protocol lock")
        methods = tuple(shortlist["selected_methods"])
        split_role = "select"
    else:
        opened_path = resolve_path(config.paths.confirm_opened)
        if opened_path.exists():
            opened = load_signed_artifact(
                opened_path,
                signature_field="confirm_open_sha256",
                expected_protocol_id=config.protocol_id,
                verify_hash_fields=("input_hashes",),
            )
            methods = tuple(
                method
                for method in opened["allowed_methods"]
                if method not in {"b_l2_cem", "b_dh_cem"}
            )
        else:
            n_lock = load_signed_artifact(
                config.paths.confirmation_n_lock,
                signature_field="confirmation_n_sha256",
                expected_protocol_id=config.protocol_id,
                verify_hash_fields=("input_hashes",),
            )
            _, finalist, _, shortlist = load_confirmation_selection(config, n_lock)
            for name, artifact in (
                ("finalist decision", finalist),
                ("shortlist", shortlist),
                ("confirmation n lock", n_lock),
            ):
                if (
                    artifact.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
                    or artifact.get("protocol_lock_sha256")
                    != lock["protocol_lock_sha256"]
                ):
                    raise ValueError(f"{name} uses another protocol lock")
            methods = (
                (finalist["selected_method"],)
                if n_lock["claim_route"] == "positive"
                else tuple(
                    method
                    for method in finalist["ranked_methods"]
                    if method in shortlist["selected_methods"]
                )
            )
        split_role = "confirm"
    evaluation_methods = ("b_l2_cem", "b_dh_cem", *methods)
    jobs: dict[str, Job] = {}
    all_training_jobs: list[str] = []
    for backbone in backbones:
        backbone_job: str | None = None
        backbone_path = source_backbone_path(config, backbone)
        if args.phase == "seed10":
            if backbone_path.exists():
                payload = torch.load(
                    backbone_path, map_location="cpu", weights_only=False
                )
                validate_backbone_protocol_binding(
                    config,
                    payload,
                    backbone_seed=backbone,
                    protocol_lock=lock,
                )
            else:
                backbone_job = f"backbone_{backbone}"
                _add(
                    jobs,
                    Job(
                        job_id=backbone_job,
                        command=(
                            "python",
                            "-m",
                            "distance_head_study.train_backbone",
                            "--config",
                            args.config,
                            "--backbone-seed",
                            str(backbone),
                        ),
                        dependencies=(),
                        outputs=(backbone_path.as_posix(),),
                    ),
                )
                all_training_jobs.append(backbone_job)
        elif not backbone_path.exists():
            raise FileNotFoundError(backbone_path)
        cache_jobs: dict[str, str | None] = {}
        cache_roles = (
            ("train", "cal") if args.phase == "seed10" else ("train", "cal", split_role)
        )
        for role in cache_roles:
            if _existing_cache(
                config,
                lock,
                split_role=role,
                backbone_seed=backbone,
            ):
                cache_jobs[role] = None
                continue
            job_id = f"cache_{role}_{backbone}"
            cache_jobs[role] = job_id
            _add(
                jobs,
                Job(
                    job_id=job_id,
                    command=(
                        "python",
                        "-m",
                        "distance_head_study.build_cache",
                        "--config",
                        args.config,
                        "--split-role",
                        role,
                        "--backbone-seed",
                        str(backbone),
                    ),
                    dependencies=(() if backbone_job is None else (backbone_job,)),
                    outputs=(
                        cache_index_path(
                            config, split_role=role, backbone_seed=backbone
                        ).as_posix(),
                    ),
                ),
            )
        bank_job: str | None = None
        if not _existing_candidate_bank(config, lock, backbone_seed=backbone):
            bank_job = f"candidate_bank_{backbone}"
            _add(
                jobs,
                Job(
                    job_id=bank_job,
                    command=(
                        "python",
                        "-m",
                        "distance_head_study.candidates",
                        "--config",
                        args.config,
                        "--split-role",
                        "train",
                        "--backbone-seed",
                        str(backbone),
                    ),
                    dependencies=(() if backbone_job is None else (backbone_job,)),
                    outputs=(
                        candidate_bank_path(
                            config, split_role="train", backbone_seed=backbone
                        ).as_posix(),
                    ),
                    gpu_required=False,
                ),
            )
        owners = {"b_dh_cem"}
        for method in methods:
            owner, _ = _method_owner(config, lock, method)
            if owner:
                owners.add(owner)
        head_jobs: dict[tuple[str, int], str | None] = {}
        for owner in sorted(owners):
            owner_method, _, _ = load_and_resolve_method(
                config.paths.method_catalog,
                owner,
                decision_root=config.paths.decision_root,
                protocol_lock=lock,
            )
            if not owner_method.trainable or owner_method.reuse_parent_checkpoint:
                raise ValueError(f"checkpoint owner is not directly trainable: {owner}")
            for head in heads:
                if _existing_head(
                    config,
                    lock,
                    owner=owner,
                    backbone_seed=backbone,
                    head_seed=head,
                ):
                    head_jobs[(owner, head)] = None
                    continue
                job_id = f"head_{owner}_{backbone}_{head}"
                head_jobs[(owner, head)] = job_id
                head_dependencies = [
                    dependency
                    for dependency in (
                        cache_jobs["train"],
                        cache_jobs["cal"],
                        bank_job,
                    )
                    if dependency is not None
                ]
                if owner_method.initialization_parent:
                    parent_job = head_jobs.get(
                        (owner_method.initialization_parent, head)
                    )
                    if parent_job is not None:
                        head_dependencies.append(parent_job)
                _add(
                    jobs,
                    Job(
                        job_id=job_id,
                        command=(
                            "python",
                            "-m",
                            "distance_head_study.train_head",
                            "--config",
                            args.config,
                            "--method",
                            owner,
                            "--backbone-seed",
                            str(backbone),
                            "--head-seed",
                            str(head),
                            "--resume",
                        ),
                        dependencies=tuple(head_dependencies),
                        outputs=(
                            head_checkpoint_path(
                                config,
                                method=owner,
                                backbone_seed=backbone,
                                head_seed=head,
                            ).as_posix(),
                        ),
                    ),
                )
                all_training_jobs.append(job_id)
        for method in evaluation_methods:
            selected_heads = (0,) if method == "b_l2_cem" else heads
            owner, _ = _method_owner(config, lock, method)
            for head in selected_heads:
                dependencies = [
                    dependency
                    for dependency in (cache_jobs.get(split_role),)
                    if dependency is not None
                ]
                if owner:
                    owner_job = head_jobs.get((owner, head))
                    if owner_job is not None:
                        dependencies.append(owner_job)
                if args.phase != "seed10":
                    diagnostic_id = f"diagnose_{split_role}_{method}_{backbone}_{head}"
                    diagnostic_output = resolve_path(
                        f"distance_head_study_runs/diagnostics/{split_role}/{method}/"
                        f"backbone{backbone}_head{head}.json"
                    )
                    if not _existing_diagnostic(
                        config,
                        lock,
                        split_role=split_role,
                        method=method,
                        backbone_seed=backbone,
                        head_seed=head,
                    ):
                        _add(
                            jobs,
                            Job(
                                job_id=diagnostic_id,
                                command=(
                                    "python",
                                    "-m",
                                    "distance_head_study.diagnose",
                                    "--config",
                                    args.config,
                                    "--method",
                                    method,
                                    "--split-role",
                                    split_role,
                                    "--backbone-seed",
                                    str(backbone),
                                    "--head-seed",
                                    str(head),
                                ),
                                dependencies=tuple(
                                    dependencies
                                    + ([] if bank_job is None else [bank_job])
                                ),
                                outputs=(diagnostic_output.as_posix(),),
                            ),
                        )
                for protocol in config.planner.action_protocols:
                    job_id = f"eval_{split_role}_{method}_{backbone}_{head}_{protocol}"
                    directory = result_directory(
                        config,
                        split_role=split_role,
                        method=method,
                        backbone_seed=backbone,
                        head_seed=head,
                        action_protocol=protocol,
                    )
                    if _existing_result(
                        config,
                        lock,
                        split_role=split_role,
                        method=method,
                        backbone_seed=backbone,
                        head_seed=head,
                        action_protocol=protocol,
                    ):
                        continue
                    _add(
                        jobs,
                        Job(
                            job_id=job_id,
                            command=(
                                "python",
                                "-m",
                                "distance_head_study.evaluate",
                                "--config",
                                args.config,
                                "--method",
                                method,
                                "--split-role",
                                split_role,
                                "--backbone-seed",
                                str(backbone),
                                "--head-seed",
                                str(head),
                                "--action-protocol",
                                protocol,
                                "--resume",
                            ),
                            dependencies=tuple(dependencies),
                            outputs=tuple(
                                (directory / name).as_posix()
                                for name in (
                                    "metadata.json",
                                    "rows.jsonl",
                                    "summary.json",
                                )
                            ),
                        ),
                    )
        if args.phase == "seed3":
            power_directory = result_directory(
                config,
                split_role="legacy",
                method="b_dh_cem",
                backbone_seed=backbone,
                head_seed=0,
                action_protocol="corrected_v1",
            )
            if not _existing_result(
                config,
                lock,
                split_role="legacy",
                method="b_dh_cem",
                backbone_seed=backbone,
                head_seed=0,
                action_protocol="corrected_v1",
            ):
                owner_job = head_jobs.get(("b_dh_cem", 0))
                _add(
                    jobs,
                    Job(
                        job_id=f"eval_legacy_power_b_dh_cem_{backbone}_0_corrected_v1",
                        command=(
                            "python",
                            "-m",
                            "distance_head_study.evaluate",
                            "--config",
                            args.config,
                            "--method",
                            "b_dh_cem",
                            "--split-role",
                            "legacy",
                            "--backbone-seed",
                            str(backbone),
                            "--head-seed",
                            "0",
                            "--action-protocol",
                            "corrected_v1",
                            "--resume",
                        ),
                        dependencies=(() if owner_job is None else (owner_job,)),
                        outputs=tuple(
                            (power_directory / name).as_posix()
                            for name in (
                                "metadata.json",
                                "rows.jsonl",
                                "summary.json",
                            )
                        ),
                    ),
                )
    if args.phase == "seed10":
        opened_path = resolve_path(config.paths.confirm_opened)
        open_dependency: str | None = None
        if opened_path.exists():
            opened = load_signed_artifact(
                opened_path,
                signature_field="confirm_open_sha256",
                expected_protocol_id=config.protocol_id,
                verify_hash_fields=("input_hashes",),
            )
            if (
                opened.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]
                or opened.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
                or tuple(opened.get("backbone_seeds", ())) != backbones
                or set(opened.get("allowed_methods", ())) != set(evaluation_methods)
            ):
                raise ValueError("existing confirmation-open artifact differs")
            for path, expected_hash in opened["locked_checkpoint_hashes"].items():
                if sha256_file(path) != expected_hash:
                    raise ValueError("a confirmation checkpoint changed after opening")
        else:
            open_dependency = "open_confirmation"
            open_job = Job(
                job_id=open_dependency,
                command=(
                    "python",
                    "-m",
                    "distance_head_study.open_confirmation",
                    "--config",
                    args.config,
                ),
                dependencies=tuple(sorted(set(all_training_jobs))),
                outputs=(opened_path.as_posix(),),
                gpu_required=False,
            )
            _add(jobs, open_job)
        for job_id, job in list(jobs.items()):
            if job_id.startswith("eval_confirm_") and open_dependency is not None:
                jobs[job_id] = Job(
                    job_id=job.job_id,
                    command=job.command,
                    dependencies=tuple((*job.dependencies, open_dependency)),
                    outputs=job.outputs,
                    gpu_required=job.gpu_required,
                )
    _validate_job_graph(jobs)
    output = resolve_path(args.output)
    metadata_path = job_plan_metadata_path(output)
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite job plan or metadata: {output}")
    serialized = "".join(
        json.dumps(job.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        for job in jobs.values()
    )
    atomic_text_dump(output, serialized)
    metadata = {
        "schema": "distance-head-job-plan-v1",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "config_path": resolve_path(args.config).as_posix(),
        "config_sha256": sha256_file(resolve_path(args.config)),
        "phase": args.phase,
        "split_role": split_role,
        "backbone_seeds": list(backbones),
        "head_seeds": list(heads),
        "methods": list(methods),
        "evaluation_methods": list(evaluation_methods),
        "seed_release_path": seed_release_path(config, args.phase).as_posix(),
        "seed_release_file_sha256": sha256_file(seed_release_path(config, args.phase)),
        "job_plan_path": output.as_posix(),
        "job_plan_sha256": sha256_file(output),
        "job_count": len(jobs),
        "job_ids": list(jobs),
    }
    metadata["job_plan_metadata_sha256"] = canonical_json_sha256(metadata)
    atomic_json_dump(metadata_path, metadata)
    print(Path(output))


if __name__ == "__main__":
    main()
