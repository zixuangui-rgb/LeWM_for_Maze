"""Resolve preregistered method templates and verify causal one-change contrasts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from distance_head_study import PROTOCOL_ID
from distance_head_study.common import (
    canonical_json_sha256,
    load_method_catalog,
    resolve_path,
)
from distance_head_study.gates import load_signed_artifact
from distance_head_study.schemas import MethodCatalog, MethodTemplate, ResolvedMethod

_META_FIELDS = {"name", "stage", "role", "description"}


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor: dict[str, Any] = payload
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"override path is not a mapping: {path}")
        cursor = child
    cursor[keys[-1]] = copy.deepcopy(value)


def _get_path(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing scientific field: {path}")
        value = value[key]
    return value


def _template_by_name(catalog: MethodCatalog, name: str) -> MethodTemplate:
    matches = [method for method in catalog.methods if method.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one method template named {name!r}")
    return matches[0]


def _decision_parent(
    catalog: MethodCatalog,
    alias: str,
    *,
    decision_root: str | Path,
    protocol_lock: dict[str, Any] | None,
) -> tuple[str, str]:
    relative = catalog.decision_aliases[alias]
    path = resolve_path(Path(decision_root) / relative)
    payload = load_signed_artifact(
        path,
        signature_field="decision_sha256",
        expected_protocol_id=PROTOCOL_ID,
        verify_hash_fields=("input_hashes",),
    )
    if payload.get("decision_name") != path.stem:
        raise ValueError(f"decision artifact name/path mismatch: {path}")
    if protocol_lock is not None and (
        payload.get("analysis_spec_sha256") != protocol_lock["analysis_spec_sha256"]
        or payload.get("protocol_lock_sha256") != protocol_lock["protocol_lock_sha256"]
    ):
        raise ValueError(f"decision artifact uses another protocol lock: {path}")
    selected = payload.get("selected_method")
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"decision artifact has no selected_method: {path}")
    eligible = payload.get("eligible_methods")
    if not isinstance(eligible, list) or selected not in eligible:
        raise ValueError(f"selected method is outside the signed eligible set: {path}")
    return selected, str(payload["decision_sha256"])


def resolve_method(
    catalog: MethodCatalog,
    name: str,
    *,
    decision_root: str | Path,
    protocol_lock: dict[str, Any] | None = None,
    _stack: tuple[str, ...] = (),
) -> tuple[ResolvedMethod, tuple[str, ...]]:
    """Resolve dynamic parents and return the effective spec plus decision hashes."""

    if name in _stack:
        raise ValueError(f"method inheritance cycle: {_stack + (name,)}")
    template = _template_by_name(catalog, name)
    if template.parent is None:
        assert template.resolved is not None
        return template.resolved, ()
    parent_name = template.parent
    decision_hashes: tuple[str, ...] = ()
    if parent_name.startswith("@"):
        parent_name, decision_hash = _decision_parent(
            catalog,
            parent_name[1:],
            decision_root=decision_root,
            protocol_lock=protocol_lock,
        )
        decision_hashes = (decision_hash,)
    parent, inherited_hashes = resolve_method(
        catalog,
        parent_name,
        decision_root=decision_root,
        protocol_lock=protocol_lock,
        _stack=_stack + (name,),
    )
    parent_payload = parent.model_dump(mode="json")
    resolved_payload = copy.deepcopy(parent_payload)
    for path, value in template.overrides.items():
        before = _get_path(parent_payload, path)
        if before == value:
            raise ValueError(f"declared change does not change value: {name}:{path}")
        _set_path(resolved_payload, path, value)
    resolved_payload.update(
        {
            "name": template.name,
            "stage": template.stage,
            "role": template.role,
            "description": template.description,
        }
    )
    resolved = ResolvedMethod.model_validate(resolved_payload)
    verify_declared_changes(parent, resolved, template.declared_changes)
    return resolved, inherited_hashes + decision_hashes


def verify_declared_changes(
    parent: ResolvedMethod,
    child: ResolvedMethod,
    declared_changes: tuple[str, ...],
) -> None:
    parent_payload = parent.model_dump(mode="json")
    child_payload = child.model_dump(mode="json")
    for field in _META_FIELDS:
        parent_payload.pop(field, None)
        child_payload.pop(field, None)
    actual: set[str] = set()

    def visit(prefix: str, left: Any, right: Any) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                path = f"{prefix}.{key}" if prefix else key
                visit(path, left.get(key), right.get(key))
            return
        if left != right:
            actual.add(prefix)

    visit("", parent_payload, child_payload)
    expected = set(declared_changes)
    if actual != expected:
        raise ValueError(
            f"scientific diff mismatch for {child.name}: actual={sorted(actual)} "
            f"declared={sorted(expected)}"
        )


def method_sha256(method: ResolvedMethod, decision_hashes: tuple[str, ...]) -> str:
    return canonical_json_sha256(
        {
            "schema": "distance-head-effective-method-v1",
            "method": method.model_dump(mode="json"),
            "decision_sha256s": list(decision_hashes),
        }
    )


def load_and_resolve_method(
    catalog_path: str | Path,
    name: str,
    *,
    decision_root: str | Path,
    protocol_lock: dict[str, Any] | None = None,
) -> tuple[ResolvedMethod, str, tuple[str, ...]]:
    catalog = load_method_catalog(catalog_path)
    method, decision_hashes = resolve_method(
        catalog,
        name,
        decision_root=decision_root,
        protocol_lock=protocol_lock,
    )
    return method, method_sha256(method, decision_hashes), decision_hashes


def validate_static_catalog(catalog: MethodCatalog) -> dict[str, int]:
    """Validate root methods and every static inheritance chain without decisions."""

    def has_dynamic_ancestor(name: str, stack: tuple[str, ...] = ()) -> bool:
        if name in stack:
            raise ValueError(f"method inheritance cycle: {stack + (name,)}")
        template = _template_by_name(catalog, name)
        if template.parent is None:
            return False
        if template.parent.startswith("@"):
            alias = template.parent[1:]
            if alias not in catalog.decision_aliases:
                raise ValueError(f"unknown decision alias: {alias}")
            return True
        return has_dynamic_ancestor(template.parent, stack + (name,))

    roots = 0
    static = 0
    for template in catalog.methods:
        if template.parent is None:
            roots += 1
            continue
        if has_dynamic_ancestor(template.name):
            continue
        resolve_method(catalog, template.name, decision_root=Path("."))
        static += 1
    return {
        "method_count": len(catalog.methods),
        "root_count": roots,
        "static_derived_count": static,
        "dynamic_count": len(catalog.methods) - roots - static,
    }


__all__ = [
    "load_and_resolve_method",
    "method_sha256",
    "resolve_method",
    "validate_static_catalog",
    "verify_declared_changes",
]
