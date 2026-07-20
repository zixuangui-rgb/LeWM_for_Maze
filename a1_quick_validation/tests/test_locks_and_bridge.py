from __future__ import annotations

import copy

import pytest
import torch

from a1_quick_validation.checkpoint_bridge import state_dict_sha256
from a1_quick_validation.common import (
    DEFAULT_PACKAGE_LOCK,
    DEFAULT_PROFILE,
    atomic_json_dump,
    load_json,
)
from a1_quick_validation.profile import build_package_lock, verify_package_lock


def test_package_lock_matches_every_current_package_file() -> None:
    _, lock, _ = verify_package_lock()
    assert lock == build_package_lock()
    assert "a1_quick_validation/docs/EXPERIMENT_DESIGN.zh.md" in lock["package_files"]
    assert "a1_quick_validation/tests/test_selection.py" in lock["package_files"]
    assert "a1_quick_validation/configs/protocol_lock.json" in lock["package_files"]


def test_package_lock_rejects_tampered_signature(tmp_path) -> None:
    payload = copy.deepcopy(load_json(DEFAULT_PACKAGE_LOCK))
    payload["claim_boundary"] += " tampered"
    path = tmp_path / "package_lock.json"
    atomic_json_dump(path, payload)
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_package_lock(DEFAULT_PROFILE, path)


def test_tensor_state_hash_is_order_stable_and_parameter_sensitive() -> None:
    first = {"b": torch.ones(2), "a": torch.arange(3, dtype=torch.float32)}
    reordered = {"a": first["a"].clone(), "b": first["b"].clone()}
    assert state_dict_sha256(first) == state_dict_sha256(reordered)
    changed = {name: value.clone() for name, value in first.items()}
    changed["a"][0] = 9.0
    assert state_dict_sha256(first) != state_dict_sha256(changed)
