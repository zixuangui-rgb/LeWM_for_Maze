"""Fail-fast A1 DistanceHead validation on the locked Procgen Maze protocol."""

from __future__ import annotations

PROFILE_ID = "procgen-maze-a1-quick-validation-v1"
PROFILE_SCHEMA = "a1-quick-validation-profile-v1"
PACKAGE_LOCK_SCHEMA = "a1-quick-validation-package-lock-v1"
DECISION_SCHEMA = "a1-quick-validation-decision-v1"
JOB_PLAN_SCHEMA = "a1-quick-validation-job-plan-v1"

REFERENCE_METHODS = ("b_l2_cem", "b_dh_cem", "a1_log")
NEW_METHODS = ("a1_bellman", "a1_predicted", "a1_hcond", "a1_reach")
PROMOTABLE_METHODS = ("a1_bellman", "a1_predicted", "a1_reach")
ALL_METHODS = REFERENCE_METHODS + NEW_METHODS

__all__ = [
    "ALL_METHODS",
    "DECISION_SCHEMA",
    "JOB_PLAN_SCHEMA",
    "NEW_METHODS",
    "PACKAGE_LOCK_SCHEMA",
    "PROMOTABLE_METHODS",
    "PROFILE_ID",
    "PROFILE_SCHEMA",
    "REFERENCE_METHODS",
]
