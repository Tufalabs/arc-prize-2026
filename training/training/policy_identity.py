"""Immutable provenance for a native weight publication served under the base name.

Two kinds exist: ``full_weight`` (the NCCL full-weight path) and
``policy_weights`` (Flash merged weights, training/flash_policy_weights.py).
Base-model and LoRA identities carry neither.
"""

import re
from typing import Any


FULL_WEIGHT_IDENTITY_FIELDS = {"kind", "weight_version", "manifest_sha256"}
POLICY_WEIGHTS_IDENTITY_FIELDS = {"kind", "policy_sha256", "policy_iteration"}
PUBLISHED_IDENTITY_FIELD_SETS = (FULL_WEIGHT_IDENTITY_FIELDS, POLICY_WEIGHTS_IDENTITY_FIELDS)
_PUBLISHED_FIELDS = FULL_WEIGHT_IDENTITY_FIELDS | POLICY_WEIGHTS_IDENTITY_FIELDS


def published_identity(adapter: dict[str, Any]) -> dict[str, Any] | None:
    """Validate an explicit publication of either kind; None for base/LoRA identities."""
    present = _PUBLISHED_FIELDS.intersection(adapter)
    if not present:
        return None
    base = type(adapter.get("id")) is int and adapter["id"] == 0 and adapter.get("path") is None
    kind = adapter.get("kind")
    if kind == "policy_weights" and present == POLICY_WEIGHTS_IDENTITY_FIELDS:
        if (
            base
            and isinstance(adapter.get("policy_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", adapter["policy_sha256"]) is not None
            and type(adapter.get("policy_iteration")) is int
            and adapter["policy_iteration"] >= -1
        ):
            return {name: adapter[name] for name in POLICY_WEIGHTS_IDENTITY_FIELDS}
        raise ValueError("native engine returned invalid policy weights identity")
    if (
        kind == "full_weight"
        and present == FULL_WEIGHT_IDENTITY_FIELDS
        and base
        and isinstance(adapter.get("weight_version"), str)
        and re.fullmatch(r"[0-9a-f]{32}", adapter["weight_version"]) is not None
        and isinstance(adapter.get("manifest_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", adapter["manifest_sha256"]) is not None
    ):
        return {name: adapter[name] for name in FULL_WEIGHT_IDENTITY_FIELDS}
    raise ValueError("native engine returned invalid full-weight publication identity")


def full_weight_identity(adapter: dict[str, Any]) -> dict[str, str] | None:
    """A full-weight publication's identity; None for any other identity."""
    identity = published_identity(adapter)
    return identity if identity is not None and identity["kind"] == "full_weight" else None


def policy_weights_identity(adapter: dict[str, Any]) -> dict[str, Any] | None:
    """A merged policy weights publication's identity; None for any other identity."""
    identity = published_identity(adapter)
    return identity if identity is not None and identity["kind"] == "policy_weights" else None
