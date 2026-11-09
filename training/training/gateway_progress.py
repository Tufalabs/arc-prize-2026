"""Ordered, task-opaque reward feedback at the trainer's serving boundary.

Scores and stages are harness claims. The gateway proves their causal order
and reconciliation with the final callback, not the environment's semantics.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any


class ProgressError(RuntimeError):
    """A progress snapshot cannot be reconciled with the captured rollout."""


def validate_progress(
    value: Any, *, require_decision_clock: bool = False
) -> dict[str, Any]:
    """Accept bounded JSON with an explicit previous-response identity."""
    if not isinstance(value, dict) or set(value) != {
        "version",
        "after_nonce",
        "score",
        "stage",
        "data",
    }:
        raise ProgressError("explicit arc3_progress snapshot is required")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ProgressError("unsupported progress version")
    nonce = value["after_nonce"]
    if nonce is not None and (not isinstance(nonce, str) or not nonce):
        raise ProgressError("progress after_nonce must be null or a nonempty string")
    score = value["score"]
    if type(score) not in (int, float) or not math.isfinite(score):
        raise ProgressError("progress score must be finite")
    if type(value["stage"]) not in (int, str):
        raise ProgressError("progress stage must be an opaque integer or string")
    if not isinstance(value["data"], dict):
        raise ProgressError("progress data must be an object")
    clock = value["data"].get("decision_elapsed_seconds")
    if require_decision_clock or "decision_elapsed_seconds" in value["data"]:
        if type(clock) not in (int, float) or not math.isfinite(clock) or clock < 0:
            raise ProgressError(
                "progress decision_elapsed_seconds must be finite and nonnegative"
            )
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, RecursionError) as error:
        raise ProgressError("progress must contain finite JSON data") from error
    if len(encoded.encode("utf-8")) > 16384:
        raise ProgressError("progress snapshot exceeds 16384 bytes")
    return deepcopy(value)


class ProgressLedger:
    """Remember feedback before generation, including a pending terminal draw."""

    def __init__(self) -> None:
        self.snapshots: dict[str, dict[str, dict[str, Any]]] = {}

    def validate_admission(
        self,
        run_id: str,
        nonce: str,
        value: Any,
        *,
        previous_nonce: str | None,
        level_boundary: bool = False,
        require_decision_clock: bool = False,
    ) -> dict[str, Any]:
        """Link this feedback to exactly the last response in this game."""
        snapshot = validate_progress(
            value, require_decision_clock=require_decision_clock
        )
        if snapshot["after_nonce"] != previous_nonce:
            raise ProgressError(
                f"run {run_id}: progress must reference the immediately preceding response"
            )
        prior = self.snapshots.get(run_id, {})
        if nonce in prior:
            raise ProgressError(f"run {run_id}: duplicate progress nonce")
        if level_boundary and (
            previous_nonce is None
            or previous_nonce not in prior
            or type(snapshot["stage"]) is not int
            or type(prior[previous_nonce]["stage"]) is not int
            or snapshot["stage"] <= prior[previous_nonce]["stage"]
        ):
            raise ProgressError(
                f"run {run_id}: boundary compaction requires a newly reported stage transition to a higher completed level"
            )
        previous = prior.get(previous_nonce) if previous_nonce is not None else None
        if previous is not None:
            old_clock = previous["data"].get("decision_elapsed_seconds")
            new_clock = snapshot["data"].get("decision_elapsed_seconds")
            if (
                old_clock is not None
                and new_clock is not None
                and new_clock < old_clock
            ):
                raise ProgressError(
                    f"run {run_id}: decision_elapsed_seconds must not decrease"
                )
        return snapshot

    def record_admission(
        self, run_id: str, nonce: str, snapshot: dict[str, Any]
    ) -> None:
        """Commit previously validated feedback after lineage admission succeeds."""
        self.snapshots.setdefault(run_id, {})[nonce] = deepcopy(snapshot)

    def admit(
        self,
        run_id: str,
        nonce: str,
        value: Any,
        *,
        previous_nonce: str | None,
        level_boundary: bool = False,
        require_decision_clock: bool = False,
    ) -> dict[str, Any]:
        snapshot = self.validate_admission(
            run_id,
            nonce,
            value,
            previous_nonce=previous_nonce,
            level_boundary=level_boundary,
            require_decision_clock=require_decision_clock,
        )
        self.record_admission(run_id, nonce, snapshot)
        return snapshot

    def reconcile(
        self,
        run_id: str,
        terminal: Any,
        steps: list[dict[str, Any]],
        episode_return: float,
        *,
        terminal_reward: float = 0.0,
    ) -> dict[str, Any]:
        """Reconcile the next-call scores with every processed draw's reward.

        A deadline draw may still be in flight. Its admission snapshot exists,
        but it executes no further actions and receives zero immediate reward.
        """
        final = validate_progress(terminal)
        last = steps[-1]["nonce"] if steps else None
        if final["after_nonce"] != last:
            raise ProgressError(
                f"run {run_id}: terminal progress must reference the last processed response"
            )
        snapshots = self.snapshots.get(run_id, {})
        initial = next(iter(snapshots.values()))["score"] if snapshots else 0.0
        cumulative = float(initial)
        for step in steps:
            snapshot = snapshots.get(step["nonce"])
            if snapshot is None or not math.isclose(
                snapshot["score"], cumulative, rel_tol=1e-9, abs_tol=1e-7
            ):
                raise ProgressError(
                    f"run {run_id}: progress score disagrees with ordered callback rewards"
                )
            cumulative += step["reward"]
        for nonce, snapshot in list(snapshots.items())[len(steps) :]:
            if not math.isclose(
                snapshot["score"], cumulative, rel_tol=1e-9, abs_tol=1e-7
            ):
                raise ProgressError(
                    f"run {run_id}: unprocessed draw has inconsistent progress"
                )
        if type(terminal_reward) not in (int, float) or not math.isfinite(
            terminal_reward
        ):
            raise ProgressError(f"run {run_id}: terminal reward must be finite")
        cumulative += terminal_reward
        if not math.isclose(final["score"], cumulative, rel_tol=1e-9, abs_tol=1e-7):
            raise ProgressError(
                f"run {run_id}: terminal progress disagrees with callback rewards"
            )
        if not math.isclose(
            final["score"] - initial, episode_return, rel_tol=1e-9, abs_tol=1e-7
        ):
            raise ProgressError(
                f"run {run_id}: terminal progress disagrees with episode return"
            )
        return final

    def close(self, run_id: str, processed_nonces: list[str]) -> dict[str, Any]:
        """Record the gateway's authoritative terminal anchor before discarding state."""
        snapshots = self.snapshots.pop(run_id, {})
        nonces = list(snapshots)
        return {
            "captured_after_nonce": nonces[-1] if nonces else None,
            "unprocessed_nonces": nonces[len(processed_nonces) :],
        }
