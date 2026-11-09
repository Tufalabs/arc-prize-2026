"""Checkpoint-identity facts the harness reports from its rollouts.

Part of what a checkpoint is bound to is the harness's, not the recipe's: the
ARC-2 Duck's prompt injection is whatever its engine preset resolves, and the
trainer cannot import the Duck to ask. The workers read such settings off the
agent they built and report them with every run; the harness adapter's
``rollout_identity`` gathers them from a window's rollouts, and the trainer
pins them in its checkpoint identity under ``HARNESS_AGENT_KEY`` at the first
window. An identity built from the config never holds the key: a resumed
trainer adopts the saved checkpoint's, and every later window must report the
same values, or training stops naming the setting that changed.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

HARNESS_AGENT_KEY = "harness_agent"
_ABSENT = object()


def without_harness_agent(identity: Any) -> Any:
    """The identity a config determines, without the facts rollouts pinned."""
    if not isinstance(identity, dict):
        return identity
    return {key: value for key, value in identity.items() if key != HARNESS_AGENT_KEY}


def pin_harness_agent(
    identity: dict[str, Any] | None, reported: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    """IDENTITY with REPORTED pinned; refuse a setting that differs from its pin.

    ``source`` names where REPORTED comes from in the error.
    """
    if not isinstance(identity, dict):
        raise RuntimeError(
            f"{source} reports harness agent settings, but this run has no checkpoint identity"
        )
    if not isinstance(reported, Mapping) or not reported:
        raise RuntimeError(f"{source} reports no harness agent settings")
    pinned = identity.get(HARNESS_AGENT_KEY)
    if pinned is None:
        return {**identity, HARNESS_AGENT_KEY: copy.deepcopy(dict(reported))}
    for name in sorted(set(pinned) | set(reported)):
        expected = pinned.get(name, _ABSENT)
        actual = reported.get(name, _ABSENT)
        if expected != actual:
            raise RuntimeError(
                f"harness agent setting {name} changed: {source} reports "
                f"{'nothing' if actual is _ABSENT else repr(actual)}, the checkpoint "
                f"identity pinned {'nothing' if expected is _ABSENT else repr(expected)} "
                "at this run's first window"
            )
    return identity


def adopt_harness_agent(
    expected: dict[str, Any] | None, saved: Any
) -> dict[str, Any] | None:
    """EXPECTED (a config's identity) with the pin of a SAVED identity whose
    config facts matched, for a resumed trainer."""
    if not isinstance(saved, dict) or HARNESS_AGENT_KEY not in saved:
        return expected
    return pin_harness_agent(
        expected, saved[HARNESS_AGENT_KEY], source="the resumed checkpoint"
    )
