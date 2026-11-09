"""``numpy_scalar_repr``: which repr the kernel gives a NumPy scalar.

NumPy 2 reprs a scalar as its constructor, so a printed coordinate list --
what ``np.argwhere`` produces, on a character board as much as a numeric one
-- reads ``[(np.int64(37), np.int64(7)), ...]`` instead of ``[(37, 7), ...]``.
Measured over a baseline run, that wrapper was about a third of the text in
every tool result carrying coordinates, inside a result capped at a couple of
thousand tokens.

The setting is independent of ``frame_api``: the two were welded together
before, which left the character arms -- the shipped ones -- paying a cost
that had already been fixed for the numeric arm.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from duck import DuckAgent
from duck.python_tool import DuckPythonTool

#: A snippet whose output differs between the two settings: scalars inside a
#: container are repr'd, and ``argwhere`` yields them whatever the board is.
PROBE = "import numpy as np\nprint([tuple(p) for p in np.argwhere(np.eye(2, dtype='int64'))])"


async def _run(repr_mode: str | None, code: str) -> str:
    """Run one snippet under a state payload carrying ``repr_mode``.

    ``None`` omits the key entirely, which is what a default agent sends.
    """
    tool = DuckPythonTool(timeout=30.0, reset_before_each_call=False)

    def _get_state() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_api": "ergonomic",
            "timeline": [],
            "valid_actions": [],
            "last_action_result": {},
        }
        if repr_mode is not None:
            payload["numpy_scalar_repr"] = repr_mode
        return payload

    def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        return {"ack": {}}

    tool.host_functions = {"get_state": _get_state, "action": _action}
    try:
        return (await tool._execute({"code": code})).to_string()  # pyright: ignore[reportPrivateUsage]
    finally:
        tool.reset(keep_tmpdir_contents=False)


async def test_the_np2_setting_prints_the_constructor() -> None:
    """The verbose form spells the wrapper out — the behaviour a default run
    ships, exercised through the kernel rather than read off the field."""
    assert "np.int64(" in await _run("np2", PROBE)


async def test_the_legacy_setting_drops_the_constructor() -> None:
    """The whole point: the same coordinates, without the wrapper."""
    out = await _run("legacy", PROBE)

    assert "np.int64(" not in out
    assert "(0, 0)" in out and "(1, 1)" in out


async def test_an_absent_key_means_the_shipped_default() -> None:
    """A payload from before this setting existed still runs, and runs as it
    did -- which is what lets a default agent omit the key and keep its
    model-visible payload byte-identical."""
    assert "np.int64(" in await _run(None, PROBE)


async def test_the_setting_is_independent_of_the_frame_api() -> None:
    """The two were welded together; the character arms are the ones that were
    left paying for it. A character board must be able to ask for either."""
    agent = DuckAgent(
        reply_cap_tokens=None, compaction_strategy="prune", frame_api="ergonomic", numpy_scalar_repr="legacy"
    )

    payload = agent._state_payload()  # pyright: ignore[reportPrivateUsage]

    assert payload["frame_api"] == "ergonomic"
    assert payload["numpy_scalar_repr"] == "legacy"


def test_an_unknown_repr_mode_is_refused_at_construction() -> None:
    """A typo would otherwise construct fine and fail inside the kernel on the
    first Python call of every game — a configuration error priced as a
    broken arm rather than as a construction error."""
    with pytest.raises(ValueError, match="unknown numpy_scalar_repr"):
        DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", numpy_scalar_repr=cast(Any, "np3"))


def test_the_payload_announces_only_a_repr_numpy_would_not_have_chosen() -> None:
    """The payload's key set is model-visible -- a snippet can print it -- so
    it carries the setting only where the model would otherwise guess wrong.

    The rule is keyed to NumPy's own behaviour, not to the duck's default: a
    kernel rendering scalars the way NumPy 2 does needs no note, and one
    rendering them the old way does. Stated over both values rather than over
    whichever one ships, so moving the default cannot make this pass by
    accident or fail without a reason."""
    assert (
        DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", numpy_scalar_repr="legacy")._state_payload()[
            "numpy_scalar_repr"
        ]
        == "legacy"
    )  # pyright: ignore[reportPrivateUsage]
    assert (
        "numpy_scalar_repr"
        not in DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", numpy_scalar_repr="np2")._state_payload()
    )  # pyright: ignore[reportPrivateUsage]
