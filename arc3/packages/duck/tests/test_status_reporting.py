"""The duck's refinement of the coarse ``running`` state.

Infra can only see that a game is running; what it is running is the
agent's to say. The duck's kernel work is the half infra could never
infer — the pool is machine-wide, so a host counting its semaphores knows
how many games are queued but not which.
"""

from __future__ import annotations

from typing import Any

import pytest

from duck import DuckAgent
from platypus.status import StatusHandle


def _fake_ctx(published: list[str]) -> Any:
    """Just enough context for the wiring: the handle is all the tool takes."""

    class _Ctx:
        status = StatusHandle(published.append)

    return _Ctx()


def _stub_host_functions() -> dict[str, Any]:
    """The duck prelude calls back for runtime state; a stand-in is enough
    here, where only the reported labels matter."""

    def get_state() -> dict[str, Any]:
        return {
            "frame_api": "ergonomic",
            "timeline": [],
            "valid_actions": [],
            "last_action_result": {},
        }

    def action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        return {"ack": {"queued": 0}}

    return {"get_state": get_state, "action": action}


async def test_the_ducks_kernel_work_reaches_the_run_status() -> None:
    """Three ways this can silently stop working, all of them caught here:
    the duck not handing its tool the context handle, ``DuckPythonTool``'s
    own ``_execute`` not reaching ullm's, and the hook being dropped
    somewhere between the two."""
    published: list[str] = []
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", persistence="round")
    tool = agent._ensure_repl(_fake_ctx(published))
    tool.host_functions = _stub_host_functions()
    try:
        await tool._execute({"code": "print(6 * 7)"})
    finally:
        await tool.clear_state()

    assert "running kernel" in published, published
    assert "waiting for kernel slot" in published, published
    # Back to the agent's own default once the kernel is done: leaving the
    # label behind would report every later LLM call as kernel work.
    assert published[-1] == "running"


def test_the_wiring_is_redone_per_move_not_at_construction() -> None:
    """The handle closes over the hosting process, so it never survives the
    agent's own pickle or deepcopy (A1). An agent that wired it once at
    construction would report nothing at all in subprocess mode — the only
    mode that ships by default."""
    first: list[str] = []
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", persistence="round")
    tool = agent._ensure_repl(_fake_ctx(first))
    assert tool.status_hook is not None

    second: list[str] = []
    rewired = agent._ensure_repl(_fake_ctx(second))
    hook = rewired.status_hook
    assert hook is not None
    with hook("running kernel"):
        pass
    assert second and not first, "the tool kept reporting to the previous move's context"


if __name__ == "__main__":
    pytest.main([__file__])
