"""The duck's refinement of the coarse ``running`` state.

Infra can only see that a game is running; what it is running is the agent's
to say. Kernel work is the half infra could never infer — the pool is
machine-wide, so a host counting its semaphores knows how many games are
queued but not which.
"""

from __future__ import annotations

from typing import Any

from duck import DuckAgent

from platypus.status import StatusHandle


def _fake_ctx(published: list[str]) -> Any:
    """Just enough context for the wiring: the handle is all the tool takes."""

    class _Ctx:
        status = StatusHandle(published.append)

    return _Ctx()


async def test_the_ducks_kernel_work_reaches_the_run_status() -> None:
    """Three ways this can silently stop working, all caught here: the duck
    not handing its tool the context handle, ``DuckPythonTool``'s own
    ``_execute`` not reaching ullm's, and the hook being dropped between the
    two."""
    published: list[str] = []
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    tool = agent._ensure_repl(_fake_ctx(published))
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
    agent's own pickle or deepcopy. An agent that wired it once at
    construction would report nothing at all in subprocess mode — the only
    mode that ships by default."""
    first: list[str] = []
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    tool = agent._ensure_repl(_fake_ctx(first))
    assert tool.status_hook is not None

    second: list[str] = []
    rewired = agent._ensure_repl(_fake_ctx(second))
    hook = rewired.status_hook
    assert hook is not None
    with hook("running kernel"):
        pass
    assert second and not first, "the tool kept reporting to the previous move's context"
