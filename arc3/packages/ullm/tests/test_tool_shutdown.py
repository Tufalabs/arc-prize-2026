"""Tearing a tool down while one of its calls is still in flight.

``clear_state`` refuses to run against a live call and is right to: freeing a
subprocess, a temp directory and a semaphore slot underneath the code still
using them is a race. The consequence was that a caller tearing down after
its own work had ended — an agent whose game finished while a REPL cell was
executing — could only skip the tool and leak all three.

``shutdown`` is the way through: cancel the call, wait for it to unwind, then
clear. These cover the three states it has to handle, and that it reaches the
"cancelled" state properly rather than forcing past the assertion.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
import ullm.conv
from ullm.tool import Tool


@dataclass
class _BlockingTool(Tool):
    """A tool whose call blocks until released, so a teardown can catch it
    mid-execution. Records whether its resources were ever freed."""

    name: str = "blocking"
    description: str = "Blocks until released."
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    cleared: bool = field(default=False, init=False)
    started: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Signal that the call is live, then block indefinitely."""
        self.started.set()
        await asyncio.sleep(3600)
        return ullm.conv.StringContent(text="never reached")

    async def _clear_state(self) -> None:
        """Stand-in for killing a kernel and dropping its tmpdir."""
        self.cleared = True


async def test_shutdown_frees_resources_of_a_call_still_in_flight() -> None:
    """The case the old code could only skip: a live call at teardown.

    Asserted on the resource, not on the mechanism — what went wrong before
    was a kernel and a semaphore slot surviving the game that owned them.
    """
    tool = _BlockingTool()
    task = asyncio.create_task(tool.run())
    await tool.started.wait()
    assert tool.state == "running"

    await tool.shutdown()

    assert tool.cleared, "teardown left the tool's resources allocated"
    assert task.done(), "the in-flight call was left running after shutdown"


async def test_shutdown_reaches_the_state_clear_state_demands() -> None:
    """It cancels and waits rather than forcing past the assertion, so the
    tool ends in a state ``clear_state`` would itself accept."""
    tool = _BlockingTool()
    task = asyncio.create_task(tool.run())
    await tool.started.wait()

    await tool.shutdown()

    assert tool.state == "cancelled"
    with pytest.raises(asyncio.CancelledError):
        task.result()


async def test_shutdown_on_an_idle_tool_is_just_a_clear() -> None:
    """Nothing in flight: no cancellation, no waiting, resources still freed."""
    tool = _BlockingTool()
    assert tool.state == "ready"

    await tool.shutdown()

    assert tool.cleared
    assert tool.state == "ready", "an idle tool should not be moved to cancelled"
