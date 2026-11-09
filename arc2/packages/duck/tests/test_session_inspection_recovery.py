"""A session listing that fails leaves a usable ``python`` session and is
said to be unavailable, with no more of the error than its head, and the
conversation rebuilt around a summary says when that session was reset
rather than calling it intact."""

import asyncio
from pathlib import Path

import pytest
import ullm.conv
from duck import DuckAgent
from duck.python_tool import LISTING_ERROR_MAX_CHARS, DuckPythonTool, SessionInspectionError
from test_files_after_a_summary import _agent, _summarize


async def test_listing_failure_is_visible_and_existing_session_still_executes() -> None:
    target = await _agent()
    tool = target._ensure_repl()
    try:
        assert tool._tmpdir is not None
        Path(tool._tmpdir, "inventory_helper.py").write_text(
            "class InventoryFails(list):\n    def __len__(self):\n        raise RuntimeError('inventory unavailable')\n"
        )
        initial = await tool._execute(
            {"code": "from inventory_helper import InventoryFails\nscratch = InventoryFails([7])\ncarried = 42"}
        )
        assert not tool.is_error and tool._process is not None, initial.to_string()
        pid = tool._process.pid
        tool.is_error = True
        listing = await target._files_and_repl_listing()
        assert "listing unavailable" in listing and "session was not cleared" in listing
        assert "inventory unavailable" in listing and "(none)" not in listing
        # The exception line alone: the listing's frames are the harness's.
        assert "Traceback" not in listing
        assert "Your files:" in listing and "`candidate.py` (" in listing
        assert tool.is_error
        result = await tool._execute({"code": "print(carried, scratch[0])"})
        assert not tool.is_error and "42 7" in result.to_string()
        assert tool._process is not None and tool._process.pid == pid
    finally:
        await tool.clear_state()


async def test_a_long_listing_error_reaches_the_rebuilt_conversation_cut() -> None:
    """An object whose description raises with a million-character message:
    the summary still installs, and its closing message carries the
    exception's head, not the whole line."""
    target = await _agent()
    tool = target._ensure_repl()
    try:
        assert tool._tmpdir is not None
        Path(tool._tmpdir, "inventory_helper.py").write_text(
            "class InventoryFails(list):\n"
            "    def __len__(self):\n"
            "        raise RuntimeError('inventory unavailable ' + 'x' * 1_000_000)\n"
        )
        initial = await tool._execute(
            {"code": "from inventory_helper import InventoryFails\nscratch = InventoryFails([7])"}
        )
        assert not tool.is_error, initial.to_string()
        await _summarize(target)
        (closing,) = [
            message.content.to_string()
            for message in target.messages
            if isinstance(message, ullm.conv.UserPrompt)
            and message.content.to_string().startswith("[context compacted — your")
        ]
        assert "listing unavailable" in closing and "RuntimeError: inventory unavailable xxx" in closing
        assert len(closing) < LISTING_ERROR_MAX_CHARS + 1_000
    finally:
        await tool.clear_state()


async def test_reset_during_inspection_does_not_claim_the_rebuilt_session_is_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = await _agent()
    tool = target._ensure_repl()

    async def reset_inventory(self: DuckPythonTool) -> list[tuple[str, str, bool]]:
        await self.clear_state()
        raise SessionInspectionError("inspection reset the kernel")

    monkeypatch.setattr(DuckPythonTool, "session_objects", reset_inventory)
    try:
        await _summarize(target)
        (closing,) = [
            message.content.to_string()
            for message in target.messages
            if isinstance(message, ullm.conv.UserPrompt)
            and message.content.to_string().startswith("[context compacted — your")
        ]
        assert tool.is_fresh
        assert closing.startswith("[context compacted — your `python` session was reset; continue from the summary]")
        assert "may have been lost" in closing and "intact" not in closing
    finally:
        await tool.clear_state()


async def test_listing_does_not_swallow_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DuckAgent()

    async def cancelled(self: DuckPythonTool) -> list[tuple[str, str, bool]]:
        raise asyncio.CancelledError

    monkeypatch.setattr(DuckPythonTool, "session_objects", cancelled)
    try:
        with pytest.raises(asyncio.CancelledError):
            await agent._files_and_repl_listing()
    finally:
        if agent.repl is not None:
            await agent.repl.clear_state()
