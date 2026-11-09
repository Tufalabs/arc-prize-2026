"""A4: the optional compaction inventory failing leaves a usable ``python``
session, puts no more of the error in the rebuilt conversation than its
head, and a rebuilt conversation says when that session was reset."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
import ullm.backend
import ullm.conv

from seal import SealAgent
from seal.prompts import SealPrompts
from seal.repl import LISTING_ERROR_MAX_CHARS, GameREPL, SessionInspectionError


def _host_functions() -> dict[str, Any]:
    """The minimal state the kernel's prelude asks the host for, standalone."""

    def get_state() -> dict[str, Any]:
        return {"level": 1, "turns": [], "available_actions": []}

    def action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        return {"ack": {"queued": 0}}

    return {"get_state": get_state, "action": action}


async def test_listing_failure_is_visible_and_existing_session_still_executes() -> None:
    agent = SealAgent(reply_cap_tokens=512, compaction_strategy="summarize", persistence="level")
    tool = agent._ensure_repl()  # pyright: ignore[reportPrivateUsage]
    tool.isolate_network = False
    tool.host_functions = _host_functions()
    try:
        assert tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage]
        Path(tool._tmpdir, "inventory_helper.py").write_text(  # pyright: ignore[reportPrivateUsage]
            "class InventoryFails(list):\n    def __len__(self):\n        raise RuntimeError('inventory unavailable')\n"
        )
        Path(tool._tmpdir, "solution.py").write_text("# saved solution\n")  # pyright: ignore[reportPrivateUsage]
        initial = await tool._execute(  # pyright: ignore[reportPrivateUsage]
            {"code": "from inventory_helper import InventoryFails\nscratch = InventoryFails([7])\ncarried = 42"}
        )
        assert not tool.is_error and tool._process is not None, initial.to_string()  # pyright: ignore[reportPrivateUsage]
        pid = tool._process.pid  # pyright: ignore[reportPrivateUsage]
        listing = await agent._files_and_repl_listing()  # pyright: ignore[reportPrivateUsage]
        assert "listing unavailable" in listing and "session was not cleared" in listing
        assert "inventory unavailable" in listing and "(none)" not in listing
        assert "Your files:" in listing and "`solution.py` (17 B)" in listing
        result = await tool._execute({"code": "print(carried, scratch[0])"})  # pyright: ignore[reportPrivateUsage]
        assert not tool.is_error and "42 7" in result.to_string()
        assert tool._process is not None and tool._process.pid == pid  # pyright: ignore[reportPrivateUsage]
    finally:
        await tool.clear_state()


async def test_reset_during_inspection_does_not_claim_the_rebuilt_session_is_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = SealAgent(compaction_strategy="summarize")
    tool = agent._ensure_repl()  # pyright: ignore[reportPrivateUsage]

    async def reset_inventory(self: GameREPL, *, timeout: float) -> list[tuple[str, str, bool]]:
        await self.clear_state()
        raise SessionInspectionError("inspection reset the kernel")

    async def count(*args: object, **kwargs: object) -> int:
        return 100

    def discard(elements: list[ullm.conv.ConversationElement]) -> None:
        pass

    monkeypatch.setattr(GameREPL, "session_objects", reset_inventory)
    monkeypatch.setattr(agent, "_request_prompt_tokens", count)
    try:
        listing = await agent._files_and_repl_listing()  # pyright: ignore[reportPrivateUsage]
        assert tool.is_fresh and "may have been lost" in listing
        prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="task"))
        agent.messages = [prompt, ullm.conv.ResponseTurn()]
        agent._turn_prompt = prompt  # pyright: ignore[reportPrivateUsage]
        agent._live_params = ullm.backend.GenericSamplingParameters(max_tokens=128)  # pyright: ignore[reportPrivateUsage]
        await agent._install_summary("saved plan", listing, 128, record_pruned_round=discard)  # pyright: ignore[reportPrivateUsage]
        rendered = "\n".join(message.to_string() for message in agent.messages)
        assert SealPrompts().compaction_closing_line_reset in rendered
        assert "intact; continue from here" not in rendered
    finally:
        await tool.clear_state()


async def test_a_long_listing_error_reaches_the_rebuilt_conversation_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """An object whose description raises with a million-character message:
    the summary still installs, and its closing message carries the
    exception's head, not the whole line."""
    agent = SealAgent(reply_cap_tokens=512, compaction_strategy="summarize", persistence="level")
    tool = agent._ensure_repl()  # pyright: ignore[reportPrivateUsage]
    tool.isolate_network = False
    tool.host_functions = _host_functions()

    async def count(*args: object, **kwargs: object) -> int:
        return 100

    def discard(elements: list[ullm.conv.ConversationElement]) -> None:
        pass

    monkeypatch.setattr(agent, "_request_prompt_tokens", count)
    try:
        assert tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage]
        Path(tool._tmpdir, "inventory_helper.py").write_text(  # pyright: ignore[reportPrivateUsage]
            "class InventoryFails(list):\n"
            "    def __len__(self):\n"
            "        raise RuntimeError('inventory unavailable ' + 'x' * 1_000_000)\n"
        )
        initial = await tool._execute(  # pyright: ignore[reportPrivateUsage]
            {"code": "from inventory_helper import InventoryFails\nscratch = InventoryFails([7])"}
        )
        assert not tool.is_error, initial.to_string()
        listing = await agent._files_and_repl_listing()  # pyright: ignore[reportPrivateUsage]
        prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="task"))
        agent.messages = [prompt, ullm.conv.ResponseTurn()]
        agent._turn_prompt = prompt  # pyright: ignore[reportPrivateUsage]
        agent._live_params = ullm.backend.GenericSamplingParameters(max_tokens=128)  # pyright: ignore[reportPrivateUsage]
        await agent._install_summary("saved plan", listing, 128, record_pruned_round=discard)  # pyright: ignore[reportPrivateUsage]
        (closing,) = [
            message.content.to_string()
            for message in agent.messages
            if isinstance(message, ullm.conv.UserPrompt)
            and message.content.to_string().startswith("[context compacted — your `python`")
        ]
        assert "listing unavailable" in closing and "RuntimeError: inventory unavailable xxx" in closing
        assert len(closing) < LISTING_ERROR_MAX_CHARS + 1_000
    finally:
        await tool.clear_state()


async def test_listing_does_not_swallow_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = SealAgent(reply_cap_tokens=512, compaction_strategy="summarize")

    async def cancelled(self: GameREPL, *, timeout: float) -> list[tuple[str, str, bool]]:
        raise asyncio.CancelledError

    monkeypatch.setattr(GameREPL, "session_objects", cancelled)
    try:
        with pytest.raises(asyncio.CancelledError):
            await agent._files_and_repl_listing()  # pyright: ignore[reportPrivateUsage]
    finally:
        if agent.repl is not None:
            await agent.repl.clear_state()
