"""A4: optional compaction inventory failure preserves a usable Python session."""

import asyncio
from pathlib import Path

import pytest
import ullm.backend
import ullm.conv

from duck import DuckAgent
from duck.python_tool import DuckPythonTool


async def test_listing_failure_is_visible_and_existing_session_still_executes() -> None:
    agent = DuckAgent(reply_cap_tokens=512, compaction_strategy="summarize", persistence="level")
    tool = agent._ensure_repl()
    tool.isolate_network = False
    tool.host_functions = {
        "get_state": lambda: {
            "frame_api": "ergonomic",
            "file_tools": True,
            "timeline": [],
            "valid_actions": [],
            "last_action_result": {},
        }
    }
    try:
        assert tool._tmpdir is not None
        Path(tool._tmpdir, "inventory_helper.py").write_text(
            "class InventoryFails(list):\n    def __len__(self):\n        raise RuntimeError('inventory unavailable')\n"
        )
        Path(tool._tmpdir, "solution.py").write_text("# saved solution\n")
        initial = await tool._execute(
            {"code": "from inventory_helper import InventoryFails\nscratch = InventoryFails([7])\ncarried = 42"}
        )
        assert not tool.is_error and tool._process is not None, initial.to_string()
        pid = tool._process.pid
        records = len(tool._execution_records)
        tool.is_error = True
        listing = await agent._files_and_repl_listing()
        assert "listing unavailable" in listing and "session was not cleared" in listing
        assert "inventory unavailable" in listing and "(none)" not in listing
        assert "Your files:" in listing and "`solution.py` (17 B)" in listing
        assert tool.is_error and len(tool._execution_records) == records
        result = await tool._execute({"code": "print(carried, scratch[0])"})
        assert not tool.is_error and "42 7" in result.to_string()
        assert tool._process is not None and tool._process.pid == pid
    finally:
        await tool.clear_state()


async def test_reset_during_inspection_does_not_claim_the_rebuilt_session_is_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = DuckAgent(compaction_strategy="summarize")
    tool = agent._ensure_repl()

    async def reset_inventory(self: DuckPythonTool) -> list[tuple[str, str, bool]]:
        from duck.python_tool import SessionInspectionError

        await self.clear_state()
        raise SessionInspectionError("inspection reset the kernel")

    async def count(*args: object, **kwargs: object) -> int:
        return 100

    def discard(elements: list[ullm.conv.ConversationElement]) -> None:
        pass

    monkeypatch.setattr(DuckPythonTool, "session_objects", reset_inventory)
    monkeypatch.setattr(agent, "_request_prompt_tokens", count)
    try:
        listing = await agent._files_and_repl_listing()
        assert tool.is_fresh and "may have been lost" in listing
        prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="task"))
        agent.messages = [prompt, ullm.conv.ResponseTurn()]
        agent._turn_prompt = prompt
        agent._live_params = ullm.backend.GenericSamplingParameters(max_tokens=128)
        await agent._install_summary("saved plan", listing, 128, record_pruned_round=discard)
        rendered = "\n".join(message.to_string() for message in agent.messages)
        assert "session was reset" in rendered and "session and files are intact" not in rendered
    finally:
        await tool.clear_state()


async def test_listing_does_not_swallow_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DuckAgent(reply_cap_tokens=512, compaction_strategy="summarize")

    async def cancelled(self: DuckPythonTool) -> list[tuple[str, str, bool]]:
        raise asyncio.CancelledError

    monkeypatch.setattr(DuckPythonTool, "session_objects", cancelled)
    try:
        with pytest.raises(asyncio.CancelledError):
            await agent._files_and_repl_listing()
    finally:
        if agent.repl is not None:
            await agent.repl.clear_state()
