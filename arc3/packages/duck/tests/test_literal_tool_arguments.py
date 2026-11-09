"""Duck's opt-out keeps sampled tool arguments literal (U9).

Malformed arguments return tool errors and remain available for a later
model correction; file contents are written exactly as JSON decoded them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import ullm.tool

from duck.agent import DuckAgent


async def _call(tool: ullm.tool.Tool, arguments: dict[str, Any]) -> str:
    if tool.state != "ready":
        tool.rearm()
    tool.arguments = arguments
    await tool.run()
    return tool.result.to_string()


async def test_every_offered_tool_rejects_repaired_argument_forms() -> None:
    agent = DuckAgent(
        journal=True, auto_correct_tool_arguments=False, expose_tool_search_files=True, expose_tool_undo_edit=True
    )
    tools = agent._turn_tools()
    try:
        for tool in tools:
            # A second JSON parse would turn this into valid arguments for
            # every tool, including the zero-required-argument undo tool.
            arguments = {"_raw": json.dumps({key: "example" for key in tool.parameters.get("required", [])})}
            output = await _call(tool, arguments)
            assert tool.is_error, (tool.name, output)
            assert "Invalid arguments" in output
            assert tool.arguments == arguments
        memory = next(tool for tool in tools if tool.name == "memory")
        output = await _call(memory, {"notes": "do not save this guessed argument"})
        assert memory.is_error and "Invalid arguments" in output
        assert agent.memory_entries == []
    finally:
        assert agent.repl is not None
        await agent.repl.clear_state()


@pytest.mark.parametrize("auto_correct", [False, True])
async def test_create_and_replace_obey_literal_content_policy(auto_correct: bool) -> None:
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", auto_correct_tool_arguments=auto_correct)
    tools = {tool.name: tool for tool in agent._turn_tools()}
    assert agent.workspace is not None and agent.repl is not None
    path = Path(agent.workspace.resolve("notes.txt"))
    try:
        original = "one\\ntwo"
        created = await _call(tools["create_file"], {"path": "notes.txt", "content": original})
        assert not tools["create_file"].is_error, created
        before = "one\ntwo" if auto_correct else original
        assert path.read_text() == before
        replacement = "three\\tfour"
        edited = await _call(
            tools["str_replace"], {"path": "notes.txt", "old_string": before, "new_string": replacement}
        )
        assert not tools["str_replace"].is_error, edited
        assert path.read_text() == ("three\tfour" if auto_correct else replacement)

        # Properly encoded JSON whitespace needs no heuristic repair.
        valid = json.loads('{"path":"notes.txt","content":"five\\nsix"}')
        await _call(tools["create_file"], valid)
        assert path.read_text() == "five\nsix"
    finally:
        await agent.repl.clear_state()


async def test_bad_python_arguments_and_syntax_do_not_execute() -> None:
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", auto_correct_tool_arguments=False)
    tool = agent._turn_tools()[0]
    try:
        for arguments in ({"input": "print(1)"}, {"code": "x = 1\\nprint(x)"}):
            output = await _call(tool, arguments)
            assert tool.is_error, output
            assert tool.arguments == arguments
            assert agent.repl is not None and agent.repl.is_fresh
        assert "Python syntax error" in output
    finally:
        await tool.clear_state()


@pytest.mark.parametrize("auto_correct", [False, True])
async def test_blank_python_is_rejected_without_starting_a_kernel(auto_correct: bool) -> None:
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", auto_correct_tool_arguments=auto_correct)
    tool = agent._turn_tools()[0]
    try:
        output = await _call(tool, {"code": "   \n"})
        assert tool.is_error and "non-empty `code`" in output
        assert agent.repl is not None and agent.repl.is_fresh
    finally:
        await tool.clear_state()
