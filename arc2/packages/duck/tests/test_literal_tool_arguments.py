"""Duck's opt-out keeps sampled tool arguments literal.

Malformed arguments return tool errors and remain available for a later
model correction.
"""

from __future__ import annotations

import json
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
    agent = DuckAgent(reply_cap_tokens=None, auto_correct_tool_arguments=False)
    tools = agent._turn_tools()
    try:
        for tool in tools:
            # A second JSON parse would turn this into valid arguments for
            # every tool.
            arguments = {"_raw": json.dumps({key: "example" for key in tool.parameters.get("required", [])})}
            output = await _call(tool, arguments)
            assert tool.is_error, (tool.name, output)
            assert "Invalid arguments" in output
            assert tool.arguments == arguments
    finally:
        assert agent.repl is not None
        await agent.repl.clear_state()


async def test_bad_python_arguments_and_syntax_do_not_execute() -> None:
    agent = DuckAgent(reply_cap_tokens=None, auto_correct_tool_arguments=False)
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
    agent = DuckAgent(reply_cap_tokens=None, auto_correct_tool_arguments=auto_correct)
    tool = agent._turn_tools()[0]
    try:
        output = await _call(tool, {"code": "   \n"})
        assert tool.is_error and "non-empty `code`" in output
        assert agent.repl is not None and agent.repl.is_fresh
    finally:
        await tool.clear_state()
