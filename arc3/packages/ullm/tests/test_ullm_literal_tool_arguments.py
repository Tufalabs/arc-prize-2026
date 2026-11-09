"""Literal execution policy prevents heuristic repairs without blocking recovery (U9)."""

import json
import pathlib
from typing import Any

import pytest
import ullm.tool
import ullm.tools.file
import ullm.tools.python


async def _call(tool: ullm.tool.Tool, arguments: dict[str, Any]) -> str:
    if tool.state != "ready":
        tool.rearm()
    tool.arguments = arguments
    await tool.run()
    return tool.result.to_string()


@pytest.mark.parametrize("auto_correct", [False, True])
async def test_argument_repair_policy_preserves_default_and_rejects_wrong_names(
    tmp_path: pathlib.Path, auto_correct: bool
) -> None:
    real = tmp_path / "note.txt"
    real.write_text("payload")
    tool = ullm.tools.file.ReadFileTool(
        allowed_files=["note.txt"], path_mapping={"note.txt": str(real)}, auto_correct_arguments=auto_correct
    )
    for arguments in ({"filename": "note.txt"}, {"_raw": json.dumps({"path": "note.txt"})}):
        result = await _call(tool, arguments)
        assert tool.is_error is not auto_correct, result
        if auto_correct:
            assert "payload" in result
        else:
            assert "Invalid arguments" in result
            assert tool.arguments == arguments
    result = await _call(tool, {"path": "note.txt"})
    assert not tool.is_error and "payload" in result


async def test_bound_python_rejects_repairs_and_recovers_on_a_later_call() -> None:
    target = ullm.tools.python.PythonTool(auto_correct_arguments=False, isolate_network=False)
    tool = target.bound(host_functions={})
    try:
        output = await _call(tool, {"input": "print('not executed')"})
        assert tool.is_error and "Invalid arguments" in output
        assert target.is_fresh
        bad = {"code": "value = 1\\nprint(value)"}
        output = await _call(tool, bad)
        assert tool.is_error and "SyntaxError" in output
        assert tool.arguments == bad
        output = await _call(tool, {"code": "value = 2\nprint(value)"})
        assert not tool.is_error and output.strip() == "2"
    finally:
        await target.clear_state()


async def test_linked_file_tools_keep_literal_content() -> None:
    target = ullm.tools.python.PythonTool(auto_correct_arguments=False)
    read, write, edit = target.linked_file_tools()
    try:
        literal = "one\\ntwo"
        await _call(write, {"path": "notes.txt", "content": literal})
        assert not write.is_error
        await _call(edit, {"path": "notes.txt", "old_string": literal, "new_string": "three\\tfour"})
        assert not edit.is_error
        output = await _call(read, {"path": "notes.txt"})
        assert "three\\tfour" in output and "three\tfour" not in output
    finally:
        await target.clear_state()
