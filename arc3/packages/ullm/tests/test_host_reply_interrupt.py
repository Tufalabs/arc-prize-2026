"""I9: interrupting a cell must leave the persistent kernel RPC stream framed."""

import base64
import json
import os
import pickle
import signal
from typing import Any

import pytest
import ullm.tools.python


async def test_sigint_during_large_host_reply_preserves_next_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = ullm.tools.python.PythonTool(
        timeout=10.0, initial_code="read_state = __host_call_trusted__", isolate_network=False
    )

    async def interrupted_reply(self: ullm.tools.python.PythonTool, call: dict[str, Any]) -> None:
        assert call["name"] == "get_state"
        assert self._process is not None and self._process.stdin is not None
        payload = base64.b64encode(pickle.dumps({"history": "x" * 2_000_000})).decode("ascii")
        frame = (json.dumps({"host_result": payload}) + "\n").encode()
        halfway = len(frame) // 2
        self._process.stdin.write(frame[:halfway])
        self._process.stdin.flush()
        os.kill(self._process.pid, signal.SIGINT)
        self._process.stdin.write(frame[halfway:])
        self._process.stdin.flush()

    monkeypatch.setattr(ullm.tools.python.PythonTool, "_handle_host_call", interrupted_reply)
    try:
        result = await tool._execute({"code": "carried = 42\nread_state('get_state')"})
        assert tool.is_error and "KeyboardInterrupt" in result.to_string()
        assert tool._process is not None
        pid = tool._process.pid
        result = await tool._execute({"code": "print(carried)"})
        assert not tool.is_error and result.to_string().strip() == "42", result.to_string()[:500]
        assert tool._process is not None and tool._process.pid == pid
    finally:
        await tool.clear_state()
