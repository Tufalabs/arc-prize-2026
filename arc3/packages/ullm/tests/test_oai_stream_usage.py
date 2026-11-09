"""A streamed reply cut off at its cap keeps its usage, tool or no tool.

The server sends the usage in the chunk after the one that finishes the
choice. The SDK's stream helper raises at a ``length`` finish whenever a tool
is strict — ``OpenAIServer`` marks them all — before it reads that chunk, so
the backend reads the stream on past it: a truncated reply offered a tool is
still billed its prompt. A thinking model that spends its cap on reasoning
meets this on every such call, and a caller anchoring its context accounting
on the billed prompt needs that anchor.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import ullm.backend
import ullm.conv
import ullm.tool
from ullm.backends import oai

#: The usage chunk vLLM sent after a 16-token reply the cap cut off mid-reasoning.
USAGE = {"prompt_tokens": 280, "completion_tokens": 16, "total_tokens": 296}


def _chunks() -> list[dict[str, Any]]:
    """What vLLM streams for a reply the cap stops inside its reasoning: the
    role, reasoning deltas, the ``length`` finish, then the usage on its own."""
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    return [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {**base, "choices": [{"index": 0, "delta": {"reasoning": "The user asks"}}]},
        {**base, "choices": [{"index": 0, "delta": {"reasoning": " for ok."}, "finish_reason": "length"}]},
        {**base, "choices": [], "usage": USAGE},
    ]


class _StreamingServer(BaseHTTPRequestHandler):
    """Answers every chat request with ``_chunks`` as server-sent events."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        """Record the request, stream the reply."""
        type(self).requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in _chunks():
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet: the test reads the requests, not the access log."""


@pytest.fixture
def server() -> Iterator[str]:
    """The streaming server's base URL."""
    _StreamingServer.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.mark.parametrize("with_tool", [False, True], ids=["no-tool", "tool"])
async def test_a_truncated_streamed_reply_keeps_its_usage(server: str, with_tool: bool) -> None:
    # As ``InprocessServerFactory`` builds it for a vLLM server, streaming.
    backend = oai.OpenAIServer(base_url=server, model="m", streaming=True, reasoning_history_field="reasoning")
    tools = (
        [
            ullm.tool.Tool(
                name="lookup",
                description="Look a value up by key.",
                parameters={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
            )
        ]
        if with_tool
        else []
    )
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="You are terse."), tools=tools),
        elements=[
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="Reply with the single word: ok.")),
            ullm.conv.ResponseTurn(elements=[], status="intermediate"),
        ],
        status="intermediate",
    )
    try:
        async for _ in backend.respond(
            ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=16), conversation
        ):
            pass
    finally:
        await backend.close()
    assert ("tools" in _StreamingServer.requests[-1]) is with_tool
    response = conversation.last_turn.last_response
    assert response.token_usage is not None
    assert (response.token_usage.prompt_tokens, response.token_usage.completion_tokens) == (280, 16)
    assert response.stop_reason == "max_tokens"
    assert response.reasoning_content.to_string() == "The user asks for ok."
