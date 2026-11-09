"""``/tokenize`` route detection when the endpoint is not a vLLM server.

``_token_count`` is exact-or-nothing: it returns None when the endpoint has no
``/tokenize``, and the duck then measures with the billed-usage + character
upper-bound ruler (the chars/3 estimate this line once named is gone). Detecting
absence by status code alone is not enough. ``_server_root()`` strips ``/v1``,
so OpenRouter's base URL becomes ``https://openrouter.ai/api`` — a Next.js app
that answers the unknown ``/tokenize`` path with **200 and an HTML page**, not
404. That reached ``response.json()`` and raised JSONDecodeError out through
``token_count``, killing a run on its first context trim.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from ullm.backends import oai


class _Response:
    def __init__(self, status: int, body: str, content_type: str) -> None:
        self.status_code = status
        self._body = body
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status at {self.status_code}")

    def json(self) -> Any:
        return json.loads(self._body)  # raises exactly as httpx does on HTML


class _Client:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls = 0

    async def request(self, method: str, url: str, json: Any = None) -> _Response:  # noqa: A002
        self.calls += 1
        return self._response


def _server(response: _Response) -> tuple[oai.OpenAIServer, _Client]:
    """The instance is frozen, so the fake client is returned rather than
    stashed on it."""
    server = oai.OpenAIServer(base_url="https://openrouter.ai/api/v1", api_key="k", model="m", streaming=False)
    client = _Client(response)
    server._server_client = lambda: client  # type: ignore[method-assign]
    server._prepare_input = lambda *a, **k: {"messages": [{"role": "user", "content": "x"}]}  # type: ignore[method-assign]
    return server, client


HTML = "<!DOCTYPE html><html><body>Not the route you wanted</body></html>"

# Deliberately no sampling params: _token_count never reads them on these paths.
_NO_PARAMS: Any = None


async def test_a_200_html_page_means_the_route_is_absent() -> None:
    """The regression: OpenRouter answers 200 text/html and this used to die
    with JSONDecodeError instead of falling back to the estimate."""
    server, _client = _server(_Response(200, HTML, "text/html; charset=utf-8"))
    assert await server._token_count(_NO_PARAMS, _conversation()) is None
    assert server._server_tokenize is False


async def test_a_404_still_means_absent() -> None:
    server, _client = _server(_Response(404, HTML, "text/html"))
    assert await server._token_count(_NO_PARAMS, _conversation()) is None
    assert server._server_tokenize is False


async def test_json_without_a_count_is_absent_too() -> None:
    """A JSON 200 from something that is not vLLM."""
    server, _client = _server(_Response(200, '{"detail": "nope"}', "application/json"))
    assert await server._token_count(_NO_PARAMS, _conversation()) is None
    assert server._server_tokenize is False


async def test_a_real_vllm_answer_still_counts() -> None:
    server, _client = _server(_Response(200, '{"count": 1234}', "application/json"))
    assert await server._token_count(_NO_PARAMS, _conversation()) == 1234
    assert server._server_tokenize is True


async def test_absence_is_remembered_so_it_costs_one_request() -> None:
    server, _client = _server(_Response(200, HTML, "text/html"))
    assert await server._token_count(_NO_PARAMS, _conversation()) is None
    assert await server._token_count(_NO_PARAMS, _conversation()) is None
    assert _client.calls == 1


class _Conversation:
    last_turn = None


def _conversation() -> Any:
    return _Conversation()


@pytest.fixture(autouse=True)
def resolve_send_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin reasoning send-back to "none": these tests exercise /tokenize only."""

    def _none(self: oai.OpenAIServer, params: Any) -> str:
        """Fixed answer regardless of params."""
        return "none"

    monkeypatch.setattr(oai.OpenAIServer, "resolve_send_back_reasoning", _none)
