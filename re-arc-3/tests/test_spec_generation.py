from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request

from pipeline.spec_generation import generate_structured_response


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_generate_structured_response_background_polls_until_completed(monkeypatch):
    requests: list[Request] = []
    payloads = [
        {"id": "resp_123", "status": "queued"},
        {"id": "resp_123", "status": "in_progress"},
        {"id": "resp_123", "status": "completed", "output_text": "# Spec\n"},
    ]
    recorded_response_ids: list[str] = []

    def fake_urlopen(request: Request, timeout: int):
        del timeout
        requests.append(request)
        return _FakeResponse(payloads.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    output = generate_structured_response(
        api_key="test-key",
        model="gpt-5.5-pro",
        prompt="Write a spec.",
        reasoning_effort="xhigh",
        background=True,
        poll_interval_seconds=0.01,
        response_id_callback=recorded_response_ids.append,
    )

    assert output == "# Spec"
    assert recorded_response_ids == ["resp_123"]
    assert [request.get_method() for request in requests] == ["POST", "GET", "GET"]
    assert requests[1].full_url == "https://api.openai.com/v1/responses/resp_123"
    create_payload = json.loads((requests[0].data or b"{}").decode("utf-8"))
    assert create_payload["background"] is True
    assert create_payload["store"] is True
    assert create_payload["reasoning"] == {"effort": "xhigh"}


def test_generate_structured_response_retries_background_poll_without_new_job(monkeypatch):
    requests: list[Request] = []
    payloads: list[dict[str, Any] | Exception] = [
        {"id": "resp_123", "status": "queued"},
        HTTPError(
            "https://api.openai.com/v1/responses/resp_123", 524, "timeout", hdrs={}, fp=BytesIO(b"error code: 524")
        ),
        {"id": "resp_123", "status": "completed", "output_text": "# Spec\n"},
    ]

    def fake_urlopen(request: Request, timeout: int):
        del timeout
        requests.append(request)
        payload = payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    output = generate_structured_response(
        api_key="test-key",
        model="gpt-5.5-pro",
        prompt="Write a spec.",
        reasoning_effort="xhigh",
        background=True,
        poll_interval_seconds=0.01,
    )

    assert output == "# Spec"
    assert [request.get_method() for request in requests] == ["POST", "GET", "GET"]
