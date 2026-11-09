from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable


class OpenAIError(RuntimeError):
    pass


def _extract_response_text(payload: dict[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if str(block.get("type") or "") != "output_text":
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    return ""


def _request_json(
    *, api_key: str, method: str, url: str, body: dict[str, object] | None = None, timeout_seconds: int
) -> dict[str, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw_payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OpenAIError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise OpenAIError(f"OpenAI API timeout after {timeout_seconds}s: {exc}") from exc
    except urllib.error.URLError as exc:
        raise OpenAIError(f"OpenAI API network error: {exc}") from exc

    if not isinstance(raw_payload, dict):
        raise OpenAIError("OpenAI response payload was not a JSON object.")
    return {str(key): value for key, value in raw_payload.items()}


def _is_retryable_openai_error(exc: OpenAIError) -> bool:
    message = str(exc)
    return (
        "OpenAI API HTTP 502:" in message
        or "OpenAI API HTTP 503:" in message
        or "OpenAI API HTTP 504:" in message
        or "OpenAI API HTTP 524:" in message
        or "OpenAI API timeout" in message
        or "OpenAI API network error" in message
    )


def _terminal_error(payload: dict[str, object]) -> OpenAIError:
    status = str(payload.get("status") or "unknown")
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code")
        if message:
            return OpenAIError(f"OpenAI background response ended with status {status}: {message}")
        if code:
            return OpenAIError(f"OpenAI background response ended with status {status}: {code}")
    incomplete_details = payload.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        reason = incomplete_details.get("reason")
        if reason:
            return OpenAIError(f"OpenAI background response ended with status {status}: {reason}")
    return OpenAIError(f"OpenAI background response ended with status {status}.")


def _create_response(
    *, api_key: str, model: str, prompt: str, reasoning_effort: str, background: bool, timeout_seconds: int
) -> dict[str, object]:
    body: dict[str, object] = {"model": model, "input": prompt, "reasoning": {"effort": reasoning_effort}}
    if background:
        body.update({"background": True, "store": True})
    return _request_json(
        api_key=api_key,
        method="POST",
        url="https://api.openai.com/v1/responses",
        body=body,
        timeout_seconds=timeout_seconds,
    )


def _retrieve_response(*, api_key: str, response_id: str, timeout_seconds: int) -> dict[str, object]:
    return _request_json(
        api_key=api_key,
        method="GET",
        url=f"https://api.openai.com/v1/responses/{response_id}",
        timeout_seconds=timeout_seconds,
    )


def _poll_background_response(
    *, api_key: str, response_id: str, timeout_seconds: int, poll_interval_seconds: float
) -> dict[str, object]:
    while True:
        payload = _retrieve_response(api_key=api_key, response_id=response_id, timeout_seconds=timeout_seconds)
        status = str(payload.get("status") or "")
        if status == "completed":
            return payload
        if status not in {"queued", "in_progress"}:
            raise _terminal_error(payload)
        time.sleep(poll_interval_seconds)


def generate_structured_response(
    *,
    api_key: str,
    model: str,
    prompt: str,
    reasoning_effort: str = "high",
    timeout_seconds: int = 3600,
    background: bool = False,
    poll_interval_seconds: float = 10.0,
    resume_response_id: str | None = None,
    response_id_callback: Callable[[str], None] | None = None,
) -> str:
    payload: dict[str, object] | None = None
    last_error: OpenAIError | None = None
    active_response_id = resume_response_id
    while True:
        try:
            if background:
                response_id = active_response_id
                if response_id is None:
                    created_payload = _create_response(
                        api_key=api_key,
                        model=model,
                        prompt=prompt,
                        reasoning_effort=reasoning_effort,
                        background=True,
                        timeout_seconds=timeout_seconds,
                    )
                    raw_response_id = created_payload.get("id")
                    if not isinstance(raw_response_id, str) or not raw_response_id.strip():
                        raise OpenAIError("OpenAI background response did not include a response id.")
                    response_id = raw_response_id.strip()
                    active_response_id = response_id
                    if response_id_callback is not None:
                        response_id_callback(response_id)
                    if str(created_payload.get("status") or "") == "completed":
                        payload = created_payload
                    else:
                        payload = _poll_background_response(
                            api_key=api_key,
                            response_id=response_id,
                            timeout_seconds=timeout_seconds,
                            poll_interval_seconds=poll_interval_seconds,
                        )
                else:
                    payload = _poll_background_response(
                        api_key=api_key,
                        response_id=response_id,
                        timeout_seconds=timeout_seconds,
                        poll_interval_seconds=poll_interval_seconds,
                    )
            else:
                payload = _create_response(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    reasoning_effort=reasoning_effort,
                    background=False,
                    timeout_seconds=timeout_seconds,
                )
            break
        except OpenAIError as exc:
            last_error = exc
            if active_response_id is None or not _is_retryable_openai_error(exc):
                raise
            time.sleep(poll_interval_seconds)

    if payload is None:
        raise last_error or OpenAIError("OpenAI response generation failed.")

    output_text = _extract_response_text(payload)
    if not output_text:
        raise OpenAIError("OpenAI response did not contain spec text output.")
    return output_text.rstrip()
