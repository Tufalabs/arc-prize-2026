"""The 200-with-error body that PARSES — the shape that cost five game tails.

Its sibling in ``test_oai_transient_retry.py`` covers the case where the SDK
rejects OpenRouter's error body and raises ``pydantic.ValidationError``. The
same upstream condition can instead parse into a completion whose ``choices``
is None or empty. That used to sail past the retry and surface later as
``TypeError: 'NoneType' object is not subscriptable`` from ``_process_output``,
which runs outside the retry wrapper — killing the run. On 2026-07-30 it ended
5 of 26 games early, one of them at 3/4 levels.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import pytest
from ullm.backends import oai


def _completion(choices: Any, error: Any = None) -> Any:
    """A parsed response object with whatever ``choices`` the provider sent."""
    return types.SimpleNamespace(choices=choices, model_extra={"error": error} if error else {})


# --- the guard itself ---------------------------------------------------------


def test_a_normal_completion_passes_through_unchanged() -> None:
    good = _completion([types.SimpleNamespace(message="hi")])
    assert oai._require_choices(good) is good


@pytest.mark.parametrize("choices", [None, []])
def test_choiceless_bodies_raise_the_transient_error(choices: Any) -> None:
    with pytest.raises(oai.UpstreamErrorBodyError):
        oai._require_choices(_completion(choices))


def test_the_providers_error_is_surfaced_in_the_message() -> None:
    """Rate-limit diagnosis should not require re-running with a debugger."""
    body = {"code": 429, "message": "rate limited", "metadata": {"limit_source": "upstream_provider_shared_pool"}}
    with pytest.raises(oai.UpstreamErrorBodyError, match="upstream_provider_shared_pool"):
        oai._require_choices(_completion(None, error=body))


def test_it_is_registered_as_transient() -> None:
    """The guard is pointless if ``_generate`` does not recognise what it raises."""
    assert oai.UpstreamErrorBodyError in oai._TRANSIENT_ERRORS


def test_plain_typeerror_is_still_fatal() -> None:
    """Guard against the lazy fix: blanket-retrying TypeError would mask real
    bugs anywhere in the generate path."""
    assert TypeError not in oai._TRANSIENT_ERRORS


# --- end to end through the real ``_generate`` --------------------------------


@dataclass
class _FakeBackend:
    """Drives the real ``_generate`` with scripted per-attempt responses.

    Not an ``OpenAIServer`` subclass, for the reason given in
    ``test_oai_transient_retry``: constructing one builds a live client per
    instance. ``_generate`` only needs the retry bounds,
    ``_generate_once``, and the open-request count its retry lines report.
    """

    outcomes: list[Any] = field(default_factory=lambda: list[Any]())
    max_transient_retries: int = 3
    transient_retry_window_s: float | None = None
    attempts: int = 0
    _open_requests: int = 0

    async def _generate_once(self, sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        outcome = self.outcomes[self.attempts]
        self.attempts += 1
        # Mirrors the real path: the guard runs before anything is yielded.
        yield oai._require_choices(outcome)

    _generate = oai.OpenAIServer._generate
    _generate_with_transient_retry = oai.OpenAIServer._generate_with_transient_retry
    _retry_context = oai.OpenAIServer._retry_context
    _out_of_retries = oai.OpenAIServer._out_of_retries
    _attempt_label = oai.OpenAIServer._attempt_label


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the backoff. Patches this module's ``_retry_after_seconds``,
    never ``asyncio.sleep`` — that is process-wide and breaks unrelated tests."""

    def _zero(error: BaseException, attempt: int = 0) -> float:
        """``_retry_after_seconds``'s signature, without the waiting."""
        return 0.0

    monkeypatch.setattr(oai, "_retry_after_seconds", _zero)


async def _drain(backend: _FakeBackend) -> list[Any]:
    """Run the real ``_generate`` to completion; params are never read here."""
    params: Any = None
    return [item async for item in backend._generate(params, None)]


async def test_a_choiceless_body_is_retried_and_recovers(no_sleep: None) -> None:
    """The fix, in one test: what killed five games is now one retry."""
    good = _completion([types.SimpleNamespace(message="hi")])
    backend = _FakeBackend(outcomes=[_completion(None), good])
    assert await _drain(backend) == [good]
    assert backend.attempts == 2


async def test_repeated_choiceless_bodies_still_fail_loudly(no_sleep: None) -> None:
    """Retrying must not become hanging: a persistently broken upstream has to
    surface rather than be swallowed."""
    backend = _FakeBackend(outcomes=[_completion(None)] * 3, max_transient_retries=2)
    with pytest.raises(oai.UpstreamErrorBodyError):
        await _drain(backend)
    assert backend.attempts == 3


async def test_an_empty_choices_list_takes_the_same_path(no_sleep: None) -> None:
    good = _completion([types.SimpleNamespace(message="hi")])
    backend = _FakeBackend(outcomes=[_completion([]), good])
    assert await _drain(backend) == [good]
    assert backend.attempts == 2
