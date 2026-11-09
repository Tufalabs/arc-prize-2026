"""Token accounting on the paths that do not finish.

A call cancelled mid-generation must land in ``Backend.total_usage``; a
malformed reply's discarded attempt must NOT, until the diagnostics span
stream records the rolled-back element too (requirements I1, roadmap).
"""

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import pytest
import ullm.backend
import ullm.conv

pytestmark = pytest.mark.asyncio


@dataclass
class _StreamingBackend(ullm.backend.Backend):
    """Emits chunks forever, counting one token per chunk, so a test can
    cancel it mid-stream. ``partial_usage`` off makes it a backend whose
    wire format cannot report usage until the end."""

    partial_usage: bool = True
    chunks: Optional[int] = None
    emitted: int = 0
    malformed: bool = False

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        return "prompt"

    async def _generate(self, sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        while self.chunks is None or self.emitted < self.chunks:
            self.emitted += 1
            yield self.emitted
            await asyncio.sleep(0)

    def _process_partial_output(self, sampling_params: Any, llm_output: Any, intermediate: Any) -> None:
        intermediate.final_content = ullm.conv.StringContent(text="x" * int(llm_output))

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
        if not self.malformed:  # left empty -> respond raises MalformedOutputError
            element.final_content = ullm.conv.StringContent(text="done")
        element.token_usage = ullm.backend.TokenUsage(
            prompt_tokens=10, completion_tokens=int(llm_output), num_requests=1
        )

    def _usage_from_partial(self, llm_output: Any) -> Optional[ullm.backend.TokenUsage]:
        if not self.partial_usage or llm_output is None:
            return None
        return ullm.backend.TokenUsage(prompt_tokens=10, completion_tokens=int(llm_output), num_requests=1)


def _conversation() -> ullm.conv.Conversation:
    """A one-prompt conversation with an open turn, ready for ``respond``."""
    conv = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="s"), tools=[]),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="u"))],
        status="intermediate",
    )
    conv.elements.append(ullm.conv.ResponseTurn(elements=[], status="intermediate"))
    return conv


async def _cancel_after(backend: _StreamingBackend, chunks: int) -> ullm.conv.Conversation:
    """Consume ``chunks`` streamed snapshots, then abandon the generator the
    way a cancelled game does."""
    conversation = _conversation()
    agen = backend.respond(ullm.backend.GenericSamplingParameters(), conversation)
    seen = 0
    async for _ in agen:
        seen += 1
        if seen >= chunks:
            break
    await agen.aclose()
    return conversation


async def test_cancelled_generation_still_counts_its_tokens() -> None:
    backend = _StreamingBackend()
    conversation = await _cancel_after(backend, chunks=5)
    # 5 chunks were generated before the caller walked away; without this the
    # total reads zero and a run under-reports every deadline-killed request.
    assert backend.total_usage.completion_tokens == 5
    assert backend.total_usage.prompt_tokens == 10
    response = conversation.last_turn.last_response
    assert response.token_usage == backend.total_usage
    assert response.final_content.to_string(), "the latest parsed partial should remain inspectable"


async def test_backend_without_partial_usage_counts_nothing() -> None:
    backend = _StreamingBackend(partial_usage=False)
    await _cancel_after(backend, chunks=5)
    # Documented asymmetry: a stream that reports usage only at the end has
    # nothing to report when it never gets there.
    assert backend.total_usage.completion_tokens == 0


async def test_completed_call_counts_once() -> None:
    backend = _StreamingBackend(chunks=3)
    async for _ in backend.respond(ullm.backend.GenericSamplingParameters(), _conversation()):
        pass
    # The finally must not double-count a call that already accounted itself.
    assert backend.total_usage.completion_tokens == 3
    assert backend.total_usage.num_requests == 1


async def test_malformed_attempt_is_not_counted() -> None:
    backend = _StreamingBackend(chunks=3, malformed=True)
    with pytest.raises(ullm.backend.MalformedOutputError):
        async for _ in backend.respond(ullm.backend.GenericSamplingParameters(), _conversation()):
            pass
    # Deliberately zero: ResponseManager rolls this element out of the turn,
    # so the span stream never sees it, and counting it here alone would make
    # the three-way reconciliation fail on an otherwise clean run.
    assert backend.total_usage.completion_tokens == 0
