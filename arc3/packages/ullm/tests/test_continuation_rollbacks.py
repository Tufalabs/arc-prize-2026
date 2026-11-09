"""Rollbacks the manager performs on served history are declared rewrites
(requirements U5): a malformed retry restores the perturbed prompt text the
retried answer was sampled after, and a cap-lift removes a served round —
both bump ``Conversation.history_epoch`` so the next call renders afresh by
declaration instead of tripping exact history."""

from __future__ import annotations

from typing import Any

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.manager

from duck import DuckAgent


def _conversation() -> ullm.conv.Conversation:
    conversation = ullm.conv.Conversation(status="intermediate")
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="q")))
    conversation.elements.append(ullm.conv.ResponseTurn(status="intermediate"))
    return conversation


class _MalformedOnce(ullm.backends.stub.StubBackend):
    """A backend whose first answer fails to parse (the retry trigger)."""

    failed: bool = False

    def _process_output(self, *args: Any, **kwargs: Any) -> None:
        if not self.failed:
            self.__dict__["failed"] = True
            raise ullm.backend.MalformedOutputError("unparseable first answer")
        super()._process_output(*args, **kwargs)


@pytest.mark.asyncio
async def test_a_recovered_malformed_retry_declares_the_rewrite() -> None:
    backend = _MalformedOnce.from_script("bad", "ok")
    conversation = _conversation()
    manager = ullm.manager.ResponseManager(
        backend=backend,
        conversation=conversation,
        sampling_params=ullm.backend.GenericSamplingParameters(temperature=0.0),
    )
    await manager.run()
    answer = conversation.last_turn.elements[-1]
    assert isinstance(answer, ullm.conv.LLMResponse) and answer.malformed_retries == 1
    assert conversation.history_epoch == 1


@pytest.mark.asyncio
async def test_a_clean_call_declares_nothing() -> None:
    backend = ullm.backends.stub.StubBackend.from_script("ok")
    conversation = _conversation()
    manager = ullm.manager.ResponseManager(
        backend=backend,
        conversation=conversation,
        sampling_params=ullm.backend.GenericSamplingParameters(temperature=0.0),
    )
    await manager.run()
    assert conversation.history_epoch == 0


def test_the_duck_moves_past_a_conversation_that_is_ahead() -> None:
    """A manager-side declaration on the live conversation is not lost to
    the duck's own counter, which seeds the next block's Conversation."""
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    agent.history_epoch = 3
    live = ullm.conv.Conversation(history_epoch=5)  # the manager declared twice in this block
    agent._live_conversation = live
    agent._mark_history_rewritten()
    assert agent.history_epoch == 6 and live.history_epoch == 6
