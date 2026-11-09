"""Failed training calls retain sampled history and never perturb or replay it."""

from dataclasses import dataclass
from typing import Any

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.manager


@dataclass
class FailingParser(ullm.backends.stub.StubBackend):
    """Return a sampled response and then fail during final parsing."""

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Retain real producer output before raising its parse failure."""
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        raise ullm.backend.MalformedOutputError("parser failed after a sampled draw")


async def test_failed_training_backend_never_pops_or_replays_a_draw() -> None:
    backend = FailingParser(
        responses=[ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="<tool_call>{"))]
    )
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="puzzle"))
    conversation = ullm.conv.Conversation(elements=[prompt])
    params = ullm.backend.GenericSamplingParameters(temperature=1, top_p=1, seed=81)
    response = ullm.manager.ResponseManager(
        backend=backend,
        conversation=conversation,
        sampling_params=params,
        max_retries=0,
        perturb_retry_prompt=False,
        preserve_failed_draws=True,
    )
    manager = ullm.manager.TurnManager(response_manager=response, conversation=conversation)
    with pytest.raises(ullm.backend.MalformedOutputError, match="parser failed"):
        await manager.run()
    assert backend._call_index == 1
    assert params.seed == 81
    assert prompt.content.to_string() == "puzzle"
    assert conversation.last_turn.last_response.final_content.to_string() == "<tool_call>{"
    assert len(conversation.last_turn.elements) == 1
