"""Tier 2/3 for exact token counting: does OUR rendering weigh what the
backend actually bills?

``token_count`` re-derives the prompt through ``_prepare_input`` and
tokenizes it. Everything about that is a claim that can only be checked
against a real tokenizer and a real billing number, so these rows run on the
GPU columns and compare ``token_count`` to the ``prompt_tokens`` the same
request reports.

The axes are the ones that historically diverge, each a way for the counted
prompt and the sent prompt to stop being the same prompt:

- **tools** — schemas are rendered INTO the prompt by the chat template, so
  a count that omits them is short by the whole tool block.
- **images** — the template emits a placeholder that expands to a
  model-specific token count; the text alone cannot reveal it.
- **reasoning passback** — whether past reasoning is re-sent changes the
  input, and it is resolved per call.

Tier 3 (cross-backend agreement) falls out of the same assertion: the
direct/proxy/server columns serve one model, so each column agreeing with
its OWN billing means they agree with each other. The server column is the
one that can fail alone — it counts with a LOCAL tokenizer against a prompt
the SERVER renders, so a chat-template or tool-schema mismatch shows up here
and nowhere else in the cheap tier.
"""

from typing import Any

import numpy as np
import pytest
import ullm.backend
import ullm.conv
import ullm.tool

# Matrix tests share the session event loop: the GPU columns' engines bind
# background tasks to the loop they start on.
pytestmark = pytest.mark.asyncio(loop_scope="session")


# Dedicated columns rather than rows on the shared matrix: these rows are
# about ONE claim (our rendering weighs what the backend bills), and the
# claim has exactly two shapes — an in-process engine we tokenize with
# directly, and an HTTP server we tokenize with a LOCAL copy of its
# tokenizer. The proxy adds no rendering of its own (it forwards to the
# wrapped backend), and its forwarding is already covered end-to-end in the
# duck's Tier-1 mutation tests, so a third column here would buy nothing and
# cost a GPU.
_TC_COLUMNS = [
    pytest.param("tcengine", id="tcengine", marks=pytest.mark.gpu),
    pytest.param("tchttp", id="tchttp", marks=pytest.mark.gpu),
]


@pytest.fixture(params=_TC_COLUMNS)
def counting_column(request: pytest.FixtureRequest) -> tuple[str, ullm.backend.Backend]:
    """(name, backend) for a column that can count exactly."""
    fixture_name = {"tcengine": "_vllm_direct_backend", "tchttp": "_vllm_server_backend"}[request.param]
    return request.param, request.getfixturevalue(fixture_name)


def _sampling(**kwargs: Any) -> ullm.backend.GenericSamplingParameters:
    return ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=16, **kwargs)


def _conversation(
    *,
    tools: list[ullm.tool.Tool] | None = None,
    image: np.ndarray | None = None,
    text: str = "Reply with the single word: ok.",
) -> ullm.conv.Conversation:
    elements: list[ullm.conv.Content] = [ullm.conv.StringContent(text=text)]
    if image is not None:
        elements.append(ullm.conv.ImageContent(data=image))
    content = ullm.conv.ListContent(elements=elements) if image is not None else elements[0]
    return ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="You are terse."), tools=tools or []),
        elements=[ullm.conv.UserPrompt(content=content)],
        status="intermediate",
    )


async def _count_then_send(
    backend: ullm.backend.Backend, conversation: ullm.conv.Conversation, sampling: ullm.backend.SamplingParameters
) -> tuple[int | None, int]:
    """Count the prompt, then send it, and return (counted, billed).

    Order matters: ``respond`` appends its output element to the turn, so a
    count taken afterwards would be describing a different conversation than
    the one that was sent."""
    turn = ullm.conv.ResponseTurn(elements=[], status="intermediate")
    conversation.elements.append(turn)
    counted = await backend.token_count(sampling, conversation)
    async for _ in backend.respond(sampling, conversation):
        pass
    response = conversation.last_turn.last_response
    assert response.token_usage is not None, "column reported no usage; cannot verify the count"
    # Close the turn so the conversation can take another user prompt (rows
    # that ask for a second exchange rely on this).
    turn.status = "processed"
    return counted, response.token_usage.prompt_tokens


def _require_exact(counted: int | None, column_name: str) -> int:
    """These columns are configured to count; None means the wiring that makes
    them exact (the server column's local tokenizer, above all) came undone,
    which is a failure rather than a reason to skip. Returns the count, so
    callers that go on to compare it are working with the exact number."""
    assert counted is not None, (
        f"{column_name} returned no exact count — the column is configured to tokenize, so this is "
        f"broken wiring, not an unsupported backend."
    )
    return counted


async def test_count_matches_billing_text_only(counting_column: tuple[str, ullm.backend.Backend]):
    name, backend = counting_column
    conversation = _conversation()
    counted, billed = await _count_then_send(backend, conversation, _sampling())
    counted = _require_exact(counted, name)
    assert counted == billed, (
        f"{name}: counted {counted} input tokens, billed {billed} — our rendering of the prompt "
        f"is not the one the backend tokenized."
    )


async def test_count_matches_billing_with_tools(counting_column: tuple[str, ullm.backend.Backend]):
    """Tool schemas render into the prompt, so omitting them from the count
    is short by the whole tool block — a large, silent error."""
    name, backend = counting_column
    tools = [
        ullm.tool.Tool(
            name="lookup",
            description="Look a value up by key.",
            parameters={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        )
    ]
    conversation = _conversation(tools=tools)
    counted, billed = await _count_then_send(backend, conversation, _sampling())
    counted = _require_exact(counted, name)
    assert counted == billed, (
        f"{name}: counted {counted} but billed {billed} with a tool in the system prompt — "
        f"the tool schema is rendered into the prompt on one path and not the other."
    )


async def test_count_matches_billing_with_an_image(counting_column: tuple[str, ullm.backend.Backend]):
    """The duck is multimodal by default, and image tokens are exactly the
    part the prompt TEXT cannot reveal: the template emits a placeholder that
    the processor expands to a model-specific count."""
    name, backend = counting_column
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    conversation = _conversation(image=image, text="Name the dominant color in one word.")
    counted, billed = await _count_then_send(backend, conversation, _sampling())
    counted = _require_exact(counted, name)
    assert counted == billed, (
        f"{name}: counted {counted} but billed {billed} with one image — the image placeholder "
        f"expansion is counted differently than it is sent."
    )


async def test_count_stays_exact_across_repeated_and_resized_images(
    counting_column: tuple[str, ullm.backend.Backend],
):
    """The in-process column caches the image placeholder expansion per
    image size (the duck re-counts the same-shape grid every move). What
    makes the cache legal is that the expansion is text-independent — so a
    cache-hit count must still match billing under a DIFFERENT text, and a
    different image size must earn its own delta rather than reuse a stale
    one. The texts vary in length between the two same-size rounds so a
    cached total (instead of a cached delta) fails here."""
    name, backend = counting_column
    small = np.zeros((64, 64, 3), dtype=np.uint8)
    large = np.zeros((128, 96, 3), dtype=np.uint8)
    prompts = [
        "Name the dominant color in one word.",
        "Name the dominant color of this differently sized image in one word.",
        "Once more, with entirely different words this time: what single color word best describes the image?",
    ]
    for image, text in zip([small, large, small], prompts, strict=True):
        conversation = _conversation(image=image, text=text)
        counted, billed = await _count_then_send(backend, conversation, _sampling())
        counted = _require_exact(counted, name)
        assert counted == billed, (
            f"{name}: counted {counted} but billed {billed} for image shape {image.shape} with "
            f"text {text!r} — the per-size expansion cache returned a number that does not "
            f"describe this prompt."
        )


@pytest.mark.parametrize("send_back_reasoning", ["none", "all"])
async def test_count_matches_billing_across_reasoning_passback(
    counting_column: tuple[str, ullm.backend.Backend], send_back_reasoning: str
):
    """Re-sending past reasoning changes the input size, and the policy is
    resolved per call. A count that read the backend default instead of the
    per-call value would be wrong by the whole reasoning block on exactly the
    calls the duck makes (it pins ``all``)."""
    name, backend = counting_column
    sampling = _sampling(send_back_reasoning=send_back_reasoning)
    conversation = _conversation(text="Think briefly, then reply with: ok.")
    # First exchange supplies the past turn whose reasoning may be re-sent.
    await _count_then_send(backend, conversation, sampling)
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="Again: ok.")))
    counted, billed = await _count_then_send(backend, conversation, sampling)
    counted = _require_exact(counted, name)
    assert counted == billed, (
        f"{name}: counted {counted} but billed {billed} with send_back_reasoning="
        f"{send_back_reasoning!r} — the count and the request disagree about what history is sent."
    )


async def test_count_tracks_the_conversation_growing(counting_column: tuple[str, ullm.backend.Backend]):
    """Eviction only works if the count MOVES with the prompt. A count that
    accidentally described a fixed prefix would satisfy a single-shot
    equality check while being useless for the thing it exists to drive."""
    name, backend = counting_column
    conversation = _conversation()
    first, _ = await _count_then_send(backend, conversation, _sampling())
    first = _require_exact(first, name)
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="Another question. " * 50)))
    second, billed = await _count_then_send(backend, conversation, _sampling())
    second = _require_exact(second, name)
    assert second > first, f"{name}: count did not grow with the conversation ({first} -> {second})"
    assert second == billed
