"""The backend x test matrix (see tests/conftest.py for columns/gating).

Rows run against every column whose capabilities support them. Scripted
columns get a per-test script mirroring what a real model would do, so
one row exercises the same ullm code path everywhere; real-LLM assertions
are semantic and deliberately loose.
"""

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, AsyncGenerator

import pytest
import ullm.backend
import ullm.backend_matrix
import ullm.backends.stub
import ullm.conv
import ullm.manager
import ullm.tool
import ullm.tools.python
import ullm.utils
from ullm.backend_matrix import BackendColumn

# Matrix tests share the session event loop: the GPU columns' engines bind
# background tasks to the loop they start on.
pytestmark = pytest.mark.asyncio(loop_scope="session")


def _make_tm(
    backend: ullm.backend.Backend, *, tools: list[ullm.tool.Tool] | None = None, max_tokens: int = 2048
) -> ullm.manager.TurnManager:
    return ullm.utils.make_turn_manager(backend, tools=tools, reasoning_effort="low", max_tokens=max_tokens)


async def _run_prompt(tm: ullm.manager.TurnManager, text: str) -> str:
    tm.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text)))
    if tm.state != "ready":
        tm.rearm()
    await tm.run()
    return tm.conversation.last_turn.last_response.final_content.to_string()


def _tool_call_response(tool_name: str, arguments: dict[str, Any]) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name=tool_name, arguments=arguments))


async def test_two_turn_reasoning_and_live_pickle(backend_column: BackendColumn):
    """Row lives in ullm.backend_matrix so a live run can call it too."""
    try:
        await ullm.backend_matrix.two_turn_reasoning_and_live_pickle(backend_column)
    except ullm.backend_matrix.RowNotSupported as exc:
        pytest.skip(str(exc))


async def test_tool_call(backend_column: BackendColumn):
    """Row lives in ullm.backend_matrix so a live run can call it too."""
    try:
        await ullm.backend_matrix.tool_call(backend_column)
    except ullm.backend_matrix.RowNotSupported as exc:
        pytest.skip(str(exc))


async def test_python_tool(backend_column: BackendColumn, kernel_pool: ullm.tools.python.KernelPool):
    """Row lives in ullm.backend_matrix so a live run can call it too."""
    try:
        await ullm.backend_matrix.python_tool(backend_column, kernel_pool)
    except ullm.backend_matrix.RowNotSupported as exc:
        pytest.skip(str(exc))


async def test_per_turn_cap_exhausted(backend_column: BackendColumn):
    """Row lives in ullm.backend_matrix so a live run can call it too."""
    try:
        await ullm.backend_matrix.per_turn_cap_exhausted(backend_column)
    except ullm.backend_matrix.RowNotSupported as exc:
        pytest.skip(str(exc))


async def test_response_injection(backend_column: BackendColumn):
    """U1/U5: injected text is served as prompt ids after the generation
    prompt — the answer starts with it, it is billed as input, the witness
    carries it, and the next turn continues from the served ids. A column
    that cannot serve it refuses the call before appending anything."""
    col = backend_column
    text = "The fruit I choose is"
    backend = col.make_backend(["an apple.", "Yes."])
    tm = _make_tm(backend, max_tokens=64)
    tm.response_manager.sampling_params.response_injection = text
    if not col.response_injection:
        with pytest.raises(ValueError, match="response_injection"):
            await _run_prompt(tm, "Name a fruit in one short sentence.")
        assert tm.conversation.last_turn.elements == []
        return
    plain = _make_tm(backend, max_tokens=64)
    # Past reasoning goes back whole on both managers: a witness is kept only
    # under that setting (an answer whose reasoning was stripped cannot be
    # re-rendered as the prefix a continued call extends), and the prompt
    # comparison below needs the two calls configured alike.
    for manager in (plain, tm):
        manager.response_manager.sampling_params.send_back_reasoning = "all"
    await _run_prompt(plain, "Name a fruit in one short sentence.")
    answer = await _run_prompt(tm, "Name a fruit in one short sentence.")
    response1 = tm.conversation.last_turn.last_response
    # The reply opens with the text: in its reasoning where the template
    # leaves a think block open for the model to fill (the model reasons on
    # from the injected words), else in its answer.
    opening = response1.reasoning_content.to_string() or answer
    assert opening.startswith(text) and response1.response_injection == text
    usage_plain = plain.conversation.last_turn.last_response.token_usage
    assert usage_plain is not None and response1.token_usage is not None
    # The served text is billed as input wherever the backend can count it;
    # a backend that cannot tokenise raw text says so rather than estimating.
    injected_tokens = await backend.text_token_count(text)
    if injected_tokens is not None:
        assert response1.token_usage.prompt_tokens == usage_plain.prompt_tokens + injected_tokens
    if not backend.supports_token_continuation():
        return  # no witness to carry the text, and no continued call to check
    assert response1._served is not None and response1._served.injected == text
    tm.response_manager.sampling_params.response_injection = None
    await _run_prompt(tm, "Is that fruit edible? Answer yes or no.")
    response2 = tm.conversation.last_turn.last_response
    assert response2._served is not None and response2._served.continued, "turn 2 rendered afresh"


#: The answer the answer-sampling row asks for: a tool call, as a duck's move
#: is, whose argument is free text — a call the question pins down entirely
#: (``add(17, 25)``) comes out the same even at the hottest temperature vLLM
#: takes, so it could not show which settings sampled it.
_NOTE_TOOL = ullm.tool.Tool(
    name="note",
    description="Leave the user a short note.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
)
_NOTE_PROMPT = "What is 17 + 25? Work it out, then call the note tool with a one-sentence remark about the result."
#: Room for the reasoning to close and the call after it.
_ANSWER_ROW_CAP = 3072
#: What the duck serves after the generation prompt to have a summary written
#: without thinking: the think block closed in the prompt.
_THINK_CLOSE = "\n</think>\n\n"


async def _note_reply(
    backend: ullm.backend.Backend, params: ullm.backend.GenericSamplingParameters
) -> ullm.conv.LLMResponse:
    """One call, one reply: the reasoning, then the tool call it settles on."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(
            content=ullm.conv.StringContent(text="You are a careful assistant. Use the tools you are given."),
            tools=[_NOTE_TOOL],
        ),
        elements=[
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=_NOTE_PROMPT)),
            ullm.conv.ResponseTurn(elements=[], status="intermediate"),
        ],
        status="intermediate",
    )
    async for _ in backend.respond(params, conversation):
        pass
    return conversation.last_turn.last_response


def _read(response: ullm.conv.LLMResponse) -> tuple[str, str]:
    """A reply as the caller receives it: its reasoning, and its answer."""
    return response.reasoning_content.to_string(), response.final_content.to_string()


async def test_answer_sampling(backend_column: BackendColumn):
    """U1: the reasoning is sampled with the call's own settings, the answer —
    what the model writes once its reasoning closes, here a tool call — with
    ``after_reasoning``'s. Greedy decoding makes both readable: a greedy
    reasoning must come out as the control's whatever the answer is sampled
    with, and a greedy answer as the control's however the reply is cut. A
    column that cannot serve it refuses the call before appending anything."""
    col = backend_column
    backend = col.make_backend(["unused"])
    greedy = ullm.backend.GenericSamplingParameters(temperature=0.0, top_p=1.0, max_tokens=_ANSWER_ROW_CAP)
    hot = ullm.backend.AnswerSampling(temperature=2.0, extra_params={"top_k": -1})
    if not col.after_reasoning:
        conversation = ullm.conv.Conversation(
            elements=[
                ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="Say hi.")),
                ullm.conv.ResponseTurn(elements=[], status="intermediate"),
            ],
            status="intermediate",
        )
        with pytest.raises(ValueError, match="after_reasoning"):
            async for _ in backend.respond(replace(greedy, after_reasoning=hot), conversation):
                pass
        assert conversation.last_turn.elements == []
        return
    # Greedy twice: the first warms the prefix cache, so the second is served
    # as every call after it is. They must agree, or greedy decoding does not
    # repeat on this engine and nothing below can be read.
    first = await _note_reply(backend, greedy)
    control = await _note_reply(backend, greedy)
    assert _read(first) == _read(control), "greedy decoding did not repeat: the row cannot judge the answer's sampling"
    assert control.final_content.to_string(), "the control's reasoning did not close within the cap"
    # Cut, greedy on both sides: the answer's top_k differs, so the reply is
    # served as two requests, but greedy picks the same tokens under it. The
    # cut alone must not change the reply, nor what it bills.
    cut = await _note_reply(
        backend,
        replace(greedy, after_reasoning=ullm.backend.AnswerSampling(temperature=0.0, extra_params={"top_k": 1})),
    )
    assert _read(cut) == _read(control), "cutting the reply in two changed it"
    assert cut.token_usage is not None and control.token_usage is not None
    assert (cut.token_usage.prompt_tokens, cut.token_usage.completion_tokens) == (
        control.token_usage.prompt_tokens,
        control.token_usage.completion_tokens,
    )
    # A hot answer: the reasoning is still the control's, the answer is not. A
    # hot answer may be nothing at all, which stays the reply here.
    split = await _note_reply(backend, replace(greedy, seed=7, keep_empty_reply=True, after_reasoning=hot))
    assert split.reasoning_content.to_string() == control.reasoning_content.to_string()
    assert split.final_content.to_string() != control.final_content.to_string()
    # A prompt that has closed the reasoning, as the duck closes it for a
    # summary: all of the reply is answer. The call hot and the answer greedy,
    # it must be the greedy reply of the same prompt.
    closed = replace(greedy, response_injection=_THINK_CLOSE)
    await _note_reply(backend, closed)
    plain = await _note_reply(backend, closed)
    inverted = await _note_reply(
        backend,
        replace(
            closed,
            temperature=2.0,
            seed=7,
            extra_params={"top_k": -1},
            after_reasoning=ullm.backend.AnswerSampling(temperature=0.0),
        ),
    )
    assert _read(inverted) == _read(plain), "a closed prompt's reply was not sampled with the answer's settings"


async def test_prefix_cache(backend_column: BackendColumn):
    """Row lives in ullm.backend_matrix so a live run can call it too."""
    try:
        await ullm.backend_matrix.prefix_cache(backend_column)
    except ullm.backend_matrix.RowNotSupported as exc:
        pytest.skip(str(exc))


async def test_image_input(backend_column: BackendColumn):
    """Row lives in ullm.backend_matrix so a live run can call it too."""
    try:
        await ullm.backend_matrix.image_input(backend_column)
    except ullm.backend_matrix.RowNotSupported as exc:
        pytest.skip(str(exc))


# --- Cheap-tier-only rows (fault injection / scripted edge cases) ---


async def test_replayed_history(backend_column: BackendColumn):
    """Row lives in ullm.backend_matrix so a live run can call it too."""
    try:
        await ullm.backend_matrix.replayed_history(backend_column)
    except ullm.backend_matrix.RowNotSupported as exc:
        pytest.skip(str(exc))


async def test_bad_json_tool_arguments_fallback():
    """T14: backends wrap unparseable tool arguments as {'_raw': text};
    the Tool layer unwraps them and the call still runs."""
    backend = ullm.backends.stub.StubBackend.from_script(
        _tool_call_response("dummy_compute", {"_raw": '{"x": 2, "y": 3}'}),
        "done",
    )
    tm = _make_tm(backend, tools=[ullm.tool.DummyTool()])
    await _run_prompt(tm, "irrelevant")
    results = [e for e in tm.conversation.last_turn.elements if isinstance(e, ullm.conv.ToolResult)]
    assert results and not results[0].is_error
    assert "12" in results[0].content.to_string()


async def test_prompt_too_long_yields_synthetic_response():
    """T8 (cheap form): PromptTooLongError becomes a graceful synthetic
    response — callers see stop_reason='max_tokens' and a flagged
    placeholder, never an exception. Uses DummyBackend's 1M-char limit."""
    backend = ullm.backend.DummyBackend()
    tm = _make_tm(backend)
    answer = await _run_prompt(tm, "x" * 1_100_000)
    response = tm.conversation.last_turn.last_response
    assert "too long" in answer
    assert response.stop_reason == "max_tokens"
    assert response.do_not_send_to_agent  # never sent back to the model
    assert any(w["type"] == "prompt_too_long" for w in backend.warning_log)
    # The backend stays usable (a fresh conversation succeeds; the old one
    # is over-limit forever since the huge prompt remains part of it).
    tm2 = _make_tm(backend)
    answer2 = await _run_prompt(tm2, "hello")
    assert answer2 and not tm2.conversation.last_turn.last_response.do_not_send_to_agent


@dataclass
class _HangingBackend(ullm.backend.DummyBackend):
    """DummyBackend whose generation never completes on its own."""

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        yield "partial output before the hang"
        await asyncio.sleep(600)

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        raise AssertionError("generation was supposed to be cancelled before processing")


async def test_cancellation_lands_mid_generate():
    """T13 (I5's foundation): asyncio cancellation of an in-flight turn
    lands at the generate await, propagates out as CancelledError, and
    leaves the backend with no in-flight request (the generator is closed
    on the way out)."""
    backend = _HangingBackend()
    tm = _make_tm(backend)
    tm.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="hi")))
    task = asyncio.create_task(tm.run())
    # Let the turn reach the hanging generate before cancelling.
    for _ in range(100):
        if backend._in_flight:
            break
        await asyncio.sleep(0.01)
    assert backend._in_flight == 1, "turn never reached the backend"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend._in_flight == 0  # generator closed, nothing orphaned
    assert tm.state == "cancelled"


@dataclass
class _FlakyBackend(ullm.backend.DummyBackend):
    """DummyBackend whose first N outputs fail to parse."""

    failures_left: int = 2
    seen_prompts: list[str] = field(default_factory=list)
    seen_transcripts: list[str] = field(default_factory=list)
    seen_tool_results: list[str] = field(default_factory=list)
    seen_seeds: list[int | None] = field(default_factory=list)

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        prompt = next(e for e in reversed(conversation.elements) if isinstance(e, ullm.conv.UserPrompt))
        self.seen_prompts.append(prompt.content.to_string())
        self.seen_transcripts.append(conversation.to_string())
        self.seen_seeds.append(getattr(sampling_params, "seed", None))
        self.seen_tool_results.append(
            next(
                e.content.to_string()
                for outer in reversed(conversation.elements)
                if isinstance(outer, ullm.conv.ResponseTurn)
                for e in reversed(outer.elements)
                if isinstance(e, ullm.conv.ToolResult)
            )
            if any(
                isinstance(e, ullm.conv.ToolResult)
                for outer in conversation.elements
                if isinstance(outer, ullm.conv.ResponseTurn)
                for e in outer.elements
            )
            else ""
        )
        if self.failures_left > 0:
            self.failures_left -= 1
            raise ullm.backend.MalformedOutputError("injected parse failure")
        super()._process_output(sampling_params, llm_output, conversation, turn, element)


async def test_malformed_output_retry_recovers():
    """T15: ResponseManager retries MalformedOutputError, records the
    recovery, and stamps the retry count on the surviving response."""
    backend = _FlakyBackend(failures_left=2)
    tm = _make_tm(backend)
    answer = await _run_prompt(tm, "hello")
    assert answer  # recovered after injected failures
    assert tm.conversation.last_turn.last_response.malformed_retries == 2
    warning_types = [w["type"] for w in backend.warning_log]
    assert warning_types.count("malformed_output_retry") == 2
    assert "malformed_output_recovered" in warning_types


async def test_malformed_retry_can_preserve_the_prompt_prefix():
    """Turning perturbation off leaves prior input byte-identical across retries."""
    backend = _FlakyBackend(failures_left=2)
    tm = _make_tm(backend)
    tm.response_manager.perturb_retry_prompt = False
    await _run_prompt(tm, "cache this exact prefix")
    assert backend.seen_prompts == ["cache this exact prefix"] * 3


async def _retry_after_multimodal_prompt(*, perturb: bool) -> tuple[_FlakyBackend, ullm.manager.TurnManager, str]:
    """Run a fault-injected retry after the text-plus-image request shape."""
    import numpy as np

    backend = _FlakyBackend(failures_left=2)
    tm = _make_tm(backend)
    retained = ullm.conv.ToolResult(
        call_id="prior-call",
        tool_name="python",
        content=ullm.conv.StringContent(text="retained tool result"),
        status="processed",
    )
    tm.conversation.elements.extend(
        [
            ullm.conv.UserPrompt(
                content=ullm.conv.StringContent(text="establish retained history"),
                status="processed",
            ),
            ullm.conv.ResponseTurn(
                elements=[
                    ullm.conv.LLMResponse(
                        final_content=ullm.conv.ToolCallContent(
                            call_id="prior-call",
                            tool_name="python",
                            arguments={"code": "print('state')"},
                        ),
                        status="processed",
                    ),
                    retained,
                ],
                status="processed",
            ),
        ]
    )
    assert isinstance(tm.response_manager.sampling_params, ullm.backend.GenericSamplingParameters)
    tm.response_manager.sampling_params.seed = 41
    tm.response_manager.perturb_retry_prompt = perturb
    tm.conversation.elements.append(
        ullm.conv.UserPrompt(
            content=ullm.conv.ListContent(
                elements=[
                    ullm.conv.StringContent(text="current state"),
                    ullm.conv.ImageContent(data=np.zeros((2, 2, 3), dtype=np.uint8)),
                ]
            )
        )
    )
    tm.rearm()
    await tm.run()
    return backend, tm, retained.content.to_string()


async def test_malformed_retry_preserves_retained_multimodal_history():
    """A recovered multimodal retry keeps its full causal request byte-identical.

    The generic perturbation policy cannot select a ListContent user prompt,
    so it reaches back to the prior StringContent tool result. The Duck policy
    must recover by seed variation alone rather than rewriting that retained
    block in the middle of the request.
    """
    stable, stable_tm, stable_result = await _retry_after_multimodal_prompt(perturb=False)
    perturbed, perturbed_tm, restored_result = await _retry_after_multimodal_prompt(perturb=True)

    assert stable.seen_transcripts[0] == stable.seen_transcripts[1] == stable.seen_transcripts[2]
    assert stable.seen_tool_results == ["retained tool result"] * 3
    assert stable.seen_seeds == [41, 42, 43]
    assert stable_result == "retained tool result"
    assert getattr(stable_tm.response_manager.sampling_params, "seed", None) == 41
    assert stable_tm.conversation.last_turn.last_response.malformed_retries == 2

    assert perturbed.seen_tool_results == [
        "retained tool result",
        "retained tool result\n",
        "retained tool result\n\n",
    ]
    assert len(set(perturbed.seen_transcripts)) == 3
    first_difference = next(
        i
        for i, (original, retry) in enumerate(
            zip(perturbed.seen_transcripts[0], perturbed.seen_transcripts[1], strict=False)
        )
        if original != retry
    )
    assert first_difference < perturbed.seen_transcripts[0].index("current state")
    assert restored_result == "retained tool result"
    assert perturbed_tm.conversation.last_turn.last_response.malformed_retries == 2
