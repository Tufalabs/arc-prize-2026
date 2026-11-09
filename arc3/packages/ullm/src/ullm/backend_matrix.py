"""The backend x capability matrix, as callable checks.

These are the rows ``tests/test_ullm_backend_matrix.py`` runs; they live here
rather than in the test module so a live run can put its own started backend
through the same checks. Deployed jobs ship ``src`` and not ``tests``, so an
inline caller has nothing to import from there — and a run that wants this
wants it against the engine it is about to play on, not a second one stood up
by a fixture.

Each row takes a :class:`BackendColumn` — a backend factory plus what that
backend can do — and raises ``AssertionError`` on failure, exactly as it did
as a test. Rows a column cannot support raise :class:`RowNotSupported`, which
callers skip: pytest turns it into ``pytest.skip``, a live run counts it.

Real-LLM assertions are semantic and deliberately loose; scripted columns pass
a per-row script so one row exercises the same ullm code path everywhere.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import ullm.backend
import ullm.conv
import ullm.manager
import ullm.tool
import ullm.tools.python
import ullm.utils


class RowNotSupported(Exception):
    """This column cannot support this row; the caller skips rather than fails."""


@dataclass
class BackendColumn:
    """One column of the matrix: a backend plus its capabilities."""

    name: str
    make_backend: Callable[[Optional[list[Any]]], ullm.backend.Backend]
    real_llm: bool  # genuine model output: assertions are semantic, prompts matter
    tool_calls: bool
    reasoning: bool  # supports sending reasoning content back (send_back_reasoning input path)
    picklable: bool = True  # backend survives a pickle round-trip (False: direct-engine)
    prefix_cache: bool = False  # reports cached_tokens on a repeated long prefix
    images: bool = False  # accepts ImageContent in UserPrompts (multimodal input)
    response_injection: bool = False  # serves GenericSamplingParameters.response_injection
    after_reasoning: bool = False  # serves GenericSamplingParameters.after_reasoning
    # How far short of the previous prompt a call's cache hit may stop: the
    # provider's cache granularity. None: the column states none.
    cache_slack_tokens: Optional[int] = None


def live_column(
    backend: ullm.backend.Backend,
    *,
    name: str = "live",
    tool_calls: bool = True,
    reasoning: bool = True,
    prefix_cache: bool = True,
    images: bool = True,
    cache_slack_tokens: Optional[int] = None,
) -> BackendColumn:
    """A column over one already-started backend.

    ``make_backend`` hands the same instance back and ignores the script: a
    live engine has no script to follow. ``picklable`` is False because an
    in-process engine is not picklable and the two-turn row must not try —
    it falls back to continuing on the same object, which is what a real run
    does anyway.
    """
    return BackendColumn(
        name=name,
        make_backend=lambda _script: backend,
        real_llm=True,
        tool_calls=tool_calls,
        reasoning=reasoning,
        picklable=False,
        prefix_cache=prefix_cache,
        images=images,
        cache_slack_tokens=cache_slack_tokens,
    )


def make_tm(
    backend: ullm.backend.Backend,
    *,
    tools: list[ullm.tool.Tool] | None = None,
    max_tokens: int = 2048,
) -> ullm.manager.TurnManager:
    """A turn manager at low effort: the rows test plumbing, not reasoning depth."""
    return ullm.utils.make_turn_manager(backend, tools=tools, reasoning_effort="low", max_tokens=max_tokens)


async def run_prompt(tm: ullm.manager.TurnManager, text: str) -> str:
    """Append a user prompt, run the turn, return the final text."""
    tm.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text)))
    if tm.state != "ready":
        tm.rearm()
    await tm.run()
    return tm.conversation.last_turn.last_response.final_content.to_string()


def tool_call_response(tool_name: str, arguments: dict[str, Any]) -> ullm.conv.LLMResponse:
    """A scripted response that calls ``tool_name`` -- what a stub column replays."""
    return ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name=tool_name, arguments=arguments))


async def two_turn_reasoning_and_live_pickle(col: BackendColumn) -> None:
    """T2+T11+T9: a fact from turn 1 survives into turn 2 through a
    mid-conversation pickle of (backend, conversation); where supported,
    turn 2 is built with full reasoning passback enabled."""
    import ullm.backends.stub

    backend = col.make_backend(["Noted: the magic word is banana.", "banana"])
    backend2 = backend
    original_sbr = backend.send_back_reasoning
    try:
        if col.reasoning:
            backend.send_back_reasoning = "all"
        tm1 = make_tm(backend)
        await run_prompt(tm1, "Remember this: the magic word is banana. Acknowledge briefly.")

        if col.picklable:
            backend2, conversation = pickle.loads(pickle.dumps((backend, tm1.conversation)))
            if isinstance(backend, ullm.backends.stub.StubBackend):
                # The restored stub must resume mid-script, not restart it.
                backend2._call_index = backend._call_index
        else:
            backend2, conversation = backend, tm1.conversation

        tm2 = make_tm(backend2)
        tm2.conversation = conversation
        answer = await run_prompt(tm2, "What is the magic word? Answer with only the word.")

        assert len(conversation.elements) == 4  # user, turn, user, turn
        if col.real_llm:
            assert "banana" in answer.lower()
        else:
            assert answer  # flow-level check only; dummy/stub don't have memory semantics
    finally:
        backend.send_back_reasoning = original_sbr
        if backend2 is not backend:
            # The unpickled copy rebuilt its own live client; the session
            # fixture only closes the original.
            await backend2.close()


async def tool_call(col: BackendColumn) -> None:
    """T3: model calls a tool; the result reaches the final answer."""
    if not col.tool_calls:
        raise RowNotSupported(f"{col.name} cannot emit tool calls")
    backend = col.make_backend([tool_call_response("dummy_compute", {"x": 2, "y": 3}), "The result is 12."])
    tm = make_tm(backend, tools=[ullm.tool.DummyTool()])
    answer = await run_prompt(tm, "Call the dummy_compute tool with x=2 and y=3, then report the numeric result.")
    turn = tm.conversation.last_turn
    results = [e for e in turn.elements if isinstance(e, ullm.conv.ToolResult)]
    assert results and "12" in results[0].content.to_string()  # 3*2 + 2*3 + N(=0)
    assert "12" in answer


async def python_tool(col: BackendColumn, kernel_pool: ullm.tools.python.KernelPool) -> None:
    """T6: PythonTool end-to-end through the tool-call loop."""
    if not col.tool_calls:
        raise RowNotSupported(f"{col.name} cannot emit tool calls")
    backend = col.make_backend([tool_call_response("python", {"code": "print(7 * 191)"}), "1337"])
    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    try:
        tm = make_tm(backend, tools=[tool])
        answer = await run_prompt(tm, "Use the python tool to compute 7*191, then report just the number.")
        results = [e for e in tm.conversation.last_turn.elements if isinstance(e, ullm.conv.ToolResult)]
        assert results and "1337" in results[0].content.to_string()
        assert "1337" in answer
    finally:
        await tool.clear_state()


async def per_turn_cap_exhausted(col: BackendColumn) -> None:
    """T12: an exhausted per-turn completion cap yields the synthetic
    truncated placeholder instead of an LLM call."""
    backend = col.make_backend(["This response must never be produced."])
    tm = make_tm(backend, max_tokens=512)
    tm.max_generated_tokens_per_turn = 0
    await run_prompt(tm, "Write a very long story.")
    response = tm.conversation.last_turn.last_response
    assert response.truncated_by_per_turn_cap
    assert response.do_not_send_to_agent  # synthetic: nothing was sent to the model


async def prefix_cache(col: BackendColumn) -> None:
    """T4: a second turn re-sending a long identical prefix reports a
    nonzero prefix-cache hit in token usage — the underlying signal
    agent-side pacing and cost accounting rely on."""
    if not col.prefix_cache:
        raise RowNotSupported(f"{col.name} does not report prefix-cache hits")
    backend = col.make_backend(None)
    # ~4-5k tokens: above Anthropic's minimum cacheable prompt length, and
    # with real headroom under the 8192 test context (600 facts came out
    # ~7k tokens on the VL tokenizer and tripped the too-long path).
    filler = " ".join(f"Fact {i}: the sky is blue on day {i}." for i in range(400))
    tm = make_tm(backend, max_tokens=128)
    await run_prompt(tm, filler + "\nReply with just: OK")
    response1 = tm.conversation.last_turn.last_response
    assert not response1.do_not_send_to_agent, "turn 1 hit the too-long path; shrink the filler"
    usage1 = response1.token_usage
    await run_prompt(tm, "Reply with just: OK")
    usage2 = tm.conversation.last_turn.last_response.token_usage
    assert usage1 is not None and usage2 is not None
    assert usage2.cached_tokens > 0, f"no prefix-cache hit on turn 2; usage={usage2.to_string()}"
    assert usage2.cached_tokens <= usage2.prompt_tokens


async def image_input(col: BackendColumn) -> None:
    """T7: an image in the UserPrompt reaches the model (VL engine via the
    HF-processor path, PIL over the proxy socket, base64 through the OAI
    server / Anthropic blocks) and influences the answer."""
    import numpy as np

    if not col.images:
        raise RowNotSupported(f"{col.name} does not accept image input")
    backend = col.make_backend(["red"])
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # solid red
    prompt = ullm.conv.UserPrompt(
        content=ullm.conv.ListContent(
            elements=[
                ullm.conv.StringContent(text="What is the dominant color of this image? Answer with one word."),
                ullm.conv.ImageContent(data=image),
            ]
        )
    )
    tm = make_tm(backend)
    tm.conversation.elements.append(prompt)
    await tm.run()
    answer = tm.conversation.last_turn.last_response.final_content.to_string()
    if col.real_llm:
        assert "red" in answer.lower()
    else:
        assert answer  # wiring check: the image didn't break the pipeline


async def replayed_history(col: BackendColumn) -> None:
    """T15: ``continuation_and_cache_survival``'s counterpart for a backend
    that returns no token ids (a hosted API), by the provider's own counts:
    over four turns with past reasoning sent back, every call's prompt is the
    previous prompt plus the previous reply -- reasoning included, token for
    token -- plus the new message, and the cache holds the previous prompt,
    short of at most ``cache_slack_tokens``. A reply stripped of its
    reasoning, or replayed otherwise than it was generated, shows as a prompt
    that grew by other than the reply."""
    if not col.prefix_cache:
        raise RowNotSupported(f"{col.name} does not report prefix-cache hits")
    if not col.reasoning:
        raise RowNotSupported(f"{col.name} cannot send reasoning back")
    backend = col.make_backend(None)
    if backend.supports_token_continuation():
        raise RowNotSupported(f"{col.name} is checked by its served ids (continuation_and_cache_survival)")
    slack = col.cache_slack_tokens
    if slack is None:
        raise RowNotSupported(f"{col.name} states no cache granularity")
    # Turn 1 runs past the slack (~12 tokens a fact), so the cache bound holds something.
    facts = " ".join(f"Fact {i}: the sky is blue on day {i}." for i in range(slack // 12 + 60))
    questions = [
        f"{facts}\nRemember the code word banana. Think it over, then reply with just OK.",
        "Write a four-line poem about the code word, thinking it through first.",
        "Count the letters in each line of your poem and list the counts.",
        "Which line was longest? Answer briefly.",
    ]
    tm = make_tm(backend, max_tokens=8192)
    tm.response_manager.sampling_params.send_back_reasoning = "all"
    previous: Optional[ullm.backend.TokenUsage] = None
    reasoned = 0
    for index, question in enumerate(questions, start=1):
        await run_prompt(tm, question)
        response = tm.conversation.last_turn.last_response
        assert not response.do_not_send_to_agent, f"turn {index} hit a synthetic path; shrink the row"
        assert response.stop_reason != "max_tokens", f"turn {index} ran out of tokens: its replay is not a whole reply"
        usage = response.token_usage
        assert usage is not None and usage.prompt_tokens, f"turn {index} reported no usage"
        reasoned += bool(response.reasoning_content.to_string().strip())
        if previous is not None:
            # The new message costs no more tokens than it has characters,
            # plus its role framing.
            growth = usage.prompt_tokens - previous.prompt_tokens - previous.completion_tokens
            assert 0 <= growth <= len(question) + 32, (
                f"turn {index}: the prompt grew by {usage.prompt_tokens - previous.prompt_tokens} after a reply "
                f"of {previous.completion_tokens} and a {len(question)}-character message"
            )
            assert usage.cached_tokens >= previous.prompt_tokens - slack, (
                f"turn {index}: {usage.cached_tokens} cached of the previous prompt's {previous.prompt_tokens} "
                f"(slack {slack}); usage={usage.to_string()}"
            )
        previous = usage
    if col.real_llm:
        assert reasoned >= 2, f"only {reasoned} replies carried reasoning: the replay went largely unexercised"


async def reasoning_recall(col: BackendColumn) -> None:
    """T14: past reasoning reaches the model: a number picked in turn 1's
    reasoning and kept out of its answer is named in turn 2. The number
    exists only in the reasoning, so this checks passback by its effect, on
    any backend; ``continuation_and_cache_survival`` checks it by the served
    ids, which a hosted API does not return. Stripped, the model has nothing
    to name it from (Mistral Large 4 picked anew: 5926 for 7391)."""
    if not col.reasoning:
        raise RowNotSupported(f"{col.name} cannot send reasoning back")
    backend = col.make_backend(["OK", "4821"])
    tm = make_tm(backend, max_tokens=8192)
    tm.response_manager.sampling_params.send_back_reasoning = "all"
    await run_prompt(
        tm,
        "In your private reasoning only, pick a random four-digit number and memorise it. "
        "Do not write the number in your visible answer: reply with just OK.",
    )
    first = tm.conversation.last_turn.last_response
    answer = await run_prompt(tm, "What was the number you picked? Answer with just the number.")
    if not col.real_llm:
        assert answer  # flow-level check only: a script has no reasoning to recall
        return
    # The range's ends are reasoned about too, and a blind guess could land
    # on one.
    picked = set(re.findall(r"\b\d{4}\b", first.reasoning_content.to_string())) - {"1000", "9999"}
    if not picked:
        raise RowNotSupported(f"{col.name}'s reasoning named no four-digit number, so there is nothing to recall")
    shown = picked & set(re.findall(r"\b\d{4}\b", first.final_content.to_string()))
    assert not shown, f"turn 1's answer gave the number away: {sorted(shown)}"
    named = re.findall(r"\b\d{4}\b", answer)
    assert named and named[-1] in picked, f"turn 2 named {answer!r}; turn 1 reasoned about {sorted(picked)}"


#: The rows a live caller runs, in the order pytest reports them. ``python_tool``
#: is absent: it needs a KernelPool fixture, and standing a kernel pool up inside
#: a starting run buys less than it risks.
LIVE_ROWS: tuple[tuple[str, Any], ...] = (
    ("two_turn_reasoning_and_live_pickle", two_turn_reasoning_and_live_pickle),
    ("tool_call", tool_call),
    ("per_turn_cap_exhausted", per_turn_cap_exhausted),
    ("prefix_cache", prefix_cache),
    ("image_input", image_input),
    ("replayed_history", replayed_history),
    ("reasoning_recall", reasoning_recall),
)


async def run_live_matrix(backend: ullm.backend.Backend, **column_kwargs: Any) -> dict[str, str]:
    """Run the matrix against an already-started backend.

    Returns row name -> "ok" / "skipped: ...". Raises on the first row that
    fails, with the row named, because a caller running this at startup wants
    to stop rather than play a run out on a backend that cannot answer.
    """
    col = live_column(backend, **column_kwargs)
    results: dict[str, str] = {}
    for name, row in LIVE_ROWS:
        try:
            await row(col)
        except RowNotSupported as exc:
            results[name] = f"skipped: {exc}"
        except Exception as exc:
            raise RuntimeError(f"backend matrix row {name!r} failed on column {col.name!r}: {exc}") from exc
        else:
            results[name] = "ok"
    return results
