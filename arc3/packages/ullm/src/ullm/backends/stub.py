"""
Stub backend that replays a fixed, pre-scripted sequence of responses.

Where ``DummyBackend`` (backend.py) returns one hardcoded string forever, this
backend replays an ordered list of ``LLMResponse`` objects — the same
user-format type every backend produces — one per LLM call, ignoring the
conversation entirely: call N always yields ``responses[N]``. The intended use
is prototyping and exercising ``TurnManager`` / ``ResponseManager`` loops
(including multi-round tool-call flows) deterministically, without a live model.

The scripted unit is a plain ``LLMResponse`` (carrying ``final_content`` and/or
tool calls and optional ``reasoning_content``); ``stop_reason`` and
``token_usage`` are filled in by the backend when left unset, so authoring stays
light. A separate input->output mapping backend (for regression testing) is
planned; this one is the fixed-sequence variant.
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal, Optional

import ullm.backend
import ullm.conv


@dataclass
class StubBackend(ullm.backend.Backend):
    """Replays ``responses`` (``LLMResponse`` objects) in order, one per LLM
    call, ignoring the input.

    Entries may be plain strings (coerced to text-only ``LLMResponse``s) or
    ``LLMResponse`` instances; build a tool-call response with
    ``LLMResponse(final_content=ToolCallContent(tool_name=..., arguments=...))``
    (blank ``call_id``s are filled deterministically at emit time so they're
    unique even when looping). For each emitted response a missing
    ``stop_reason`` is derived (``"tool_use"`` iff it carries tool calls, else
    ``"stop"``) and a missing ``token_usage`` is estimated from sizes.

    ``on_exhausted`` (``"raise"`` default / ``"repeat_last"`` / ``"loop"``)
    governs behavior once the script runs out; :meth:`reset` rewinds. Like
    ``DummyBackend`` it is pickle-safe, so it can stand in for a real backend in
    subprocess-mode prototyping."""

    responses: list[ullm.conv.LLMResponse] = field(default_factory=lambda: list[ullm.conv.LLMResponse]())
    on_exhausted: Literal["raise", "repeat_last", "loop"] = "raise"
    _call_index: int = field(default=0, init=False, repr=False)  # Number of calls served so far

    def __post_init__(self) -> None:
        # Coerce bare strings to text responses before the instance freezes.
        self.responses = [self._coerce_response(r) for r in self.responses]
        super().__post_init__()

    @staticmethod
    def _coerce_response(r: Any) -> ullm.conv.LLMResponse:
        """Accept a raw string (-> text response) or an LLMResponse as-is."""
        if isinstance(r, ullm.conv.LLMResponse):
            return r
        if isinstance(r, str):
            return ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=r))
        raise TypeError(f"StubBackend.responses entries must be str or LLMResponse, got {type(r).__name__}")

    @classmethod
    def from_script(
        cls,
        *responses: "str | ullm.conv.LLMResponse",
        on_exhausted: Literal["raise", "repeat_last", "loop"] = "raise",
        **kwargs: Any,
    ) -> "StubBackend":
        """Ergonomic, type-checker-friendly constructor: pass responses as
        positional args (bare strings allowed). Extra ``Backend`` kwargs
        (``diagnostics``, ``send_back_reasoning``, ...) are forwarded."""
        return cls(responses=[cls._coerce_response(r) for r in responses], on_exhausted=on_exhausted, **kwargs)

    @classmethod
    def from_conversation(cls, conversation: ullm.conv.Conversation, **kwargs: Any) -> "StubBackend":
        """Build a stub that replays the assistant ``LLMResponse``s of a recorded
        conversation in order (synthetic ``do_not_send_to_agent`` placeholders
        skipped). Handy for record/replay prototyping."""
        responses = list[ullm.conv.LLMResponse]()
        for el in conversation.elements:
            if isinstance(el, ullm.conv.ResponseTurn):
                responses.extend(
                    copy.deepcopy(sub)
                    for sub in el.elements
                    if isinstance(sub, ullm.conv.LLMResponse) and not sub.do_not_send_to_agent
                )
            elif isinstance(el, ullm.conv.LLMResponse) and not el.do_not_send_to_agent:
                responses.append(copy.deepcopy(el))
        return cls(responses=responses, **kwargs)

    def _check_constraints(self) -> None:
        """Every scripted response must be emittable. Checked up front, because the
        base class would otherwise reject an empty one at run time as a malformed
        output and burn the whole retry budget on it.
        """
        assert len(self.responses) > 0, "StubBackend requires at least one scripted response"
        # Catch un-emittable responses up front rather than after the base
        # process_output triggers max_retries MalformedOutputErrors at run time.
        for i, r in enumerate(self.responses):
            empty = ullm.backend._is_empty_content(r.final_content)
            # The base backstop lets an empty reply through on exactly these
            # two stops: a budget cut, or a draw a trainer kept as malformed.
            assert not empty or r.stop_reason in ("max_tokens", "malformed"), (
                f"responses[{i}] has empty final_content; give it text/tool calls "
                "(or stop_reason='max_tokens' to simulate truncation, "
                "stop_reason='malformed' for a retained malformed draw)"
            )
        super()._check_constraints()

    def reset(self) -> None:
        """Rewind to the first scripted response (handy for re-running a loop)."""
        self._call_index = 0

    def supports_response_injection(self) -> bool:
        """Served, not continued: the script decides the answer whatever the
        prompt says, so injected text cannot steer this backend -- but it can
        still open the answer, which is the part a recording has to reproduce.
        Refusing would keep every scripted reference off the configuration the
        agent actually ships."""
        return True

    @staticmethod
    def _open_with(element: ullm.conv.LLMResponse, injection: str) -> None:
        """Put the served text where the answer starts, as an engine's template
        does: inside the reasoning when the response thinks, at the head of the
        text when it does not. A bare tool call has no text of its own to open,
        so the reasoning carries it."""
        if not ullm.backend._is_empty_content(element.reasoning_content):
            element.reasoning_content = ullm.conv.StringContent(text=injection + element.reasoning_content.to_string())
        elif isinstance(element.final_content, ullm.conv.StringContent):
            element.final_content = ullm.conv.StringContent(text=injection + element.final_content.text)
        else:
            element.reasoning_content = ullm.conv.StringContent(text=injection)

    def _select_response(self) -> ullm.conv.LLMResponse:
        """Pick the response for the current call index, applying the
        exhaustion policy once the index runs past the scripted list."""
        n = len(self.responses)
        assert n > 0, "StubBackend has no scripted responses"
        i = self._call_index
        if i < n:
            return self.responses[i]
        if self.on_exhausted == "loop":
            return self.responses[i % n]
        if self.on_exhausted == "repeat_last":
            return self.responses[-1]
        raise RuntimeError(
            f"StubBackend exhausted: {n} scripted response(s) but call #{i + 1} was requested. "
            "Add more responses or set on_exhausted='loop'/'repeat_last'."
        )

    def _fill_into(
        self,
        src: ullm.conv.LLMResponse,
        element: ullm.conv.LLMResponse,
        conversation: ullm.conv.Conversation,
        injection: Optional[str] = None,
    ) -> None:
        """Copy a scripted response's content into the live element, filling in
        blank tool-call ids, a missing stop_reason, and a missing token_usage.
        ``injection`` opens the answer and is charged as input, never as
        completion: it was served, not sampled."""
        element.reasoning_content = copy.deepcopy(src.reasoning_content)
        final_content = copy.deepcopy(src.final_content)
        # tool_calls() returns the live ToolCallContent nodes in the tree, so
        # assigning here mutates the deepcopy in place.
        tool_calls = final_content.tool_calls()
        for j, tc in enumerate(tool_calls):
            if not tc.call_id:
                tc.call_id = f"stub-{self._call_index}-{j}"
        element.final_content = final_content
        element.stop_reason = src.stop_reason if src.stop_reason is not None else ("tool_use" if tool_calls else "stop")
        element.token_usage = (
            copy.deepcopy(src.token_usage)
            if src.token_usage is not None
            else self._estimate_usage(src, conversation, injection)
        )
        element.do_not_send_to_agent = src.do_not_send_to_agent
        if injection:
            self._open_with(element, injection)
        element.status = "processed"

    def _estimate_usage(
        self, src: ullm.conv.LLMResponse, conversation: ullm.conv.Conversation, injection: Optional[str] = None
    ) -> ullm.backend.TokenUsage:
        """Rough ~4-chars-per-token estimate from the prompt and emitted content,
        so token-budget paths exercise even without explicit usage."""
        reasoning_chars = len(src.reasoning_content.to_string())
        completion_chars = len(src.final_content.to_string()) + reasoning_chars
        return ullm.backend.TokenUsage(
            prompt_tokens=(len(conversation.to_string()) + len(injection or "")) // 4,
            completion_tokens=max(1, completion_chars // 4),
            reasoning_tokens=reasoning_chars // 4,
            num_requests=1,
        )

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        # Selection (and the exhaustion check) happens here so an exhausted
        # script fails before _generate; the chosen response rides through as
        # the llm_input. The index only advances on a successful _process_output.
        """Pick the next scripted response; the conversation is not consulted."""
        return self._select_response()

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        # llm_input is the LLMResponse selected in _prepare_input; emit it whole.
        """Emit the selected response whole — no streaming to simulate."""
        yield llm_input

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Copy the scripted response onto the element, filling in the fields a real
        backend would supply (stop reason, usage estimate, call ids).
        """
        self._fill_into(llm_output, element, conversation, self.resolve_response_injection(sampling_params))
        self._call_index += 1  # Advance only after a fully successful emission.

    def _process_partial_output(
        self, sampling_params: ullm.backend.SamplingParameters, llm_output: Any, intermediate: ullm.conv.LLMResponse
    ) -> None:
        # The whole response is yielded in one shot, so a peek shows it in full.
        """A peek shows the whole response, since it is yielded in one shot."""
        self._fill_into(
            llm_output, intermediate, ullm.conv.Conversation(), self.resolve_response_injection(sampling_params)
        )


@dataclass
class ArithmeticStubBackend(ullm.backend.Backend):
    """Answers the last small sum ("a+b") found in the final user prompt.

    Unlike ``StubBackend`` this stub actually READS the request, so tests
    whose prompts vary per call (e.g. random-sum seam checks) get correct,
    request-dependent answers without a model or an API key. Deterministic
    output and token estimate; pickle-safe like the other stubs."""

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        """Read the last user prompt and answer the arithmetic in it — the
        request-dependent variant, so a test can tell a real round trip from a
        replayed one.
        """
        prompts = [e for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)]
        assert prompts, "ArithmeticStubBackend needs a UserPrompt to answer"
        text = prompts[-1].content.to_string()
        sums = re.findall(r"(\d+)\s*\+\s*(\d+)", text)
        assert sums, f"ArithmeticStubBackend found no 'a+b' in the prompt: {text!r}"
        a, b = sums[-1]
        return str(int(a) + int(b))

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """Emit the computed answer whole."""
        yield llm_input

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Copy the computed response onto the element."""
        element.final_content = ullm.conv.StringContent(text=str(llm_output))
        element.stop_reason = "stop"
        element.token_usage = ullm.backend.TokenUsage(
            prompt_tokens=len(conversation.to_string()) // 4,
            completion_tokens=max(1, len(str(llm_output)) // 4),
            num_requests=1,
        )
        element.status = "processed"
