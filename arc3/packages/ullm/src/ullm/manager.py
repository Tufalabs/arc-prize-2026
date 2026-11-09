import asyncio
import contextlib
import contextvars
import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generator, Optional, Sequence, Union

import ullm.backend
import ullm.conv
import ullm.support as uls
import ullm.tool

"""
Processors to manage conversations and turns.
"""


TurnElementsObserver = Callable[[Sequence[ullm.conv.ConversationElement]], None]
DiscardedRoundObserver = Callable[[Sequence[ullm.conv.ConversationElement], str], None]

_TURN_ELEMENTS_OBSERVER: contextvars.ContextVar[Optional[TurnElementsObserver]] = contextvars.ContextVar(
    "ullm_turn_elements_observer", default=None
)
_DISCARDED_ROUND_OBSERVER: contextvars.ContextVar[Optional[DiscardedRoundObserver]] = contextvars.ContextVar(
    "ullm_discarded_round_observer", default=None
)


@contextlib.contextmanager
def observe_turn_activity(
    elements: TurnElementsObserver,
    discarded_round: DiscardedRoundObserver,
) -> Generator[None, None, None]:
    """Route nested plain-manager activity to the active instrumented turn.

    Context-local routing keeps concurrent turns isolated. A nested
    instrumented manager installs its own observers, so its usage belongs to
    its child span instead of being duplicated in the parent.
    """
    elements_token = _TURN_ELEMENTS_OBSERVER.set(elements)
    discarded_token = _DISCARDED_ROUND_OBSERVER.set(discarded_round)
    try:
        yield
    finally:
        _DISCARDED_ROUND_OBSERVER.reset(discarded_token)
        _TURN_ELEMENTS_OBSERVER.reset(elements_token)


def _find_last_perturbable_content(conversation: ullm.conv.Conversation) -> Optional[ullm.conv.StringContent]:
    """Walk the conversation (including inside ResponseTurns) in reverse and
    return the most recent ``StringContent`` carried by a ``UserPrompt`` or
    ``ToolResult``. Used by the retry loop to append invisible whitespace on
    retry — shifting tokenization and activation patterns enough to dodge any
    input-deterministic kernel trigger (e.g. NaN-producing MoE path on
    gpt-oss-120b) while preserving task semantics. Returns ``None`` if no
    perturbable target exists, in which case the retry falls back to seed
    bump only."""

    def walk(elem: Any) -> Optional[ullm.conv.StringContent]:
        if isinstance(elem, ullm.conv.ResponseTurn):
            for sub in reversed(elem.elements):
                result = walk(sub)
                if result is not None:
                    return result
            return None
        if isinstance(elem, (ullm.conv.UserPrompt, ullm.conv.ToolResult)):
            if isinstance(elem.content, ullm.conv.StringContent):
                return elem.content
        return None

    for elem in reversed(conversation.elements):
        result = walk(elem)
        if result is not None:
            return result
    return None


def _request_id_of(element: Any) -> Optional[int]:
    """The backend request behind a response, if it got as far as claiming
    one (a prompt that never fits does not)."""
    return getattr(element, "request_id", None)


def _record_request_id(*logs: Any) -> None:
    """Append a request id to each log, skipping the call that has none."""
    request_id = logs[-1]
    if request_id is None:
        return
    for log in logs[:-1]:
        log.append(request_id)


@dataclass
class ResponseManager(uls.Processor):
    """Manages a single LLM response. The response could be for example an answer to the user or a tool call."""

    backend: ullm.backend.Backend = field(default_factory=ullm.backend.Backend)
    conversation: ullm.conv.Conversation = field(default_factory=ullm.conv.Conversation)
    sampling_params: ullm.backend.SamplingParameters = field(default_factory=ullm.backend.GenericSamplingParameters)
    max_retries: int = 5  # Retries on MalformedOutputError; each retry bumps the seed to decorrelate.
    perturb_retry_prompt: bool = True  # Allow temporary trailing newlines on malformed-output retries.
    preserve_failed_draws: bool = False  # Keep evidence intact when a training backend fails.
    # Runs after retry prompt/seed adjustments, immediately before each backend call.
    before_attempt: Optional[Callable[[int], Union[None, Awaitable[None]]]] = None
    _request_ids: list[int] = field(
        default_factory=lambda: list[int](), init=False, repr=False
    )  # request IDs used by this processor
    _latest_partial: Any = field(default=None, init=False, repr=False)  # latest partial output from the generator
    _intermediate_output: Optional[ullm.conv.LLMResponse] = field(
        default=None, init=False, repr=False
    )  # shadow element for peeking

    def _check_constraints(self) -> None:
        """A ResponseManager needs a concrete backend, not the abstract base."""
        assert type(self.backend) is not ullm.backend.Backend, "Backend must be configured for ResponseManager"
        super()._check_constraints()

    async def _run(self) -> None:
        """Produce the response. On MalformedOutputError, rolls back the appended
        element and retries up to max_retries times with a bumped seed; the seed
        is restored before returning. When ``perturb_retry_prompt`` is on,
        each retry also appends N trailing newlines to the last user/tool-result
        content, where N is the attempt number — cheap invisible perturbation
        to dodge any input-deterministic kernel trigger (the NaN-producing MoE
        path on gpt-oss-120b/SM100a recurs when a retry walks the exact same
        prompt)."""
        original_len = len(self.conversation.last_turn.elements)
        original_seed = getattr(self.sampling_params, "seed", None)
        first_retry_trigger: Optional[ullm.backend.MalformedOutputError] = None
        attempt_request_ids: list[int] = []

        perturb_target = _find_last_perturbable_content(self.conversation) if self.perturb_retry_prompt else None
        perturb_original = perturb_target.text if perturb_target is not None else None

        attempt = 0
        try:
            for attempt in range(self.max_retries + 1):
                # Apply retry-count-proportional newline perturbation to the
                # last user-side text. Attempt 0 is untouched; subsequent
                # attempts append `attempt` newlines. Restored in the outer
                # finally so the conversation isn't mutated across the call.
                if perturb_target is not None and perturb_original is not None:
                    perturb_target.text = perturb_original + ("\n" * attempt)
                if self.before_attempt is not None:
                    hook_result = self.before_attempt(attempt)
                    if inspect.isawaitable(hook_result):
                        await hook_result
                attempt_rid: Optional[int] = None
                try:
                    try:
                        # backend.respond() runs the whole call: it appends the
                        # output element, streams already-parsed partial
                        # snapshots for peeking, and leaves the processed
                        # element as the turn's last response.
                        # Hold the async generator in a local so we can aclose() it
                        # explicitly on any unwind path. Without this, Jupyter (which
                        # never runs asyncio.shutdown_asyncgens) defers cleanup of the
                        # abandoned generator to GC — which can be arbitrarily late,
                        # leaving backend-owned resources (vLLM requests, HTTP streams)
                        # dangling after a cancellation.
                        agen = self.backend.respond(self.sampling_params, self.conversation)
                        try:
                            async for partial in agen:
                                self._latest_partial = partial
                                self._intermediate_output = partial
                        finally:
                            await agen.aclose()
                        attempt_rid = _request_id_of(self.conversation.last_turn.elements[-1])
                    except ullm.backend.MalformedOutputError as e:
                        if self.preserve_failed_draws:
                            raise
                        if first_retry_trigger is None:
                            first_retry_trigger = e
                        # Roll back the element we appended so the conversation looks
                        # like nothing happened this attempt.
                        removed_element = self.conversation.last_turn.elements.pop()
                        attempt_rid = _request_id_of(removed_element)
                        assert len(self.conversation.last_turn.elements) == original_len, (
                            "element list length drift after MalformedOutputError rollback"
                        )
                        self._latest_partial = None
                        self._intermediate_output = None
                        if attempt == self.max_retries:
                            raise
                        _record_request_id(self._request_ids, attempt_request_ids, attempt_rid)
                        print(f"[MalformedOutputError, retrying ({attempt + 1}/{self.max_retries}): {e}]", flush=True)
                        self.backend.log_warning(
                            {
                                "type": "malformed_output_retry",
                                "attempt": attempt + 1,
                                "max_retries": self.max_retries,
                                "error": str(e),
                                "request_id": attempt_rid,
                                "removed_element": copy.deepcopy(removed_element),
                            }
                        )
                        # Bump seed to decorrelate the retry from the failing draw.
                        if hasattr(self.sampling_params, "seed"):
                            base = original_seed if original_seed is not None else 0
                            self.sampling_params.seed = base + attempt + 1
                        continue
                except ullm.backend.PromptTooLongError as e:
                    if self.preserve_failed_draws:
                        raise
                    # Prompt exceeded context window — produce a synthetic response.
                    # Roll back any element already appended during this attempt.
                    if len(self.conversation.last_turn.elements) > original_len:
                        self.conversation.last_turn.elements.pop()
                    output_element = ullm.conv.LLMResponse(
                        final_content=ullm.conv.StringContent(text="Failed due to too long input prompt"),
                        stop_reason="max_tokens",
                        status="processed",
                        do_not_send_to_agent=True,
                    )
                    self.conversation.last_turn.elements.append(output_element)
                    self.backend.log_warning({"type": "prompt_too_long", "error": str(e)})
                    break
                _record_request_id(self._request_ids, attempt_request_ids, attempt_rid)
                break
        finally:
            self._latest_partial = None
            self._intermediate_output = None
            if hasattr(self.sampling_params, "seed"):
                self.sampling_params.seed = original_seed
            if perturb_target is not None and perturb_original is not None:
                if attempt > 0:
                    # A retried answer was sampled after the perturbed text;
                    # restoring the original rewrites history the model was
                    # served, and the served-history chain must know
                    # (ullm.continuation: declared, not drifted).
                    self.conversation.mark_history_rewritten()
                perturb_target.text = perturb_original

        # If we had to retry, record how many attempts it took to recover, the
        # error that triggered the retry sequence, the request IDs issued (so
        # the raw inputs/outputs can be fetched via backend._request_log when
        # diagnostics is enabled), and a snapshot of the sampling params.
        if attempt > 0:
            self.backend.log_warning(
                {
                    "type": "malformed_output_recovered",
                    "retries": attempt,
                    "first_error": str(first_retry_trigger),
                    "request_ids": list(attempt_request_ids),
                    "sampling_params": copy.deepcopy(self.sampling_params),
                }
            )

        assert len(self.conversation.last_turn.elements) == original_len + 1, (
            "ResponseManager._run must append exactly one element on success"
        )

        # Stamp the retry count onto the surviving LLMResponse so downstream
        # analysis can tell recovered-after-N-retries from first-try-success.
        # The appended element is normally an LLMResponse; guard anyway.
        tail = self.conversation.last_turn.elements[-1]
        if isinstance(tail, ullm.conv.LLMResponse):
            tail.malformed_retries = attempt

        # Check output status
        for e in self.conversation.last_turn.elements:
            assert e.status == "processed", (
                "Issue with output: all elements of the last run must be `processed` after running ResponseManager"
            )

    def _peek(self) -> None:
        """Nothing to compute: ``respond()`` streams already-parsed partial
        snapshots, and ``_run`` keeps ``_intermediate_output`` pointing at the
        latest one as they arrive."""

    def print_all_requests(self) -> None:
        """Print all requests made by this processor."""
        for request_id in self._request_ids:
            self.backend.print_request(request_id)
            print()


def _skipped_tool_results(tool_calls: "list[ullm.conv.ToolCallContent]", reason: str) -> "list[ullm.conv.ToolResult]":
    """Error ToolResults for tool calls that were parsed but deliberately not
    executed (truncated response, or a stop_when hook ended the turn)."""
    return [
        ullm.conv.ToolResult(
            call_id=tc.call_id,
            tool_name=tc.tool_name,
            content=ullm.conv.StringContent(text=reason),
            is_error=True,
        )
        for tc in tool_calls
    ]


@dataclass
class TurnManager(uls.Processor):
    """Manages a full LLM turn, including tool calls.
    When the LLM responds with tool calls, TurnManager executes them,
    appends results, and loops until the LLM produces a final response without tool calls."""

    response_manager: ResponseManager = field(default_factory=ResponseManager)
    conversation: ullm.conv.Conversation = field(default_factory=ullm.conv.Conversation)
    max_tool_rounds: int = 200  # Safety limit on tool call rounds per turn
    # Cap on the SUM of completion tokens generated within ONE turn (a
    # turn can fan out into many LLM responses via tool calls). Enforced
    # by lowering ``response_manager.sampling_params.max_tokens`` for the
    # next call within the turn so the per-turn total can't exceed this.
    # ``None`` disables the cap. Lives on TurnManager (not on
    # sampling_params) because it's a per-turn concept, not a per-call
    # sampling knob — the response_manager has no notion of "the turn".
    max_generated_tokens_per_turn: Optional[int] = None
    # Asked before every call, and again when a call stops on the cap: return
    # True and the cap stops binding this turn. It exists because a turn is
    # long — many calls with tool work between them — while the reason for a
    # cap can expire at any instant inside one. Without it the whole rule is
    # decided once, at turn start, and a turn that outlives its own cap is cut
    # short by a bound nobody would now impose. A round already truncated that
    # way is re-run rather than kept, since a truncated round cannot be
    # salvaged. ``None`` means the cap, if any, holds for the whole turn.
    per_turn_cap_lifted: Optional[Callable[[], bool]] = None
    # Called before every LLM call in the turn (including the first).
    # Callers use it to re-shape the conversation just-in-time — e.g. an
    # agent evicting old history against a context budget before each
    # request, the way it would if it drove the calls itself.
    # May be async: an agent counting tokens against a remote backend cannot
    # answer synchronously, and an awaited result is awaited here.
    before_round: Optional[Callable[[], Union[None, Awaitable[None]]]] = None
    # Early-stop hook. When set, the round's tool calls execute SERIALLY
    # (in call order; the default path runs them in parallel) and the hook
    # runs after each ToolResult is appended. Returning a string ends the
    # turn: every not-yet-executed tool call of the response receives an
    # error ToolResult carrying that string (each tool_use must get a
    # result — a dangling pair is malformed input on the next request),
    # and the turn closes. Return None to continue.
    stop_when: Optional[Callable[[ullm.conv.LLMResponse, ullm.conv.ToolResult], Optional[str]]] = None
    # Called once per response the backend actually produced, with that
    # response, before the manager branches on tool calls, truncation or a
    # cap-lift retry. Every branch below can end the round without a
    # ToolResult, so a caller needing per-response bookkeeping cannot get it
    # from ``stop_when`` — which fires per result, and not at all for a
    # response that carries no tool call or is discarded and re-run.
    after_response: Optional[Callable[[ullm.conv.LLMResponse], None]] = None
    # Called with each tool call's result once the call has been handled --
    # run, or refused as an unknown tool or a malformed call -- and returns
    # the result the turn records in its place. Lets a caller add to what the
    # model reads after any call without wrapping every tool. Results standing
    # in for calls that never ran (a truncated response, a ``stop_when`` stop)
    # do not pass through it.
    after_tool_result: Optional[Callable[[ullm.conv.ToolResult], ullm.conv.ToolResult]] = None
    classify_tool_calls: bool = False  # Classify sampled calls without treating Python failures as syntax failures.

    def _observe_turn_elements(self, elements: Sequence[ullm.conv.ConversationElement]) -> None:
        """Observe elements after this manager appends them to the current turn.

        The base manager forwards to an ambient instrumented parent, when one
        exists. Subclasses override this protected seam for their own
        append-only accounting.
        """
        observer = _TURN_ELEMENTS_OBSERVER.get()
        if observer is not None:
            observer(elements)

    def _observe_discarded_round(self, elements: Sequence[ullm.conv.ConversationElement], *, reason: str) -> None:
        """Observe a completed round immediately before this manager removes it."""
        observer = _DISCARDED_ROUND_OBSERVER.get()
        if observer is not None:
            observer(elements, reason)

    def _training_call_errors(
        self, response: ullm.conv.LLMResponse, tools: dict[str, ullm.tool.Tool]
    ) -> dict[int, str]:
        """Count recoverable malformed calls separately from tool-body failures."""
        errors: dict[int, str] = {}
        for index, call in enumerate(response.final_content.tool_calls()):
            tool = tools.get(call.tool_name)
            error = call.malformed_error
            if tool is None:
                error = f"Unknown tool: '{call.tool_name}'"
            elif error is None:
                error = tool.argument_error(call.arguments)
            if error is not None:
                errors[index] = error
        response.malformed_call_count += len(errors)
        return errors

    async def _run(self) -> None:
        """Produce the turn, looping through tool calls until a final response."""
        if self.classify_tool_calls and self.per_turn_cap_lifted is not None:
            raise ValueError("classify_tool_calls cannot replay a response after lifting its episode cap")
        assert not self.conversation.status == "processed", (
            "Conversation must be `ready` or `intermediate` to run TurnManager"
        )
        assert isinstance(self.conversation.elements[-1], ullm.conv.UserPrompt), (
            "Last element of conversation must be a UserPrompt to run TurnManager"
        )

        # Set up the turn
        self.response_manager.conversation = self.conversation
        self.conversation.status = "intermediate"
        turn = ullm.conv.ResponseTurn(status="intermediate")
        self.conversation.elements.append(turn)

        # Build tool lookup from system prompt
        tools_by_name = dict[str, ullm.tool.Tool]()
        for t in self.conversation.system_prompt.tools:
            tools_by_name[t.name] = t

        # Per-turn token cap setup. ``max_generated_tokens_per_turn`` lives
        # on the TurnManager (it's a per-turn concept, not a per-call
        # sampling knob); we enforce it by lowering
        # ``sampling_params.max_tokens`` for each call within the turn so
        # the running sum can't exceed it. The original ``max_tokens`` is
        # restored in the finally block so the next turn starts clean.
        sampling_params = self.response_manager.sampling_params
        original_max_tokens: Optional[int] = getattr(sampling_params, "max_tokens", None)
        per_turn_cap: Optional[int] = self.max_generated_tokens_per_turn
        tokens_this_turn = 0
        cap_applied = False  # whether max_tokens was lowered, i.e. needs restoring

        # Rounds are counted by the ones that SURVIVE, not by iterations: a
        # round discarded below never reached the model, so charging it
        # against the allowance would spend a round on nothing.
        #
        # A discard cannot repeat without bound even though nothing counts
        # them. A truncated call spends everything the cap had left, so the
        # round after a discard either finds the cap lifted and runs
        # uncapped, or finds it back with nothing left and ends the turn.
        rounds_done = 0
        try:
            # Loop: get LLM response, check for tool calls, execute, repeat
            while rounds_done < self.max_tool_rounds:
                _round = rounds_done
                if self.before_round is not None:
                    # May be async: an agent that counts tokens against a
                    # remote backend cannot answer synchronously.
                    _hook_result = self.before_round()
                    if inspect.isawaitable(_hook_result):
                        await _hook_result
                # Per-turn cap pre-call adjustment. Asked again every round
                # rather than once per turn: a turn can run for hours, and a
                # cap whose reason has expired must stop binding the rounds
                # that follow rather than only the next turn.
                active_cap: Optional[int] = per_turn_cap
                if active_cap is not None and self.per_turn_cap_lifted is not None and self.per_turn_cap_lifted():
                    active_cap = None
                effective_max: Optional[int] = original_max_tokens
                if active_cap is not None:
                    cap_left = active_cap - tokens_this_turn
                    if cap_left <= 0:
                        # Cap was already exhausted by prior responses in this
                        # turn (e.g. last response was a tool_use that hit the
                        # cap exactly, then tools ran, now we'd start another
                        # response with no budget). Append a synthetic marker
                        # so downstream can detect the truncation; skip the
                        # call entirely.
                        placeholder = ullm.conv.LLMResponse(
                            final_content=ullm.conv.StringContent(
                                text=f"[Synthetic placeholder: per-turn token cap exhausted before next call ({tokens_this_turn}/{active_cap})]"
                            ),
                            stop_reason="other",
                            status="processed",
                            do_not_send_to_agent=True,
                            truncated_by_per_turn_cap=True,
                        )
                        turn.elements.append(placeholder)
                        self._observe_turn_elements([placeholder])
                        break
                    if original_max_tokens is None:
                        effective_max = cap_left
                    else:
                        effective_max = min(original_max_tokens, cap_left)
                    sampling_params.max_tokens = effective_max
                    cap_applied = True
                elif cap_applied:
                    # The cap bound an earlier round and has since lifted:
                    # put the sampling ceiling back, or the lowered value
                    # would go on capping every round of this turn.
                    sampling_params.max_tokens = original_max_tokens
                    cap_applied = False

                # Where this round's elements start, so a round retried
                # below leaves nothing of its first attempt behind.
                round_started_at = len(turn.elements)
                try:
                    await self.response_manager.run()
                except BaseException:
                    # A cancellation or backend failure can leave an
                    # incomplete response carrying partial usage. It still
                    # belongs to this turn; malformed attempts are popped by
                    # ResponseManager before they reach here.
                    self._observe_turn_elements(list(turn.elements[round_started_at:]))
                    raise
                self.response_manager.rearm()

                # Check if the response contains tool calls
                last_response = turn.last_response
                tool_calls = last_response.final_content.tool_calls()

                # Update per-turn token tally and check whether THIS call hit
                # the per-turn cap. The truncation is ours when the backend
                # stopped at max_tokens under a budget WE lowered: with no
                # sampling cap of its own every such stop is ours, with one
                # only a stop under a strictly smaller budget is. A synthetic
                # placeholder was never generated at all — the prompt-too-long
                # stand-in reports max_tokens without a call having run — so it
                # is excluded rather than mislabelled. Flag the response so
                # downstream callers can surface a give-up; the loop then exits
                # via the existing tool-call / break logic.
                if last_response.token_usage is not None:
                    tokens_this_turn += last_response.token_usage.completion_tokens
                if last_response.stop_reason == "max_tokens" and not last_response.do_not_send_to_agent:
                    # Equality belongs to the episode budget too: both ceilings
                    # can be exhausted by the same final generated token.
                    last_response.truncation_cap_source = (
                        "episode" if active_cap is not None and tokens_this_turn >= active_cap else "response"
                    )
                malformed_calls: dict[int, str] = {}
                unparseable_calls = last_response.malformed_call_count
                empty_reply = False
                if self.classify_tool_calls:
                    if last_response.truncation_cap_source == "episode":
                        last_response.malformed_call_count = 0
                        unparseable_calls = 0
                    else:
                        # A reply the backend kept as malformed with nothing in
                        # it (no text, no tool call: the draw ended its turn
                        # without acting) costs what a malformed tool call
                        # costs. A budget stop is not this: it keeps its own
                        # stop reason and stays unpenalized.
                        empty_reply = last_response.stop_reason == "malformed" and ullm.backend._is_empty_content(
                            last_response.final_content
                        )
                        if empty_reply and not last_response.malformed_call_count:
                            last_response.malformed_call_count = 1
                            unparseable_calls = 1
                        malformed_calls = self._training_call_errors(last_response, tools_by_name)
                self._observe_turn_elements([last_response])
                if self.after_response is not None:
                    self.after_response(last_response)
                if self.classify_tool_calls and unparseable_calls:
                    if empty_reply:
                        feedback_text = (
                            "Malformed reply: no text and no tool call. "
                            "Reply with a valid tool call: a tool name and its argument object."
                        )
                    else:
                        feedback_text = (
                            f"Malformed tool call: {last_response.malformed_error or 'invalid tool-call syntax'}. "
                            "Use a valid tool name and argument object."
                        )
                    feedback = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=feedback_text))
                    turn.elements.append(feedback)
                    self._observe_turn_elements([feedback])
                hit_per_turn_cap = (
                    active_cap is not None
                    and last_response.stop_reason == "max_tokens"
                    and not last_response.do_not_send_to_agent
                    and effective_max is not None
                    and (original_max_tokens is None or effective_max < original_max_tokens)
                )
                cap_lifted_after_response = bool(
                    hit_per_turn_cap and self.per_turn_cap_lifted is not None and self.per_turn_cap_lifted()
                )
                if hit_per_turn_cap:
                    last_response.truncated_by_per_turn_cap = True
                if cap_lifted_after_response:
                    # The cap retired while this round was in flight, so the
                    # truncation is an artefact of a bound that no longer
                    # applies — and a truncated round cannot be salvaged, only
                    # run again.
                    #
                    # The agent's surviving-element budget deliberately omits
                    # this output; append-only observers see it before removal
                    # so spend accounting does not depend on that budget view.
                    self._observe_discarded_round(list(turn.elements[round_started_at:]), reason="per_turn_cap_lifted")
                    # The removed round was served: the next prompt no longer
                    # extends it, which the served-history chain must hear as
                    # a declared rewrite rather than drift.
                    self.conversation.mark_history_rewritten()
                    del turn.elements[round_started_at:]
                    continue
                # Past the discard: this round reached the model and counts.
                rounds_done += 1

                if not tool_calls:
                    # No tool calls — turn is complete (whether truncated or
                    # naturally ended).
                    break

                # If the LLM was cut off (max_tokens, content_filter, etc.,
                # incl. our per-turn cap), don't execute potentially truncated
                # tool calls — end the turn. The truncated_by_per_turn_cap
                # flag is preserved on last_response so downstream can
                # distinguish a turn-cap give-up from other truncations.
                if last_response.stop_reason not in (None, "end_turn", "stop", "tool_use"):
                    self.response_manager.backend.log_warning(
                        {
                            "type": "truncated_tool_calls",
                            "stop_reason": last_response.stop_reason,
                            "n_tool_calls": len(tool_calls),
                            "round": _round,
                            "truncated_by_per_turn_cap": last_response.truncated_by_per_turn_cap,
                        }
                    )
                    # Answered rather than left dangling. The calls are NOT
                    # executed -- a truncated call's arguments are not to be
                    # trusted -- but a tool_use with no result is malformed
                    # input on the next request (Anthropic 400s on it), and
                    # the model reads the reason, which is the honest one.
                    skipped_results = _skipped_tool_results(
                        tool_calls,
                        f"not executed: the response was truncated (stop_reason={last_response.stop_reason})",
                    )
                    turn.elements.extend(skipped_results)
                    self._observe_turn_elements(skipped_results)
                    break

                # TODO: add a 'terminate' tool

                # Execute tool calls in parallel (async), serializing calls to the same tool.
                # Each result is appended in the original call order.
                results = list[ullm.conv.ToolResult]([None] * len(tool_calls))  # type: ignore[list-item]
                tool_locks = dict[str, asyncio.Lock]()  # serialize concurrent calls to same tool

                async def _execute_tool_call(idx: int, tc: ullm.conv.ToolCallContent) -> None:
                    await _handle_tool_call(idx, tc)
                    if self.after_tool_result is not None:
                        results[idx] = self.after_tool_result(results[idx])

                async def _handle_tool_call(idx: int, tc: ullm.conv.ToolCallContent) -> None:
                    tool = tools_by_name.get(tc.tool_name)
                    if tool is None:
                        results[idx] = ullm.conv.ToolResult(
                            call_id=tc.call_id,
                            tool_name=tc.tool_name,
                            content=ullm.conv.StringContent(text=f"Unknown tool: '{tc.tool_name}'"),
                            is_error=True,
                            malformed_call_count=int(self.classify_tool_calls),
                        )
                        self.response_manager.backend.log_warning(
                            {
                                "type": "tool_error",
                                "tool_name": tc.tool_name,
                                "call_id": tc.call_id,
                                "arguments": copy.deepcopy(tc.arguments),
                                "content": f"Unknown tool: '{tc.tool_name}'",
                            }
                        )
                        return

                    if idx in malformed_calls:
                        error = malformed_calls[idx]
                        results[idx] = ullm.conv.ToolResult(
                            call_id=tc.call_id,
                            tool_name=tc.tool_name,
                            content=ullm.conv.StringContent(text=error),
                            is_error=True,
                            malformed_call_count=1,
                        )
                        self.response_manager.backend.log_warning(
                            {
                                "type": "malformed_tool_call",
                                "tool_name": tc.tool_name,
                                "call_id": tc.call_id,
                                "arguments": copy.deepcopy(tc.arguments),
                                "content": error,
                            }
                        )
                        return

                    # Serialize calls to the same tool instance (shared state)
                    if tc.tool_name not in tool_locks:
                        tool_locks[tc.tool_name] = asyncio.Lock()
                    async with tool_locks[tc.tool_name]:
                        tool.arguments = tc.arguments
                        await tool.run()

                        # Drain any warnings the tool accumulated during this call.
                        for w in tool.pop_pending_warnings():
                            self.response_manager.backend.log_warning(w)

                        # Build the result and clean up before releasing the lock
                        is_error = tool.is_error
                        result_content = tool.result
                        results[idx] = ullm.conv.ToolResult(
                            call_id=tc.call_id,
                            tool_name=tc.tool_name,
                            content=result_content,
                            is_error=is_error,
                            malformed_call_count=int(idx in malformed_calls),
                        )
                        if is_error:
                            # Every error a tool REPORTS: schema validation, a REPL
                            # traceback, a timeout. An exception RAISED out of
                            # _execute never arrives here — Processor.run re-raises
                            # it — and that is deliberate: the three that escape a
                            # PythonTool (a raising host function, initial_code
                            # failing, unusable network isolation) are the harness
                            # breaking, not something to hand the model as a result.
                            # tool.arguments is captured post-`_maybe_remap_single_arg`, which
                            # is what the tool actually ran on — more useful than tc.arguments.
                            self.response_manager.backend.log_warning(
                                {
                                    "type": "tool_error",
                                    "tool_name": tc.tool_name,
                                    "call_id": tc.call_id,
                                    "raw_arguments": copy.deepcopy(tc.arguments),
                                    "arguments": copy.deepcopy(tool.arguments),
                                    "tool_state": tool.state,
                                    "content": result_content.to_string()
                                    if hasattr(result_content, "to_string")
                                    else str(result_content),
                                }
                            )
                        if tool.clear_state_on == "run":
                            await tool.clear_state()
                        tool.rearm()

                if self.stop_when is None:
                    await asyncio.gather(*[_execute_tool_call(i, tc) for i, tc in enumerate(tool_calls)])
                    turn.elements.extend(results)
                    self._observe_turn_elements(results)
                else:
                    # Serial path: execute in call order, consulting the
                    # stop hook after each result so a caller can end the
                    # turn mid-response; the remainder gets explicit
                    # skipped results.
                    stop_text: Optional[str] = None
                    for idx, tc in enumerate(tool_calls):
                        await _execute_tool_call(idx, tc)
                        turn.elements.append(results[idx])
                        self._observe_turn_elements([results[idx]])
                        stop_when = self.stop_when
                        assert stop_when is not None
                        stop_text = stop_when(last_response, results[idx])
                        if stop_text is not None:
                            skipped_results = _skipped_tool_results(tool_calls[idx + 1 :], stop_text)
                            turn.elements.extend(skipped_results)
                            self._observe_turn_elements(skipped_results)
                            break
                    if stop_text is not None:
                        break
            else:
                # Exceeded max rounds — mark turn as processed anyway
                pass

            # Clear state for tools configured to clear per turn
            for t in self.conversation.system_prompt.tools:
                if t.clear_state_on == "turn":
                    await t.clear_state()

            turn.status = "processed"
        finally:
            # Restore the per-call sampling cap so the next turn starts fresh:
            # a turn that lowered ``max_tokens`` to enforce the per-turn cap
            # would otherwise leak that lower value into subsequent turns.
            # Only when we actually lowered it — SamplingParameters subclasses
            # need not carry a ``max_tokens`` field at all, and assigning one
            # to a frozen instance would raise from this finally, replacing
            # whatever exception is already unwinding.
            if cap_applied:
                sampling_params.max_tokens = original_max_tokens

    def live_snapshot(self) -> dict[str, Any]:
        """Point-in-time view of a running turn for on-demand observers
        (live viewers): responses completed so far plus the current
        partial model output — populated whenever the backend streams
        (``respond()`` feeds ``_process_partial_output`` snapshots into
        ``_intermediate_output``), empty between calls (tool execution)
        or on non-streaming backends. Side-effect-free and cheap;
        callable from another task at any moment, any state."""
        n_responses = 0
        if self.conversation.elements:
            tail = self.conversation.elements[-1]
            if isinstance(tail, ullm.conv.ResponseTurn):
                n_responses = sum(
                    1
                    for element in tail.elements
                    if isinstance(element, ullm.conv.LLMResponse) and element.status == "processed"
                )
        partial = self.response_manager._intermediate_output
        return {
            "state": self.state,
            "n_responses": n_responses,
            "partial_reasoning": partial.reasoning_content.to_string() if partial is not None else "",
            "partial_final": partial.final_content.to_string() if partial is not None else "",
        }

    def _peek(self) -> None:
        """Peek at the intermediate response; this can be used for example to update a UI with partial responses."""
        if self.response_manager.state == "running":
            self.response_manager.peek()

    def _rearm(self) -> None:
        """Propagate rearm to ``response_manager`` so the next ``run()``
        doesn't trip the ``state == 'ready'`` assertion when an upstream
        caller recovers from an exception that propagated through
        ``response_manager.run()`` — a caught ``MalformedOutputError``
        patched up and retried, say.

        Inside ``_run()`` the per-round ``response_manager.rearm()`` already
        leaves it ``ready``; this only changes behavior on the error path."""
        if self.response_manager.state != "ready":
            self.response_manager.rearm()
