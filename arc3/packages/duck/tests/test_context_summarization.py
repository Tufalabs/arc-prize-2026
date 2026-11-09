"""Gated model-written context summarization and its accounting contract.

Forked from arc2-platypus `packages/duck/tests/test_context_summarization.py`,
the suite of the mechanism this duck ported as it was. Every test is ARC2's,
bar what differs underneath it:

- The move context is ARC3's (`LLMMoveContext`, a history and a game, with one
  prompt a turn block rather than a list), and so is the observation an
  analyzer entry is handed.
- The summarizing agent pins what ARC3's duck has and ARC2's does not, at the
  values that leave ARC2's premises standing: no file tools and no journal
  (the one tool is `python`), no image, the compact summary request that
  states a size, and no listing after a summary.
- The pruning fixtures are ARC3's (`test_context_pruning.py`): its decision
  ruler charges a larger fixed prefix, which ARC2's constants would leave
  every fixture comfortably under.
- ARC3's summary request keeps an empty reply (`keep_empty_reply`) whatever
  the backend's policy, so an empty summary is always billed and refused as
  no usable summary, and the retained-draw tests reach the malformed branch
  through a scripted parser refusal instead.

Not carried, each for a subject ARC3 does not have: the requeued check/show
feedback of ARC2's answer tools; and the request gap re-decided before every
LLM call (ARC3 decides it once a move, in the solver).
"""

from __future__ import annotations

import random
import threading
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import arcengine
import numpy as np
import pytest
import taaf.game
import ullm.backend
import ullm.backends.stub
import ullm.continuation
import ullm.conv

import platypus.diagnostics
from duck import DuckAgent
from duck.prompts import DuckPrompts
from duck.python_tool import DuckPythonTool
from platypus.contract import Action, GameInfo, GiveUpError, Move, Observation
from platypus.llm import LLMConfig, LLMMoveContext

WINDOW = 4096
RATIO = 16
SUMMARY = "VALID SUMMARY"
LATEST = "LATEST TAIL"
INSTRUCTION = "Your context window is nearly full"
THINK_CLOSE = "\n</think>\n\n"


@dataclass(frozen=True)
class _SeenRequest:
    summary: bool
    history_epoch: int
    tools: tuple[str, ...]
    reasoning_effort: str | None
    max_tokens: int | None
    seed: int | None
    extra_params: dict[str, Any]
    request_context: dict[str, Any]
    response_injection: str | None
    system_text: str
    element_texts: tuple[str, ...]
    user_texts: tuple[str, ...]
    turn: ullm.conv.ResponseTurn


@dataclass
class _CountingScript(ullm.backends.stub.StubBackend):
    """A scripted backend whose exact count exposes each compaction branch.

    A summary request carries the live tools, so it is told apart by its
    instruction. The script also holds the one exact-history rule a summary
    can break: the rebuilt history is a rewrite of what was served, so
    counting it on the epoch the summary request was served on is the drift
    the engine refuses (it ended 5 of 105 games of the mode's first run). And
    it can fail a summary request on cue: past the engine window, or as a
    request that does not extend the served ids."""

    ordinary_count: int = 100
    compacted_count: int = 100
    summary_count: int = 100
    summary_with_current_count: int | None = None
    single_user_count: int | None = None
    compacted_with_tail_count: int | None = None
    wrapper_marker: str = SUMMARY
    tail_marker: str = LATEST
    return_none: bool = False
    overflow_remaining: int = 0
    drift_remaining: int = 0
    summary_epochs: list[int] = field(default_factory=lambda: list[int](), init=False)
    seen_requests: list[_SeenRequest] = field(default_factory=lambda: list[_SeenRequest](), init=False)
    counted: list[int | None] = field(default_factory=lambda: list[int | None](), init=False)
    first_live_turn: ullm.conv.ResponseTurn | None = field(default=None, init=False)

    @staticmethod
    def _is_summary(conversation: ullm.conv.Conversation) -> bool:
        return any(
            isinstance(element, ullm.conv.UserPrompt) and INSTRUCTION in element.content.to_string()
            for element in conversation.elements
        )

    def _prompt_weight(self, conversation: ullm.conv.Conversation) -> int:
        text = conversation.to_string()
        if self._is_summary(conversation):
            if self.summary_with_current_count is not None and "CURRENT REQUEST" in text:
                return self.summary_with_current_count
            return self.summary_count
        if self.wrapper_marker in text:
            if self.compacted_with_tail_count is not None and self.tail_marker in text:
                return self.compacted_with_tail_count
            return self.compacted_count
        if self.single_user_count is not None:
            users = sum(isinstance(element, ullm.conv.UserPrompt) for element in conversation.elements)
            if users == 1:
                return self.single_user_count
        return self.ordinary_count

    async def _token_count(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        conversation: ullm.conv.Conversation,
    ) -> int | None:
        del sampling_params
        rebuilt = self.wrapper_marker in conversation.to_string() and not self._is_summary(conversation)
        if rebuilt and self.summary_epochs and conversation.history_epoch <= self.summary_epochs[-1]:
            raise ullm.continuation.HistoryDriftError("the rebuilt history was counted on the summary's epoch")
        if (
            not self._is_summary(conversation)
            and self.wrapper_marker not in conversation.to_string()
            and isinstance(conversation.elements[-1], ullm.conv.ResponseTurn)
            and self.first_live_turn is None
        ):
            self.first_live_turn = conversation.last_turn
        count = None if self.return_none else self._prompt_weight(conversation)
        self.counted.append(count)
        return count

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        del send_back_reasoning
        summary = self._is_summary(conversation)
        self.seen_requests.append(
            _SeenRequest(
                summary=summary,
                history_epoch=conversation.history_epoch,
                tools=tuple(tool.name for tool in conversation.system_prompt.tools),
                reasoning_effort=conversation.system_prompt.reasoning_effort,
                max_tokens=getattr(sampling_params, "max_tokens", None),
                seed=getattr(sampling_params, "seed", None),
                extra_params=dict(getattr(sampling_params, "extra_params", {})),
                request_context=deepcopy(sampling_params.request_context),
                response_injection=getattr(sampling_params, "response_injection", None),
                system_text=conversation.system_prompt.content.to_string(),
                element_texts=tuple(element.to_string() for element in conversation.elements),
                user_texts=tuple(
                    element.content.to_string()
                    for element in conversation.elements
                    if isinstance(element, ullm.conv.UserPrompt)
                ),
                turn=turn,
            )
        )
        prepared = super()._prepare_input(sampling_params, turn, conversation, "all")
        if summary:
            self.summary_epochs.append(conversation.history_epoch)
            if self.drift_remaining:
                self.drift_remaining -= 1
                raise ullm.continuation.HistoryDriftError("scripted drift")
            if self.overflow_remaining:
                self.overflow_remaining -= 1
                raise ullm.backend.PromptTooLongError("scripted summary overflow")
        return prepared

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        assert element.token_usage is not None
        element.token_usage.prompt_tokens = self._prompt_weight(conversation)
        element.token_usage.cached_tokens = 0


@dataclass
class _InvalidCountScript(_CountingScript):
    """A backend that violates the runtime token-count contract."""

    invalid_count: Any = True

    async def _token_count(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        conversation: ullm.conv.Conversation,
    ) -> Any:
        del sampling_params, conversation
        self.counted.append(self.invalid_count)
        return self.invalid_count


@dataclass
class _MalformedCountingScript(_CountingScript):
    """Allow empty scripted responses so the caller-owned validator sees them."""

    def _check_constraints(self) -> None:
        assert self.responses
        ullm.backend.Backend._check_constraints(self)


def _response(
    text: str,
    *,
    completion_tokens: int = 1,
    stop_reason: ullm.conv.StopReason | None = None,
) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason=stop_reason,
        token_usage=ullm.backend.TokenUsage(completion_tokens=completion_tokens, num_requests=1),
    )


def _tool_response(*, completion_tokens: int = 1) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[
                ullm.conv.ToolCallContent(
                    tool_name="python",
                    arguments={"code": "print('INVALID SUMMARY TOOL CALL')"},
                )
            ]
        ),
        token_usage=ullm.backend.TokenUsage(completion_tokens=completion_tokens, num_requests=1),
    )


def _reasoning_response(text: str, *, completion_tokens: int = 1) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=text),
        final_content=ullm.conv.StringContent(text=""),
        stop_reason="max_tokens",
        token_usage=ullm.backend.TokenUsage(completion_tokens=completion_tokens, num_requests=1),
    )


def _processed_response(text: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(completion_tokens=1, num_requests=1),
        status="processed",
    )


def _summarizing_agent(
    *,
    profile_max_tokens: int = 256,
    max_tokens_per_game: int | None = 10_000,
    compaction_forbidden_token_ids: tuple[int, ...] = (),
    window: int = WINDOW,
    summary_tail: Literal["full", "without_reasoning", "none"] | None = None,
) -> DuckAgent:
    """The summarize arm as the vLLM presets deploy it: token-native, so the
    detached summary requests are measured by the backend's own count."""
    return DuckAgent(
        # What ARC3's duck has and ARC2's does not, at ARC2's premises.
        file_tools="none",
        journal=False,
        multimodal_context="",
        add_files_and_repl_to_summary=False,
        prompts=DuckPrompts(thorough_summary=False),
        reply_cap_tokens=profile_max_tokens,
        compaction_strategy="summarize",
        compaction_ratio=RATIO,
        summary_tail=summary_tail or DuckAgent.summary_tail,
        compaction_forbidden_token_ids=compaction_forbidden_token_ids,
        token_native_context=True,
        context_window_tokens=window,
        compact_to_tokens=1000,
        max_tokens_per_game=max_tokens_per_game,
        reasoning_effort="high",
        llm_config=LLMConfig(
            sampling_profiles={
                "play": ullm.backend.GenericSamplingParameters(
                    temperature=0.0,
                    max_tokens=None,
                )
            }
        ),
    )


def _tool_round(text: str, *, reasoning: str = "") -> tuple[ullm.conv.LLMResponse, ullm.conv.ToolResult]:
    """A processed response that called a tool, and the call's result: the
    tail ARC3's rebuild keeps is the most recent tool call and its round,
    where ARC2's kept the latest response whatever it was."""
    response = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=reasoning) if reasoning else ullm.conv.EmptyContent(),
        final_content=ullm.conv.ListContent(
            elements=[
                ullm.conv.StringContent(text=text),
                ullm.conv.ToolCallContent(call_id="tail", tool_name="python", arguments={"code": "print('tail')"}),
            ]
        ),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(completion_tokens=1, num_requests=1),
        status="processed",
    )
    result = ullm.conv.ToolResult(
        call_id="tail", tool_name="python", content=ullm.conv.StringContent(text="tail result")
    )
    return response, result


def _history(target: DuckAgent) -> tuple[ullm.conv.UserPrompt, ullm.conv.LLMResponse, ullm.conv.ResponseTurn]:
    task = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="PINNED TASK"))
    stale = _processed_response("STALE WORK")
    latest, result = _tool_round(LATEST)
    prior = ullm.conv.ResponseTurn(elements=[stale, latest, result], status="processed")
    target.messages = [task, prior]
    return task, latest, prior


def _keeps_the_status(target: DuckAgent) -> None:
    """ARC3's rebuild opens on the summary and keeps the prompt that started
    the current turn, the current status, where ARC2's kept its task prompt
    first (`sweep12_branch_notes.md`, what the port does differently)."""
    assert SUMMARY in target.messages[0].to_string()
    assert any(
        isinstance(element, ullm.conv.UserPrompt) and "CURRENT REQUEST" in element.to_string()
        for element in target.messages
    )


def _observation(*, outcome: str = "playing", level: int = 0, index: int = 1, colour: int = 0) -> Observation:
    raw = arcengine.FrameDataRaw(
        game_id="summarization",
        state=arcengine.GameState.NOT_FINISHED,
        levels_completed=level,
        win_levels=9,
        available_actions=[1, 2, 3, 4],
    )
    raw.frame = [np.full((8, 8), colour, dtype=np.int8)]
    return Observation.from_state(
        taaf.game.GameState(raw=raw),
        outcome=outcome,  # type: ignore[arg-type]
        level=level,
        index=index,
        caused_by=Action("ACTION1"),
    )


def _context(backend: ullm.backend.Backend) -> Any:
    return LLMMoveContext(
        history=[Move(action=Action("ACTION1"), observations=(_observation(),))],
        game=GameInfo(game_id="summarization", hint=None, action_space=("ACTION1", "ACTION2"), number_of_levels=9),
        rng=random.Random(7),
        shared={},
        backend=backend,
    )


async def _run_block(
    target: DuckAgent,
    backend: ullm.backend.Backend,
    *,
    remaining_calls: int | None = None,
) -> int:
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="CURRENT REQUEST"))
    return await _turn_block(target, _context(backend), prompt, remaining_calls=remaining_calls)


async def _turn_block(target: DuckAgent, ctx: Any, prompt: ullm.conv.UserPrompt, *, remaining_calls: int | None) -> int:
    """One turn block, with the prompt it starts from recorded as ARC3's
    analyzer entry records it: a summary's rebuild keeps that prompt, and
    ARC2's turn block, which these tests drive directly, has no such record."""
    target._turn_prompt = prompt
    return await target._run_turn_block(ctx, prompt, remaining_calls=remaining_calls)


# The pruning fixtures, ARC3's (`test_context_pruning.py`): the window clears
# the ~12.4k fixed prefix the decision ruler charges, and one 12-round block
# crosses its budget while one round does not.
PRUNE_WINDOW = 20480
PRUNE_TARGET = 15500


def _turn(rounds: int) -> ullm.conv.ResponseTurn:
    elements: list[Any] = []
    for i in range(rounds):
        elements.append(ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=f"thinking {i} " * 40)))
        elements.append(
            ullm.conv.ToolResult(
                call_id=f"c{i}", tool_name="python", content=ullm.conv.StringContent(text=f"result {i} " * 40)
            )
        )
    return ullm.conv.ResponseTurn(elements=elements, status="intermediate")


def pruning_agent(**kwargs: Any) -> DuckAgent:
    kwargs.setdefault("reply_cap_tokens", None)
    kwargs.setdefault("compaction_strategy", "prune")
    kwargs.setdefault("multimodal_context", "")
    kwargs.setdefault("file_tools", "none")
    kwargs.setdefault("compact_to_tokens", PRUNE_TARGET)
    return DuckAgent(context_window_tokens=PRUNE_WINDOW, **kwargs)


def single_block(target: DuckAgent, rounds: int = 12) -> None:
    target.messages = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")), _turn(rounds)]


def current_turn(target: DuckAgent) -> ullm.conv.ResponseTurn:
    tail = target.messages[-1]
    assert isinstance(tail, ullm.conv.ResponseTurn)
    return tail


async def test_default_strategy_keeps_the_existing_pruning_path() -> None:
    target = pruning_agent()
    single_block(target)
    before = len(current_turn(target).elements)
    backend = _CountingScript(responses=[_response("unused")], return_none=True)
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(
            content=ullm.conv.StringContent(text=target._system_prompt_text()),
            reasoning_effort=target.reasoning_effort,
            tools=target._turn_tools(),
        ),
        elements=target.messages,
        status="intermediate",
    )
    target._live_backend = backend
    target._live_params = ullm.backend.GenericSamplingParameters(max_tokens=512)
    target._live_conversation = conversation

    await target._trim_live_messages()

    assert len(current_turn(target).elements) < before
    assert backend.seen_requests == []
    assert backend.counted == [], "the prune arm never asks a backend to count"
    (bound,) = target._pending_bounds
    assert bound == target._decision_prompt_tokens(target.messages)


def test_pruning_ignores_summary_only_configuration() -> None:
    target = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        compaction_ratio=1,
        compaction_forbidden_token_ids=(-1,),
    )

    assert target._uses_context_summary() is False
    assert "Your context is finite" not in target._system_prompt_text()


def test_public_target_preserves_summary_policy_and_current_budget() -> None:
    target = _summarizing_agent()
    assert target.context_budget_tokens() == 3328
    assert target.compact_target_tokens() == 3328
    target.reply_cap_tokens = 512
    assert target.context_budget_tokens() == 3072
    assert target.compact_target_tokens() == 3072
    target.compact_to_tokens = 0  # Inactive prune settings do not constrain summaries.
    assert target.compact_target_tokens() == 3072
    target.reply_cap_tokens = None
    with pytest.raises(ValueError, match="requires a positive reply_cap_tokens"):
        target.compact_target_tokens()


def test_summary_strategy_announces_the_compaction_contract() -> None:
    target = _summarizing_agent()

    assert "Your context is finite" in target._system_prompt_text()
    assert "compact summary" in target._system_prompt_text()


def test_summary_request_backstop_is_game_wide_and_gated() -> None:
    target = _summarizing_agent()

    assert target._summary_request_backstop_remaining(0) == 512
    assert target._summary_request_backstop_remaining(511) == 1
    with pytest.raises(GiveUpError, match="request backstop exhausted"):
        target._summary_request_backstop_remaining(512)

    assert pruning_agent()._summary_request_backstop_remaining(10_000) is None


async def test_a_detached_request_is_bounded_like_the_live_one() -> None:
    """With no backend to count, a summary-arm request is measured by the
    same provable bound the live turn's cold start uses: an image weighs its
    engine-side cost — by its cells, not its bytes — never a flat charge and
    never its base64 length."""

    def conversation(data: np.ndarray[Any, Any]) -> ullm.conv.Conversation:
        return ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="system")),
            elements=[ullm.conv.UserPrompt(content=ullm.conv.ImageContent(data=data))],
        )

    small = conversation(np.zeros((1, 1, 3), dtype=np.uint8))
    large = conversation(np.random.default_rng(3).integers(0, 256, size=(480, 480, 3), dtype=np.uint8))
    target = _summarizing_agent()
    params = ullm.backend.GenericSamplingParameters(max_tokens=256)

    small_summary = await target._request_prompt_tokens(params, small)
    large_summary = await target._request_prompt_tokens(params, large)
    assert small_summary == target._bounded_conversation_tokens(small)
    assert large_summary - small_summary == target._image_bound_tokens(
        ullm.conv.ImageContent(data=np.zeros((480, 480, 3), dtype=np.uint8))
    ) - target._image_bound_tokens(ullm.conv.ImageContent(data=np.zeros((1, 1, 3), dtype=np.uint8)))


def test_invalid_summary_reply_cap_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="reply_cap_tokens.*positive"):
        _summarizing_agent(profile_max_tokens=0)


async def test_analyzer_failure_does_not_retain_live_backend_handles(monkeypatch: Any) -> None:
    class _Repl:
        on_answer: Any = None

        async def setup_task(self, _view: Any) -> None:
            pass

    target = _summarizing_agent()
    backend = _CountingScript()
    ctx = _context(backend)

    def repl(_ctx: Any) -> _Repl:
        return _Repl()

    monkeypatch.setattr(target, "_ensure_repl", repl)

    async def fail(_ctx: Any, _obs: Any, *, remaining_move_calls: int | None = None) -> Any:
        del remaining_move_calls
        target._live_backend = backend
        target._live_params = ullm.backend.GenericSamplingParameters(max_tokens=10)
        target._live_conversation = ullm.conv.Conversation()
        target._status_handle = ctx.status
        raise RuntimeError("scripted analyzer failure")

    monkeypatch.setattr(target, "_run_analyzer_entry", fail)
    with pytest.raises(RuntimeError, match="scripted analyzer failure"):
        await target._do_move(ctx, _observation())

    assert target._live_backend is None
    assert target._live_params is None
    assert target._live_conversation is None
    assert target._status_handle is None


@pytest.mark.parametrize("invalid_ids", [(True,), (-1,), ("248058",), (248058.0,)])
def test_summary_token_bans_require_nonnegative_integer_ids(invalid_ids: Any) -> None:
    with pytest.raises(ValueError, match="compaction_forbidden_token_ids"):
        _summarizing_agent(compaction_forbidden_token_ids=invalid_ids)


@pytest.mark.parametrize(("offset", "expected_summaries"), [(0, [False]), (1, [True, False])])
async def test_summary_trigger_uses_exact_window_arithmetic(
    offset: int,
    expected_summaries: list[bool],
) -> None:
    target = _summarizing_agent()
    _history(target)
    budget = WINDOW - target._reply_reserve_tokens() - 512
    script = [_response("DONE")] if offset == 0 else [_response(SUMMARY), _response("DONE")]
    backend = _CountingScript(
        responses=script,
        ordinary_count=budget + offset,
        compacted_count=500,
        summary_count=500,
    )

    await _run_block(target, backend)

    assert [request.summary for request in backend.seen_requests] == expected_summaries
    if offset == 0:
        assert backend.counted == [budget]


async def test_the_summary_rides_the_live_request_and_rebuilds_the_live_history() -> None:
    """Same system prompt, tools and effort as the ordinary call, the whole
    history, thinking closed by injection, and room for twice the target."""
    target = _summarizing_agent(summary_tail="full")
    _task, latest, _prior = _history(target)
    messages = target.messages
    backend = _CountingScript(
        responses=[_response(SUMMARY, completion_tokens=7), _response("DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    await _run_block(target, backend)

    summary_request, ordinary_request = backend.seen_requests
    assert summary_request.summary and not ordinary_request.summary
    assert summary_request.tools == ordinary_request.tools == ("python",)
    assert summary_request.reasoning_effort == ordinary_request.reasoning_effort == "high"
    assert summary_request.system_text == ordinary_request.system_text
    assert summary_request.response_injection == THINK_CLOSE
    assert summary_request.max_tokens == 2 * (WINDOW // RATIO)
    assert "PINNED TASK" in "\n".join(summary_request.element_texts)
    assert "STALE WORK" in "\n".join(summary_request.element_texts), "the whole history"
    assert INSTRUCTION in summary_request.user_texts[-1]
    assert "do NOT call any tools" in summary_request.user_texts[-1]
    assert f"{WINDOW // RATIO:,d}" in summary_request.user_texts[-1]

    assert target.messages is messages
    _keeps_the_status(target)
    assert backend.first_live_turn is not None
    assert target.messages[-1] is backend.first_live_turn
    assert "STALE WORK" not in "\n".join(element.to_string() for element in target.messages)
    assert LATEST in "\n".join(element.to_string() for element in target.messages)
    assert any(
        element is latest
        for message in target.messages
        if isinstance(message, ullm.conv.ResponseTurn)
        for element in message.elements
    )
    wrappers = [
        message
        for message in target.messages
        if isinstance(message, ullm.conv.UserPrompt) and SUMMARY in message.content.to_string()
    ]
    assert len(wrappers) == 1


@pytest.mark.parametrize(
    ("mode", "frame_api", "colour_legend"),
    [
        ("current_grid", "ergonomic", True),
        ("current_grid", "numeric", True),
        ("current_grid", "numeric", False),
        ("level_start", "ergonomic", True),
        ("level_start", "ergonomic", False),
        ("level_start", "numeric", False),
        ("current_grid", "ergonomic", False),
        ("", "numeric", True),
    ],
)
@pytest.mark.parametrize("retain_level_start", [False, True])
async def test_summary_restores_reference_images_and_counts_them_before_installing(
    mode: str,
    frame_api: Literal["ergonomic", "numeric"],
    colour_legend: bool,
    retain_level_start: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference images survive repeated summaries, later turns and level changes,
    while level-start mode keeps the original board as the current board changes."""
    target = _summarizing_agent(summary_tail="none")
    target.multimodal_context = mode
    target.frame_api = frame_api
    target.color_name_legend = colour_legend
    obs = _observation()
    target._record_observation(_context(_CountingScript()), obs)
    first = target._build_user_prompt_element(obs)
    if not retain_level_start:
        obs = _observation(index=2, colour=1)
        target._record_observation(_context(_CountingScript()), obs)
    status = first if retain_level_start else target._build_user_prompt_element(obs)
    target._turn_prompt = status
    target.messages = [first, ullm.conv.ResponseTurn(status="processed")] if status is not first else []
    target.messages.extend([status, ullm.conv.ResponseTurn(status="intermediate")])
    target._live_params = ullm.backend.GenericSamplingParameters(max_tokens=256)
    caption = target.prompts.color_name_legend_caption if colour_legend else target.prompts.palette_legend_caption
    expected_legend = bool(mode) and (colour_legend or frame_api == "numeric")

    def images(content: ullm.conv.Content) -> list[ullm.conv.ImageContent]:
        elements = content.elements if isinstance(content, ullm.conv.ListContent) else [content]
        return [element for element in elements if isinstance(element, ullm.conv.ImageContent)]

    original_images = images(first.content)
    legend = original_images[0] if expected_legend else None
    level_board = original_images[-1] if mode == "level_start" else None
    measured: list[int] = []
    original_content = status.content

    async def count(params: ullm.backend.GenericSamplingParameters, conversation: ullm.conv.Conversation) -> int:
        del params
        candidate = next(
            element
            for element in conversation.elements
            if isinstance(element, ullm.conv.UserPrompt) and "Current state:" in element.to_string()
        )
        # Token accounting must see the image before the live prompt is changed.
        assert status.content is original_content
        assert candidate.to_string().count(caption) == int(expected_legend)
        count = len(images(candidate.content))
        measured.append(count)
        return 500 + 256 * count

    monkeypatch.setattr(target, "_request_prompt_tokens", count)
    for iteration in range(5):
        if iteration >= 2:
            obs = _observation(level=int(iteration >= 3), index=iteration + 2, colour=iteration)
            target._record_observation(_context(_CountingScript()), obs)
            status = target._build_user_prompt_element(obs)
            target._turn_prompt = status
            target.messages.extend([status, ullm.conv.ResponseTurn(status="intermediate")])
            if iteration == 3 and mode == "level_start":
                next_board = images(status.content)[-1]
                assert level_board is not None and not np.array_equal(next_board.data, level_board.data)
                level_board = next_board
            if iteration != 3:
                assert caption not in status.to_string()
                if mode == "level_start":
                    assert images(status.content) == []
        before = images(status.content)
        original_content = status.content
        boards = [image for image in before if legend is None or not np.array_equal(image.data, legend.data)]
        if level_board is not None:
            boards = [level_board]
        dropped, tokens = await target._install_summary(SUMMARY, "", 256, record_pruned_round=lambda _: None)
        after = images(status.content)
        assert not dropped
        assert tokens == 500 + 256 * len(after)
        assert measured[-1] == len(after) == len(boards) + int(expected_legend)
        assert status.to_string().count(caption) == int(expected_legend)
        assert all(any(image is board for image in after) for board in boards)
        if legend is not None:
            assert np.array_equal(after[0].data, legend.data)


@dataclass
class _SummaryObserver:
    events: list[tuple[Any, ...]] = field(default_factory=list[tuple[Any, ...]])
    request_id: int = 0
    counts: list[int] = field(default_factory=list[int])

    def prepare_request(self, params: ullm.backend.GenericSamplingParameters) -> None:
        self.events.append(("prepare",))
        self.request_id += 1
        self.counts = []
        params.request_context.pop("observed", None)

    def prompt_measured(self, count: int, *, epoch: int, trigger: int, compacting: bool) -> None:
        self.events.append(("count", count, epoch, trigger, compacting))
        self.counts.append(count)

    def history_rewritten(self, *, epoch: int) -> None:
        self.events.append(("rewrite", epoch))

    def request_prepared(
        self,
        params: ullm.backend.GenericSamplingParameters,
        *,
        epoch: int,
        prompt_tokens: int,
        target: int,
    ) -> None:
        self.events.append(("ready", epoch, prompt_tokens, target))
        assert self.counts[-1] == prompt_tokens
        params.request_context["observed"] = {"id": self.request_id, "counts": list(self.counts)}


@pytest.mark.parametrize("summary_attempts", [0, 1, 2])
async def test_ordinary_observer_is_reprepared_before_post_summary_measurements(summary_attempts: int) -> None:
    target = _summarizing_agent()
    target.context_observer = _SummaryObserver()
    _history(target)
    responses = [_tool_response()] if summary_attempts == 2 else []
    if summary_attempts:
        responses.append(_response(SUMMARY))
    responses.append(_response("DONE"))
    backend = _CountingScript(
        responses=responses,
        ordinary_count=target._context_budget_tokens() + 1 if summary_attempts else 100,
        compacted_count=300,
        compacted_with_tail_count=WINDOW,
        summary_count=500,
    )

    await _run_block(target, backend)

    observed = [request.request_context["observed"] for request in backend.seen_requests]
    assert len(observed) == summary_attempts + 1
    assert [entry["id"] for entry in observed] == (list(range(2, summary_attempts + 3)) if summary_attempts else [1])
    ordinary_counts = observed[-1]["counts"]
    if summary_attempts:
        # Include both candidate counts during installation, before the final live counts.
        assert ordinary_counts == [WINDOW, 300, 300, 300]
    else:
        assert ordinary_counts == [100, 100]


async def test_the_register_injection_rides_the_summarize_arm_and_not_its_summary() -> None:
    """Both compaction arms arm the injection in the one ``before_round``
    hook: the summarize arm's ordinary request opens on the served text and
    counts as the game's first call, while the summary request it fired
    first opens on the end of thinking instead of the register."""
    target = _summarizing_agent()
    target.inject_text_on_first_prompt = "Okay, so"
    target.inject_text_on_later_prompts = "We need"
    _history(target)
    backend = _CountingScript(
        responses=[_response(SUMMARY, completion_tokens=7), _response("DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    await _run_block(target, backend)

    summary_request, ordinary_request = backend.seen_requests
    assert summary_request.response_injection == THINK_CLOSE
    assert ordinary_request.response_injection == "Okay, so"
    assert target.response_step == 1


async def test_a_later_compaction_summarizes_the_previous_wrapper() -> None:
    class _RepeatedCompactionScript(_CountingScript):
        """Grow again between summaries while each rebuilt request fits."""

        def _prompt_weight(self, conversation: ullm.conv.Conversation) -> int:
            text = conversation.to_string()
            if self._is_summary(conversation):
                return super()._prompt_weight(conversation)
            if "SUMMARY TWO" in text:
                return 500
            if "SUMMARY ONE" in text:
                first_ordinary_billing = self._call_index == 2 and bool(conversation.last_turn.elements)
                return 500 if self._call_index == 1 or first_ordinary_billing else self.ordinary_count
            return super()._prompt_weight(conversation)

    target = _summarizing_agent()
    _history(target)
    backend = _RepeatedCompactionScript(
        responses=[
            _response("SUMMARY ONE"),
            _response("FIRST DONE"),
            _response("SUMMARY TWO"),
            _response("SECOND DONE"),
        ],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    await _run_block(target, backend)
    await _run_block(target, backend)

    second_summary_request = backend.seen_requests[2]
    assert second_summary_request.summary
    assert "SUMMARY ONE" in "\n".join(second_summary_request.element_texts)
    durable = "\n".join(element.to_string() for element in target.messages)
    assert "SUMMARY TWO" in durable
    assert target.context_summaries == 2


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(_response("   ", completion_tokens=2), id="whitespace"),
        pytest.param(_response("", completion_tokens=2, stop_reason="max_tokens"), id="empty"),
        pytest.param(_tool_response(completion_tokens=2), id="tool-call"),
        pytest.param(_response("<tool_call>{}</tool_call>", completion_tokens=2), id="tool-markup"),
    ],
)
async def test_invalid_summary_gets_one_reseeded_retry(
    invalid: ullm.conv.LLMResponse, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And a tool call made in place of a summary never runs: the kernel is
    the live game's, and a call the retry discards would leave its variables
    and answers behind."""
    ran: list[str] = []
    monkeypatch.setattr(DuckPythonTool, "run", _spy_run(ran))
    target = _summarizing_agent()
    _history(target)
    backend = _CountingScript(
        responses=[invalid, _response(SUMMARY, completion_tokens=3), _response("DONE", completion_tokens=4)],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    await _run_block(target, backend)

    assert ran == [], "a summary reply's tool call reached the kernel"

    first, retry, ordinary = backend.seen_requests
    assert first.summary and retry.summary
    assert first.element_texts == retry.element_texts
    assert first.seed is not None
    assert retry.seed == first.seed + 1
    # The refused reply was served, so it is a step of a trace, and the retry
    # does not extend it: a declared rewrite. Undeclared, exact history stops
    # the game at the retry.
    assert retry.history_epoch > first.history_epoch
    assert ordinary.tools == ("python",)
    history = "\n".join(element.to_string() for element in target.messages)
    assert SUMMARY in history
    assert "INVALID SUMMARY TOOL CALL" not in history
    assert backend.total_usage.completion_tokens == 9


def _spy_run(ran: list[str]) -> Any:
    """A ``DuckPythonTool.run`` that records the code it was handed and runs nothing."""

    async def run(self: DuckPythonTool) -> None:
        ran.append(str(self.arguments.get("code")))

    return run


@dataclass
class _EmptySummaryOnceScript(_CountingScript):
    """The engine's reading of a reply that is empty after the injected end of
    thinking: its reasoning parser finds nothing on either side of the marker,
    and ullm refuses the reply as malformed before it is billed (the sweep's
    "no usable summary" crashes, twice in a row). The stub cannot produce that
    reply itself -- it opens every answer with the injection -- so the
    refusal is scripted."""

    empty_remaining: int = 1

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        if self._is_summary(conversation) and self.empty_remaining:
            # What the parser leaves of an empty reply after the marker; ullm's
            # own check then refuses it, as it does on the engine.
            self.empty_remaining -= 1
            element.final_content = ullm.conv.StringContent(text="")
            element.reasoning_content = ullm.conv.StringContent(text="")


async def test_genuinely_empty_summary_gets_one_reseeded_retry() -> None:
    target = _summarizing_agent()
    _history(target)
    backend = _EmptySummaryOnceScript(
        responses=[
            _response("EMPTIED BY THE PARSER", completion_tokens=7),
            _response(SUMMARY, completion_tokens=3),
            _response("DONE", completion_tokens=4),
        ],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    await _run_block(target, backend)

    assert [request.summary for request in backend.seen_requests] == [True, True, False]
    # ARC3's summary request keeps an empty reply (`keep_empty_reply`, so a
    # summary left in the reasoning can be read), so the emptied reply is
    # billed before the duck refuses it; ARC2's engine refused it unbilled.
    assert backend.total_usage.completion_tokens == 14
    assert backend.total_usage.num_requests == 3
    # The retry starts over: the same request, not the refused reply left in
    # front of it, and only the seed moves.
    first, retry, _ = backend.seen_requests
    assert retry.element_texts == first.element_texts
    assert "EMPTIED BY THE PARSER" not in "\n".join(retry.element_texts)
    assert retry.seed != first.seed
    # Kept and billed, the emptied reply was served: the retry is declared.
    assert retry.history_epoch > first.history_epoch


@pytest.mark.parametrize("retry", ["empty", "tool", "drift"])
async def test_viewer_closes_only_the_context_installed_after_a_summary_retry(tmp_path: Path, retry: str) -> None:
    """Rejected attempts from the real agent remain in the original context."""
    from platypus.viewer_data import load_game_full

    target = _summarizing_agent()
    _history(target)
    responses = [_response(SUMMARY), _response("DONE")]
    if retry != "drift":
        responses.insert(0, _tool_response() if retry == "tool" else _response("EMPTY"))
    backend_type = _EmptySummaryOnceScript if retry == "empty" else _CountingScript
    backend = backend_type(
        responses=responses,
        drift_remaining=int(retry == "drift"),
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )
    store = platypus.diagnostics.BlockStore(tmp_path / "diag_blocks")
    logger = platypus.diagnostics.SpanLogger(tmp_path / "games" / "retry" / "spans.jsonl", store)
    (tmp_path / "run_spans.jsonl").touch()
    ctx = _context(backend)
    ctx.diag = logger
    with closing(logger), logger.span("infra/game", "retry-game") as game:
        game.artifact("game_metadata", {"number_of_levels": 1}, codec="json")
        with logger.span("infra/move", "move 0"):
            await _turn_block(
                target,
                ctx,
                ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="CURRENT REQUEST")),
                remaining_calls=None,
            )

    payload = load_game_full(tmp_path, None, 0)
    sections = [section for event in payload["events"] for section in event.get("transcript_sections", [])]
    attempts = [section for section in sections if section["label"] == "SUMMARY ATTEMPT"]
    boundaries = [section for section in sections if "compaction" in section]
    assert len(attempts) == len(boundaries) == 1
    assert attempts[0]["context"] == boundaries[0]["context"] == 0
    assert "summary installed" not in attempts[0]["content"]
    assert boundaries[0]["compaction"]["number"] == 1
    assert len(payload["contexts"]) == 2
    assert payload["contexts"][0]["compaction"]["mode"] == "summarize"


async def test_reasoning_only_summary_is_accepted_and_billed() -> None:
    target = _summarizing_agent()
    _history(target)
    backend = _MalformedCountingScript(
        responses=[_reasoning_response("REASONING SUMMARY", completion_tokens=5), _response("DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
        wrapper_marker="REASONING SUMMARY",
    )

    await _run_block(target, backend)

    assert "REASONING SUMMARY" in "\n".join(element.to_string() for element in target.messages)
    assert backend.total_usage.completion_tokens == 6
    assert backend.total_usage.num_requests == 2


async def test_two_invalid_summaries_fail_without_mutating_durable_history() -> None:
    target = _summarizing_agent()
    task, latest, prior = _history(target)
    messages = target.messages
    prior_elements = prior.elements
    prior_element_ids = [id(element) for element in prior.elements]
    backend = _CountingScript(
        responses=[_response(" ", completion_tokens=2), _response("\t", completion_tokens=3)],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    with pytest.raises(RuntimeError, match="compaction|summary"):
        await _run_block(target, backend)

    assert target.messages is messages
    assert target.messages[0] is task
    assert target.messages[1] is prior
    assert prior.elements is prior_elements
    assert [id(element) for element in prior.elements] == prior_element_ids
    assert latest in prior.elements
    assert SUMMARY not in "\n".join(element.to_string() for element in target.messages)
    assert backend.seen_requests[0].element_texts == backend.seen_requests[1].element_texts
    assert backend.seen_requests[1].seed == backend.seen_requests[0].seed + 1  # type: ignore[operator]
    assert backend.total_usage.completion_tokens == 5


async def test_summary_token_bans_are_scoped_to_compaction_calls() -> None:
    target = _summarizing_agent(compaction_forbidden_token_ids=(248058,))
    _history(target)
    backend = _CountingScript(
        responses=[_response(SUMMARY), _response("DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    await _run_block(target, backend)

    summary_request, ordinary_request = backend.seen_requests
    assert summary_request.extra_params["logit_bias"] == {248058: -100.0}
    assert "logit_bias" not in ordinary_request.extra_params


async def test_recency_tail_is_dropped_when_the_rebuilt_request_is_still_too_large() -> None:
    target = _summarizing_agent()
    _task, _latest, _prior = _history(target)
    messages = target.messages
    backend = _CountingScript(
        responses=[_response(SUMMARY), _response("DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        compacted_with_tail_count=target._context_budget_tokens() + 1,
        summary_count=500,
    )

    await _run_block(target, backend)

    assert target.messages is messages
    _keeps_the_status(target)
    assert backend.first_live_turn is target.messages[-1]
    history = "\n".join(element.to_string() for element in target.messages)
    assert SUMMARY in history
    assert LATEST not in history
    assert LATEST not in "\n".join(backend.seen_requests[-1].element_texts)


async def test_one_call_allowance_can_be_spent_entirely_on_compaction() -> None:
    target = _summarizing_agent()
    _history(target)
    backend = _CountingScript(
        responses=[_response(SUMMARY), _response("NEXT DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    assert await _run_block(target, backend, remaining_calls=1) == 1

    assert [request.summary for request in backend.seen_requests] == [True]
    assert SUMMARY in "\n".join(element.to_string() for element in target.messages)
    _keeps_the_status(target)
    next_prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="NEXT REQUEST"))
    assert await _turn_block(target, _context(backend), next_prompt, remaining_calls=1) == 1
    target._ordinary_conversation(target.messages).check_constraints()


async def test_invalid_summary_spending_the_allowance_leaves_no_unsent_turn() -> None:
    target = _summarizing_agent()
    _history(target)
    backend = _CountingScript(
        responses=[_response(" ", completion_tokens=2), _response("NEXT DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    assert await _run_block(target, backend, remaining_calls=1) == 1

    assert target.context_summaries == 0
    assert "CURRENT REQUEST" not in "\n".join(element.to_string() for element in target.messages)
    assert not any(isinstance(element, ullm.conv.ResponseTurn) and not element.elements for element in target.messages)
    backend.ordinary_count = 100
    next_prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="NEXT REQUEST"))
    assert await _turn_block(target, _context(backend), next_prompt, remaining_calls=1) == 1
    target._ordinary_conversation(target.messages).check_constraints()


async def test_summary_usage_reduces_the_ordinary_request_budget() -> None:
    target = _summarizing_agent(profile_max_tokens=100, max_tokens_per_game=100)
    _history(target)
    backend = _CountingScript(
        responses=[_response(SUMMARY, completion_tokens=7), _response("DONE", completion_tokens=11)],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )
    ctx = _context(backend)
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="CURRENT REQUEST"))

    await _turn_block(target, ctx, prompt, remaining_calls=2)

    summary_request, ordinary_request = backend.seen_requests
    assert summary_request.max_tokens == 100
    assert ordinary_request.max_tokens == 93
    assert backend.total_usage.completion_tokens == 18
    assert backend.total_usage.num_requests == 2
    assert target._game_tokens_used(ctx) == 18


async def test_pre_summary_fit_uses_the_remaining_game_budget() -> None:
    target = _summarizing_agent(profile_max_tokens=256, max_tokens_per_game=100)
    _history(target)
    backend = _CountingScript(
        responses=[_response("DONE")],
        ordinary_count=3400,
    )

    await _run_block(target, backend)

    (ordinary_request,) = backend.seen_requests
    assert ordinary_request.tools == ("python",)
    assert ordinary_request.max_tokens == 100
    assert target.context_summaries == 0


async def test_post_summary_fit_uses_the_remaining_game_budget() -> None:
    target = _summarizing_agent(profile_max_tokens=256, max_tokens_per_game=100)
    _history(target)
    backend = _CountingScript(
        responses=[_response(SUMMARY, completion_tokens=90), _response("DONE")],
        ordinary_count=3500,
        compacted_count=3500,
        summary_count=500,
    )

    await _run_block(target, backend)

    summary_request, ordinary_request = backend.seen_requests
    assert summary_request.max_tokens == 100
    assert ordinary_request.max_tokens == 10


async def test_invalid_count_falls_back_to_the_bound() -> None:
    """A backend answering something that is not a count is treated as one
    that cannot count: the detached request is measured by the bound."""
    target = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", token_native_context=True)
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="system")),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="request"))],
        status="intermediate",
    )
    backend = _InvalidCountScript(responses=[_response("unused")], invalid_count=True)
    target._live_backend = backend
    target._live_params = ullm.backend.GenericSamplingParameters(max_tokens=64)
    target._live_conversation = conversation

    measured = await target._request_prompt_tokens(target._live_params, conversation)

    assert backend.counted == [True], "the backend was asked, and answered nonsense"
    assert measured == target._bounded_conversation_tokens(conversation)


async def test_summary_diagnostics_carry_the_post_summary_reading(monkeypatch: Any) -> None:
    class _Span:
        def __init__(self) -> None:
            self.values: list[dict[str, Any]] = []

        def kv(self, **values: Any) -> None:
            self.values.append(values)

        def event(self, _name: str, **_values: Any) -> None:
            pass

    # A backend that cannot count leaves every reading to the bound, so the
    # window has to clear the bound's fixed floor (the system prompt and tool
    # schema at their character count) and the stale work has to push the
    # ordinary request past the budget on that same ruler.
    target = _summarizing_agent(window=32768)
    _task, _latest, prior = _history(target)
    stale = prior.elements[0]
    assert isinstance(stale, ullm.conv.LLMResponse)
    stale.final_content = ullm.conv.StringContent(text="X" * 40_000)
    backend = _CountingScript(
        responses=[_response(SUMMARY), _response("DONE")],
        return_none=True,
    )
    span = _Span()
    monkeypatch.setattr(platypus.diagnostics, "current_span", lambda: span)

    await _run_block(target, backend)

    compaction = next(values for values in span.values if values.get("compaction_strategy") == "summarize")
    # One reading, on the decision ruler: the rebuilt request as the next
    # ordinary call would send it, under the budget the summary was made
    # to reach. There is no second ruler for it to be labelled with.
    assert set(compaction) & {"context_after_est", "context_after_exact"} == set()
    assert 0 < compaction["context_after"] <= target._context_budget_tokens()
    assert compaction["context_after"] < 40_000, "the stale work the summary replaced is not in the reading"


async def test_a_summary_declares_both_of_its_re_renders() -> None:
    """The summary request extends the served ids, so it goes out on the served
    epoch. The rebuilt history is the rewrite, and it is declared twice: before
    the install first counts it (the script refuses a count of it on the
    summary's epoch, as the engine did in 5 of 105 games of the mode's first
    run), and after, so the next ordinary call renders it afresh rather than
    failing to extend the summary's own served ids. So the epoch moves twice
    across one summary, both times after its request."""
    target = _summarizing_agent()
    _history(target)
    before = target.history_epoch
    backend = _CountingScript(
        responses=[_response(SUMMARY, completion_tokens=7), _response("DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )

    await _run_block(target, backend)

    assert target.context_summaries == 1
    assert target.history_epoch == before + 2, (before, target.history_epoch)
    summary_request, ordinary_request = backend.seen_requests
    assert summary_request.history_epoch == before, "the request is not a rewrite"
    assert ordinary_request.history_epoch == before + 2
    assert ordinary_request.turn is not summary_request.turn


# --------------------------------------------------------------------------
# the summary request is never reduced, and exceeds the agent's window
# --------------------------------------------------------------------------


class _RecordingSpan:
    def __init__(self) -> None:
        self.values: list[dict[str, Any]] = []

    def kv(self, **values: Any) -> None:
        self.values.append(values)

    def event(self, _name: str, **_values: Any) -> None:
        pass


def _summary_backend(target: DuckAgent, **kwargs: Any) -> _CountingScript:
    kwargs.setdefault("responses", [_response(SUMMARY, completion_tokens=7), _response("DONE")])
    return _CountingScript(
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
        **kwargs,
    )


async def test_the_summary_request_never_drops_history_to_fit() -> None:
    target = _summarizing_agent()
    task, latest, prior = _history(target)
    backend = _summary_backend(target, overflow_remaining=1)

    with pytest.raises(RuntimeError, match="never drops history"):
        await _run_block(target, backend)

    assert len(backend.seen_requests) == 1
    assert target.messages[0] is task
    assert target.messages[1] is prior
    assert latest in prior.elements
    assert "STALE WORK" in "\n".join(element.to_string() for element in target.messages)


async def test_the_summary_request_is_not_held_to_the_agents_window() -> None:
    """The request exceeds the window by design; the engine bounds it."""
    target = _summarizing_agent()
    _history(target)
    backend = _summary_backend(target, responses=[_response(SUMMARY, completion_tokens=50), _response("DONE")])
    backend.summary_count = WINDOW - 1

    await _run_block(target, backend)

    assert target.context_summaries == 1


async def test_summary_diagnostics_report_its_length(monkeypatch: Any) -> None:
    target = _summarizing_agent()
    _history(target)
    span = _RecordingSpan()
    monkeypatch.setattr(platypus.diagnostics, "current_span", lambda: span)

    await _run_block(target, _summary_backend(target))

    compaction = next(values for values in span.values if values.get("compacted"))
    assert compaction["compaction_strategy"] == "summarize"
    assert compaction["compaction_max_tokens"] == 2 * (WINDOW // RATIO)
    assert compaction["summary_tokens"] == 7
    assert compaction["summary_request_rerendered"] is False


async def test_a_drifted_summary_request_is_sent_afresh_once(monkeypatch: Any) -> None:
    """A request that does not extend the served ids falls back to a declared
    re-render -- once, and on the record."""
    target = _summarizing_agent()
    _history(target)
    before = target.history_epoch
    span = _RecordingSpan()
    monkeypatch.setattr(platypus.diagnostics, "current_span", lambda: span)
    backend = _summary_backend(target, drift_remaining=1)

    await _run_block(target, backend)

    drifted, resent, ordinary = backend.seen_requests
    assert drifted.element_texts == resent.element_texts
    assert not ordinary.summary
    # The re-render's declaration, then the install's two.
    assert backend.summary_epochs == [before, before + 1]
    assert target.history_epoch == before + 3
    compaction = next(values for values in span.values if values.get("compacted"))
    assert compaction["summary_request_rerendered"] is True


async def test_a_second_drift_is_not_papered_over() -> None:
    target = _summarizing_agent()
    _history(target)
    backend = _summary_backend(target, drift_remaining=2)

    with pytest.raises(ullm.continuation.HistoryDriftError):
        await _run_block(target, backend)


# --------------------------------------------------------------------------
# summary_thinking and summary_tail
# --------------------------------------------------------------------------


async def test_a_thinking_summary_opens_on_the_generation_prompt_with_room_to_reason() -> None:
    target = _summarizing_agent()
    target.summary_thinking = True
    _history(target)
    backend = _summary_backend(target)

    await _run_block(target, backend)

    summary_request, ordinary_request = backend.seen_requests
    assert summary_request.summary
    assert summary_request.response_injection is None, "neither the register nor the end of thinking"
    assert summary_request.max_tokens == 6 * (WINDOW // RATIO)
    assert summary_request.tools == ordinary_request.tools == ("python",)
    assert summary_request.reasoning_effort == ordinary_request.reasoning_effort == "high"
    assert SUMMARY in "\n".join(element.to_string() for element in target.messages)


async def test_summary_tail_none_keeps_only_the_task_and_the_summary() -> None:
    target = _summarizing_agent()
    target.summary_tail = "none"
    _task, latest, _prior = _history(target)
    backend = _summary_backend(target)

    await _run_block(target, backend)

    history = "\n".join(element.to_string() for element in target.messages)
    _keeps_the_status(target)
    assert SUMMARY in history
    assert LATEST not in history
    assert not any(
        element is latest
        for message in target.messages
        if isinstance(message, ullm.conv.ResponseTurn)
        for element in message.elements
    )


async def test_summary_tail_without_reasoning_keeps_a_reasoning_free_copy() -> None:
    target = _summarizing_agent()
    target.summary_tail = "without_reasoning"
    task = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="PINNED TASK"))
    stale = _processed_response("STALE WORK")
    latest, result = _tool_round(LATEST, reasoning="LATEST REASONING")
    # What the engine sampled, which a real backend renders a past response
    # from; the stub never reads it.
    latest._model_contents = SimpleNamespace(outputs=[SimpleNamespace(text=f"LATEST REASONING\n</think>\n\n{LATEST}")])
    target.messages = [task, ullm.conv.ResponseTurn(elements=[stale, latest, result], status="processed")]
    backend = _summary_backend(target)

    await _run_block(target, backend)

    history = "\n".join(element.to_string() for element in target.messages)
    assert LATEST in history
    assert "LATEST REASONING" not in history
    assert "STALE WORK" not in history
    (holder,) = [
        message
        for message in target.messages
        if isinstance(message, ullm.conv.ResponseTurn)
        and any(
            isinstance(element, ullm.conv.LLMResponse) and LATEST in element.to_string() for element in message.elements
        )
    ]
    # The response leads its round, its tool result after it.
    kept = holder.elements[0]
    assert kept is not latest, "a copy, not the served object"
    assert latest.reasoning_content.to_string() == "LATEST REASONING", "the original is untouched"

    # The rebuilt history as vLLM hands it to the chat template, which
    # renders a past response from its engine output: a copy without it is
    # refused.
    vllmx = pytest.importorskip("ullm.backends.vllmx")

    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
    )
    conversation.elements.extend(target.messages[: target.messages.index(holder) + 1])
    messages, _ = _vllm_backend(vllmx)._history_messages(holder, conversation, "all")
    # The kept response, its tool result rendered after it.
    assert messages[-2] == {"role": "assistant", "content": LATEST}


class _ThinkParser:
    """vLLM's qwen3 reasoning parser as the render uses it (ullm's
    `test_vllmx_history_render.py`)."""

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
        del request
        if "</think>" not in text:
            return text, ""
        reasoning, content = text.split("</think>", 1)
        return reasoning, content


def _vllm_backend(vllmx: Any) -> Any:
    """A VLLM with only the fields ``_history_messages`` reads."""
    backend = vllmx.VLLM.__new__(vllmx.VLLM)
    backend.__dict__.update(
        reasoning_effort_prompts=None,
        chat_template_kwargs={},
        native_reasoning_effort=False,
        _reasoning_parser=_ThinkParser(),
        _reasoning_lock=threading.Lock(),
        _reasoning_parser_no_thinking=None,
        _tool_parser=object(),
    )
    return backend


async def test_summary_tail_full_keeps_the_reasoning() -> None:
    target = _summarizing_agent(summary_tail="full")
    task = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="PINNED TASK"))
    latest, result = _tool_round(LATEST, reasoning="LATEST REASONING")
    target.messages = [
        task,
        ullm.conv.ResponseTurn(elements=[_processed_response("STALE WORK"), latest, result], status="processed"),
    ]

    await _run_block(target, _summary_backend(target))

    assert "LATEST REASONING" in "\n".join(element.to_string() for element in target.messages)


def test_an_unknown_summary_tail_is_refused() -> None:
    with pytest.raises(ValueError, match="summary_tail"):
        DuckAgent(summary_tail="half")  # type: ignore[arg-type]


@dataclass
class _ClosingScript(_CountingScript):
    """Records each thread close with how many requests came before it."""

    closes: list[tuple[str, int]] = field(default_factory=lambda: list[tuple[str, int]](), init=False)

    async def close_conversation(self, conversation_id: str, run_id: str | None = None) -> None:
        self.closes.append((conversation_id, len(self.seen_requests)))


async def test_a_summary_closes_the_served_thread_before_the_next_request() -> None:
    """The summary rewrites the history the last request was served on: the
    engine hears that thread is over after the summary request, which extends
    it, and before the ordinary request opens the new epoch."""
    target = _summarizing_agent()
    _history(target)
    vars(target)["_thread_epoch"] = target.history_epoch  # served at this epoch
    backend = _ClosingScript(
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
        responses=[_response(SUMMARY, completion_tokens=7), _response("DONE")],
    )

    await _run_block(target, backend)

    assert [request.summary for request in backend.seen_requests] == [True, False]
    assert backend.closes == [(target._conversation_id(), 1)]


async def test_a_round_that_rewrites_nothing_closes_nothing() -> None:
    target = _summarizing_agent()
    _history(target)
    vars(target)["_thread_epoch"] = target.history_epoch
    backend = _ClosingScript(responses=[_response("DONE")])

    await _run_block(target, backend)

    assert [request.summary for request in backend.seen_requests] == [False]
    assert backend.closes == []


class _MalformedSummaryOnceScript(_CountingScript):
    """A summary reply the backend refuses as malformed after producing it --
    a parser failure, which ``keep_empty_reply`` does not lift -- once."""

    malformed_remaining: int = 1

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        if self._is_summary(conversation) and self.malformed_remaining:
            self.malformed_remaining -= 1
            raise ullm.backend.MalformedOutputError("scripted parser refusal")


def _charge_recorder(target: DuckAgent, charges: list[int]) -> Any:
    """``_charge_budget`` recording what each charge billed the game."""
    charge_budget = target._charge_budget

    def charge(ctx: Any, started_on: int, started_used: int, game_tokens_before: int) -> None:
        charges.append(target._game_tokens_used(ctx) - game_tokens_before)
        charge_budget(ctx, started_on, started_used, game_tokens_before)

    return charge


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(_response(" ", completion_tokens=7), id="empty"),
        pytest.param(_response("<tool_call>{}</tool_call>", completion_tokens=7), id="tool-markup"),
        pytest.param(_tool_response(completion_tokens=7), id="tool-call"),
    ],
)
async def test_retained_invalid_summary_gives_up_without_retry(
    invalid: ullm.conv.LLMResponse, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _summarizing_agent()
    target.preserve_failed_draws = True
    target.malformed_retries = 0
    _history(target)
    messages = target.messages
    epoch = target.history_epoch
    observed: list[ullm.conv.LLMResponse] = []
    history_at_response: list[ullm.conv.ConversationElement] = []
    checked: list[ullm.conv.LLMResponse] = []
    charges: list[int] = []
    check_usage = target._check_one_usage

    def observe(response: ullm.conv.LLMResponse) -> None:
        observed.append(response)
        history_at_response.extend(deepcopy(target.messages))

    def check(measured: int, response: ullm.conv.LLMResponse, *, enforce_window: bool = True) -> int:
        checked.append(response)
        return check_usage(measured, response, enforce_window=enforce_window)

    target.response_observer = observe
    observer = _SummaryObserver()
    target.context_observer = observer
    monkeypatch.setattr(target, "_check_one_usage", check)
    monkeypatch.setattr(target, "_charge_budget", _charge_recorder(target, charges))
    backend = _summary_backend(target, responses=[invalid])

    with pytest.raises(GiveUpError, match="no usable summary.*retained draw, no retry"):
        await _run_block(target, backend)

    (request,) = backend.seen_requests
    assert request.summary
    assert target._live_params is not None
    assert request.seed == target._live_params.seed
    assert target.messages is messages
    assert target.messages == history_at_response
    assert target.history_epoch == epoch
    assert not any(event[0] == "rewrite" for event in observer.events)
    assert len(observed) == 1
    assert checked == observed
    assert observed[0] in request.turn.elements
    assert target.context_summaries == 0
    assert charges == [7]
    assert backend.total_usage.completion_tokens == 7


async def test_retention_does_not_change_a_valid_summary() -> None:
    outcomes: list[tuple[Any, ...]] = []
    for retain in (False, True):
        target = _summarizing_agent()
        target.preserve_failed_draws = retain
        target.malformed_retries = 0
        _history(target)
        observer = _SummaryObserver()
        target.context_observer = observer
        backend = _summary_backend(target)

        await _run_block(target, backend)

        assert target.history_epoch == 2
        assert target.context_summaries == 1
        outcomes.append(
            (
                [message.to_string() for message in target.messages],
                target.history_epoch,
                observer.events,
                [request.element_texts for request in backend.seen_requests],
            )
        )
    assert outcomes[0] == outcomes[1]


async def test_retained_summary_drift_propagates_without_rerender() -> None:
    target = _summarizing_agent()
    target.preserve_failed_draws = True
    target.malformed_retries = 0
    _history(target)
    epoch = target.history_epoch
    observer = _SummaryObserver()
    target.context_observer = observer
    backend = _summary_backend(target, drift_remaining=1)

    with pytest.raises(ullm.continuation.HistoryDriftError, match="scripted drift"):
        await _run_block(target, backend)

    (request,) = backend.seen_requests
    assert request.summary
    assert target.history_epoch == epoch
    assert tuple(element.to_string() for element in target.messages[:-1]) == request.element_texts[:-2]
    tail = target.messages[-1]
    assert isinstance(tail, ullm.conv.ResponseTurn) and not tail.elements
    assert not any(event[0] == "rewrite" for event in observer.events)


@pytest.mark.parametrize("accept_malformed_output", [False, True])
async def test_retained_empty_summary_gives_up_without_rollback(
    monkeypatch: pytest.MonkeyPatch,
    accept_malformed_output: bool,
) -> None:
    """The backend's policy does not decide it here: the summary request keeps
    an empty reply either way, so the draw is observed, billed and refused as
    no usable summary."""
    target = _summarizing_agent()
    target.preserve_failed_draws = True
    target.malformed_retries = 0
    _history(target)
    messages = target.messages
    epoch = target.history_epoch
    observed: list[ullm.conv.LLMResponse] = []
    checked: list[ullm.conv.LLMResponse] = []
    charges: list[int] = []
    check_usage = target._check_one_usage

    def check(measured: int, response: ullm.conv.LLMResponse, *, enforce_window: bool = True) -> int:
        checked.append(response)
        return check_usage(measured, response, enforce_window=enforce_window)

    target.response_observer = observed.append
    monkeypatch.setattr(target, "_check_one_usage", check)
    monkeypatch.setattr(target, "_charge_budget", _charge_recorder(target, charges))
    backend = _EmptySummaryOnceScript(
        responses=[_response("EMPTIED BY THE PARSER", completion_tokens=7)],
        accept_malformed_output=accept_malformed_output,
        ordinary_count=target._context_budget_tokens() + 1,
        summary_count=500,
    )

    with pytest.raises(GiveUpError, match="no usable summary.*retained draw, no retry") as caught:
        await _run_block(target, backend)

    assert caught.value.__cause__ is None
    (request,) = backend.seen_requests
    assert request.summary
    assert target.messages is messages
    assert tuple(element.to_string() for element in target.messages[:-1]) == request.element_texts[:-2]
    tail = target.messages[-1]
    assert isinstance(tail, ullm.conv.ResponseTurn) and not tail.elements
    assert target.history_epoch == epoch
    assert len(checked) == 1
    assert request.turn.elements == checked
    assert observed == checked
    assert charges == [7]
    assert backend.total_usage.completion_tokens == 7


async def test_retained_malformed_summary_gives_up_without_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply the backend refuses is left where it was drawn, checked against
    its bound, and the game gives up on it: no retry and no rewrite. It never
    reached ullm's response callback or usage meter."""
    target = _summarizing_agent()
    target.preserve_failed_draws = True
    target.malformed_retries = 0
    _history(target)
    messages = target.messages
    epoch = target.history_epoch
    observed: list[ullm.conv.LLMResponse] = []
    checked: list[ullm.conv.LLMResponse] = []
    charges: list[int] = []
    check_usage = target._check_one_usage

    def check(measured: int, response: ullm.conv.LLMResponse, *, enforce_window: bool = True) -> int:
        checked.append(response)
        return check_usage(measured, response, enforce_window=enforce_window)

    target.response_observer = observed.append
    monkeypatch.setattr(target, "_check_one_usage", check)
    monkeypatch.setattr(target, "_charge_budget", _charge_recorder(target, charges))
    backend = _MalformedSummaryOnceScript(
        responses=[_response(SUMMARY, completion_tokens=7)],
        ordinary_count=target._context_budget_tokens() + 1,
        summary_count=500,
    )

    with pytest.raises(GiveUpError, match="malformed summary.*retained draw, no retry") as caught:
        await _run_block(target, backend)

    assert isinstance(caught.value.__cause__, ullm.backend.MalformedOutputError)
    (request,) = backend.seen_requests
    assert request.summary
    assert target.messages is messages
    assert tuple(element.to_string() for element in target.messages[:-1]) == request.element_texts[:-2]
    tail = target.messages[-1]
    assert isinstance(tail, ullm.conv.ResponseTurn) and not tail.elements
    assert target.history_epoch == epoch
    assert len(checked) == 1
    assert observed == []
    assert target.context_summaries == 0
