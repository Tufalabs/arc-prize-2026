"""LLM-agent contract surface (requirements.md §1, §4).

``LLMAgent`` extends ``Agent`` with ``llm_config``; ``LLMMoveContext``
adds the shared backend. The backend topology is fixed by the solver
(``platypus.llm_solver``): the config holds a *factory*, the host builds
the real backend from it and always serves it over a Unix socket, and
every game — in-process or subprocess — talks to a per-game
``ullm.backends.proxy.ProxyBackend`` client. Agents construct ullm's
turn-runner per use over ``ctx.backend`` (there is no bespoke chat
client); sampling comes from named profiles with per-call seeds derived
from the per-game RNG (U8) — agents never pick seeds.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import random
import traceback
import weakref
from abc import abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence, cast

import numpy as np
import taaf.game
import ullm.backend
import ullm.capture
import ullm.conv
import ullm.factory
import ullm.introspect
import ullm.manager
import ullm.tool

from platypus.agent import Agent
from platypus.context import HistoryView, MoveContext, no_yield
from platypus.contract import Action, GameInfo, GiveUp, Move, Observation
from platypus.diagnostics import BlockStore, Span, SpanLogger
from platypus.slots import SlotView
from platypus.status import NULL_STATUS, StatusHandle

_ARC_PALETTE = np.array(
    [[round(channel * 255) for channel in taaf.game.ARC_COLORS[i]] for i in range(16)], dtype=np.uint8
)

_ACTIVE_DIAGNOSTIC_TURN: ContextVar[tuple[object, object] | None] = ContextVar(
    "platypus_active_diagnostic_turn", default=None
)


def frame_image(frame: taaf.game.Frame, *, scale: int = 1, alt_text: str = "") -> ullm.conv.ImageContent:
    """Render a game frame as RGB ``ImageContent`` via the ARC palette —
    the one frame→pixels mapping multimodal agents share, so two agents
    showing the model "the board" show identical pixels. ``scale``
    nearest-upscales (VL models resolve tiny grids poorly; pixel-exact,
    so replay verification is unaffected by the choice)."""
    rgb = _ARC_PALETTE[frame.data]
    if scale != 1:
        rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)
    return ullm.conv.ImageContent(data=rgb, alt_text=alt_text)


@dataclass
class LLMConfig:
    """LLM knobs; lives ON the agent (§1), read by ``LLMAgentSolver`` from
    the template. A plain mutable dataclass, like ``RunConfig`` — edit
    while composing, before the run starts. ``backend_factory`` is the
    picklable spec — the solver starts a copy of it host-side, never the
    template's own instance, so the template and the host's per-game agent
    copies keep carrying an unstarted (picklable) spec. The copy a game
    CHILD receives carries none (``LLMAgentSolver._child_agent``): the
    child reaches inference through its proxy client, and unpickling the
    spec would import the engine into every game.

    ``sampling_profiles``: named profiles (U8). Calls select a profile by
    name — the turn-runner surface offers no per-call override knobs, so
    variation is normally another profile. Profiles must not pin ``seed``
    — per-call seeds are derived from the per-game RNG.
    """

    backend_factory: ullm.factory.BackendFactory | None = None
    sampling_profiles: Mapping[str, ullm.backend.SamplingParameters] = field(
        default_factory=lambda: dict[str, ullm.backend.SamplingParameters]()
    )
    # Capture (U3, reference generation only): record every LLM call, one
    # stream per (game, pass) COPY — each pass is its own GameRun, so each
    # gets its own `games/<run_id>/capture/` under the run's job_dir.
    capture: bool = False
    # Replay (U4): a directory holding one `<game_id>/` subdir of recorded
    # traffic per game. When set, no backend_factory/socket is used — each
    # game gets its own ReplayBackend over its recorded stream. Keyed by
    # game_id, so n_passes > 1 is NOT yet supported here (every pass copy
    # would replay the same stream and the off-seed passes would mismatch
    # loudly; reference COLLECTION refuses duplicates earlier anyway —
    # the designed extension is (game_id, occurrence-ordinal) keying).
    replay_dir: str | None = None
    # Traces (U5): write one taaf R5 training trace per game run. Requires a
    # trace-capable host backend (in-process vLLM engine, or OpenAIServer
    # against a vLLM server with streaming=False) — anything else is a loud
    # error at setup, never a silent skip. The trace lands on the game's
    # trace_root (Benchmark.trace_root, default job_dir/traces).
    traces: bool = False
    # What those traces store (taaf R5.39): "full" both representations, or
    # "ids" the exact side alone -- token ids, images and sampling, with no
    # messages or output text -- for a consumer that replays the engine's own
    # traffic (ullm.traffic). An ids-only step costs the host no re-render.
    trace_content: Literal["full", "ids"] = "full"
    # Smoke-test the host backend once it is up, before any game runs, and
    # fail at setup rather than after a run's worth of empty moves. Off by
    # default: it costs a live call, and a replay or proxy backend has
    # nothing useful to prove.
    #
    # Worth the call because the failures it catches are total and silent:
    # a backend that cannot serve a second turn, a tool call, the per-turn
    # cap or a cache hit takes every game down on its FIRST turn with zero
    # actions and zero tokens, and the harness still exits COMPLETED. The
    # matrix tests the backend, not the agent: its rows run at low effort,
    # and the agent's own request configuration (its reasoning_effort, say)
    # is that configuration's to get right.
    verify_backend: bool = False
    # Exact history (ullm.continuation, U5). A capable host backend — the
    # in-process vLLM engine, or OpenAIServer against a vLLM server (probed
    # at setup) with streaming=False — always extends the served history by
    # the sampled ids instead of re-rendering it, so a game's steps nest as
    # one token sequence. With this set, that becomes a promise: the agent
    # declares every rewrite of sent history
    # (``Conversation.mark_history_rewritten``), a call that cannot be
    # continued without one raises instead of rendering afresh, the trace
    # collector refuses a non-nesting step within an epoch, and a backend
    # that cannot continue at all is a loud error at setup. Where the
    # promise cannot hold — hosted APIs, streaming, replay, images over a
    # server topology, send_back_reasoning other than "all" — see "Token
    # continuation" in packages/ullm/design.md; leave this off there.
    exact_history: bool = False


@dataclass
class LLMMoveContext(MoveContext):
    """Context for ``LLMAgent`` moves: the base services plus the shared
    LLM backend (§4). ``backend`` is this game's own proxy client to the
    host-side backend — per-game accounting lives on it."""

    backend: ullm.backend.Backend = field(kw_only=True)


@dataclass(kw_only=True)
class LLMAgent(Agent):
    """Agent with an LLM config. State stays pure (A1): the backend is
    never stored — it arrives per call on the context."""

    llm_config: LLMConfig = field(default_factory=LLMConfig)

    def _validate_host_backend(self, backend: ullm.backend.Backend) -> None:
        """Validate agent requirements after the host's capability probes."""

    async def run_on_host(self, backend: ullm.backend.Backend, job_dir: Path | None) -> bool:
        """Take the run over on the host, in place of its games, once the
        backend is up; return whether it did.

        The default declines and the solver hosts the games. An agent that
        measures the engine rather than playing (the traffic replay) takes
        over here: the engine has been started, verified and gated exactly as
        a game run's would be, and the games stay unstarted.
        """
        del backend, job_dir
        return False

    # Deliberate covariant narrowing (§1): LLMAgents are always hosted by
    # LLMAgentSolver, which guarantees an LLMMoveContext — redeclared here
    # once so every subclass hook is typed against the LLM context.
    @abstractmethod
    async def _do_move(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, ctx: LLMMoveContext, obs: Observation
    ) -> Action | GiveUp | None:
        """The agent's move, over the LLM context. Re-declared here so every
        subclass hook is typed against `LLMMoveContext`."""
        ...

    def resolve_sampling(self, ctx: LLMMoveContext, profile: str) -> ullm.backend.GenericSamplingParameters:
        """Resolve a named sampling profile (U8): a fresh copy with the
        per-call seed stamped from ``ctx.rng``. Unknown names fail loudly
        on first use (there is no declaration mechanism)."""
        profiles = self.llm_config.sampling_profiles
        if profile not in profiles:
            raise KeyError(f"unknown sampling profile {profile!r}; llm_config defines {sorted(profiles)}")
        params = copy.deepcopy(profiles[profile])
        if not isinstance(params, ullm.backend.GenericSamplingParameters):
            raise TypeError(
                f"sampling profile {profile!r} is {type(params).__name__}; "
                "only GenericSamplingParameters profiles are supported"
            )
        if params.seed is not None:
            raise ValueError(
                f"sampling profile {profile!r} pins seed={params.seed}; "
                "per-call seeds are derived from the per-game RNG (U8) — leave seed unset"
            )
        params.seed = ctx.rng.getrandbits(31)
        return params

    def prepare_for_backend(self) -> None:
        """Settle configuration that depends on other configuration, once,
        immediately before the backend is built.

        The solver calls this at the last moment an engine argument can still
        be changed: after every override the caller made, before the factory
        spec is copied and started. A no-op today — every value is final at
        construction. It exists as the seam for settings DERIVED from others
        (the engine window from the agent's window and reply reserve, the
        sequence cap from the harness concurrency), which must resolve here
        rather than in a constructor if they are to see a caller's overrides.

        Idempotent by contract: a caller may run it to inspect the settled
        configuration, and the solver will run it again.
        """

    def new_turn_manager(
        self,
        ctx: LLMMoveContext,
        *,
        profile: str,
        system_prompt: str,
        tools: list[ullm.tool.Tool] | None = None,
        max_tool_rounds: int | None = None,
    ) -> ullm.manager.TurnManager:
        """The per-use turn-runner (§4): a fresh Conversation +
        ResponseManager + TurnManager stack over ``ctx.backend`` with the
        named profile resolved. Append a ``UserPrompt`` to
        ``.conversation.elements`` and ``await .run()``; the manager-layer
        defenses (malformed retry, prompt-too-long, truncation detection)
        ride along — and so does diagnostics: the returned manager is a
        :class:`DiagnosticTurnManager` recording the turn as an
        ``llm/turn`` span on ``ctx.diag`` (a no-op on the null stream)."""
        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(
                content=ullm.conv.StringContent(text=system_prompt),
                tools=tools if tools is not None else [],
            )
        )
        response_manager = ullm.manager.ResponseManager(
            backend=ctx.backend, sampling_params=self.resolve_sampling(ctx, profile)
        )
        if max_tool_rounds is None:
            return DiagnosticTurnManager(
                response_manager=response_manager, conversation=conversation, diag=ctx.diag, label=profile
            )
        return DiagnosticTurnManager(
            response_manager=response_manager,
            conversation=conversation,
            max_tool_rounds=max_tool_rounds,
            diag=ctx.diag,
            label=profile,
        )


def _remap_block_refs(structure: Any, mapping: Mapping[str, str]) -> Any:
    """Rewrite capture-format ``{"$block": h}`` references onto the
    diagnostics block-store keys they were stored under."""
    if isinstance(structure, dict):
        mapping_dict = cast("dict[str, Any]", structure)
        out: dict[str, Any] = {key: _remap_block_refs(value, mapping) for key, value in mapping_dict.items()}
        if "$block" in out:
            out["$block"] = mapping[mapping_dict["$block"]]
        if "block" in out and mapping_dict.get("c") == "image":
            out["block"] = mapping[mapping_dict["block"]]
        return out
    if isinstance(structure, list):
        return [_remap_block_refs(item, mapping) for item in cast("list[Any]", structure)]
    return structure


def serialize_conversation_blocks(conversation: ullm.conv.Conversation, block_store: BlockStore) -> Any:
    """Serialize a whole conversation (capture format, bookkeeping
    included) into the diagnostics block store: large text/images become
    dedup'd blocks, the returned structure references them by store key.
    Shared by the turn-close artifact and the live-conversation writer, so
    both reference the same blocks. Serialization is synchronous — call it
    on the event loop (no concurrent mutation), then do file I/O off it."""
    blocks: dict[str, Any] = {}
    structure = ullm.capture.serialize_conversation(conversation, blocks)
    stored = {h: block_store.store(payload, "json") for h, payload in blocks.items()}
    return _remap_block_refs(structure, stored)


@dataclass
class DiscardedRound:
    """One generated round removed before its diagnostic turn closed.

    ``reason`` distinguishes context compaction from a response retried after
    a per-turn cap lifted. ``available_until_round`` applies to context
    compaction: the number of model-visible response rounds that existed when
    this oldest round left the prompt.
    """

    elements: list[ullm.conv.ConversationElement]
    reason: str
    available_until_round: int | None = None


@dataclass
class DiagnosticTurnManager(ullm.manager.TurnManager):
    """``TurnManager`` wrapped in an ``llm/turn`` span.

    The span opens live around the turn (a crash mid-turn closes it with
    outcome ``error`` and whatever completed is recorded), and at close
    carries one ``llm/response`` event per observed response, the response
    usage as span self totals, one ``tool/result`` event per model-visible
    result, one ``kernel/exec`` event per Python-tool execution drained from
    the turn's tools, and the surviving conversation as a block-deduped
    ``conversation`` artifact (capture's canonical serialization,
    bookkeeping included).

    Generated rounds removed before close still contribute events and usage;
    their content and removal metadata live in ``discarded_rounds``. Discard
    serialization is deferred to the best-effort turn-close recorder, so the
    compaction path performs no diagnostics I/O. With the null stream this is
    exactly a ``TurnManager``."""

    diag: SpanLogger = field(default_factory=SpanLogger)
    label: str = ""
    _recorded_elements: list[ullm.conv.ConversationElement] = field(
        default_factory=lambda: list[ullm.conv.ConversationElement](), init=False, repr=False
    )
    _discarded_rounds: list[DiscardedRound] = field(
        default_factory=lambda: list[DiscardedRound](), init=False, repr=False
    )
    # The conversation's history epoch when each response was observed: a
    # call continues the one before it exactly when both share an epoch
    # (``Conversation.mark_history_rewritten``). Taken at observation, not at
    # close, since a summary mid-turn moves the epoch on.
    _response_epochs: dict[int, int] = field(default_factory=lambda: dict[int, int](), init=False, repr=False)
    _observation_generation: object | None = field(default=None, init=False, repr=False)

    def _observe_turn_elements(self, elements: Sequence[ullm.conv.ConversationElement]) -> None:
        """Keep this turn's appended elements in execution order until close,
        and check what each response re-prefilled."""
        self._check_continuation_prefill(elements)
        if not self.diag.active:
            super()._observe_turn_elements(elements)
        elif self._observation_generation is not None:
            self._note_response_epochs(elements)
            self._recorded_elements.extend(elements)

    def _check_continuation_prefill(self, elements: Sequence[ullm.conv.ConversationElement]) -> None:
        """``check_continuation_prefill`` for each response, named by game and label."""
        game = self.diag.stream_dir.name if self.diag.stream_dir is not None else None
        where = f"{game} {self.label}" if game else self.label
        for element in elements:
            if isinstance(element, ullm.conv.LLMResponse):
                check_continuation_prefill(self.conversation, element, where=where)

    def _note_response_epochs(self, elements: Sequence[ullm.conv.ConversationElement]) -> None:
        """Remember the conversation's history epoch for each response
        observed: a nested request (a summary's) extends the same history, so
        it shares the epoch the rewrite that follows it moves on from."""
        for element in elements:
            if isinstance(element, ullm.conv.LLMResponse):
                self._response_epochs[id(element)] = self.conversation.history_epoch

    def _observe_discarded_round(self, elements: Sequence[ullm.conv.ConversationElement], *, reason: str) -> None:
        """Remember a manager-owned discard without doing diagnostics I/O."""
        if not self.diag.active:
            super()._observe_discarded_round(elements, reason=reason)
        elif self._observation_generation is not None:
            self._remember_discarded_round(elements, reason=reason)

    def record_context_pruned_round(self, elements: Sequence[ullm.conv.ConversationElement]) -> None:
        """Remember one oldest round Duck removed from the live prompt.

        This method is active only while this manager is running with a real
        diagnostics stream, so a retained callback or an out-of-turn trim is
        inert rather than leaking content into another span.
        """
        active = _ACTIVE_DIAGNOSTIC_TURN.get()
        if (
            not self.diag.active
            or active is None
            or active[0] is not self
            or active[1] is not self._observation_generation
        ):
            return
        turn = self.conversation.elements[-1] if self.conversation.elements else None
        visible_responses = (
            sum(isinstance(element, ullm.conv.LLMResponse) for element in turn.elements)
            if isinstance(turn, ullm.conv.ResponseTurn)
            else 0
        )
        visible_responses += sum(
            sum(isinstance(element, ullm.conv.LLMResponse) for element in record.elements)
            for record in self._discarded_rounds
            if record.reason == "context_compaction"
        )
        self._remember_discarded_round(
            elements,
            reason="context_compaction",
            available_until_round=visible_responses,
        )

    def _remember_discarded_round(
        self,
        elements: Sequence[ullm.conv.ConversationElement],
        *,
        reason: str,
        available_until_round: int | None = None,
    ) -> None:
        """Retain a removed round for best-effort recording at turn close."""
        if not self.diag.active or self._observation_generation is None:
            return
        self._discarded_rounds.append(
            DiscardedRound(
                elements=list(elements),
                reason=reason,
                available_until_round=available_until_round,
            )
        )

    async def _run(self) -> None:
        """Run the turn, publishing live snapshots alongside it when
        diagnostics are on. Falls through to the plain turn otherwise, so the
        null stream costs nothing."""
        if not self.diag.active:
            await super()._run()
            return
        import platypus.live

        self._recorded_elements.clear()
        self._discarded_rounds.clear()
        self._response_epochs.clear()
        game_dir = self.diag.stream_dir
        with self.diag.span("llm/turn", self.label) as span:
            writer: asyncio.Task[None] | None = None
            if game_dir is not None:
                # Spawn the on-demand live-snapshot writer for this turn:
                # it only writes while a viewer is asking (job-dir files,
                # so live viewing reaches across hosts over NFS).
                writer = platypus.live.start_live_writer(game_dir, span_id=span.span_id, label=self.label, manager=self)
            try:
                generation = object()
                self._observation_generation = generation

                def observe_nested_elements(elements: Sequence[ullm.conv.ConversationElement]) -> None:
                    if self._observation_generation is generation:
                        self._check_continuation_prefill(elements)
                        self._note_response_epochs(elements)
                        self._recorded_elements.extend(elements)

                def observe_nested_discarded(elements: Sequence[ullm.conv.ConversationElement], reason: str) -> None:
                    if self._observation_generation is generation:
                        self._remember_discarded_round(elements, reason=reason)

                active_token = _ACTIVE_DIAGNOSTIC_TURN.set((self, generation))
                try:
                    with ullm.manager.observe_turn_activity(
                        observe_nested_elements,
                        observe_nested_discarded,
                    ):
                        await super()._run()
                finally:
                    _ACTIVE_DIAGNOSTIC_TURN.reset(active_token)
                    if self._observation_generation is generation:
                        self._observation_generation = None
            finally:
                if writer is not None:
                    writer.cancel()
                    try:
                        with contextlib.suppress(asyncio.CancelledError):
                            await writer
                    except Exception:
                        traceback.print_exc()
                # Best-effort by design: a recording failure in this finally
                # would otherwise REPLACE the turn's real exception.
                try:
                    self._record_turn(span)
                except Exception:
                    traceback.print_exc()
                finally:
                    self._recorded_elements.clear()
                    self._discarded_rounds.clear()
                    self._response_epochs.clear()

    def _record_turn(self, span: Span) -> None:
        """Stamp completed responses, results, tools, and artifacts onto a turn."""
        discarded_by_element = {id(element): record for record in self._discarded_rounds for element in record.elements}
        generated = 0
        uncached = 0
        for element in self._recorded_elements:
            discarded = discarded_by_element.get(id(element))
            if isinstance(element, ullm.conv.LLMResponse):
                summary = _response_summary(element)
                _record_response_summary(
                    span,
                    summary,
                    discarded=discarded,
                    incomplete=element.status != "processed",
                    history_epoch=self._response_epochs.get(id(element)),
                )
                generated += summary.completion_tokens
                uncached += summary.uncached_input_tokens
            elif isinstance(element, ullm.conv.ToolResult):
                _record_tool_result(span, element, discarded=discarded)
        span.add_usage(generated, uncached)
        for tool in self.conversation.system_prompt.tools:
            drain = getattr(tool, "pop_execution_records", None)
            if drain is None:
                continue
            for record in drain():
                span.event(
                    "kernel/exec",
                    t=record["t"],
                    duration_s=record["duration_s"],
                    is_error=record["is_error"],
                    fresh_kernel=record["fresh_kernel"],
                )
        assert self.diag.block_store is not None  # active stream: block store present
        structure = serialize_conversation_blocks(self.conversation, self.diag.block_store)
        span.artifact("conversation", structure, codec="json")
        self._record_discarded_rounds(span)

    def _record_discarded_rounds(self, span: Span) -> None:
        """Store removed round content and metadata in generation order."""
        if not self._discarded_rounds:
            return
        position = {id(element): index for index, element in enumerate(self._recorded_elements)}
        records = sorted(
            self._discarded_rounds,
            key=lambda record: min((position.get(id(element), len(position)) for element in record.elements)),
        )
        dropped = ullm.conv.Conversation(
            system_prompt=self.conversation.system_prompt,
            elements=[ullm.conv.ResponseTurn(elements=record.elements) for record in records],
        )
        metadata = []
        for record in records:
            item: dict[str, Any] = {"reason": record.reason}
            if record.available_until_round is not None:
                item["available_until_round"] = record.available_until_round
            metadata.append(item)
        assert self.diag.block_store is not None  # active stream: block store present
        span.artifact(
            "discarded_rounds",
            {
                "records": metadata,
                "conversation": serialize_conversation_blocks(dropped, self.diag.block_store),
            },
            codec="json",
        )


def _response_summary(response: ullm.conv.LLMResponse) -> ullm.introspect.ResponseSummary:
    """Summarize one response through ullm's stable introspection surface."""
    (summary,) = ullm.introspect.turn_response_summaries(ullm.conv.ResponseTurn(elements=[response]))
    return summary


def _discard_fields(discarded: DiscardedRound | None) -> dict[str, Any]:
    """Event fields describing why a response or result left the conversation."""
    if discarded is None:
        return {}
    fields: dict[str, Any] = {"discarded_reason": discarded.reason}
    if discarded.available_until_round is not None:
        fields["available_until_round"] = discarded.available_until_round
    return fields


RECOMPUTE_WARN_TOKENS = 200
"""A call that continues the one before it in its conversation (the same
history epoch) brings only its prompt less that call's prompt and reply; the
engine should have the rest cached. Uncached tokens past that by more than
this are announced on stdout: the engine re-prefilled what it once had. A
continuation resumes at the previous reply's decode tail, a
``prefix_match_unit`` boundary (64 or 80 tokens) within a unit of the reply's
end, or one unit further back when the reply's last boundary was a block
boundary or was crossed by its final step; two units at 80 stay under it."""

# Per conversation (by id, checked against a weak reference so a reused id is
# not mistaken for it): the epoch, prompt and completion tokens of its last
# call. One game's conversations are few, so nothing needs evicting.
_last_calls: dict[int, tuple[weakref.ref[ullm.conv.Conversation], int, int, int]] = {}


def check_continuation_prefill(
    conversation: ullm.conv.Conversation, response: ullm.conv.LLMResponse, *, where: str
) -> None:
    """Announce a continuation whose uncached prompt exceeds what it adds by
    more than ``RECOMPUTE_WARN_TOKENS``.

    A continuation is a call in the same history epoch as the conversation's
    previous call. What it adds is its prompt less that call's prompt and
    completion, when that is not negative; a negative one is a history not
    extended exactly, whose recompute is the history's, not the cache's, and
    is left alone.
    """
    usage = response.token_usage
    if response.do_not_send_to_agent or usage is None or not usage.prompt_tokens:
        return
    epoch = conversation.history_epoch
    prompt, completion = usage.prompt_tokens, usage.completion_tokens
    last = _last_calls.get(id(conversation))
    _last_calls[id(conversation)] = (weakref.ref(conversation), epoch, prompt, completion)
    if last is None:
        return
    last_conversation, last_epoch, last_prompt, last_completion = last
    if last_conversation() is not conversation or last_epoch != epoch:
        return
    added = prompt - last_prompt - last_completion
    if added < 0:
        return
    uncached = prompt - usage.cached_tokens
    if uncached - added > RECOMPUTE_WARN_TOKENS:
        print(
            f"[recompute] {where}: {uncached} uncached prompt tokens where the call adds {added} "
            f"(prompt {prompt}, previous prompt {last_prompt} + reply {last_completion}): "
            f"{uncached - added} re-prefilled",
            flush=True,
        )


def _record_response_summary(
    span: Span,
    summary: ullm.introspect.ResponseSummary,
    *,
    discarded: DiscardedRound | None,
    incomplete: bool = False,
    history_epoch: int | None = None,
) -> None:
    """Emit one response summary with optional discard/incomplete metadata,
    and the conversation's history epoch at the call: consecutive calls in
    one epoch extend each other's history exactly, so what a call brings is
    its prompt less the previous call's prompt and completion."""
    extra = _discard_fields(discarded)
    if incomplete:
        extra["incomplete"] = True
    if history_epoch is not None:
        extra["history_epoch"] = history_epoch
    if summary.response_injection is not None:
        extra["response_injection"] = summary.response_injection
    span.event(
        "llm/response",
        completion_tokens=summary.completion_tokens,
        uncached_input_tokens=summary.uncached_input_tokens,
        cached_input_tokens=summary.cached_input_tokens,
        reasoning_tokens=summary.reasoning_tokens,
        stop_reason=summary.stop_reason,
        malformed_retries=summary.malformed_retries,
        n_tool_calls=summary.n_tool_calls,
        synthetic=summary.synthetic,
        truncated_by_per_turn_cap=summary.truncated_by_per_turn_cap,
        **extra,
    )


def _record_tool_result(span: Span, result: ullm.conv.ToolResult, *, discarded: DiscardedRound | None) -> None:
    """Emit one model-visible tool result without duplicating its stored text."""
    span.event(
        "tool/result",
        tool_name=result.tool_name,
        is_error=result.is_error,
        **_discard_fields(discarded),
    )


class TokenMeter:
    """Per-game token deltas off a backend's accumulated usage (I1).

    ``take()`` returns ``(generated, uncached_input)`` spent since the
    previous ``take()`` — the mapping onto TAAF's per-action kwargs:
    generated = completion tokens, uncached input = prompt minus cached.
    The backend must be this game's private client (each game gets its own
    ``ProxyBackend``), so its ``total_usage`` is per-game by construction.
    """

    def __init__(self, backend: ullm.backend.Backend):
        self._backend = backend
        self._generated = 0
        self._uncached = 0

    def take(self) -> tuple[int, int]:
        """Tokens spent since the last call, as (generated, uncached). Read
        immediately before each action so a batch's cost lands on the action
        it was spent for (I1)."""
        usage = self._backend.total_usage
        generated = usage.completion_tokens
        uncached = usage.prompt_tokens - usage.cached_tokens
        delta = (generated - self._generated, uncached - self._uncached)
        self._generated, self._uncached = generated, uncached
        return delta


REPLAY_CONSUMPTION_FILE = "replay_consumption.json"


def record_replay_consumption(backend: Any, diag: SpanLogger) -> None:
    """Record how much of a replay stream this game left unread.

    Written by whichever process HOSTS the game, because that is where the
    consuming client lives: in subprocess mode the ``ReplayBackend`` is
    pickled into the child, so the parent's copy never advances and could
    only ever report the whole stream as unread. Live backends have no
    ``unconsumed``, so this is a no-op for them.

    A sidecar file beside the span stream rather than a span: the game's
    stream has exactly one root by construction, and consumers rely on that
    (``reference_support.replay_spans`` unpacks it). This is also written
    after the game span has closed, so there is nothing left to attach to.

    Recorded rather than raised: a replay that stops early is a fact about
    the run, and it is the reference assertions that decide it is a failure.
    """
    count = getattr(backend, "unconsumed", None)
    game_dir = diag.stream_dir
    if count is None or game_dir is None:
        return
    (game_dir / REPLAY_CONSUMPTION_FILE).write_text(json.dumps({"unconsumed": int(count())}), encoding="utf-8")


def build_llm_context(
    moves: Sequence[Move],
    game_info: GameInfo,
    rng: random.Random,
    backend: ullm.backend.Backend,
    diag: SpanLogger | None = None,
    status: StatusHandle | None = None,
    slots: SlotView | None = None,
    yield_slot: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
) -> tuple[LLMMoveContext, TokenMeter]:
    """One construction site for the LLM context + its meter, shared by
    the in-process host and the subprocess child so both process modes
    build the identical surface (I9)."""
    handle = status if status is not None else NULL_STATUS
    # An LLM agent's unclaimed time is real work of its own — prompt
    # building, parsing, the agent's own bookkeeping — so name it rather
    # than let it read back as the coarse "running".
    handle.set_base("other")
    ctx = LLMMoveContext(
        history=HistoryView(moves),
        game=game_info,
        rng=rng,
        shared={},
        diag=diag if diag is not None else SpanLogger(),
        slots=slots if slots is not None else SlotView(),
        status=handle,
        yield_slot=yield_slot if yield_slot is not None else no_yield,
        backend=backend,
    )
    # The wait for a model response happens inside the backend, where the
    # agent cannot see it; wiring it here means every LLM agent reports it
    # without opting in. Tools are agent-held, so those they wire themselves.
    backend.status_hook = handle
    return ctx, TokenMeter(backend)
