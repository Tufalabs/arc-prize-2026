"""
Backend interface for ULLM. Each backend subclass implements the conversion between
user format (conversation objects) and model format (backend-specific), and handles inference.
"""

# TODO: use caching to speed up tokenization

import asyncio
import contextlib
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Literal, Optional

import ullm.conv
import ullm.support as uls

# A host's "say what you are waiting on" hook: takes a short label and
# returns a context manager scoped to the wait. Structural on purpose —
# ullm never imports the host that supplies it (platypus.status).
StatusHook = Callable[[str], contextlib.AbstractContextManager[None]]

if TYPE_CHECKING:
    import ullm.capture
    import ullm.engine_monitor
    import ullm.tracing


class MalformedOutputError(Exception):
    """Raised by backends when model output cannot be parsed into a valid LLMResponse.
    ResponseManager catches this to drive retry logic; other exceptions propagate."""

    pass


class DegenerateOutputError(MalformedOutputError):
    """Subclass for runaway-repetition degeneracy (e.g. '!!!!!' spam from a
    broken kernel). Backends add a jittered sleep on this path before
    propagating, so the 64-way solver fleet doesn't all reconverge on the
    same poisoned batch composition. Caught by the normal retry loop
    because it's a MalformedOutputError."""

    pass


class ForbiddenTokenError(RuntimeError):
    """A completion contains a token its request banned (logit bias -100).

    Not a MalformedOutputError: a retry would hide a serving path that lets
    banned tokens through, so it fails the call, and with it the game."""

    pass


class PromptTooLongError(Exception):
    """Raised when the input prompt exceeds the model's context window.
    ResponseManager catches this and produces a synthetic LLMResponse instead of failing."""

    pass


def _is_empty_content(content: "ullm.conv.Content") -> bool:
    """Return True iff a conversation `Content` carries no user-facing payload.
    Used by `Backend.process_output` to enforce the non-empty-final invariant."""
    if isinstance(content, ullm.conv.EmptyContent):
        return True
    if isinstance(content, ullm.conv.StringContent):
        return not content.text
    if isinstance(content, ullm.conv.ListContent):
        return all(_is_empty_content(e) for e in content.elements)
    # Tool call or other non-text content: treat as present.
    return False


def _raw_text_from_model_contents(model_contents: Any) -> str:
    """Best-effort extraction of the raw assistant text from a backend's
    ``_model_contents`` object — bypassing the per-backend reasoning /
    tool-call parsers that produce ullm's split user-format Content. Used
    for diagnostic messages (e.g. the empty-final invariant) where we
    want to see what the model literally emitted before any ullm parsing.

    Walks the well-known shapes produced by each backend (vLLM
    ``RequestOutput``, OpenAI ``ChatCompletion``, Anthropic ``Message``)
    and falls back to a truncated ``repr()``. Fully wrapped in a single
    try/except so a malformed ``_model_contents`` can't mask the
    underlying error we're trying to report."""
    try:
        # vLLM RequestOutput: outputs[0].text is the raw text.
        if hasattr(model_contents, "outputs") and getattr(model_contents, "outputs"):
            return getattr(model_contents.outputs[0], "text", "") or ""
        # OpenAI ChatCompletion: choices[0].message.content (+ any reasoning prefix).
        if hasattr(model_contents, "choices") and getattr(model_contents, "choices"):
            msg = getattr(model_contents.choices[0], "message", None)
            if msg is not None:
                reasoning = getattr(msg, "reasoning", None) or ""
                content = getattr(msg, "content", None) or ""
                return f"[reasoning] {reasoning}\n[content] {content}" if reasoning else content
        # Anthropic Message: walk content blocks (text / thinking / tool_use).
        if hasattr(model_contents, "content") and isinstance(model_contents.content, list):
            parts = list[str]()
            for block in model_contents.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    parts.append(getattr(block, "text", "") or "")
                elif btype == "thinking":
                    parts.append(f"[thinking] {getattr(block, 'thinking', '') or ''}")
                elif btype == "tool_use":
                    parts.append(f"[tool_use {getattr(block, 'name', '?')}]({getattr(block, 'input', '')})")
                else:
                    parts.append(f"[{btype or '?'}]")
            return "\n".join(parts)
        # Unknown shape — show the repr, capped.
        r = repr(model_contents)
        return r if len(r) <= 2000 else r[:2000] + "...<truncated>"
    except Exception as e:
        return f"<raw-text extraction failed: {type(e).__name__}: {e}>"


@dataclass
class TokenUsage(uls.BaseClass):
    """Token usage counters for a single request or accumulated across requests."""

    prompt_tokens: int = 0  # Total input tokens (including cached reads and cache writes)
    completion_tokens: int = 0  # Output tokens (including reasoning)
    cached_tokens: int = 0  # Prompt tokens served from prefix cache (subset of prompt_tokens)
    cache_write_tokens: int = 0  # Prompt tokens written to cache (subset of prompt_tokens)
    reasoning_tokens: int = 0  # Completion tokens spent on reasoning (subset of completion_tokens)
    num_requests: int = 0  # Number of requests aggregated into this usage

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.num_requests += other.num_requests
        return self

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            num_requests=self.num_requests + other.num_requests,
        )

    def to_string(self) -> str:
        """One-line usage summary, omitting the fields that are zero."""
        parts = [f"prompt={self.prompt_tokens}"]
        if self.cached_tokens:
            parts.append(f"cached={self.cached_tokens}")
        if self.cache_write_tokens:
            parts.append(f"cache_write={self.cache_write_tokens}")
        parts.append(f"completion={self.completion_tokens}")
        if self.reasoning_tokens:
            parts.append(f"reasoning={self.reasoning_tokens}")
        return ", ".join(parts)


@dataclass
class TokenPricing(uls.BaseClass):
    """Per-token pricing in dollars per million tokens. Set to NaN for unknown pricing."""

    input_per_mtok: float = float("nan")  # Regular input tokens
    output_per_mtok: float = float("nan")  # Output tokens
    cache_read_per_mtok: float = float("nan")  # Cached input tokens (reads)
    cache_write_per_mtok: float = float("nan")  # Cache write tokens

    def cost(self, usage: TokenUsage) -> float:
        """Compute cost in dollars from token usage. Returns NaN if pricing is not configured."""
        regular_input = usage.prompt_tokens - usage.cached_tokens - usage.cache_write_tokens
        return (
            regular_input * self.input_per_mtok
            + usage.cached_tokens * self.cache_read_per_mtok
            + usage.cache_write_tokens * self.cache_write_per_mtok
            + usage.completion_tokens * self.output_per_mtok
        ) / 1_000_000


# Known pricing tables (dollars per million tokens)
ANTHROPIC_PRICING: dict[str, TokenPricing] = {
    "claude-haiku-4-5": TokenPricing(
        input_per_mtok=1, output_per_mtok=5, cache_read_per_mtok=0.1, cache_write_per_mtok=1.25
    ),
    "claude-sonnet-4-6": TokenPricing(
        input_per_mtok=3, output_per_mtok=15, cache_read_per_mtok=0.3, cache_write_per_mtok=3.75
    ),
    "claude-opus-4-6": TokenPricing(
        input_per_mtok=5, output_per_mtok=25, cache_read_per_mtok=0.5, cache_write_per_mtok=6.25
    ),
    "claude-opus-4-7": TokenPricing(
        input_per_mtok=5, output_per_mtok=25, cache_read_per_mtok=0.5, cache_write_per_mtok=6.25
    ),
}


@dataclass
class SamplingParameters(uls.BaseClass):
    """Base class for sampling parameters."""

    # Opaque application metadata travels beside the model request, never
    # into decoding parameters or the model's token prompt.
    request_context: dict[str, Any] = field(default_factory=dict, kw_only=True)


class TokenCount(int):
    """An exact count with the token prompt that was measured, when available."""

    prompt_token_ids: tuple[int, ...]
    media: dict[str, Any] | None

    def __new__(cls, count: int, prompt_token_ids: Any, media: dict[str, Any] | None = None) -> "TokenCount":
        value = super().__new__(cls, count)
        value.prompt_token_ids = tuple(prompt_token_ids)
        value.media = media
        if len(value.prompt_token_ids) != count:
            raise ValueError("token count differs from its prompt evidence")
        return value

    def __reduce__(self) -> Any:
        return type(self), (int(self), self.prompt_token_ids, self.media)


#: The ``extra_params`` the answer may set: per-token shaping that carries no
#: state across the reply. Every other key either bounds or seeds the whole
#: reply (a cap, stops, a seed, a minimum), bans tokens the joined reply is
#: checked against, or counts the request's own output (presence and
#: frequency penalties), which in the answer request leaves the reasoning out.
_ANSWER_EXTRA_KEYS = frozenset({"top_k", "min_p"})


@dataclass
class AnswerSampling(uls.BaseClass):
    """How the answer is sampled: what the model writes once its reasoning
    closes (``GenericSamplingParameters.after_reasoning``). A field left None
    keeps the call's own value; ``extra_params`` merge over the call's key by
    key and take only ``top_k`` and ``min_p``. Everything else stays the
    call's."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    extra_params: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def __post_init__(self) -> None:
        other = sorted(set(self.extra_params) - _ANSWER_EXTRA_KEYS)
        if other:
            raise ValueError(
                f"after_reasoning cannot set {other}: the answer's extra_params take only "
                f"{sorted(_ANSWER_EXTRA_KEYS)}, and the rest stays the call's"
            )
        super().__post_init__()


@dataclass
class GenericSamplingParameters(SamplingParameters):
    """Generic parameters that any backend must support.
    extra_params can be used to pass additional backend-specific parameters;
    unsupported extra_params raise an error by default.
    Any property left at None uses that backend's default."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = 30000
    stop_sequences: Optional[list[str]] = None
    seed: Optional[int] = None
    # Reasoning-passback policy for THIS call ("none"/"current_turn"/"all");
    # None uses the backend's ``send_back_reasoning`` default. Per-call
    # request-construction config, so it lives here with the other per-call
    # knobs (and rides named sampling profiles / the proxy wire for free).
    send_back_reasoning: Optional[Literal["none", "current_turn", "all"]] = None
    # Text appended AFTER the chat template's generation prompt, so the
    # model continues from it as the start of its answer (on a thinking
    # template: right inside the opened think block). Served as tokens of
    # its own after the rendered prompt, never re-rendered through the
    # template. Only a backend whose ``supports_response_injection()`` is
    # True accepts it; every other backend refuses the call.
    response_injection: Optional[str] = None
    # Keep a reply with no final content (no text, no tool call) as THIS
    # call's answer, marked ``stop_reason="malformed"`` with the reason on
    # ``malformed_error``, instead of raising MalformedOutputError for the
    # caller to re-sample. For a caller that has a use for such a reply --
    # one that reads the answer out of the reasoning when the model never
    # closed its thinking. Narrower than ``Backend.accept_malformed_output``:
    # this call only, and only the empty reply; the degeneracy and parser
    # checks stand.
    keep_empty_reply: bool = False
    extra_params: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    # Sample the answer — the tokens after the reasoning closes — apart
    # from the reasoning, or None to sample the whole reply alike. A call
    # whose prompt already closes the reasoning samples all of it this way.
    # Only a backend whose ``supports_after_reasoning()`` is True accepts
    # it; every other backend refuses the call.
    after_reasoning: Optional[AnswerSampling] = None

    def for_answer(self) -> "GenericSamplingParameters":
        """These parameters with ``after_reasoning`` applied: how the answer
        is sampled."""
        answer = self.after_reasoning
        if answer is None:
            return self
        return replace(
            self,
            temperature=self.temperature if answer.temperature is None else answer.temperature,
            top_p=self.top_p if answer.top_p is None else answer.top_p,
            extra_params={**self.extra_params, **answer.extra_params},
            after_reasoning=None,
        )

    def samples_answer_apart(self) -> bool:
        """Whether ``after_reasoning`` samples the answer any differently from
        the call's own settings. Settings that change nothing sample the whole
        reply alike, so a backend serves them as one request."""
        answer = self.for_answer()
        return (answer.temperature, answer.top_p, answer.extra_params) != (
            self.temperature,
            self.top_p,
            self.extra_params,
        )


@dataclass
class Backend(uls.BaseClass):
    """Base class for LLM backends. Subclasses implement specific backends (vLLM, OpenAI, human, etc.).
    Backends are shared resources (hold HTTP clients, engine connections) and are not deepcopied —
    multiple conversations/TurnManagers can safely share the same backend instance."""

    diagnostics: bool = False  # If True, store inputs and outputs for each request
    track_warnings: bool = True  # If True, append each warning dict to warning_log
    warning_log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    send_back_reasoning: Literal["none", "current_turn", "all"] = "current_turn"
    # DEFAULT reasoning-passback policy; a per-call value on
    # ``GenericSamplingParameters.send_back_reasoning`` overrides it:
    #   "none" — strip all reasoning from history (OpenAI reasoning models, DeepSeek-R1)
    #   "current_turn" — keep reasoning for the current ResponseTurn, strip older ones (Anthropic)
    #   "all" — keep all reasoning in history
    reasoning_effort_prompts: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    # Fallback prompts appended to the system message when the backend has no native reasoning-effort
    # API knob. Empty by default — soft text hints have unclear effects and tend to fight model-card
    # defaults; backend instances or callers can populate per-level entries (e.g.
    # ``{"high": "Think carefully..."}``) when they actually want one.
    pricing: TokenPricing = field(default_factory=TokenPricing)  # Token pricing; NaN by default (unknown)
    # Capture mode (platypus U3): when set, base respond() records every
    # call (request + outcome, incl. malformed / prompt-too-long) into the
    # sink's directory. Identity-blind — see ullm.capture.
    capture_sink: Optional["ullm.capture.CaptureSink"] = None
    # Exact history (ullm.continuation, requirements U5). A backend whose
    # ``supports_token_continuation()`` is True always extends the served
    # history by the sampled ids rather than re-rendering it; with this set
    # the caller also declares every rewrite of sent history via
    # ``Conversation.mark_history_rewritten``, and a call that cannot be
    # continued without one raises ``HistoryDriftError`` instead of quietly
    # rendering afresh.
    exact_history: bool = False
    # Accept a reply with no final content (no text, no tool call) as the
    # model's real turn instead of raising MalformedOutputError for the
    # caller to re-sample: the element stays with ``stop_reason="malformed"``
    # and the reason on ``malformed_error``, and is traced like any other
    # step. An RL trainer needs this: a re-sampled reply hides the draw the
    # policy actually made. Part of the backend's configuration (it travels
    # with the pickled backend to a proxy host), never an ambient default:
    # one fully configured agent describes the experiment.
    accept_malformed_output: bool = False
    # A proxy may tighten this policy, never enable replay forbidden by its host.
    replay_on_link_loss: bool = True
    # Trace mode (platypus U5): set host-side by the trace collector on a
    # backend whose trace_capability() confirmed R5-grade records. Every
    # successful respond() then requires a run_id and reports one
    # TraceStepRecord here. See ullm.tracing.
    trace_recorder: Optional[Any] = field(default=None, init=False, repr=False)
    # Set host-side beside ``trace_recorder`` when the run records ids-only
    # traces (taaf R5.39): records then carry the ids, the images and the
    # sampling, and no messages or output text -- so a step pays no
    # re-render, no decode and no messages copy.
    trace_ids_only: bool = field(default=False, init=False, repr=False)
    # Status reporting: when a host wires this, respond() names the wait for
    # a model response, so a run's periodic report can tell "the model is
    # generating" apart from "the agent is busy with something else". The
    # wait is only observable in here — a caller awaiting respond() cannot
    # see where inside it the time goes. Process-local; never pickles.
    status_hook: Optional[StatusHook] = field(default=None, init=False, repr=False)
    # Hold the engine's profiler windows shut, and lift the hold moving them
    # later, set by the factory that started the engine
    # (``EngineFactory.hold_profile_windows``, ``delay_profile_windows``);
    # None when no factory's profiler watches it. Process-local; never pickles.
    hold_profile_windows: Optional[Callable[[], None]] = field(default=None, init=False, repr=False)
    delay_profile_windows: Optional[Callable[[float], None]] = field(default=None, init=False, repr=False)
    _request_counter: int = field(default=0, init=False, repr=False)
    _request_log: dict[int, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    total_usage: TokenUsage = field(default_factory=TokenUsage, init=False)  # Accumulated across all requests
    # Active requests currently inside ``_generate``. Subclasses are expected
    # to bump/decrement this around their generate loops; ``status()`` reads
    # it for monitoring. See ``VLLMHarmony._generate`` for the reference pattern.
    # Seconds between parses of a streamed partial. Bounds only how stale a
    # live view can be — partials have no streaming consumer, they are read
    # on demand. 0.0 = parse every tick, required by a backend whose partials
    # are deltas rather than cumulative snapshots.
    partial_interval_s: float = 3.0
    # Floor on the gap between this instance's consecutive calls: the next
    # respond() starts no sooner than this many seconds after the previous
    # one ended, and time the caller spent elsewhere meanwhile (a kernel
    # step) counts toward it, so a caller slower than the floor never
    # waits. Per instance: a host that hands each game its own client
    # paces each game on its own. 0 is no floor.
    min_request_gap_s: float = 0.0
    _last_response_ended: float = field(default=0.0, init=False, repr=False)

    _in_flight: int = field(default=0, init=False, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        """Backends pickle as *configuration*: live resources (HTTP clients,
        engine handles — named by ``_transient_fields``) are dropped and
        rebuilt on the receiving side, and per-process accounting state
        (request log/counter, warning log, total_usage) restarts at zero —
        each process accounts for its own traffic. This path exists for
        shipping a configured backend into worker processes; snapshots
        never include backends (a resume reconstructs them anyway).

        ``copy.deepcopy`` goes through here too: a deepcopy is a detached
        configuration clone, never a share. Code that deepcopies a structure
        around a live backend and wants to KEEP the shared instance must say
        so by seeding the memo: ``copy.deepcopy(obj, {id(backend): backend})``.
        """
        assert self._in_flight == 0, f"Cannot pickle {type(self).__name__} with {self._in_flight} requests in flight"
        state = dict(self.__dict__)
        for name in self._transient_fields():
            state.pop(name, None)
        state["_request_counter"] = 0
        state["_last_response_ended"] = 0.0
        state["_request_log"] = dict[int, dict[str, Any]]()
        state["warning_log"] = list[dict[str, Any]]()
        state["total_usage"] = TokenUsage()
        state["trace_recorder"] = None  # host-side collector hook; never crosses processes
        state["trace_ids_only"] = False  # travels with the collector hook, so is dropped with it
        state["status_hook"] = None  # process-local reporting hook; rewired by the receiving host
        state["hold_profile_windows"] = None  # the starting factory's; process-local
        state["delay_profile_windows"] = None  # the starting factory's; process-local
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore via __dict__.update (setattr would trip BaseClass's
        # frozen-attribute check: the transient fields are absent until
        # _rebuild_transient recreates them). The instance is unfrozen for
        # the duration of the hook so implementations can use plain
        # attribute assignment.
        self.__dict__.update(state)
        self.__dict__["_is_frozen"] = False
        self._rebuild_transient()  # a raise here fails the unpickle; the half-built object is discarded
        self.__dict__["_is_frozen"] = True

    def _transient_fields(self) -> tuple[str, ...]:
        """Names of fields holding live, process-local resources. Dropped at
        pickle time; ``_rebuild_transient`` recreates them after unpickling."""
        return ()

    async def probe_continuation(self) -> None:
        """Resolve what continuation needs from the serving surface (a live
        query — token ids honoured, tokenizer identity), so that
        ``supports_token_continuation()`` can answer. Default: nothing to
        resolve; the answer is already known."""
        return None

    async def allows_generation_replay(self) -> bool:
        """Whether a proxy may resend a generation after losing its host link.

        ``replay_on_link_loss`` declares this backend's policy. Async so
        wrappers can combine it with the policy of their remote host.
        """
        return self.replay_on_link_loss

    def supports_token_continuation(self) -> bool:
        """Whether this backend can serve an id prompt it built from its own
        witnessed ids — then it always does; a host asking for
        ``exact_history`` checks this at setup and fails loudly otherwise."""
        return False

    @property
    def _continuing(self) -> bool:
        """Continuation is in effect: whenever the backend can."""
        return self.supports_token_continuation()

    def supports_response_injection(self) -> bool:
        """Whether this backend can serve caller text as tokens after the
        generation prompt (``GenericSamplingParameters.response_injection``).
        A request carrying one on a backend that cannot is refused."""
        return False

    def resolve_response_injection(self, sampling_params: SamplingParameters) -> Optional[str]:
        """The text this call injects after the generation prompt, or None;
        a backend that cannot serve it refuses the call here."""
        injection = getattr(sampling_params, "response_injection", None) or None
        if injection is None:
            return None
        if not self.supports_response_injection():
            raise ValueError(
                f"{type(self).__name__} does not support response_injection (only the in-process vLLM "
                "backend serves text after the generation prompt)"
            )
        return injection

    def supports_after_reasoning(self) -> bool:
        """Whether this backend can sample the answer apart from the
        reasoning (``GenericSamplingParameters.after_reasoning``). A request
        carrying it on a backend that cannot is refused."""
        return False

    def resolve_after_reasoning(self, sampling_params: SamplingParameters) -> Optional[AnswerSampling]:
        """How this call samples its answer apart from its reasoning, or
        None; a backend that cannot refuses the call here."""
        answer = getattr(sampling_params, "after_reasoning", None)
        if answer is None:
            return None
        if not self.supports_after_reasoning():
            raise ValueError(
                f"{type(self).__name__} does not support after_reasoning (only the in-process vLLM backend "
                "samples the answer apart from the reasoning)"
            )
        return answer

    def _shed_served_ids(self, element: "ullm.conv.LLMResponse") -> None:
        """Drop the prompt ids a raw completion carries once the witness and
        the trace step have consumed them: a response's ``_model_contents``
        otherwise retains the whole prompt it was sampled after, so the
        conversation grows with the sum of its prompts and every call ships
        that. Default: this backend keeps no such ids."""
        del element

    def _rebuild_transient(self) -> None:
        """Recreate the ``_transient_fields`` after unpickling. Runs with the
        instance temporarily unfrozen. Default: nothing to rebuild."""
        pass

    # What THIS backend's _prepare_input / _process_output cost, as status
    # labels. They describe only local work: a backend that renders and
    # parses in-process keeps these defaults; a client whose hooks merely
    # serialize overrides them (ProxyBackend), because the real rendering
    # then happens on its host INSIDE the client's generation wait —
    # per-phase propagation over the proxy protocol is roadmapped.
    _prepare_status_label = "preprocessing LLM response"
    _parse_status_label = "postprocessing LLM response"

    async def _pace(self) -> None:
        """Wait out what is left of ``min_request_gap_s`` since this
        instance's previous call ended; nothing when the caller already
        spent that long, or before the first call."""
        if self.min_request_gap_s <= 0 or not self._last_response_ended:
            return
        remaining = self.min_request_gap_s - (time.monotonic() - self._last_response_ended)
        if remaining > 0:
            with self._status_scope("pacing the next request"):
                await asyncio.sleep(remaining)

    def _status_scope(self, label: str) -> contextlib.AbstractContextManager[None]:
        """Scope ``label`` as what this backend is doing, for whatever host
        wired ``status_hook`` (no-op when none did). Named apart from
        ``status()``, which is this backend's monitoring one-liner."""
        return contextlib.nullcontext() if self.status_hook is None else self.status_hook(label)

    def log_warning(self, warning: dict[str, Any]) -> None:
        """Record a warning about recoverable model-output oddities (fallback
        parses, schema-invalid tool calls, retried malformed outputs, etc.).

        Each warning must be a dict with at least a `type` field identifying
        the category; additional keys are caller-specific context. Warnings
        are only retained when `track_warnings` is True, but the structural
        constraint is always checked so miswritten call sites fail loudly."""
        assert isinstance(warning, dict), f"warning must be a dict, got {type(warning).__name__}"
        assert "type" in warning, f"warning must have a 'type' field; got keys {list(warning)}"
        assert isinstance(warning["type"], str), (
            f"warning['type'] must be a string, got {type(warning['type']).__name__}"
        )
        if self.track_warnings:
            self.warning_log.append(warning)

    async def respond(
        self,
        sampling_params: SamplingParameters,
        conversation: ullm.conv.Conversation,
        run_id: Optional[str] = None,
    ) -> AsyncGenerator["ullm.conv.LLMResponse", None]:
        """The backend's single public operation: run one full model call
        against ``conversation``.

        ``run_id`` attributes the call to a game run (platypus: stamped by
        the solver via the per-game proxy client and forwarded here by the
        socket host). Plain callers may omit it; a backend with tracing
        enabled (``trace_recorder`` set) refuses an unattributed call.

        Appends the output element to the current turn, streams user-format
        partial snapshots (parsed via ``_process_partial_output``), and on
        completion leaves the processed element as
        ``conversation.last_turn.last_response``. Raises
        ``MalformedOutputError`` (with the appended element left in place for
        the caller to roll back) or ``PromptTooLongError`` (nothing appended).
        The reasoning-passback policy is resolved here — the per-call value
        on the sampling params when set, else the backend's default — and
        handed to ``_prepare_input``; no instance state is touched.

        Subclasses customize behaviour through the private hooks
        (``_prepare_input`` / ``_generate`` / ``_process_output`` /
        ``_process_partial_output``); this default composes them and owns
        the accounting (request counter/log, ``total_usage``, in-flight)."""
        assert conversation.status == "intermediate", "Conversation must be `intermediate` to run LLM"
        turn = conversation.last_turn
        assert turn.status == "intermediate", "Last element of conversation must be `intermediate` to run LLM"

        await self._pace()
        send_back_reasoning = self.resolve_send_back_reasoning(sampling_params)
        injection = self.resolve_response_injection(sampling_params)
        self.resolve_after_reasoning(sampling_params)

        # Capture (platypus U3): canonicalize the request BEFORE the output
        # element is appended, then record whichever outcome this call ends
        # in — success, malformed output, or prompt-too-long.
        captured_request: Any = None
        captured_blocks: dict[str, Any] = {}
        if self.capture_sink is not None:
            from ullm import capture as _capture_module

            captured_request = _capture_module.canonicalize_request(sampling_params, conversation, captured_blocks)

        def _capture(kind: str, element: Optional["ullm.conv.LLMResponse"], error: Optional[BaseException]) -> None:
            if self.capture_sink is not None:
                from ullm import capture as capture_module

                capture_module.capture_record(
                    self.capture_sink, captured_request, captured_blocks, element, error, kind
                )

        try:
            with self._status_scope(self._prepare_status_label):
                llm_input = await self._prepare_input_async(sampling_params, turn, conversation, send_back_reasoning)
        except PromptTooLongError as e:
            _capture("prompt_too_long", None, e)
            raise
        if run_id is not None:
            llm_input = self._stamp_llm_input(llm_input, run_id)

        request_id = self._request_counter
        self._request_counter += 1
        if self.diagnostics:
            self._request_log[request_id] = {"input": llm_input}

        # The output element joins the conversation only once prepare succeeded.
        element = ullm.conv.LLMResponse()
        element.request_id = request_id
        element.response_injection = injection
        turn.elements.append(element)

        intermediate = ullm.conv.LLMResponse()
        llm_output: Any = None
        self._in_flight += 1
        agen = self._generate(sampling_params, llm_input)
        # Accounting sits in a `finally` because the tokens exist whatever
        # happens next: a cancelled generation (game deadline, level over)
        # and a malformed reply both burned real GPU time, and a total that
        # skipped them would report less spend than the run paid for (I1).
        try:
            try:
                try:
                    # The generation wait: _prepare_input above and
                    # _process_output below publish their own scopes, named
                    # per backend for the LOCAL work they do. For a proxy
                    # client this wait therefore still contains the host's
                    # rendering and parsing, not only engine time.
                    with self._status_scope("waiting for LLM response"):
                        last_parsed_at = 0.0
                        async for partial in agen:
                            llm_output = partial
                            # A parse re-reads the whole accumulated output,
                            # so its cost grows with the response; the readers
                            # pull on demand and cannot see fresher than one
                            # interval anyway.
                            now = time.monotonic()
                            if now - last_parsed_at >= self.partial_interval_s:
                                last_parsed_at = now
                                try:
                                    self._process_partial_output(sampling_params, partial, intermediate)
                                except Exception:
                                    pass  # mid-stream parse hiccups never break streaming
                            yield intermediate
                finally:
                    self._in_flight -= 1
                    await agen.aclose()
                if self.diagnostics:
                    self._request_log[request_id]["output"] = llm_output

                with self._status_scope(self._parse_status_label):
                    self._process_output(sampling_params, llm_output, conversation, turn, element)
                # Backends must populate token usage
                assert element.token_usage is not None, f"{type(self).__name__} must set token_usage in _process_output"
                # Empty final_content is legitimate when the model was cut off by the token budget
                # (e.g. reasoning consumed the entire budget): return the response as-is with
                # stop_reason="max_tokens" instead of raising, so callers can handle truncation.
                # A reply already marked malformed (accepted by the host behind a
                # proxy) passes through: the mark IS the verdict.
                if _is_empty_content(element.final_content) and element.stop_reason not in ("max_tokens", "malformed"):
                    reasoning_tokens = element.token_usage.reasoning_tokens
                    completion_tokens = element.token_usage.completion_tokens
                    raw_text = _raw_text_from_model_contents(element._model_contents)
                    message = (
                        "Model produced no final content (no text response, no tool call). "
                        f"reasoning_tokens={reasoning_tokens}, completion_tokens={completion_tokens}. "
                        f"Raw model output: {raw_text!r}"
                    )
                    # getattr: a backend pickled before the field existed (a long-lived
                    # worker shipping its client to a fresh host) must keep raising,
                    # and a SamplingParameters without the field is not asking.
                    if not getattr(self, "accept_malformed_output", False) and not getattr(
                        sampling_params, "keep_empty_reply", False
                    ):
                        raise MalformedOutputError(message)
                    # The model's real turn, kept as such: empty content, the
                    # reason recorded, traced and counted like any reply.
                    element.stop_reason = "malformed"
                    element.malformed_error = message
            except MalformedOutputError as e:
                _capture("malformed", None, e)
                raise
            except PromptTooLongError as e:
                _capture("prompt_too_long", None, e)
                raise
            self.total_usage += element.token_usage
        finally:
            self._last_response_ended = time.monotonic()
            # A cancellation lands here before ``_process_output`` could fill
            # token_usage, so count what the last streamed chunk knows; a
            # backend that cannot say returns None and the call goes uncounted.
            # token_usage SET means the call reached ``_process_output``:
            # counted just above, or a malformed reply whose element
            # ``ResponseManager`` rolls out of the turn. The span stream never
            # sees that element, so counting it here alone would break the
            # three-way reconciliation on a clean run (roadmap.md).
            if element.token_usage is None:
                spent = self._usage_from_partial(llm_output)
                if spent is not None:
                    # Keep the request-scoped usage on the incomplete
                    # element too. Turn-level observers can then attribute a
                    # cancelled call without subtracting a backend-wide
                    # cumulative counter that may include concurrent calls.
                    element.token_usage = spent
                    element.reasoning_content = intermediate.reasoning_content
                    element.final_content = intermediate.final_content
                    self.total_usage += spent
        _capture("response", element, None)
        if self.trace_recorder is not None:
            # Tracing (platypus U5): report this call's step data to the
            # host-side collector. Only successful calls become steps —
            # malformed/too-long attempts raised above. Any failure in
            # here (missing run_id, witness ids absent, input roundtrip
            # mismatch) is deliberately fatal: a trace-enabled run must
            # not silently produce an incomplete or unfaithful trace.
            if run_id is None:
                raise RuntimeError(
                    f"{type(self).__name__} has tracing enabled but respond() was called without "
                    "run_id — every model call must be attributable to a game run"
                )
            record = await self._build_trace_step(
                run_id=run_id,
                sampling_params=sampling_params,
                llm_input=llm_input,
                element=element,
                conversation=conversation,
            )
            self.trace_recorder(record)
        # The witness and the trace step have taken what they need from the
        # raw completion; what stays on the conversation must be linear in
        # the history it describes, and every call ships it whole.
        self._shed_served_ids(element)
        # Log the processed LLMResponse (deepcopy so later mutations don't affect the log)
        if self.diagnostics:
            import copy

            self._request_log[request_id]["element"] = copy.deepcopy(element)

    def resolve_send_back_reasoning(self, sampling_params: SamplingParameters) -> str:
        """The reasoning-passback policy for one call: the per-call value when
        set, else the backend default. ``respond`` and ``token_count`` must
        resolve it identically or they describe different requests."""
        return getattr(sampling_params, "send_back_reasoning", None) or self.send_back_reasoning

    async def token_count(
        self, sampling_params: SamplingParameters, conversation: ullm.conv.Conversation
    ) -> Optional[int]:
        """EXACT input tokens ``conversation`` would cost as a request right
        now — the counterpart of ``respond``, answering "what would this
        prompt weigh" without sending it.

        Returns ``None`` when this backend cannot tokenize, which is a
        legitimate answer and the default: the caller falls back to whatever
        estimate it trusts. There is no approximate mode here — a number that
        came back is a promise that the request would really weigh that, and
        callers are entitled to hard-error when it turns out not to.

        Reasoning passback is resolved exactly as ``respond`` resolves it, so
        the count reflects whether past reasoning is actually sent.

        Async because the answer does not always come from local memory: the
        whole-backend proxy has to ask the host, and it is the topology the
        duck actually runs in. Counting drives eviction, so it is part of the
        recorded stream — a replayed run must be handed the same numbers or it
        evicts elsewhere and diverges.

        Subclasses override ``_token_count``; this owns the capture."""
        self.resolve_response_injection(sampling_params)  # refused here as respond() would refuse it
        self.resolve_after_reasoning(sampling_params)
        count = await self._token_count(sampling_params, conversation)
        if self.capture_sink is not None:
            from ullm import capture as _capture_module

            blocks: dict[str, Any] = {}
            request = _capture_module.canonicalize_request(sampling_params, conversation, blocks)
            self.capture_sink.record(request, blocks, "token_count", {"count": count})
        return count

    async def text_token_count(self, text: str) -> Optional[int]:
        """EXACT token length of raw ``text`` on this backend's tokenizer,
        or ``None`` when it cannot tokenize -- the same contract as
        ``token_count``: exact or nothing, never an estimate. It drives
        journal retention, so it is part of the recorded stream exactly like
        ``token_count``: a replayed run must be handed the same numbers or
        it retains elsewhere and diverges.

        Subclasses override ``_text_token_count``; this owns the capture."""
        count = await self._text_token_count(text)
        if self.capture_sink is not None:
            import ullm.capture as _capture_module

            request = _capture_module.canonicalize_text_request(text)
            self.capture_sink.record(request, {}, "text_token_count", {"count": count})
        return count

    async def _text_token_count(self, text: str) -> Optional[int]:
        """``None`` by default: a backend that cannot tokenize raw text says
        so, and the caller falls back to its own conservative ruler."""
        return None

    async def _token_count(
        self, sampling_params: SamplingParameters, conversation: ullm.conv.Conversation
    ) -> Optional[int]:
        """Default: this backend cannot tokenize."""
        return None

    async def _prepare_input_async(
        self,
        sampling_params: SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        """Allow CPU-heavy backends to prepare without blocking the event loop."""
        return self._prepare_input(sampling_params, turn, conversation, send_back_reasoning)

    def _prepare_input(
        self,
        sampling_params: SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        """``send_back_reasoning`` arrives resolved (per-call value or the
        backend default) — implementations must use it, not the instance field."""
        raise NotImplementedError("Subclasses must implement _prepare_input()")

    def _stamp_llm_input(self, llm_input: Any, run_id: str) -> Any:
        """Attach run attribution to the prepared request, where the wire
        format has a vehicle for it. Called with the ``run_id`` this
        ``respond()`` was given, independently of tracing. Default: no-op."""
        return llm_input

    async def trace_capability(self) -> Optional["ullm.tracing.TraceCapability"]:
        """Whether this backend can produce R5-grade trace records (exact
        token ids + a real input-token roundtrip), and the serving-stack
        identity if so. ``None`` = cannot — requesting the trace sink on
        such a backend is a setup-time error (platypus U5, never a silent
        skip). A capability is a runtime answer, not a class property:
        the same OpenAIServer class is capable against a vLLM server and
        not against real OpenAI. Default: not capable."""
        return None

    async def _build_trace_step(
        self,
        *,
        run_id: str,
        sampling_params: SamplingParameters,
        llm_input: Any,
        element: "ullm.conv.LLMResponse",
        conversation: ullm.conv.Conversation,
    ) -> "ullm.tracing.TraceStepRecord":
        """Build one call's trace record — witness token ids from the
        engine plus a PASSING input-token roundtrip (raise
        ``ullm.tracing.TraceReconstructionError`` otherwise). Only reached
        when ``trace_recorder`` is set, which the collector only does on a
        backend whose ``trace_capability()`` returned a capability."""
        raise NotImplementedError(f"{type(self).__name__} advertises no trace capability")

    def _usage_from_partial(self, llm_output: Any) -> Optional[TokenUsage]:
        """Usage for a call that never reached ``_process_output`` — cancelled
        mid-generation — derived from the last streamed chunk, or ``None`` when
        this backend's stream cannot say.

        Returning ``None`` means the call goes uncounted, so a backend whose
        wire format reports usage only in a final chunk (most OpenAI-compatible
        servers) still loses a cancelled call's tokens. Backends that can count
        from any chunk override this."""
        return None

    async def _generate(self, sampling_params: SamplingParameters, llm_input: Any) -> AsyncGenerator[Any, None]:
        """Run inference. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement _generate()")
        yield  # make this a generator

    def _process_output(
        self,
        sampling_params: SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Turn the backend's raw output into user-format content on ``element``.
        Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _process_output()")

    def _process_partial_output(
        self, sampling_params: SamplingParameters, llm_output: Any, intermediate: ullm.conv.LLMResponse
    ) -> None:
        """Default: streaming not supported. Override in subclasses."""
        intermediate.final_content = ullm.conv.StringContent(text="[Streaming not supported for this backend]")

    def status(self) -> str:
        """One-line status for monitoring. Base implementation reports the
        class name and active ``_generate`` calls. Subclasses can override to
        add backend-specific detail (engine liveness, queue depth, etc.)."""
        return f"{type(self).__name__}: in_flight={self._in_flight}"

    async def engine_metrics(self) -> Optional["ullm.engine_monitor.EngineMetrics"]:
        """Scheduler-level metrics for the engine behind this backend — what
        the periodic engine-status line is made of (running/waiting, KV usage,
        prefix-hit rate).

        ``None`` — the default — means there is no engine here to report on:
        a hosted API, a stub, or a client whose HOST does the reporting. A
        backend that can answer gets monitored wherever it runs, which is why
        this lives on the backend rather than on the factory that built it:
        the factory only knows about engines it stood up itself.

        Async because the answer is not always local — an HTTP backend reads
        its server's ``/metrics``.
        """
        return None

    async def test_alive(self):
        """Runs a basic prompt to check if the backend is responsive."""
        import ullm.conv
        import ullm.manager

        sampling_params = GenericSamplingParameters(temperature=0.7, max_tokens=256)
        response_manager = ullm.manager.ResponseManager(backend=self, sampling_params=sampling_params)
        turn_manager = ullm.manager.TurnManager(response_manager=response_manager)

        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(
                content=ullm.conv.StringContent(text="You are a helpful assistant. Answer questions concisely."),
                reasoning_effort="low",
            )
        )
        conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="What is 1+1")))
        turn_manager.conversation = conversation

        await turn_manager.run()

    async def close(self) -> None:
        """Release backend resources (HTTP clients, connections, etc.). No-op by default."""
        pass

    async def close_context(self, prefix: str) -> None:
        """Tell the engine a context has ended: ``<run_id>`` for a whole game,
        ``<run_id>/<conversation_id>`` for one conversation. Its cached prefix
        then goes before live contexts' when the engine needs room. No-op by
        default."""
        del prefix

    async def close_conversation(self, conversation_id: str, run_id: Optional[str] = None) -> None:
        """Tell the engine one conversation's thread has ended (its history was
        rewritten, or its player is done): ``close_context`` on the name its
        requests carry, ``<run_id>/<conversation_id>``. A request without a
        run_id carries no name, so there is nothing to close."""
        if run_id is not None:
            await self.close_context(f"{run_id}/{conversation_id}")

    # Diagnostics
    def decode_input(self, request_id: int) -> str:
        """Decode the input for a given request ID to a human-readable string. Override in subclasses."""
        return str(self._request_log[request_id]["input"])

    def decode_output(self, request_id: int) -> str:
        """Decode the output for a given request ID to a human-readable string. Override in subclasses."""
        return str(self._request_log[request_id]["output"])

    def print_request(self, request_id: int) -> None:
        """Print input and output for a given request."""
        print(f"--- Request {request_id} ---")
        print(f"INPUT:\n{self.decode_input(request_id)}")
        try:
            print(f"OUTPUT:\n{self.decode_output(request_id)}")
        except Exception:
            # Debug printer: an undecodable output should not stop the dump.
            # Narrow, so an interrupt still reaches the caller.
            print("OUTPUT: <absent or undecodable>")

    def print_all_requests(self) -> None:
        """Print all logged requests."""
        for request_id in self._request_log:
            self.print_request(request_id)
            print()

    def clear_log(self) -> None:
        """Clear the diagnostics log."""
        self._request_log.clear()

    def reset_usage(self) -> None:
        """Reset aggregate token usage to zero."""
        self.total_usage = TokenUsage()


@dataclass
class DummyBackend(Backend):
    """A stubbed backend for testing. Despite its simplicity it scores 100% on all known and future AI benchmarks."""

    def _prepare_input(
        self,
        sampling_params: SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        # Simulate a context window limit: reject prompts with >1M characters
        """Nothing to render — the only real work is the simulated context-window
        limit, so that prompt-too-long handling has a backend to exercise it on.
        """
        total_chars = 0
        for conv_element in conversation.elements:
            content = getattr(conv_element, "content", None)
            if content is not None:
                total_chars += len(content.to_string())
        if total_chars > 1_000_000:
            raise PromptTooLongError(f"DummyBackend: input too long ({total_chars} chars > 1M limit)")
        return None

    async def _generate(self, sampling_params: SamplingParameters, llm_input: Any) -> AsyncGenerator[Any, None]:
        """Always the same answer."""
        yield "I am Groot."

    def _process_output(
        self,
        sampling_params: SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Put the canned reply on the element."""
        element.final_content = ullm.conv.StringContent(text=llm_output)
        element.token_usage = TokenUsage(num_requests=1)
        element.status = "processed"
