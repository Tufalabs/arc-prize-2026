"""
Classes to support conversation structure in ULLM.

An important concept here is that we have two formats:
- User format: makes sense to users of ULLM, and is backend-agnostic. This is the format that users of ULLM will interact with.
- Model format: makes sense to the LLM backend, and is implemented specifically for each backend.
"""

# TODO: better management to prevent user from settings dependent fields

import contextlib
import contextvars
import hashlib
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, Optional

import numpy as np

import ullm.support as uls

if TYPE_CHECKING:
    import ullm.backend
    import ullm.tool

"""Define content types for conversation messages, always in user format."""


_HISTORY_REWRITE_OBSERVER: contextvars.ContextVar[Optional[Callable[["Conversation"], None]]] = contextvars.ContextVar(
    "ullm_history_rewrite_observer", default=None
)


@contextlib.contextmanager
def observe_history_rewrites(observer: Callable[["Conversation"], None]) -> Generator[None, None, None]:
    """Observe or reject declared rewrites in this task, including nested managers.

    The callback sees the conversation's old epoch before it advances. Keeping
    the hook context-local leaves conversation serialization and concurrent
    games independent of the observer.
    """
    token = _HISTORY_REWRITE_OBSERVER.set(observer)
    try:
        yield
    finally:
        _HISTORY_REWRITE_OBSERVER.reset(token)


@dataclass
class Content(uls.BaseClass):
    """Base class for content in conversation messages."""

    _model_contents: Any = field(
        default=None, init=False, repr=False
    )  # Used by the backend to cache content in model format

    def to_string(self) -> str:
        """Convert content to a string representation, mainly for logging."""
        raise NotImplementedError("Subclasses must implement to_string()")

    def tool_calls(self) -> list["ToolCallContent"]:
        """Extract all ToolCallContent items from this content tree."""
        return []


class EmptyContent(Content):
    """An empty content element, used when there is no content."""

    def to_string(self) -> str:
        """Empty content renders as the empty string."""
        return ""


@dataclass
class StringContent(Content):
    """Content that is a simple string."""

    text: str = field(default="")

    def to_string(self) -> str:
        """The text itself."""
        return self.text


@dataclass
class ListContent(Content):
    """Content element that is itself a list of other elements."""

    elements: list[Content] = field(default_factory=lambda: list[Content]())

    def to_string(self) -> str:
        """The elements' renderings, newline-joined."""
        return "\n".join(e.to_string() for e in self.elements)

    def tool_calls(self) -> list["ToolCallContent"]:
        """Every tool call among the elements, in order."""
        result = list["ToolCallContent"]()
        for e in self.elements:
            result.extend(e.tool_calls())
        return result


def _pixel_digest(data: np.ndarray) -> str:
    """SHA-256 over an RGB image's size and its pixels in row order."""
    digest = hashlib.sha256(f"RGB:{data.shape[1]}x{data.shape[0]}:".encode())
    digest.update(np.ascontiguousarray(data))
    return digest.hexdigest()


@dataclass(eq=False)
class ImageContent(Content):
    """An image carried as a numpy array of shape (H, W, 3), dtype uint8 (RGB).
    Backends are responsible for any encoding (PNG, base64, ...) needed by their
    wire format — this class is the canonical user-format representation.

    ``eq=False`` because numpy array equality returns an element-wise array, not
    a bool, which would break the auto-generated dataclass ``__eq__``.

    ``digest`` is the image's identity: SHA-256 over its size and pixels,
    taken once where the image is made and carried with it, so a backend that
    checks or keys every image of a long history on every call reads it
    instead of hashing the pixels again. Pixels are never changed in place;
    assigning ``data`` takes a new digest."""

    data: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.uint8))
    alt_text: str = ""  # Human-readable description; surfaced only in to_string() for logs
    digest: str = field(default="", init=False, repr=False)  # read through fingerprint()

    def __post_init__(self) -> None:
        """Take the digest once, where the image is made."""
        super().__post_init__()
        self.digest = _pixel_digest(self.data)

    def __setattr__(self, key: str, value: Any) -> None:
        """A new ``data`` is a new image, with a digest of its own."""
        super().__setattr__(key, value)
        if key == "data" and self._is_frozen:
            super().__setattr__("digest", _pixel_digest(value))

    def fingerprint(self) -> str:
        """The image's digest; taken here for one unpickled from before images carried it."""
        return self.digest or _pixel_digest(self.data)

    def _check_constraints(self) -> None:
        """Image data must be (H, W, 3) RGB."""
        super()._check_constraints()
        assert self.data.ndim == 3 and self.data.shape[2] == 3, (
            f"ImageContent.data must have shape (H, W, 3), got {self.data.shape}"
        )
        assert self.data.dtype == np.uint8, f"ImageContent.data must be dtype uint8, got {self.data.dtype}"

    def to_string(self) -> str:
        # Placeholder used in logs / human-format rendering. The pixel data is
        # never stringified — backends consume ``data`` directly.
        """A textual placeholder for the image."""
        h, w = self.data.shape[:2]
        if self.alt_text:
            return f"[image:{h}x{w} {self.alt_text}]"
        return f"[image:{h}x{w}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageContent):
            return NotImplemented
        return np.array_equal(self.data, other.data)


@dataclass
class ToolCallContent(Content):
    """A tool call produced by the LLM, appearing as content within an LLMResponse."""

    call_id: str = ""  # Unique ID for matching call to result
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    # Parser failures belong to this call even if another call was recovered.
    malformed_error: Optional[str] = None

    def to_string(self) -> str:
        """The call rendered for logs and human-format transcripts."""
        return f"[tool call {self.call_id}: {self.tool_name}({self.arguments})]"

    def tool_calls(self) -> list["ToolCallContent"]:
        """Itself: a tool call is its own only call."""
        return [self]


"""Define conversation elements; this can be single elements like a user prompt, up to the conversation as a whole."""


@dataclass
class ConversationElement(uls.BaseClass):
    """Base class for elements in a conversation."""

    _model_contents: Any = field(
        default=None, init=False, repr=False
    )  # Used by the backend to cache content in model format

    status: Literal["ready", "intermediate", "processed"] = field(
        default="ready"
    )  # State of the element in the processing pipeline
    # "ready" means has not started processing yet
    # "intermediate" means it is partial output from a running LLM
    # "processed" means done
    # Note that inputs to the LLM should always we "processed"

    def to_string(self) -> str:
        """Convert the conversation element to a string representation, mainly for logging."""
        return ""


@dataclass
class LLMInput(ConversationElement):
    """Superclass for anything that is sent as input to the LLM."""

    def _check_constraints(self) -> None:
        """Input elements are only ever handed on once processed."""
        assert self.status == "processed", "LLM input must be `processed`"
        return super()._check_constraints()

    def __post_init__(self) -> None:
        self.status = "processed"  # LLM inputs are always "processed"
        return super().__post_init__()


@dataclass
class SystemPrompt(LLMInput):
    """A system prompt in the conversation."""

    content: Content = field(default_factory=EmptyContent)

    # "xhigh" is Qwen3.8's top level (its set is low/medium/xhigh); backends
    # whose scale tops out at "high" downgrade it there, the same way "max"
    # already downgrades.
    reasoning_effort: Optional[Literal["low", "medium", "high", "max", "xhigh"]] = (
        "high"  # Not all backends support this
    )
    # TODO: properly support None (no effort specified). Also consider adding "none" and "minimal"
    #   levels (OpenAI supports these) if there's a use case.
    # If the backend has native reasoning_effort support, that is used.
    # Otherwise, the backend appends a fallback prompt to the system message content.

    tools: list["ullm.tool.Tool"] = field(
        default_factory=lambda: list["ullm.tool.Tool"]()
    )  # Tool definitions available to the LLM

    def to_string(self) -> str:
        """The system prompt with its tools and reasoning effort."""
        parts = ["=== SYSTEM PROMPT"]
        parts.append(self.content.to_string())
        if self.reasoning_effort is not None:
            parts.append(f"  reasoning_effort: {self.reasoning_effort}")
        for t in self.tools:
            parts.append(f"  tool: {t.name} — {t.description}\n    parameters: {t.parameters}")
        return "\n".join(parts)


@dataclass
class UserPrompt(LLMInput):
    """A user prompt in the conversation, i.e. content going from the user to the LLM."""

    content: Content = field(default_factory=EmptyContent)

    def to_string(self) -> str:
        """The user prompt, under its header."""
        return "=== USER PROMPT \n" + self.content.to_string()


@dataclass
class ToolResult(LLMInput):
    """Result of a tool execution, sent back to the LLM as input."""

    call_id: str = ""  # Matches the ToolCallContent.call_id
    tool_name: str = ""
    content: Content = field(default_factory=EmptyContent)
    is_error: bool = False  # Whether the tool execution failed (schema validation, runtime error, etc.)
    malformed_call_count: int = 0  # Excludes tool-body exceptions and budget cancellation.

    def to_string(self) -> str:
        """The tool result, naming its call and flagging an error."""
        error_tag = " [ERROR]" if self.is_error else ""
        return f"=== TOOL RESULT {self.call_id} ({self.tool_name}{error_tag})\n{self.content.to_string()}"


StopReason = Literal["stop", "max_tokens", "tool_use", "stop_sequence", "other", "malformed"]
# Normalized reason the model stopped generating:
#   "stop"          — natural end-of-turn
#   "max_tokens"    — generation hit the token budget
#   "tool_use"      — stopped to call a tool
#   "stop_sequence" — hit a configured stop sequence
#   "other"         — anything else (content filter, refusal, etc.)


@dataclass
class LLMResponse(ConversationElement):
    """A response from the LLM, i.e. content produced by the LLM (to the user, to a tool, etc.)."""

    reasoning_content: Content = field(default_factory=EmptyContent)
    final_content: Content = field(default_factory=EmptyContent)
    token_usage: Optional["ullm.backend.TokenUsage"] = None  # Populated by the backend in process_output
    stop_reason: Optional[StopReason] = None  # Populated by the backend in process_output
    # Set True on artificial/synthetic LLMResponses (e.g. placeholders
    # inserted to keep the conversation well-formed after an error). All
    # backends skip flagged elements during request construction, so they
    # don't appear in what's sent to the model.
    do_not_send_to_agent: bool = field(default=False)
    # Number of MalformedOutputError retries that fired before this response
    # was produced. 0 means the first attempt succeeded. Set by
    # ResponseManager at the end of its retry loop; also set on the
    # synthetic placeholder basic_learner injects when the retry budget is
    # exhausted (equal to max_retries there).
    malformed_retries: int = field(default=0)
    # The backend request that produced this response. Stamped in
    # ``respond()`` where the id is claimed, because the backend counter is
    # shared: a caller reading it around the call attributes the id of
    # whichever request claimed one in between.
    request_id: Optional[int] = None
    generation_request_id: Optional[str] = None  # Engine run identity and unique generation nonce.
    # Why an empty reply or a parser failure was accepted as a model turn
    # (``Backend.accept_malformed_output``). Parser failures keep raw text as
    # final content. None on replies accepted without a malformed marker.

    malformed_error: Optional[str] = None
    # Syntax/JSON/schema/unknown-tool failures, and a retained reply with no
    # text and no tool call; finalized by TurnManager under training.
    malformed_call_count: int = 0
    truncation_cap_source: Optional[Literal["response", "episode"]] = None
    # True iff this response (or a synthetic-placeholder one carrying this
    # flag) marks the moment the TurnManager truncated its response loop
    # because the per-turn completion-token cap was reached. Set on the
    # truncated response itself when the backend returned with
    # stop_reason='max_tokens' AND the TurnManager had lowered max_tokens
    # below the sampling-params default for that call. Also set on a
    # synthetic placeholder when the cap was already exhausted before the
    # next call (cap_left <= 0 at the top of the response loop). Downstream
    # callers (e.g. basic_learner) key off this to surface a give-up.
    truncated_by_per_turn_cap: bool = field(default=False)
    # Text the backend served after the generation prompt and the model
    # continued from (``GenericSamplingParameters.response_injection``): the
    # start of this response's text as history and readers see it, sampled
    # by nobody. None when nothing was injected.
    response_injection: Optional[str] = None
    # Token-continuation witness (``ullm.continuation.ServedTurn``): the ids
    # this answer was served after and sampled, so the next call can extend
    # the history by tokens. None when the backend keeps no witness.
    _served: Any = field(default=None, init=False, repr=False)

    def to_string(self) -> str:
        """The response, with reasoning shown separately from the final content."""
        parts = list[str]()
        if self.reasoning_content.to_string():
            if self.status == "processed" or self.final_content.to_string() != "":  # reasoning finished
                parts.append("= REASONING \n" + self.reasoning_content.to_string() + "\n= END REASONING")
            else:  # reasoning ongoing
                parts.append("= REASONING \n" + self.reasoning_content.to_string())
        if self.final_content.to_string():
            parts.append(self.final_content.to_string())
        if self.token_usage is not None or self.stop_reason is not None:
            bracket_parts = list[str]()
            if self.token_usage is not None:
                bracket_parts.append(self.token_usage.to_string())
            if self.stop_reason is not None:
                bracket_parts.append(f"stop={self.stop_reason}")
            parts.append(f"  [{', '.join(bracket_parts)}]")
        return "=== LLM RESPONSE \n" + "\n".join(parts)


@dataclass
class ResponseTurn(ConversationElement):
    """A full LLM response chain to a user prompt, including reasoning and tool calls."""

    elements: list[ConversationElement] = field(default_factory=lambda: list[ConversationElement]())
    # Typical structure: [LLMResponse, ToolCall, ToolResponse, ToolCall, ToolResponse, LLMResponse]; not enforced

    @property
    def last_response(self) -> "LLMResponse":
        """The last LLMResponse in this turn."""
        assert isinstance(self.elements[-1], LLMResponse), "Last element of turn is not an LLMResponse"
        return self.elements[-1]

    def _check_constraints(self) -> None:
        """Only the last element of a turn may still be in flight."""
        for e in self.elements[:-1]:
            assert e.status == "processed", "All but the last element of a ResponseTurn must be `processed`"
        if self.status == "processed":
            assert self.elements[-1].status == "processed", (
                "Last element of a ResponseTurn must be `processed` if the turn is `processed`"
            )
        return super()._check_constraints()

    def to_string(self) -> str:
        """The turn's elements, under its header."""
        return "===== LLM TURN\n" + "\n".join(e.to_string() for e in self.elements)


@dataclass
class Conversation(ConversationElement):
    """A full conversation, consisting of multiple turns."""

    system_prompt: SystemPrompt = field(default_factory=SystemPrompt)
    elements: list[ConversationElement] = field(default_factory=lambda: list[ConversationElement]())
    # Typical structure of elements: [UserPrompt, ResponseTurn, UserPrompt, ResponseTurn]; not enforced
    # Bumped by whoever rewrites history the model has already been sent
    # (compaction, eviction, a rewritten system prompt) — see
    # ``mark_history_rewritten``. A backend continuing the served history
    # by tokens keys on it: unchanged means the sent prefix must still hold.
    history_epoch: int = field(default=0)
    # Names this conversation's thread for the engine's prefix cache, across
    # the Conversation objects a caller builds around one history (a vLLM
    # backend tags each request with ``<run_id>/<conversation_id>`` and the
    # epoch; ``Backend.close_context`` ends it). None: untagged.
    conversation_id: Optional[str] = field(default=None)

    def mark_history_rewritten(self) -> None:
        """Declare that elements the model was already sent have changed.
        Appending new elements, or dropping elements after the last answer,
        is not a rewrite — the served prefix still stands."""
        observer = _HISTORY_REWRITE_OBSERVER.get()
        if observer is not None:
            observer(self)
        self.history_epoch += 1

    @property
    def last_turn(self) -> "ResponseTurn":
        """The last ResponseTurn in the conversation."""
        assert isinstance(self.elements[-1], ResponseTurn), "Last element of conversation is not a ResponseTurn"
        return self.elements[-1]

    def _check_constraints(self) -> None:
        """Only the last element of a conversation may still be in flight."""
        for e in self.elements[:-1]:
            assert e.status == "processed", "All but the last element of a Conversation must be `processed`"
        if self.status == "processed" and len(self.elements) > 0:
            assert self.elements[-1].status == "processed", (
                "Last element of a Conversation must be `processed` if the conversation is `processed`"
            )
        return super()._check_constraints()

    def to_string(self) -> str:
        """The whole conversation, system prompt first."""
        parts = ["======= CONVERSATION", self.system_prompt.to_string()]
        parts.extend(e.to_string() for e in self.elements)
        return "\n".join(parts)


def strip_model_contents(obj: Any) -> None:
    """Recursively clear `_model_contents` on Content / ConversationElement instances.

    Mutates in place. Intended for deepcopies destined for diagnostics / logs:
    backends cache raw model-format payloads (vLLM RequestOutput, Anthropic
    Message objects, etc.) on these fields and they can be large. Do NOT call
    on a live conversation — stripping breaks history-reconstruction paths
    that read `_model_contents` (see backends/common.py, vllmx.py)."""
    if isinstance(obj, list):
        for item in obj:
            strip_model_contents(item)
        return
    if not isinstance(obj, (Content, ConversationElement)):
        return
    obj._model_contents = None
    for f in fields(obj):
        value = getattr(obj, f.name, None)
        if isinstance(value, (Content, ConversationElement, list)):
            strip_model_contents(value)
