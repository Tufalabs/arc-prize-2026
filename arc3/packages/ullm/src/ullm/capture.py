"""Capture and replay of backend traffic (platypus U3/U4).

Capture is a mode on the ``Backend`` superclass: when ``capture_sink`` is
set, base ``respond()`` records one entry per call — the canonical
request plus the outcome (a processed response, a malformed-output
error, or a prompt-too-long error) — into two append-only JSONL streams
in the sink's directory:

- ``records.jsonl``: one line per call, in call order. Large text and
  image payloads are stored by content hash.
- ``blocks.jsonl``: one line per content block, appended at first local
  occurrence. Duplicate lines are legal (a restored sink loses its
  seen-set); readers dedup by hash.

This module is deliberately identity-blind: a sink is a directory plus a
call counter. Who a stream belongs to (game, pass, run) is expressed by
WHERE the caller points the directory.

``ReplayBackend`` is the counterpart: it reads one record stream in
order — call N must be record N — verifies the incoming canonical
request byte-for-byte against the recorded one (content is the check,
the ordinal is the key; content-keyed lookup is ambiguous because
identical requests legitimately occur), and emits the recorded outcome
faithfully, recorded token usage included. Any divergence is a hard
error carrying a readable diff.

Canonical request identity = user-format conversation + the sampling
parameters exactly as passed (unresolved: backend-side defaults such as
``send_back_reasoning`` are generation-time configuration, not agent
behavior). ``_model_contents`` is backend-native and never captured.
"""

import base64
import copy
import difflib
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import numpy as np

import ullm.backend
import ullm.conv
import ullm.support as uls

# Text shorter than this is inlined in the canonical structure; longer text
# goes to the block stream. Small enough that records stay readable, large
# enough that the block stream isn't full of fragments.
_INLINE_TEXT_LIMIT = 128


class ReplayMismatchError(Exception):
    """The incoming request differs from the recorded one at this call."""


class ReplayExhaustedError(Exception):
    """More calls were made than the record stream contains."""


# ---------------------------------------------------------------------------
# Canonical serialization (user format -> JSON structure + content blocks)
# ---------------------------------------------------------------------------


def _hash_text(text: str) -> str:
    """Content identity for a text block: sha256 of its UTF-8 bytes."""
    return hashlib.sha256(text.encode()).hexdigest()


def _add_text(text: str, blocks: dict[str, Any]) -> Any:
    """Inline short text; long text becomes a block reference."""
    if len(text) < _INLINE_TEXT_LIMIT:
        return text
    h = _hash_text(text)
    blocks[h] = text
    return {"$block": h}


def _resolve_text(obj: Any, blocks: dict[str, Any]) -> str:
    """Inverse of ``_add_text``: an inline string, or the block it refers to."""
    if isinstance(obj, dict):
        return str(blocks[obj["$block"]])
    return str(obj)


def _content_to_json(content: ullm.conv.Content, blocks: dict[str, Any]) -> Any:
    """One content object -> its canonical JSON shape, appending any large
    payload (long text, image bytes) to ``blocks`` and referencing it."""
    if isinstance(content, ullm.conv.StringContent):
        return {"c": "str", "text": _add_text(content.text, blocks)}
    if isinstance(content, ullm.conv.ListContent):
        return {"c": "list", "elements": [_content_to_json(e, blocks) for e in content.elements]}
    if isinstance(content, ullm.conv.ImageContent):
        raw = content.data.tobytes()
        h = hashlib.sha256(raw).hexdigest()
        blocks[h] = {"image_b64": base64.b64encode(raw).decode(), "shape": list(content.data.shape)}
        return {"c": "image", "block": h, "alt": content.alt_text}
    if isinstance(content, ullm.conv.ToolCallContent):
        return {
            "c": "tool_call",
            "call_id": content.call_id,
            "tool_name": content.tool_name,
            "arguments": content.arguments,
        }
    if isinstance(content, ullm.conv.EmptyContent):
        return {"c": "empty"}
    raise TypeError(f"Cannot canonicalize content of type {type(content).__name__}")


def _content_from_json(obj: Any, blocks: dict[str, Any]) -> ullm.conv.Content:
    """Inverse of ``_content_to_json``, resolving block references."""
    kind = obj["c"]
    if kind == "str":
        return ullm.conv.StringContent(text=_resolve_text(obj["text"], blocks))
    if kind == "list":
        return ullm.conv.ListContent(elements=[_content_from_json(e, blocks) for e in obj["elements"]])
    if kind == "image":
        block = blocks[obj["block"]]
        data = np.frombuffer(base64.b64decode(block["image_b64"]), dtype=np.uint8)
        return ullm.conv.ImageContent(data=data.reshape(block["shape"]).copy(), alt_text=obj["alt"])
    if kind == "tool_call":
        return ullm.conv.ToolCallContent(call_id=obj["call_id"], tool_name=obj["tool_name"], arguments=obj["arguments"])
    if kind == "empty":
        return ullm.conv.EmptyContent()
    raise ValueError(f"Unknown content kind {kind!r}")


def _element_to_json(
    element: ullm.conv.ConversationElement, blocks: dict[str, Any], include_bookkeeping: bool = False
) -> Any:
    """Serialize the fields that shape what the model sees. Bookkeeping
    fields (token_usage, retry counters) are excluded by default — they
    don't affect request construction and would create spurious replay
    mismatches. ``include_bookkeeping`` adds them, for hosts persisting
    finished conversations for display (``serialize_conversation``)."""
    if isinstance(element, ullm.conv.UserPrompt):
        return {"kind": "user", "content": _content_to_json(element.content, blocks)}
    if isinstance(element, ullm.conv.ToolResult):
        return {
            "kind": "tool_result",
            "call_id": element.call_id,
            "tool_name": element.tool_name,
            "is_error": element.is_error,
            "content": _content_to_json(element.content, blocks),
        }
    if isinstance(element, ullm.conv.LLMResponse):
        out = {
            "kind": "response",
            "reasoning": _content_to_json(element.reasoning_content, blocks),
            "final": _content_to_json(element.final_content, blocks),
            "do_not_send_to_agent": element.do_not_send_to_agent,
        }
        if include_bookkeeping:
            usage = element.token_usage
            out["stop_reason"] = element.stop_reason
            out["malformed_retries"] = element.malformed_retries
            out["response_injection"] = element.response_injection
            out["usage"] = (
                None
                if usage is None
                else {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                }
            )
        return out
    if isinstance(element, ullm.conv.ResponseTurn):
        return {
            "kind": "turn",
            "elements": [_element_to_json(e, blocks, include_bookkeeping) for e in element.elements],
        }
    raise TypeError(f"Cannot canonicalize element of type {type(element).__name__}")


def _sampling_to_json(sampling_params: ullm.backend.SamplingParameters) -> Any:
    """Resolved sampling parameters as canonical JSON. Replay compares these
    verbatim, so every field a backend acts on has to appear here."""
    if not isinstance(sampling_params, ullm.backend.GenericSamplingParameters):
        raise TypeError(f"capture/replay supports GenericSamplingParameters only, got {type(sampling_params).__name__}")
    out: dict[str, Any] = {
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "max_tokens": sampling_params.max_tokens,
        "stop_sequences": sampling_params.stop_sequences,
        "seed": sampling_params.seed,
        "extra_params": sampling_params.extra_params,
        "send_back_reasoning": sampling_params.send_back_reasoning,
        "response_injection": sampling_params.response_injection,
    }
    answer = sampling_params.after_reasoning
    if answer is not None:
        # Only when set, so every stream recorded before the field existed
        # still compares equal.
        out["after_reasoning"] = {
            "temperature": answer.temperature,
            "top_p": answer.top_p,
            "extra_params": answer.extra_params,
        }
    return out


def canonicalize_text_request(text: str) -> Any:
    """The canonical form of one raw-text count request: content-addressed,
    so the stream stays linear in unique text while a mismatched retention
    input still fails the byte-for-byte request comparison."""
    import hashlib as _hashlib

    return {
        "op": "text_token_count",
        "text_sha256": _hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }


def canonicalize_request(
    sampling_params: ullm.backend.SamplingParameters, conversation: ullm.conv.Conversation, blocks: dict[str, Any]
) -> Any:
    """The request identity: conversation as the model-input structure +
    sampling params as passed. New content blocks are added to ``blocks``."""
    system = conversation.system_prompt
    return {
        "system": {
            "content": _content_to_json(system.content, blocks),
            "reasoning_effort": system.reasoning_effort,
            "tools": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in system.tools],
        },
        # The trailing intermediate ResponseTurn (the one being generated
        # into) carries no model input yet; earlier elements are the prompt.
        "elements": [_element_to_json(e, blocks) for e in conversation.elements[:-1]],
        "sampling": _sampling_to_json(sampling_params),
    }


def serialize_conversation(conversation: ullm.conv.Conversation, blocks: dict[str, Any]) -> Any:
    """Serialize a WHOLE conversation — system prompt plus every element,
    response turns and their bookkeeping (usage, stop reasons, retry
    counts) included — to the canonical JSON structure, adding content
    blocks to ``blocks``. Superset of :func:`canonicalize_request` (which
    serializes only the model-input identity), for hosts persisting
    finished conversations for display (e.g. diagnostics)."""
    system = conversation.system_prompt
    return {
        "system": {
            "content": _content_to_json(system.content, blocks),
            "reasoning_effort": system.reasoning_effort,
            "tools": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in system.tools],
        },
        "elements": [_element_to_json(e, blocks, include_bookkeeping=True) for e in conversation.elements],
    }


def canonical_dumps(obj: Any) -> str:
    """The one serialization replay comparisons are made in: sorted keys and
    no incidental whitespace, so equal structures give equal bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _response_to_json(element: ullm.conv.LLMResponse, blocks: dict[str, Any]) -> Any:
    """A completed response as canonical JSON: content, stop reason, usage."""
    usage = element.token_usage
    assert usage is not None
    return {
        "reasoning": _content_to_json(element.reasoning_content, blocks),
        "final": _content_to_json(element.final_content, blocks),
        "stop_reason": element.stop_reason,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cached_tokens": usage.cached_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "num_requests": usage.num_requests,
        },
    }


def _response_into_element(obj: Any, blocks: dict[str, Any], element: ullm.conv.LLMResponse) -> None:
    """Fill a fresh response element from its recorded JSON, in place — replay
    hands back the element the backend appended, rather than a new one."""
    element.reasoning_content = _content_from_json(obj["reasoning"], blocks)
    element.final_content = _content_from_json(obj["final"], blocks)
    element.stop_reason = obj["stop_reason"]
    element.token_usage = ullm.backend.TokenUsage(**obj["usage"])
    element.status = "processed"


def _resolve_for_diff(obj: Any, blocks: dict[str, Any]) -> Any:
    """Inline block contents (text only; images stay refs) so mismatch
    diffs read as prose, not hashes."""
    if isinstance(obj, dict):
        if "$block" in obj:
            block = blocks.get(obj["$block"], "<missing block>")
            return block if isinstance(block, str) else obj
        return {k: _resolve_for_diff(v, blocks) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_for_diff(v, blocks) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Sink (capture side)
# ---------------------------------------------------------------------------


@dataclass
class CaptureSink(uls.BaseClass):
    """Appends call records + content blocks under ``directory``. Identity-
    blind: the caller expresses whose stream this is by where it points
    the directory. Append-and-close per record — capture runs are test
    generation, robustness beats throughput."""

    directory: str = field(kw_only=True)
    _seen_blocks: set[str] = field(default_factory=lambda: set[str](), init=False, repr=False)
    _next_index: int = field(default=0, init=False, repr=False)

    def record(self, request: Any, blocks: dict[str, Any], kind: str, outcome: Any) -> None:
        """Append one call's record, writing only the blocks this stream has not
        stored before — which is what keeps storage linear in unique content
        rather than quadratic in calls (U2).
        """
        os.makedirs(self.directory, exist_ok=True)
        new_blocks = {h: b for h, b in blocks.items() if h not in self._seen_blocks}
        if new_blocks:
            with open(os.path.join(self.directory, "blocks.jsonl"), "a", encoding="utf-8") as f:
                for h, b in new_blocks.items():
                    f.write(canonical_dumps({"hash": h, "block": b}) + "\n")
            self._seen_blocks.update(new_blocks)
        line = {"index": self._next_index, "kind": kind, "request": request, "outcome": outcome}
        with open(os.path.join(self.directory, "records.jsonl"), "a", encoding="utf-8") as f:
            f.write(canonical_dumps(line) + "\n")
        self._next_index += 1


def load_stream(directory: str) -> tuple[list[Any], dict[str, Any]]:
    """Read a capture directory back: (records in order, blocks by hash).
    Duplicate block lines are legal and deduped here."""
    blocks: dict[str, Any] = {}
    blocks_path = os.path.join(directory, "blocks.jsonl")
    if os.path.exists(blocks_path):
        with open(blocks_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                blocks[entry["hash"]] = entry["block"]
    records: list[Any] = []
    records_path = os.path.join(directory, "records.jsonl")
    if os.path.exists(records_path):
        with open(records_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    for i, rec in enumerate(records):
        assert rec["index"] == i, f"record stream corrupt: line {i} carries index {rec['index']}"
    return records, blocks


# ---------------------------------------------------------------------------
# Replay backend
# ---------------------------------------------------------------------------


@dataclass
class ReplayBackend(ullm.backend.Backend):
    """Serves one recorded stream back, in order, verifying every request.

    Call N must equal record N: the incoming canonical request is compared
    byte-for-byte (as canonical JSON) against the recorded one, and any
    divergence raises :class:`ReplayMismatchError` with a readable diff.
    Recorded outcomes are emitted faithfully — a processed response
    (recorded usage included, so token accounting reproduces bit-exact),
    a re-raised ``MalformedOutputError``, or a ``PromptTooLongError``.
    Picklable (records load lazily), so it crosses into game subprocesses
    like any backend."""

    directory: str = field(kw_only=True)
    _records: Optional[list[Any]] = field(default=None, init=False, repr=False)
    _blocks: Optional[dict[str, Any]] = field(default=None, init=False, repr=False)
    _call_index: int = field(default=0, init=False, repr=False)

    def supports_response_injection(self) -> bool:
        """Recordings replay whatever they hold: the canonical request carries
        the injection, so a stream recorded without one mismatches instead."""
        return True

    def supports_after_reasoning(self) -> bool:
        """As with injection: the canonical request carries the answer's
        sampling, so a stream recorded without it mismatches instead."""
        return True

    def _transient_fields(self) -> tuple[str, ...]:
        """The loaded stream is rebuilt on demand rather than copied."""
        return ("_records", "_blocks")

    def _rebuild_transient(self) -> None:
        """Forget the loaded stream; the next access re-reads it."""
        self._records = None
        self._blocks = None

    def _ensure_loaded(self) -> tuple[list[Any], dict[str, Any]]:
        """The stream's records and blocks, read from disk on first use."""
        if self._records is None or self._blocks is None:
            self._records, self._blocks = load_stream(self.directory)
        return self._records, self._blocks

    def unconsumed(self) -> int:
        """Records not yet served — nonzero after a run hints the agent
        under test made fewer calls than the recorded one (worth a look,
        not proof of divergence)."""
        records, _ = self._ensure_loaded()
        return len(records) - self._call_index

    async def text_token_count(self, text: str) -> Optional[int]:
        """Serve the recorded raw-text count -- ordinal and kind-checked
        exactly like ``token_count``: retention driven by a different number
        would keep a different journal and stop being the recorded run."""
        records, _ = self._ensure_loaded()
        index = self._call_index
        self._call_index += 1
        if index >= len(records):
            raise ReplayExhaustedError(
                f"replay exhausted: the recorded stream has {len(records)} call(s) but call "
                f"#{index + 1} was made -- the agent under test makes more LLM calls than the recorded one"
            )
        record = records[index]
        incoming = canonicalize_text_request(text)
        if canonical_dumps(incoming) != canonical_dumps(record["request"]):
            raise ReplayMismatchError(
                f"replay mismatch at call #{index}: the text_token_count request differs from the "
                f"recorded one -- agent behavior changed, or a determinism (A2) violation."
            )
        if record["kind"] != "text_token_count":
            raise ReplayMismatchError(
                f"replay mismatch at call #{index}: recorded a {record['kind']!r} call but the agent "
                f"made a text_token_count call -- the agent's counting and calling no longer "
                f"interleave the way the recorded run's did."
            )
        return record["outcome"]["count"]

    async def token_count(
        self, sampling_params: ullm.backend.SamplingParameters, conversation: ullm.conv.Conversation
    ) -> Optional[int]:
        """Serve the recorded count. Counting drives eviction, so it is part
        of the stream and matched by ordinal exactly like a response: a run
        that counted differently would evict elsewhere and stop being the
        recorded run."""
        record = self._next_record("token_count", sampling_params, conversation)
        return record["outcome"]["count"]

    def _next_record(
        self, expect_kind: str, sampling_params: ullm.backend.SamplingParameters, conversation: ullm.conv.Conversation
    ) -> Any:
        """Take the next record, verify the request matches it byte-for-byte,
        and check it is the kind of call the agent just made."""
        records, blocks = self._ensure_loaded()
        index = self._call_index
        self._call_index += 1
        if index >= len(records):
            raise ReplayExhaustedError(
                f"replay exhausted: the recorded stream has {len(records)} call(s) but call "
                f"#{index + 1} was made — the agent under test makes more LLM calls than the recorded one"
            )
        record = records[index]
        incoming_blocks: dict[str, Any] = {}
        incoming = canonicalize_request(sampling_params, conversation, incoming_blocks)
        if canonical_dumps(incoming) != canonical_dumps(record["request"]):
            expected = json.dumps(_resolve_for_diff(record["request"], blocks), sort_keys=True, indent=1)
            got = json.dumps(_resolve_for_diff(incoming, incoming_blocks), sort_keys=True, indent=1)
            diff = "\n".join(
                difflib.unified_diff(
                    expected.splitlines(),
                    got.splitlines(),
                    fromfile=f"recorded call #{index}",
                    tofile=f"incoming call #{index}",
                    lineterm="",
                    n=3,
                )
            )
            raise ReplayMismatchError(
                f"replay mismatch at call #{index}: the request differs from the recorded one — "
                f"agent behavior changed, or a determinism (A2) violation.\n{diff}"
            )
        is_count = record["kind"] == "token_count"
        if is_count != (expect_kind == "token_count"):
            raise ReplayMismatchError(
                f"replay mismatch at call #{index}: recorded a {record['kind']!r} call but the agent "
                f"made a {expect_kind!r} call — the agent's counting and calling no longer interleave "
                f"the way the recorded run's did."
            )
        return record

    async def respond(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        conversation: ullm.conv.Conversation,
        run_id: Optional[str] = None,  # accepted for signature parity; replay never traces
    ) -> AsyncGenerator[ullm.conv.LLMResponse, None]:
        """Serve the next recorded call. Ordinal, not associative: call N must be
        record N, and a request that does not match it is an error with a diff
        rather than a silent substitution (U4).
        """
        assert conversation.status == "intermediate", "Conversation must be `intermediate` to run LLM"
        turn = conversation.last_turn
        assert turn.status == "intermediate", "Last element of conversation must be `intermediate` to run LLM"
        _, blocks = self._ensure_loaded()
        record = self._next_record("respond", sampling_params, conversation)
        kind = record["kind"]
        if kind == "prompt_too_long":
            raise ullm.backend.PromptTooLongError(record["outcome"]["error"])
        # Both remaining kinds appended an element in the original run.
        element = ullm.conv.LLMResponse()
        turn.elements.append(element)
        if kind == "malformed":
            raise ullm.backend.MalformedOutputError(record["outcome"]["error"])
        assert kind == "response", f"unknown record kind {kind!r}"
        _response_into_element(record["outcome"], blocks, element)
        # The record matched this request, so the text it asks to inject is
        # the text the recorded call served: a replayed response names its
        # served prefix instead of presenting it as sampled.
        element.response_injection = self.resolve_response_injection(sampling_params)
        assert element.token_usage is not None
        self.total_usage += element.token_usage
        yield element


def capture_record(
    sink: CaptureSink,
    request: Any,
    blocks: dict[str, Any],
    element: Optional[ullm.conv.LLMResponse],
    error: Optional[BaseException],
    kind: str,
) -> None:
    """Assemble and write one record (called from ``Backend.respond``)."""
    if kind == "response":
        assert element is not None
        outcome = _response_to_json(element, blocks)
    else:
        outcome = {"error": str(error)}
    sink.record(request, blocks, kind, outcome)


def deepcopy_blocks(blocks: dict[str, Any]) -> dict[str, Any]:
    """Detach a block map so later mutation of the originals cannot reach a
    record already handed to a sink."""
    return copy.deepcopy(blocks)
