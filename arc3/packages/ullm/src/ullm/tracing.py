"""Trace-step production surface (platypus U5, taaf R5/R12.02).

A backend that can produce R5-grade training-trace records — exact token
ids straight from the engine plus a passing input-token roundtrip —
advertises it via ``Backend.trace_capability()`` and implements
``Backend._build_trace_step()``. Completed records land on
``Backend.trace_recorder``, a host-side collector (``platypus.tracing``)
that owns step numbering, action linkage, media splitting, and the taaf
``register_trace`` boundary. Backends REPORT step data; they never write
trace files themselves.

Validation posture (matches ARC3-Inference's ``taaf_trace``): the witness
token ids always come from the engine, never from our own tokenization,
and the backend re-renders the JSON-round-tripped stored ``messages``
through its own serving tokenizer/processor to prove the stored portable
side reproduces them. An input-token mismatch is a hard error that stops
the run (R12.02: the trace would not be reproducible for SFT). Output
reconstruction is recorded evidence, not a gate (R5.38). Identity
metadata is recorded, not exercised — the trace consumer (ARC3-Training)
trains directly off the stored token ids.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional

import ullm.backend
import ullm.conv


@dataclass(frozen=True)
class PackedJson:
    """A JSON value a step's ``extra`` holds compressed until its trace is
    written (the collector's ``finalize`` unpacks it), for a value too big to
    hold as Python objects for a whole game: an API teacher's per-token
    alternatives are some 6.8 KB a token as dicts, 1.2 KB as JSON, a fraction
    of that compressed, and a collector holds every step of every game in
    play (2026-10-29). Lossless: JSON round-trips Python's floats,
    ints and strings exactly, so the written trace is the unpacked value's."""

    data: bytes

    @classmethod
    def pack(cls, value: Any) -> PackedJson:
        """``value`` (JSON-shaped) as compressed JSON."""
        return cls(zlib.compress(json.dumps(value, separators=(",", ":")).encode(), 1))

    def unpack(self) -> Any:
        """The value packed, equal to it."""
        return json.loads(zlib.decompress(self.data))


def unpack_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """``extra`` with its ``PackedJson`` values unpacked, the form a trace is
    written in."""
    return {key: value.unpack() if isinstance(value, PackedJson) else value for key, value in extra.items()}


@dataclass(frozen=True)
class TraceCapability:
    """Identity of the serving stack behind a trace-capable backend.

    Feeds taaf ``Trace``'s mandatory identity block (R5.21) and the
    per-step ``provider_validation`` provenance keys. All strings; empty
    means "not applicable / unknown" (e.g. ``model_weight_hash`` for a
    base model — same posture as ARC3-Inference, which only fills it for
    LoRA adapters from the runtime export summary)."""

    provider_name: str
    provider_version: str
    api_kind: str
    provider_model_id: str
    validation_method: str  # immutable adapter id, e.g. "platypus.vllmx.inprocess_roundtrip.v1"
    model_id: str
    model_revision: str = ""
    model_weight_hash: str = ""
    parent_model_revision: str = ""
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    processor_id: str = ""
    processor_revision: str = ""
    vision_preprocessor_revision: str = ""
    # The trace contents this backend can produce (taaf R5.39): every
    # trace-capable backend writes "full"; "ids" needs a producer that omits
    # the portable side rather than one that ignores the flag.
    contents: tuple[str, ...] = ("full",)


@dataclass
class TraceImage:
    """One image that entered the model call, as encoded bytes.

    ``png`` holds the exact bytes the collector will write under the
    trace's ``media/`` dir (PNG unless ``mime`` says otherwise). For
    backends whose messages carry only image *markers* (vllmx), these are
    the images in marker order; for data-URL backends (OAI) the collector
    splits the URLs out of the messages itself and this list stays empty.
    """

    png: bytes
    mime: str = "image/png"
    role: str = "conversation_image"


@dataclass
class TraceStepRecord:
    """One model call's trace data, backend-produced, collector-consumed.

    ``messages`` is the JSON-round-tripped request conversation exactly as
    the serving stack rendered it (data-URL images still inline on the
    OAI path — the collector splits them). ``request_payload`` is the
    JSON-able request the payload hashes are computed over; it must
    contain ``messages`` (the collector derives ``request_args`` by
    dropping that key, per R5.31)."""

    run_id: str
    messages: list[dict[str, Any]]
    input_token_ids: list[int]
    output_token_ids: list[int]
    output_text: str  # exact detokenization of output_token_ids
    output_message: Optional[dict[str, Any]]
    sampling_params: dict[str, Any]
    request_payload: dict[str, Any]
    images: list[TraceImage] = field(default_factory=lambda: list[TraceImage]())
    # Statuses per R5.31/R5.38 — input is validated to "ok" backend-side
    # (anything else raised), output/media are recorded evidence.
    input_token_reconstruction_status: str = "ok"
    output_token_reconstruction_status: str = "unavailable"
    media_reconstruction_status: str = "not_applicable"
    uncached_input_tokens: int = 0
    # Per-output-token alternatives for ``TraceStep.logprobs``, aligned 1:1
    # to output_token_ids (taaf validates that length). None means the call
    # did not ask for them, which is the default: they are paid for on every
    # step and only a scoring consumer reads them.
    logprobs: Optional[list[Any]] = None
    # Lands in TraceStep.extra — e.g. stop_reason, which consumers need to
    # filter truncated/malformed completions out of SFT batches.
    extra: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


class TraceReconstructionError(RuntimeError):
    """An input-token roundtrip failed: the stored messages do not
    reproduce the prompt the engine consumed. The trace would be unusable
    for SFT (R12.02), so this stops the run."""


def sampling_to_dict(params: ullm.backend.SamplingParameters) -> dict[str, Any]:
    """JSON-safe resolved sampling parameters for ``TraceStep.sampling_params``."""
    if not isinstance(params, ullm.backend.GenericSamplingParameters):
        return {"type": type(params).__name__}
    out: dict[str, Any] = {}
    for name in (
        "temperature",
        "top_p",
        "max_tokens",
        "stop_sequences",
        "seed",
        "send_back_reasoning",
        "response_injection",
    ):
        value = getattr(params, name)
        if value is not None:
            out[name] = value
    for key, value in params.extra_params.items():
        out.setdefault(key, value)
    return out


def logprobs_from_vllm(completion: Any, output_token_ids: list[int]) -> Optional[list[Any]]:
    """``TraceStep.logprobs`` for one vLLM ``CompletionOutput``, or None when
    the call did not ask for logprobs.

    Per sampled position: the sampled token's own logprob and vocab rank,
    plus the requested top-k alternatives as ``[token_id, logprob, rank]``
    triples. Ids stay ints (JSON object keys could not), and the decoded
    strings are dropped — a scoring consumer works in token ids, and the
    text is already in ``output_text``.

    Length is checked here rather than at the taaf boundary: a mismatch
    fails validation for the whole trace at write time, which surfaces as
    a run that played every game and then stored nothing.
    """
    positions = getattr(completion, "logprobs", None)
    if positions is None:
        return None
    if len(positions) != len(output_token_ids):
        raise TraceReconstructionError(
            f"vLLM returned {len(positions)} logprob positions for {len(output_token_ids)} sampled "
            "tokens; TraceStep.logprobs must align one-to-one with output_token_ids"
        )
    out = list[Any]()
    for index, (token_id, position) in enumerate(zip(output_token_ids, positions)):
        # vLLM always includes the sampled token, even when it ranks below
        # the requested k — but a null-safe read keeps a future engine that
        # omits it from crashing a run over evidence.
        sampled = position.get(token_id)
        if sampled is not None and not math.isfinite(float(sampled.logprob)):
            # The direct signal of a broken sampler (NaN logits); a consumer
            # weighting by these would carry the fault into its gradient.
            raise TraceReconstructionError(
                f"vLLM returned a non-finite logprob ({sampled.logprob}) for sampled token {token_id} "
                f"at output position {index}"
            )
        out.append(
            {
                "token_id": int(token_id),
                "logprob": None if sampled is None else float(sampled.logprob),
                "rank": None if sampled is None or sampled.rank is None else int(sampled.rank),
                "top": [[int(t), float(lp.logprob), lp.rank] for t, lp in position.items()],
            }
        )
    return out


def input_mismatch_error(kind: str, reconstructed: list[int], expected: list[int]) -> TraceReconstructionError:
    """Uniform hard-error for a failed input roundtrip, with the same
    diagnostics ARC3-Inference records (lengths + first divergence)."""
    divergence = next(
        (i for i, (a, b) in enumerate(zip(reconstructed, expected)) if a != b),
        min(len(reconstructed), len(expected)),
    )
    return TraceReconstructionError(
        f"{kind}: re-rendering the stored messages does not reproduce the prompt token ids the "
        f"engine consumed (reconstructed_len={len(reconstructed)}, expected_len={len(expected)}, "
        f"first_divergence={divergence}). The stored portable side would not be a faithful SFT "
        f"source (taaf R12.02), so tracing refuses to continue."
    )


def step_extra(
    element: ullm.conv.LLMResponse,
    *,
    continued: bool,
    history_epoch: int,
    exact_history: bool,
    response_injection: Optional[str] = None,
) -> dict[str, Any]:
    """``TraceStep.extra`` for one call. ``continued``: the prompt was the
    served ids extended by tokens (False: rendered afresh — first call, or
    after a declared rewrite). ``history_epoch``: the agent's rewrite
    counter, so a consumer can tell a declared boundary from drift.
    ``exact_history``: the run promised nesting within an epoch.
    ``response_injection``: text served as the last input ids that the
    response text starts with — present only when something was injected,
    so a consumer that re-tokenises text knows the messages do not end
    where the input ids end (a token consumer needs nothing: the injected
    ids are input, the output ids are what was sampled)."""
    extra = {
        "stop_reason": element.stop_reason,
        "continued": continued,
        "history_epoch": history_epoch,
        "exact_history": exact_history,
        # Set when the reply carried no final content and was accepted as
        # the model's turn (Backend.accept_malformed_output); a consumer
        # penalizing bad generations keys on it.
        "malformed": getattr(element, "malformed_error", None),
    }
    if response_injection:
        extra["response_injection"] = response_injection
    return extra


def output_message_without_injection(output_message: dict[str, Any], injection: Optional[str]) -> dict[str, Any]:
    """``output_message`` with the injected prefix taken off the field it
    opens (reasoning first, else content), for a render compared against
    the sampled ids. A prefix the parsed text no longer starts with is left
    in place: the mismatch it then produces is the honest evidence."""
    if not injection:
        return output_message
    view = dict(output_message)
    for key in ("reasoning_content", "content"):
        text = view.get(key)
        if isinstance(text, str) and text:
            if text.startswith(injection):
                view[key] = text[len(injection) :]
            return view
    return view


def prompt_extends_previous(previous_input: list[int], previous_output: list[int], prompt: list[int]) -> bool:
    """Whether ``prompt`` extends the previous step's prompt + sampled output
    ids: the served history grew by tokens rather than being rendered anew.
    A step failing this re-tokenized text the model had already sampled (or
    the history before it was rewritten), so the run's steps do not nest as
    one token sequence — a training consumer sees a broken episode."""
    served = [*previous_input, *previous_output]
    return len(prompt) >= len(served) and prompt[: len(served)] == served


def describe_nesting_break(previous_input: list[int], previous_output: list[int], prompt: list[int]) -> str:
    """Where ``prompt`` leaves the previous step's prompt + sampled ids, for a
    step that does not nest (``prompt_extends_previous`` is False): a resend
    of the previous request, a prompt that stops inside what was served, or
    the first differing token -- inside the previous prompt or its reply --
    with the ids on either side from there."""
    served = [*previous_input, *previous_output]
    sizes = f"prompt {len(prompt)} ids; previous prompt {len(previous_input)} + reply {len(previous_output)}"
    if prompt == previous_input:
        return f"{sizes}: the previous request sent again, without its reply"
    shared = next((i for i, (a, b) in enumerate(zip(served, prompt)) if a != b), min(len(served), len(prompt)))
    if shared == len(prompt):
        where = f"stops at id {shared}, inside the previous " + ("prompt" if shared < len(previous_input) else "reply")
        return f"{sizes}: the prompt {where} (served history rolled back)"
    part = "prompt" if shared < len(previous_input) else "reply"
    offset = shared if part == "prompt" else shared - len(previous_input)
    return (
        f"{sizes}: first differs at id {shared}, id {offset} of the previous {part}; "
        f"served {served[shared : shared + 8]}, prompt {prompt[shared : shared + 8]}"
    )


def classify_output_reconstruction(reconstructed: Optional[list[int]], expected: list[int], decode: Any = None) -> str:
    """R5.38 output-reconstruction status: how well the structured
    ``output_message``, re-rendered through the serving template,
    reproduces the engine's output token ids. Evidence, never a gate —
    ``output_text`` stays authoritative regardless."""
    if reconstructed is None:
        return "unavailable"
    if list(reconstructed) == list(expected):
        return "ok"
    if decode is not None:
        try:
            if "".join(str(decode(reconstructed)).split()) == "".join(str(decode(expected)).split()):
                return "whitespace_mismatch"
        except Exception:
            pass
    return "mismatch"


def template_render_view(output_message: dict[str, Any]) -> dict[str, Any]:
    """Adapt a stored ``output_message`` for an HF chat-template render.

    The stored form follows the OpenAI wire shape — tool-call ``arguments``
    are JSON strings — but HF templates (e.g. Qwen's) iterate arguments as a
    MAPPING (jinja ``dictsort``), so the render needs them parsed. The
    stored trace keeps the string form; only the render view changes."""
    import copy
    import json

    view = copy.deepcopy(output_message)
    for call in view.get("tool_calls") or []:
        function = call.get("function") or {}
        if isinstance(function.get("arguments"), str):
            try:
                function["arguments"] = json.loads(function["arguments"])
            except json.JSONDecodeError:
                pass  # non-JSON arguments render as-is; a mismatch status is the honest outcome
    return view


def classify_output_render_hf(
    tokenizer: Any, chat_template_kwargs: dict[str, Any], output_message: dict[str, Any], output_token_ids: list[int]
) -> str:
    """R5.38 output-reconstruction status via a LOCAL HF tokenizer render —
    the one implementation both backends share (server-side re-render proved
    too dialect-fragile: wire schemas remap reasoning/argument fields before
    the template sees them).

    The prefix renders WITH the generation prompt (the template pre-seeds the
    assistant header + any think seed INTO the prompt, so generated ids start
    after them), and one trailing history-framing newline is stripped."""
    render_message = template_render_view(output_message)
    dummy = [{"role": "user", "content": "x"}]
    prefix = tokenizer.apply_chat_template(dummy, add_generation_prompt=True, tokenize=False, **chat_template_kwargs)
    full = tokenizer.apply_chat_template(
        dummy + [render_message], add_generation_prompt=False, tokenize=False, **chat_template_kwargs
    )
    if not full.startswith(prefix):
        return "mismatch"  # render diverges before generation even starts
    suffix = full[len(prefix) :].removesuffix("\n")
    rendered = tokenizer.encode(suffix, add_special_tokens=False)
    return classify_output_reconstruction([int(t) for t in rendered], output_token_ids, decode=tokenizer.decode)


def output_message_from_element(element: ullm.conv.LLMResponse) -> Optional[dict[str, Any]]:
    """Template-renderable assistant completion from a processed ullm
    element (R5.37). Reasoning is keyed ``reasoning_content`` (the chat
    template contract) and tool-call arguments are JSON strings."""
    import json

    message: dict[str, Any] = {"role": "assistant"}
    if not isinstance(element.reasoning_content, ullm.conv.EmptyContent):
        reasoning_text = element.reasoning_content.to_string()
        if reasoning_text:
            message["reasoning_content"] = reasoning_text
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    contents = (
        element.final_content.elements
        if isinstance(element.final_content, ullm.conv.ListContent)
        else [element.final_content]
    )
    for content in contents:
        if isinstance(content, ullm.conv.StringContent):
            texts.append(content.text)
        elif isinstance(content, ullm.conv.ToolCallContent):
            tool_calls.append(
                {
                    "id": content.call_id,
                    "type": "function",
                    "function": {"name": content.tool_name, "arguments": json.dumps(content.arguments)},
                }
            )
    message["content"] = "\n".join(texts) if texts else None
    if tool_calls:
        message["tool_calls"] = tool_calls
    if message["content"] is None and not tool_calls and "reasoning_content" not in message:
        return None
    return message
