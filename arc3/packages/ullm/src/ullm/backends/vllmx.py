import asyncio
import contextvars
import copy
import dataclasses
import functools
import json
import math
import random
import re
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional, Sequence, cast

import openai_harmony
import ullm.backend
import ullm.backends.common as common
import ullm.backends.engine_interface as engine_interface
import ullm.backends.engine_validation
import ullm.continuation
import ullm.conv
import ullm.engine_monitor
import ullm.tracing
import vllm
import vllm.engine.async_llm_engine
import vllm.exceptions
import vllm.lora.request


def patch_engine_core_sigint_for_jupyter() -> None:
    """Make vLLM's EngineCore subprocess ignore SIGINT, so Jupyter's interrupt
    button doesn't kill the engine.

    Background: Jupyter delivers interrupts via ``os.killpg(kernel_pgid, SIGINT)``
    (jupyter_client/provisioning/local_provisioner.py), which reaches every
    process in the kernel's process group — including EngineCore (spawned via
    ``multiprocessing.Process`` inside the kernel). vLLM's default SIGINT
    handler interprets that as a shutdown request and terminates the engine
    cleanly, but the parent's monitor thread then logs ``Engine core proc
    EngineCore died unexpectedly`` and permanently marks the engine dead
    (``EngineDeadError`` on every subsequent call).

    This patch wraps ``EngineCoreProc.run_engine_core`` so that, in the child,
    any attempt to register a SIGINT handler is silently rewritten to
    ``SIG_IGN``. SIGTERM passes through unchanged, so Jupyter kernel restart
    (which sends SIGTERM via the same killpg path) still tears the engine
    down cleanly.

    MUST be called BEFORE creating the AsyncLLMEngine, since the patch needs
    to be in place when ``run_engine_core`` is invoked in the spawned child.
    Idempotent — safe to call multiple times. No-op if vLLM v1 internals are
    not available (older vLLM, or layout changed).
    """
    try:
        import vllm.v1.engine.core as _vllm_core
    except ImportError:
        return

    global _orig_run_engine_core
    orig = _vllm_core.EngineCoreProc.run_engine_core
    if getattr(orig, "_ullm_sigint_guarded", False):
        return  # already patched

    # Module-level wrapper (not a closure) so multiprocessing.spawn can
    # pickle it by qualname when vLLM uses it as the EngineCore Process
    # target — required when CUDA gets initialized in the parent (e.g.
    # multimodal processor load) and vLLM forces spawn instead of fork.
    _orig_run_engine_core = orig
    _vllm_core.EngineCoreProc.run_engine_core = staticmethod(_sigint_guarded_run_engine_core)


# Captured by patch_engine_core_sigint_for_jupyter in the parent (and inherited
# by fork children). In a spawn child this stays None — the wrapper falls back
# to looking up EngineCoreProc.run_engine_core, which is the unpatched original
# in the freshly-imported child module.
_orig_run_engine_core: Any = None


def _sigint_guarded_run_engine_core(*args: Any, **kwargs: Any) -> Any:
    """Run vLLM's engine core with SIGINT handling neutralised, so a notebook
    interrupt does not take the engine down with the cell.
    """
    import signal as _signal

    _orig_signal_signal = _signal.signal

    def _guarded_signal(signum: int, handler: Any) -> Any:
        if signum == _signal.SIGINT:
            handler = _signal.SIG_IGN
        return _orig_signal_signal(signum, handler)

    _signal.signal = _guarded_signal  # type: ignore[assignment]
    # Pre-install SIG_IGN so any SIGINT delivered during startup, before
    # vLLM's own signal handler setup runs, is also a no-op.
    _orig_signal_signal(_signal.SIGINT, _signal.SIG_IGN)

    if _orig_run_engine_core is not None:
        target = _orig_run_engine_core
    else:
        # Spawn child: fresh import, the patch never ran here. The class
        # attribute is the unpatched original (the parent's mutation of
        # EngineCoreProc.run_engine_core doesn't survive spawn).
        import vllm.v1.engine.core as _vllm_core

        target = _vllm_core.EngineCoreProc.run_engine_core
    return target(*args, **kwargs)


_sigint_guarded_run_engine_core._ullm_sigint_guarded = True  # type: ignore[attr-defined]


def _count_reasoning_tokens(parser: Any, token_ids: list[int]) -> int:
    """Reasoning tokens in one generated sequence.

    vLLM's counter walks from an opening ``<think>`` and counts while nested,
    so it returns 0 for templates that PRE-FILL the opening tag — Qwen3 does,
    which means the model only ever generates the closing ``</think>`` and the
    depth counter never rises. Fall back to "everything before the end token",
    which is exactly the reasoning span in that layout.

    Prefer the parser's own count when it produced one: it understands nested
    and multi-span cases that the positional fallback cannot.
    """
    counted = int(parser.count_reasoning_tokens(token_ids) or 0)
    if counted:
        return counted
    end_id = getattr(parser, "end_token_id", None)
    if end_id is None:
        return 0
    try:
        return token_ids.index(end_id)
    except ValueError:
        return 0  # no end token: nothing closed, so nothing countable


def _vllm_assistant_text(model_contents: Any) -> str:
    """Extract raw assistant text from a vLLM RequestOutput stored in _model_contents."""
    return model_contents.outputs[0].text


def _vllm_assistant_final_text(
    model_contents: Any, reasoning_parser: Any, *, accept_malformed_output: bool = False
) -> str:
    """Same as ``_vllm_assistant_text`` but with the reasoning span stripped
    via the configured ``reasoning_parser``. Falls back to the raw text if
    no parser is configured. The parser preserves any tool-call markup in
    the model's native format (e.g. Qwen3 ``<tool_call>`` XML), unlike the
    legacy ``final_content.to_string()`` fallback in
    ``common.to_message_dicts`` which would re-render parsed tool calls
    via their bracket repr — see that fn's ``assistant_final_text_extractor``
    docstring."""
    raw = model_contents.outputs[0].text
    if reasoning_parser is None:
        return raw
    try:
        _reasoning, content_text = reasoning_parser.extract_reasoning(raw, request=None)
    except ValueError:
        if not accept_malformed_output:
            raise
        # Accepted parser failures must remain available to subsequent history renders.
        return raw
    return (content_text or "").strip()


def _pil_images(images: Sequence[ullm.conv.ImageContent]) -> list[Any]:
    """The PIL images the engine and the processor take, built from their contents."""
    from PIL import Image

    return [Image.fromarray(image.data, mode="RGB") for image in images]


@dataclass
class _PreparedRender:
    """One call's render of its history, made where the request is built, for
    the witness that call leaves: the same conversation and turn, unchanged
    but for the answer appended to the turn."""

    conversation: ullm.conv.Conversation
    turn: ullm.conv.ResponseTurn
    turn_size: int  # the turn's elements when the request was built
    send_back_reasoning: str
    template_kwargs: dict[str, Any]
    messages: Any  # the chat-template messages the request was built from
    images: list[ullm.conv.ImageContent]  # the images they carry, in order
    head: Optional[str] = None  # their template render with the generation prompt, when the call made one


#: The render the current call made of its history (`_PreparedRender`): set
#: where the request is built and read by the witness the call leaves, in the
#: same task.
_PREPARED_RENDER: contextvars.ContextVar[Optional[_PreparedRender]] = contextvars.ContextVar(
    "ullm_vllmx_prepared_render", default=None
)


def _collapse_image_placeholders(prompt_ids: list[int], image_token_id: int, image_count: int) -> list[int]:
    """``prompt_ids`` with each run of the image placeholder token cut to one,
    the form an engine's processor expands per image; refused unless there
    is exactly one run per image."""
    collapsed: list[int] = []
    runs = 0
    for token in prompt_ids:
        if token == image_token_id:
            if collapsed and collapsed[-1] == image_token_id:
                continue
            runs += 1
        collapsed.append(token)
    if runs != image_count:
        raise ullm.continuation.HistoryDriftError(
            f"the prompt carries {runs} image placeholder run(s) for {image_count} image(s): "
            "images and placeholders no longer line up"
        )
    return collapsed


#: A ceiling no rendered image reaches, for asking a processor's resize rule
#: what it would do with no ceiling at all.
_NO_PIXEL_CEILING = 1 << 40


def _user_content_to_hf(content: ullm.conv.Content) -> tuple[Any, list[ullm.conv.ImageContent]]:
    """Convert a ``UserPrompt``'s Content into HuggingFace processor chat-template
    content. Returns ``(message_content, images)``:
      - ``message_content`` is a plain string when there are no images, or a list
        of ``{"type":"text"|"image"}`` blocks otherwise. Image blocks carry no
        inline data — vLLM consumes the images via ``multi_modal_data``.
      - ``images`` is the in-order list of ``ImageContent`` objects in the
        content tree; a render that serves them builds their PIL images
        (`_pil_images`), one that only needs the text does not.
    Walking is order-preserving: text accumulated since the last image becomes
    one text block, then the image marker, etc."""
    blocks = list[dict[str, Any]]()
    pending_text = list[str]()
    images = list[ullm.conv.ImageContent]()
    has_image = False

    def flush_text() -> None:
        if pending_text:
            blocks.append({"type": "text", "text": "\n".join(pending_text)})
            pending_text.clear()

    def walk(c: ullm.conv.Content) -> None:
        nonlocal has_image
        if isinstance(c, ullm.conv.StringContent):
            if c.text:
                pending_text.append(c.text)
        elif isinstance(c, ullm.conv.ListContent):
            for e in c.elements:
                walk(e)
        elif isinstance(c, ullm.conv.ImageContent):
            flush_text()
            blocks.append({"type": "image"})
            images.append(c)
            has_image = True
        elif isinstance(c, ullm.conv.EmptyContent):
            pass
        elif isinstance(c, ullm.conv.ToolCallContent):
            # ToolCallContent should never appear in a UserPrompt; if it does, skip.
            pass
        else:
            raise ValueError(f"Unsupported content type for HF user message: {type(c)}")

    walk(content)
    if not has_image:
        return "\n".join(pending_text), images
    flush_text()
    return blocks, images


class _ModelContentError(ValueError):
    """A parser rejected model-controlled text, rather than its caller failing."""


def _mark_malformed(element: ullm.conv.LLMResponse, reason: str) -> None:
    """Mark accepted parser failures or empty content, preserving budget stops.

    Repetition alone remains an ordinary, unmarked policy output when accepted.
    """
    if element.stop_reason != "max_tokens":
        element.stop_reason = "malformed"
    element.malformed_error = reason
    element.status = "processed"


def _require_explicit_sampling(params: ullm.backend.SamplingParameters) -> None:
    """Refuse a profile whose temperature or top_p the engine would default
    (``VLLM.require_explicit_sampling``). ``max_tokens`` is not among them:
    None there is the stated choice of no request cap, not an omission."""
    if not isinstance(params, ullm.backend.GenericSamplingParameters):
        raise TypeError(f"require_explicit_sampling needs GenericSamplingParameters, got {type(params).__name__}")
    missing = [name for name in ("temperature", "top_p") if getattr(params, name) is None]
    if missing:
        raise ValueError(
            f"require_explicit_sampling: {', '.join(missing)} unstated in the sampling profile; "
            "the engine would fill it in silently"
        )
    # Extra kwargs otherwise override the named fields during conversion while
    # trace serialization keeps the named value: two different distributions.
    duplicated = set(params.extra_params) & {"temperature", "top_p", "max_tokens", "stop", "seed"}
    if duplicated:
        raise ValueError(f"require_explicit_sampling: extra_params overrides named fields: {sorted(duplicated)}")


def _check_output_degeneracy(text: str, token_ids: list[int] | None = None) -> None:
    """Raise DegenerateOutputError if the model's output ended in a long run
    of '!' characters — a known failure mode on gpt-oss under vllm 0.19.x
    where the model gets stuck emitting '!!!!!' until max_tokens. Checking
    only the last 10 chars keeps this narrow and high-precision; other chars
    aren't inspected since this specific pattern is what's observed in the
    wild. When available, the tail of `token_ids` is included in the error
    message — if the sampler is deterministically stuck on one token (NaN
    logits path), those tail ids will all be the same integer, which is
    strong confirmation of the kernel-level failure mode."""
    tail = text[-10:]
    if len(tail) == 10 and set(tail) == {"!"}:
        tail_ids = list(token_ids[-25:]) if token_ids else []
        raise ullm.backend.DegenerateOutputError(
            f"Output degenerated into repeated '!' (last 10 chars: !!!!!!!!!!); "
            f"total output length {len(text)} chars; "
            f"last {len(tail_ids)} token ids: {tail_ids}"
        )


def _joined_output(reasoning: Any, answer: Any, end_text: str) -> Any:
    """An answer request's output as the one reply it continues: the
    reasoning request's prompt, ids and text ahead of the answer's, with
    ``end_text`` (the reasoning's end token, which vLLM leaves out of a stop
    token's text) between the texts. The cache hits are the reasoning's less
    what the answer request prefilled again, so prompt minus cached is still
    the prefill the reply cost and a resume that missed the cache reads as a
    recompute."""
    head, tail = reasoning.outputs[0], answer.outputs[0]
    completion = copy.copy(tail)
    completion.token_ids = [*head.token_ids, *tail.token_ids]
    completion.text = head.text + end_text + tail.text
    completion.logprobs = None
    if head.logprobs is not None and tail.logprobs is not None:
        completion.logprobs = [*head.logprobs, *tail.logprobs]
    completion.cumulative_logprob = None
    if head.cumulative_logprob is not None and tail.cumulative_logprob is not None:
        completion.cumulative_logprob = head.cumulative_logprob + tail.cumulative_logprob
    joined = copy.copy(answer)
    joined.outputs = [completion]
    joined.request_id = reasoning.request_id
    joined.prompt_token_ids = reasoning.prompt_token_ids
    if hasattr(joined, "prompt_logprobs"):
        joined.prompt_logprobs = getattr(reasoning, "prompt_logprobs", None)
    prefilled = len(answer.prompt_token_ids or []) - (answer.num_cached_tokens or 0)
    joined.num_cached_tokens = max(0, (reasoning.num_cached_tokens or 0) - prefilled)
    return joined


def _capped(reasoning: Any, end_text: str) -> Any:
    """A reasoning request stopped at its end token, reported as the cap:
    how one request, for which the end token is no stop, reports the same
    tokens — the end token's text in the text, finished by the cap."""
    completion = copy.copy(reasoning.outputs[0])
    completion.text = completion.text + end_text
    completion.finish_reason, completion.stop_reason = "length", None
    capped = copy.copy(reasoning)
    capped.outputs = [completion]
    return capped


def _banned_token_ids(params: vllm.SamplingParams) -> frozenset[int]:
    """The token ids a request bans: logit bias at -100, which the OpenAI API
    defines as a ban and where vLLM clamps any lower bias."""
    return frozenset(int(token_id) for token_id, bias in (params.logit_bias or {}).items() if bias <= -100)


def _check_banned_tokens(banned: frozenset[int], token_ids: Any) -> None:
    """Raise ForbiddenTokenError if a completion holds a token its request banned.

    The ban is the engine's (logit bias), and nothing downstream re-checks
    it; under speculative decoding it rests on the verification seeing the
    biased logits, so a serving path that lost the bias would otherwise pass
    silently."""
    if not banned:
        return
    found = sorted(banned.intersection(token_ids))
    if found:
        raise ullm.backend.ForbiddenTokenError(
            f"completion contains banned token ids {found} (logit bias -100); {len(token_ids)} tokens generated"
        )


def _normalize_vllm_finish_reason(raw: Any) -> Any:
    """Map vLLM's finish_reason string to ullm's normalized StopReason, or None."""
    if raw is None:
        return None
    return {
        "stop": "stop",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(raw, "other")


#: Harmony's own scale, and nothing else: a level it does not have ("max",
#: "xhigh") is refused rather than approximated to HIGH — raise, not clamp.
_REASONING_EFFORT_MAP = {
    "low": openai_harmony.ReasoningEffort.LOW,
    "medium": openai_harmony.ReasoningEffort.MEDIUM,
    "high": openai_harmony.ReasoningEffort.HIGH,
}


def _to_harmony_messages(
    element: ullm.conv.ConversationElement, encoding: openai_harmony.HarmonyEncoding, include_reasoning: bool = True
) -> list[openai_harmony.Message]:
    """Convert ullm conversation elements to openai_harmony Message objects.
    Recursive: containers delegate to their children.
    For LLMResponses, re-parses original output tokens to preserve channel info (analysis/final).
    If include_reasoning is False, analysis channel messages are filtered out."""
    match element:
        case ullm.conv.SystemPrompt():
            # SystemContent holds config (reasoning effort, channel info)
            sc = openai_harmony.SystemContent.new()
            if element.reasoning_effort is not None:
                if element.reasoning_effort not in _REASONING_EFFORT_MAP:
                    raise ValueError(
                        f"reasoning_effort={element.reasoning_effort!r} is not a level Harmony has: "
                        "its scale is 'low' / 'medium' / 'high'. Set one of those (or None)."
                    )
                sc = sc.with_reasoning_effort(_REASONING_EFFORT_MAP[element.reasoning_effort])
            sys_msg = openai_harmony.Message.from_role_and_contents(openai_harmony.Role.SYSTEM, [sc])
            # DeveloperContent holds instructions and custom function tools
            dc = openai_harmony.DeveloperContent.new().with_instructions(common.content_to_string(element.content))
            tool_descs = [
                openai_harmony.ToolDescription.new(
                    name=tool.name, description=tool.description, parameters=tool.parameters
                )
                for tool in element.tools
            ]
            if tool_descs:
                dc = dc.with_function_tools(tool_descs)
            dev_msg = openai_harmony.Message.from_role_and_contents(openai_harmony.Role.DEVELOPER, [dc])
            return [sys_msg, dev_msg]
        case ullm.conv.UserPrompt():
            return [
                openai_harmony.Message.from_role_and_content(
                    openai_harmony.Role.USER, common.content_to_string(element.content)
                )
            ]
        case ullm.conv.LLMResponse():
            if element.do_not_send_to_agent:
                return []
            assert element._model_contents is not None, (
                "LLMResponse must have _model_contents for history reconstruction"
            )
            if include_reasoning:
                # Full reconstruction: re-parse tokens to preserve channel info
                start_tokens = encoding.encode("<|start|>assistant", allowed_special="all")
                token_ids = list(start_tokens) + list(element._model_contents.outputs[0].token_ids)
                try:
                    messages = encoding.parse_messages_from_completion_tokens(token_ids)
                except openai_harmony.HarmonyError:
                    raw_text = encoding.decode(element._model_contents.outputs[0].token_ids)
                    return [openai_harmony.Message.from_role_and_content(openai_harmony.Role.ASSISTANT, raw_text)]
                return messages
            else:
                # Strip reasoning: re-parse but filter out analysis channel
                start_tokens = encoding.encode("<|start|>assistant", allowed_special="all")
                token_ids = list(start_tokens) + list(element._model_contents.outputs[0].token_ids)
                try:
                    messages = encoding.parse_messages_from_completion_tokens(token_ids)
                except openai_harmony.HarmonyError:
                    raw_text = encoding.decode(element._model_contents.outputs[0].token_ids)
                    return [openai_harmony.Message.from_role_and_content(openai_harmony.Role.ASSISTANT, raw_text)]
                return [msg for msg in messages if msg.channel != "analysis"]
        case ullm.conv.ToolResult():
            # Tool result: from tool author to assistant via commentary channel
            msg = (
                openai_harmony.Message.from_author_and_content(
                    openai_harmony.Author.new(openai_harmony.Role.TOOL, f"functions.{element.tool_name}"),
                    common.content_to_string(element.content),
                )
                .with_channel("commentary")
                .with_recipient("assistant")
            )
            return [msg]
        case ullm.conv.ResponseTurn():
            result = list[openai_harmony.Message]()
            for e in element.elements:
                result += _to_harmony_messages(e, encoding, include_reasoning)
            return result
        case ullm.conv.Conversation():
            result = _to_harmony_messages(element.system_prompt, encoding, include_reasoning)
            for e in element.elements:
                result += _to_harmony_messages(e, encoding, include_reasoning)
            return result
        case _:
            raise ValueError(f"Unsupported conversation element type: {type(element)}")


def _hf_snapshot_commit(model_path: str) -> str:
    """The HF-hub commit hash when ``model_path`` points into a hub cache
    snapshot (`.../models--org--name/snapshots/<sha>/...`) — the directory
    name IS the commit, giving local-path models a real revision pin."""
    import re

    match = re.search(r"/snapshots/([0-9a-f]{7,40})(?:/|$)", model_path)
    return match.group(1) if match else ""


def _to_vllm_sampling_params(
    sampling_params: ullm.backend.SamplingParameters, stop_token_ids: list[int]
) -> vllm.SamplingParams:
    """Convert ullm sampling parameters to vLLM SamplingParams, injecting stop token IDs."""
    if isinstance(sampling_params, SamplingParametersVLLM):
        params = copy.copy(sampling_params.params)
    elif isinstance(sampling_params, ullm.backend.GenericSamplingParameters):
        kwargs = dict[str, Any]()
        if sampling_params.temperature is not None:
            kwargs["temperature"] = sampling_params.temperature
        if sampling_params.top_p is not None:
            kwargs["top_p"] = sampling_params.top_p
        # Always pass max_tokens: ullm None means "no request cap", and
        # vLLM honors an explicit None (generate to context end) — but
        # OMITTING the field silently applies SamplingParams' default of
        # 16 tokens, which truncates every response mid-thought.
        kwargs["max_tokens"] = sampling_params.max_tokens
        if sampling_params.stop_sequences is not None:
            kwargs["stop"] = sampling_params.stop_sequences
        if sampling_params.seed is not None:
            kwargs["seed"] = sampling_params.seed
        for k, v in sampling_params.extra_params.items():
            kwargs[k] = v
        params = vllm.SamplingParams(**kwargs)
    else:
        raise ValueError(f"Unsupported sampling parameters type: {type(sampling_params)}")

    # Inject stop token IDs
    existing = list(params.stop_token_ids) if params.stop_token_ids else []
    params.stop_token_ids = list(set(existing + stop_token_ids))
    return params


@dataclass
class SamplingParametersVLLM(ullm.backend.SamplingParameters):
    """Sampling parameters for vLLM backend, directly wrapping vLLM's class.
    Note that you can use either this or GenericSamplingParameters from ullm.backend"""

    params: vllm.SamplingParams = field(default_factory=vllm.SamplingParams)


def _restore_vllm(backend_type: Any, configuration: dict[str, Any]) -> Any:
    """Reconstruct configuration while leaving parsers and live accounting lazy."""
    return backend_type(**configuration)


@dataclass
class VLLM(ullm.backend.Backend):
    """A backend wrapping a vLLM engine, using HuggingFace chat templates.
    Supports tool calling when tool_call_parser is set (e.g. 'hermes', 'llama3_json')."""

    engine: engine_interface.EngineLike = field(kw_only=True)
    tool_call_parser: str = ""  # Name of vllm tool parser (e.g. "hermes"); empty = no tool support
    reasoning_parser: str = ""  # Name of vllm reasoning parser (e.g. "qwen3"); empty = treat all output as final
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=dict[str, Any]
    )  # forwarded to apply_chat_template (e.g. {"preserve_thinking": True} for Qwen3.6)
    # Forward SystemPrompt.reasoning_effort into the chat template per request
    # (Qwen3.8-style templates, whose effort scale is low/medium/xhigh; "high"
    # and "max" are refused rather than reinterpreted). Off by default, and
    # off for every Qwen3.6 config, so existing token streams are untouched.
    native_reasoning_effort: bool = False
    mm_processor_kwargs: dict[str, Any] = field(
        default_factory=dict[str, Any]
    )  # forwarded to vLLM's TextPrompt / TokensPrompt (e.g. {"min_pixels": 4096} to override Qwen-VL's default min_pixels)
    # Refuse a call whose profile leaves temperature or top_p unstated. Off,
    # the engine's own defaults fill them in; a consumer that computes on
    # the sampling distribution (an RL trainer) cannot correct for a value
    # it never saw, so it turns this on and states every coordinate.
    require_explicit_sampling: bool = False
    classify_tool_calls: bool = False
    # Validate final engine text against its exact prompt/output IDs before
    # parsing. This uses the engine's incremental detokenization semantics.
    verify_engine_output: bool = False
    # Opt in to carrying prompt IDs with counts across a backend proxy.
    # Ordinary counting keeps its small integer-only wire payload.
    token_count_evidence: bool = False
    # Application metadata is sampled at real request admission, never while
    # measuring a prompt. The second argument identifies the preceding draw.
    request_context_provider: Optional[Callable[[str, Optional[str]], dict[str, Any]]] = field(
        default=None, init=False, repr=False
    )
    _completed_requests: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # The adapter every generation runs under, or None for the base weights.
    # A host that swaps adapters between rollout windows sets this between
    # them; each trace step records the adapter the engine reports back.
    lora_request: Optional[vllm.lora.request.LoRARequest] = None
    _tool_parser: Any = field(init=False, default=None, repr=False)  # vllm ToolParser instance
    _reasoning_parser: Any = field(init=False, default=None, repr=False)  # vllm ReasoningParser instance
    # The same parser built for a request the template renders WITHOUT
    # thinking. vLLM's think-style parsers disambiguate an output with no
    # ``</think>`` by construction-time ``enable_thinking``: with it on,
    # such an output is all reasoning (the opening tag was pre-filled and
    # never closed); with it off, all content. A no-thinking request on a
    # thinking backend needs the second reading, or its whole answer lands
    # in ``reasoning_content`` and re-renders inside a think block the
    # prompt never opened.
    _reasoning_parser_no_thinking: Any = field(init=False, default=None, repr=False)
    # History rendering runs on the counting thread; output parsing runs on
    # the event loop. vLLM reasoning parsers own mutable per-parse state.
    _reasoning_lock: Any = field(init=False, default_factory=threading.Lock, repr=False)
    _tokenizer_setup_done: bool = field(init=False, default=False, repr=False)
    _processor: Any = field(
        init=False, default=None, repr=False
    )  # HF AutoProcessor; lazy-loaded when ImageContent first appears
    _processor_setup_done: bool = field(init=False, default=False, repr=False)
    _mm_expansion_cache: dict[tuple[Any, ...], int] = field(
        init=False, default_factory=dict, repr=False
    )  # image-token delta per tuple of image pixel sizes; see _token_count
    _processed_image_sizes: dict[tuple[int, int], Optional[tuple[int, int]]] = field(
        init=False, default_factory=dict, repr=False
    )  # (height, width) the processor turns each rendered size into; see _refuse_images_the_processor_would_shrink
    _count_executor: Any = field(init=False, default=None, repr=False)  # 1-thread pool for token counting

    def __post_init__(self) -> None:
        # Tokenizer-dependent fields are populated lazily on first
        # generation via _ensure_tokenizer_setup. Loading the tokenizer
        # eagerly here forces every unpickle to call AutoTokenizer.
        # from_pretrained, which in turn hits the HF Hub model_info API
        # (transformers' _patch_mistral_regex heuristic) — fine when
        # generating, but it 429s when a viewer loads many run pickles
        # in a row.
        super().__post_init__()

    async def probe_continuation(self) -> None:
        """Connect remote engines before using their verified tokenizer snapshot."""
        if isinstance(self.engine, engine_interface.RemoteEngineLike):
            await self.engine.connect()

    async def allows_generation_replay(self) -> bool:
        """Propagate an engine transport's no-replay policy to every outer proxy."""
        if isinstance(self.engine, engine_interface.RemoteEngineLike):
            return self.replay_on_link_loss and await self.engine.allows_generation_replay()
        return self.replay_on_link_loss

    async def respond(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        conversation: ullm.conv.Conversation,
        run_id: Optional[str] = None,
    ) -> AsyncGenerator[ullm.conv.LLMResponse, None]:
        """Resolve the engine connection, then use the shared response lifecycle."""
        await self.probe_continuation()
        async for response in super().respond(sampling_params, conversation, run_id):
            yield response

    async def _prepare_input_async(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        """Render remote media without blocking the host's network event loop."""
        if hasattr(self.engine, "prepare_media"):

            def prepare() -> tuple[Any, Optional[_PreparedRender]]:
                prepared = self._prepare_input(sampling_params, turn, conversation, send_back_reasoning)
                return prepared, _PREPARED_RENDER.get()

            # The worker sets the render in its copy of this context; the
            # witness this call leaves reads it from this one.
            prepared, render = await self._run_counting_work(prepare)
            _PREPARED_RENDER.set(render)
        else:
            prepared = self._prepare_input(sampling_params, turn, conversation, send_back_reasoning)
        if conversation.conversation_id is not None:
            # The engine's prefix cache tracks the conversation's thread
            # (vLLM's prefix_cache_context_threads); _generate tags it.
            prepared["prefix_cache_thread"] = (conversation.conversation_id, conversation.history_epoch)
        return prepared

    async def _run_counting_work(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Serialize processor access and drain active reads before cancellation cleanup."""
        work = self._counting_executor().submit(
            contextvars.copy_context().run, functools.partial(function, *args, **kwargs)
        )
        pending = asyncio.wrap_future(work)
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            if not work.cancel():
                # A running thread cannot be interrupted. Keep the conversation
                # stable until it stops reading, even if cancellation repeats.
                while not pending.done():
                    try:
                        await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        break
                if not pending.cancelled():
                    pending.exception()  # retrieve any error; cancellation remains the outcome
            raise

    async def close_context(self, prefix: str) -> None:
        """End the context's thread in an in-process engine's prefix cache
        (vLLM's ``close_prefix_cache_context``)."""
        engine_core = getattr(self.engine, "engine_core", None)
        if engine_core is not None:
            await engine_core.call_utility_async("close_prefix_cache_context", prefix)

    async def close(self) -> None:
        """Drain remote requests and close this backend's local counting resources."""
        if isinstance(self.engine, engine_interface.RemoteEngineLike):
            await self.engine.aclose()
        if self._count_executor is not None:
            self._count_executor.shutdown(wait=True)
            self._count_executor = None
        await super().close()

    def _ensure_tokenizer_setup(self) -> None:
        """Resolve the engine's tokenizer and the derived stop-token ids, once."""
        if self._tokenizer_setup_done:
            return
        tokenizer = self.engine.get_tokenizer()
        if self.tool_call_parser:
            from vllm.tool_parsers import ToolParserManager

            parser_cls = ToolParserManager.get_tool_parser(self.tool_call_parser)
            self._tool_parser = parser_cls(tokenizer)
        if self.reasoning_parser:
            from vllm.reasoning import ReasoningParserManager

            parser_cls = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
            # Mirror the chat template so the parser's "no `</think>` seen"
            # disambiguation matches what the model actually emits — per
            # request, since `_template_kwargs` can turn thinking off for
            # one conversation on a backend that has it on.
            enable_thinking = self.chat_template_kwargs.get("enable_thinking", True)
            self._reasoning_parser = parser_cls(tokenizer, chat_template_kwargs={"enable_thinking": enable_thinking})
            self._reasoning_parser_no_thinking = (
                self._reasoning_parser
                if not enable_thinking
                else parser_cls(tokenizer, chat_template_kwargs={"enable_thinking": False})
            )
        self._tokenizer_setup_done = True

    def _thinking_enabled(self, conversation: ullm.conv.Conversation) -> bool:
        """Whether the template renders ``conversation`` with a think block
        for the model to fill: the configured default, unless the native
        effort path turns it off for this conversation."""
        enabled = bool(self.chat_template_kwargs.get("enable_thinking", True))
        if self.native_reasoning_effort and conversation.system_prompt.reasoning_effort is None:
            return False
        return enabled

    def _reasoning_parser_for(self, conversation: ullm.conv.Conversation) -> Any:
        """The reasoning parser whose no-``</think>`` reading matches how
        ``conversation`` was rendered (see ``_reasoning_parser_no_thinking``)."""
        if self._reasoning_parser is not None and not self._thinking_enabled(conversation):
            return self._reasoning_parser_no_thinking or self._reasoning_parser
        return self._reasoning_parser

    def _ensure_processor_setup(self) -> None:
        """Lazy-load the HF processor for the model, built the way vLLM builds
        the engine's own. Called only on the multimodal path so text-only
        conversations don't pay the load cost.

        vLLM splits the stated processor options by key: one the processor
        takes per call goes to each call, and every other key goes to its
        construction. Some image options act only at construction -- a Qwen
        processor folds a lone ``max_pixels`` into its ``size`` there and
        ignores it per call -- so a processor loaded without them would size
        images, and count their tokens, differently from the engine."""
        if self._processor_setup_done:
            return
        get_processor = getattr(self.engine, "get_processor", None)
        if get_processor is not None:
            self._processor = get_processor()
            self._processor_setup_done = True
            return
        import vllm.transformers_utils.processor
        from transformers import AutoProcessor

        model_name = self.engine.model_config.model
        revision = getattr(self.engine.model_config, "revision", None)
        processor = AutoProcessor.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        per_call = vllm.transformers_utils.processor.get_processor_kwargs_keys(
            vllm.transformers_utils.processor.get_processor_kwargs_type(processor)
        )
        construction = {key: value for key, value in self._stated_processor_kwargs().items() if key not in per_call}
        if construction:
            processor = AutoProcessor.from_pretrained(
                model_name, revision=revision, trust_remote_code=True, **construction
            )
        self._processor = processor
        self._processor_setup_done = True

    def _stated_processor_kwargs(self) -> dict[str, Any]:
        """The processor options in force for this backend's requests: the
        engine's configuration, overlaid key by key with this backend's own,
        as vLLM merges them for each request."""
        mm_config = self.engine.model_config.get_multimodal_config()
        merged: dict[str, Any] = dict(mm_config.mm_processor_kwargs or {})
        merged.update(self.mm_processor_kwargs)
        return merged

    def _local_processor_call_kwargs(self) -> dict[str, Any]:
        """Options for the processor calls THIS process makes (counting, trace
        re-render), matching what the engine passes its own processor.

        The stated kwargs are filtered the way vLLM filters them at call
        time, so a construction-only key can't leak into the call.
        ``truncation`` is then forced off and outranks both sources: a
        checkpoint whose tokenizer_config.json carries a ``max_length`` makes
        it the default for every processor call, which cuts the expanded
        image-placeholder run and fails transformers' special-token check.
        Measuring or reproducing a prompt means seeing all of it in any case."""
        import vllm.utils.func_utils

        allowed: dict[str, Any] = vllm.utils.func_utils.get_allowed_kwarg_only_overrides(
            self._processor, self._stated_processor_kwargs(), requires_kw_only=False, allow_var_kwargs=True
        )
        allowed.update({"truncation": False, "max_length": None})
        return allowed

    def _refuse_images_the_processor_would_shrink(self, pil_images: list[Any]) -> None:
        """Raise where the engine's processor would resize an image DOWN.

        Past its pixel ceiling a processor shrinks an image rather than
        refusing it, and says nothing about having done so: the model then
        reads a picture smaller than the one that was rendered, and only a
        comparison of the two sizes finds it.

        Which ceiling holds is not worked out here. It can be stated by the
        checkpoint, the engine and this backend, as ``size`` or as
        ``max_pixels``, and which statement wins depends on whether vLLM hands
        that key to the processor's construction or to its call. So the
        processor built as the engine builds its own is handed a blank image
        of each rendered size, once per size, with the options each call
        carries, and the size it returns is compared with its resize rule's
        rounding alone: a smaller one is a shrink. An image UNDER the floor
        is enlarged instead, which loses no detail, so it passes.

        The rule is read off the processor's own module rather than restated,
        and a processor that carries none, or reports no grid, is left alone:
        this refuses what we can prove, not what we guess.
        """
        self._ensure_processor_setup()
        image_processor = getattr(self._processor, "image_processor", None)
        module = sys.modules.get(getattr(type(image_processor), "__module__", ""))
        smart_resize = getattr(module, "smart_resize", None)
        patch_size = getattr(image_processor, "patch_size", None)
        merge_size = getattr(image_processor, "merge_size", None)
        if smart_resize is None or patch_size is None or merge_size is None:
            return
        factor = int(patch_size) * int(merge_size)
        for index, image in enumerate(pil_images):
            height, width = int(image.height), int(image.width)
            if (height, width) not in self._processed_image_sizes:
                self._processed_image_sizes[height, width] = self._processed_image_size(height, width)
            processed = self._processed_image_sizes[height, width]
            if processed is None:
                return
            # No floor and no ceiling: the rule's rounding to its factor alone.
            rounded = smart_resize(height, width, factor=factor, min_pixels=1, max_pixels=_NO_PIXEL_CEILING)
            if processed[0] * processed[1] < rounded[0] * rounded[1]:
                raise ValueError(
                    f"image {index} of this request is {width}x{height}, which the engine's processor would "
                    f"resize to {processed[1]}x{processed[0]}: the model would read a picture smaller than the one "
                    "rendered for it. Render within the processor's pixel ceiling, or raise that ceiling in "
                    "mm_processor_kwargs to cover what this host renders."
                )

    def _processed_image_size(self, height: int, width: int) -> Optional[tuple[int, int]]:
        """The (height, width) the local processor turns a rendered image of
        this size into, or ``None`` where its output carries no patch grid.
        Only the size is read, so a blank image stands in for the render."""
        import PIL.Image

        processed = self._processor(
            images=[PIL.Image.new("RGB", (width, height))], **self._local_processor_call_kwargs()
        )
        grid = processed.get("image_grid_thw")
        if grid is None:
            return None
        patch_size = int(self._processor.image_processor.patch_size)
        _, rows, columns = (int(cells) for cells in grid[0])
        return rows * patch_size, columns * patch_size

    async def _text_token_count(self, text: str) -> Optional[int]:
        """Exact token length of raw ``text`` on the engine's own tokenizer --
        in-process, so the count is the engine's by construction. No special
        tokens: the text is a slice inside a larger prompt, never a prompt
        of its own. Encoded on the counting thread: concurrent calls into
        one tokenizer are not safe (see ``_token_count``)."""
        await self.probe_continuation()

        def count() -> int:
            self._ensure_tokenizer_setup()
            return len(self.engine.get_tokenizer().encode(text, add_special_tokens=False))

        return await self._run_counting_work(count)

    async def _token_count(
        self, sampling_params: ullm.backend.SamplingParameters, conversation: ullm.conv.Conversation
    ) -> Optional[int]:
        """Exact prompt length, from the same ``_prepare_input`` the request
        would use — so the chat template, tool schemas and reasoning-passback
        policy are the ones actually sent.

        The text-only path already carries token ids. The multimodal path
        carries prompt TEXT plus PIL images, where the image contribution is
        not knowable from the text: the processor expands each image into a
        model-specific number of placeholder tokens. That expansion depends
        only on the images' pixel sizes (and the fixed processor kwargs),
        not on the text around them — so the full processor runs once per
        size combination and its text-independent delta is cached; after
        that only the text is tokenized. Image preprocessing is the
        expensive half of a count, and agents that send the same-shape
        grid every move re-count it constantly, so the cache hits almost
        always.

        The tokenizer work runs on a single worker thread: it holds no
        event-loop state, the fast tokenizer releases the GIL, and off the
        loop a count does not stall request submission. ONE worker, because
        ``tokenizer.__call__`` mutates shared truncation/padding state, so
        concurrent calls into one tokenizer are not safe — a single thread
        serializes counts without blocking the loop."""
        # _prepare_input pairs the engine input with trace context; the
        # engine half is the prompt that would actually be sent.
        await self.probe_continuation()
        llm_input = (
            await self._prepare_input_async(
                sampling_params,
                conversation.last_turn,
                conversation,
                self.resolve_send_back_reasoning(sampling_params),
            )
        )["engine"]
        token_ids = llm_input.get("prompt_token_ids")
        if token_ids is not None:
            return (
                ullm.backend.TokenCount(len(token_ids), token_ids, llm_input.get("media"))
                if self.token_count_evidence
                else len(token_ids)
            )
        self._ensure_processor_setup()
        text = llm_input["prompt"]
        images = llm_input["multi_modal_data"]["image"]
        image_list = images if isinstance(images, list) else [images]
        size_key = tuple(image.size for image in image_list)
        loop = asyncio.get_running_loop()

        def _text_len() -> int:
            # truncation/max_length pinned for the same reason
            # _local_processor_call_kwargs pins them: a checkpoint-supplied
            # max_length default would silently cut what is being measured.
            return len(self._processor.tokenizer(text, truncation=False, max_length=None)["input_ids"])

        delta = self._mm_expansion_cache.get(size_key)
        if delta is None:
            call_kwargs = self._local_processor_call_kwargs()

            def _full_and_text_len() -> tuple[int, int]:
                prepared = self._processor(text=text, images=images, **call_kwargs)
                return len(prepared["input_ids"][0]), _text_len()

            full_len, text_len = await loop.run_in_executor(self._counting_executor(), _full_and_text_len)
            self._mm_expansion_cache[size_key] = full_len - text_len
            return full_len
        return delta + await loop.run_in_executor(self._counting_executor(), _text_len)

    def _counting_executor(self) -> Any:
        """The lazily created single-thread pool ``_token_count`` runs its
        tokenizer work on. Lazy so a backend that never counts (or is built
        just to be configured) starts no thread."""
        if self._count_executor is None:
            import concurrent.futures

            self._count_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ullm-token-count"
            )
        return self._count_executor

    async def engine_metrics(self) -> Optional["ullm.engine_monitor.EngineMetrics"]:
        """This process's prometheus registry — the engine's own scheduler
        numbers, readable from the frontend while EngineCore runs in its own
        subprocess."""
        import ullm.engine_monitor

        return await ullm.engine_monitor.inprocess_source()

    def status(self) -> str:
        """Extends the base ``status()`` with vLLM engine liveness flags and
        triggers vLLM's periodic stats log line as a side effect."""
        return f"{super().status()} {_vllm_engine_status_str(self.engine)}"

    def __reduce__(self) -> Any:
        if self._in_flight:
            raise TypeError("cannot pickle a backend with requests in flight")
        if not isinstance(self.engine, engine_interface.RemoteEngineLike):
            raise TypeError("cannot pickle an in-process vLLM engine; serve the backend through ProxyBackend")
        configuration = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}
        return _restore_vllm, (type(self), configuration)

    def _history_messages(
        self,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
        *,
        before: Optional[ullm.conv.LLMResponse] = None,
    ) -> tuple[Any, list[ullm.conv.ImageContent]]:
        """The conversation as chat-template message dicts (untyped: the
        processor's template signature wants its own TypedDicts), plus the
        images the user prompts carry (in message order). Pure: needs the
        parsers, not the engine. ``before`` stops the current turn at that
        answer (exclusive): the history the answer was sampled after."""
        common_kwargs = self._render_kwargs(conversation)
        # Build messages by iterating elements: UserPrompts go through the local
        # HF-style walker (which collects the images on the side); everything else
        # delegates to common.to_message_dicts. This avoids ever invoking the OAI
        # walker on multimodal input — no encode→decode round-trip.
        images = list[ullm.conv.ImageContent]()
        messages: Any = common.to_message_dicts(conversation.system_prompt, **common_kwargs)
        for element in conversation.elements:
            include_reasoning = send_back_reasoning == "all" or (
                send_back_reasoning == "current_turn" and element is turn
            )
            if isinstance(element, ullm.conv.UserPrompt):
                content, msg_images = _user_content_to_hf(element.content)
                messages.append({"role": "user", "content": content})
                images.extend(msg_images)
            elif before is not None and isinstance(element, ullm.conv.ResponseTurn) and before in element.elements:
                head = ullm.conv.ResponseTurn(elements=element.elements[: element.elements.index(before)])
                messages += common.to_message_dicts(head, include_reasoning=include_reasoning, **common_kwargs)
                break
            else:
                messages += common.to_message_dicts(element, include_reasoning=include_reasoning, **common_kwargs)
        return messages, images

    def _render_kwargs(self, conversation: ullm.conv.Conversation) -> dict[str, Any]:
        """The keywords every element of ``conversation`` renders to messages with."""
        # Use raw_tool_calls=True so tool calls stay as raw text (XML tags etc.) rather than
        # OpenAI-style structured tool_calls, which HF chat templates may not support.
        raw_tc = bool(self._tool_parser)
        # When stripping reasoning, route through the configured reasoning
        # parser so tool-call markup stays in the model's native format
        # (Qwen3 <tool_call>...</tool_call>, etc.). Without this, common.py
        # would fall back to ToolCallContent.to_string() and the model
        # would see — and parrot — its own internal Python repr.
        reasoning_parser = self._reasoning_parser_for(conversation)

        def final_extractor(model_contents: Any) -> str:
            with self._reasoning_lock:
                return _vllm_assistant_final_text(
                    model_contents,
                    reasoning_parser=reasoning_parser,
                    accept_malformed_output=self.accept_malformed_output,
                )

        # With a reasoning parser the template owns the think block: past
        # reasoning rides as the structured ``reasoning_content`` field the
        # template renders inside <think>...</think>, and the content stays
        # reasoning-free. The raw output text is not a substitute — the
        # generation prompt pre-filled its opening tag, so the text alone
        # has no think block for the template to recognise.
        reasoning_field = "reasoning_content" if reasoning_parser is not None else None
        return dict(
            reasoning_effort_prompts=self.reasoning_effort_prompts,
            assistant_text_extractor=_vllm_assistant_text,
            assistant_final_text_extractor=final_extractor,
            raw_tool_calls=raw_tc,
            reasoning_field=reasoning_field,
        )

    def _template_kwargs(self, conversation: ullm.conv.Conversation) -> dict[str, Any]:
        """The chat-template kwargs a render of ``conversation`` uses: the
        configured ones, the per-request reasoning effort, the tool schemas."""
        template_kwargs = dict[str, Any](self.chat_template_kwargs)
        if self.native_reasoning_effort:
            # The template reads `reasoning_effort` as a kwarg (Qwen3.8-style
            # templates) and injects per-level instruction text, so this is
            # where the SystemPrompt's setting becomes real on the in-process
            # path. Identity mapping only: the model's own vocabulary is
            # low/medium/xhigh, and reinterpreting "high" or "max" for it
            # would silently run a different effort than the caller named —
            # refused instead, at the first request, where the config error
            # is still cheap. None means what it means everywhere else: no
            # thinking at all.
            effort = conversation.system_prompt.reasoning_effort
            if effort is None:
                template_kwargs["enable_thinking"] = self._thinking_enabled(conversation)
            elif effort in ("high", "max"):
                raise ValueError(
                    f"reasoning_effort={effort!r} is not a level this model has: its scale is "
                    "'low' / 'medium' / 'xhigh'. Set one of those (or None for no thinking)."
                )
            else:
                template_kwargs["reasoning_effort"] = effort
        if self._tool_parser and conversation.system_prompt.tools:
            template_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in conversation.system_prompt.tools
            ]
        return template_kwargs

    def supports_response_injection(self) -> bool:
        """The engine takes an id prompt, so caller text appends as ids
        after the rendered generation prompt."""
        return True

    def supports_after_reasoning(self) -> bool:
        """The reply is served as two requests when the answer is sampled
        apart (``_generate``), the second continuing the first's ids."""
        return True

    def resolve_after_reasoning(
        self, sampling_params: ullm.backend.SamplingParameters
    ) -> Optional[ullm.backend.AnswerSampling]:
        """As the base, and refused on a traced backend when the answer is
        sampled apart: a trace step records one sampling for the whole reply.
        Refused here, where ``respond()`` resolves it, so nothing joins the
        conversation."""
        answer = super().resolve_after_reasoning(sampling_params)
        if (
            answer is not None
            and self.trace_recorder is not None
            and cast(ullm.backend.GenericSamplingParameters, sampling_params).samples_answer_apart()
        ):
            raise ValueError("after_reasoning is not traced yet: a trace step records one sampling for the whole reply")
        return answer

    def _injection_ids(self, injection: Optional[str]) -> list[int]:
        """The injected text as the ids served after the rendered prompt —
        tokenised standalone, without special tokens, so the seam never
        merges into the template's last token."""
        if injection is None:
            return []
        tokenizer = self.engine.get_tokenizer()
        return [int(t) for t in tokenizer.encode(injection, add_special_tokens=False)]

    def supports_token_continuation(self) -> bool:
        """The engine takes id prompts and reports its own prompt/output ids."""
        return True

    def _refuse_unwitnessable(self, send_back_reasoning: str) -> None:
        """Under exact history, a call whose answer could not be witnessed —
        past reasoning stripped — is an error. (Images are continued on this
        path: the delta goes through the processor.)"""
        if not self.exact_history:
            return
        if send_back_reasoning != "all":
            raise ullm.continuation.HistoryDriftError(
                f"exact_history requires send_back_reasoning='all' (got {send_back_reasoning!r}): stripping "
                "past reasoning re-renders the answers the model was served"
            )

    def _delta_encoder(self, delta_images: list[Any]) -> Callable[[str], list[int]]:
        """Token ids for a delta text through the processor, with the images
        whose placeholders it carries — the expansion each placeholder gets
        depends on its image alone (its pixel size and the fixed processor
        kwargs), never on the text around it, so a delta expands standalone
        exactly as inside the full render."""
        self._ensure_processor_setup()

        def encode(delta: str) -> list[int]:
            placeholders = delta.count(str(self._processor.image_token))
            if placeholders != len(delta_images):
                raise ullm.continuation.HistoryDriftError(
                    f"the delta carries {placeholders} image placeholder(s) but {len(delta_images)} new image(s) "
                    "arrived since the last answer: images and placeholders no longer line up"
                )
            if not delta_images:
                return [
                    int(t) for t in self._processor.tokenizer(delta, truncation=False, max_length=None)["input_ids"]
                ]
            encoded = self._processor(text=[delta], images=delta_images, **self._local_processor_call_kwargs())
            ids = encoded["input_ids"]
            first = ids[0] if not isinstance(ids[0], int) else ids
            return [int(t) for t in (first.tolist() if hasattr(first, "tolist") else first)]

        return encode

    def _continuation_input(
        self,
        conversation: ullm.conv.Conversation,
        messages: Any,
        images: list[ullm.conv.ImageContent],
        pil_images: list[Any],
        template_kwargs: dict[str, Any],
        send_back_reasoning: str,
        render: Optional[_PreparedRender] = None,
    ) -> Optional["ullm.continuation.ContinuationPrompt"]:
        """The id prompt extending the served history, or None when this call
        renders afresh (first call, or a declared rewrite). Under exact
        history every other reason to render afresh is an error. The
        template's render of ``messages`` goes on ``render``, whose witness
        renders the same history."""
        # Configuration a witness cannot cover is refused on EVERY call under
        # exact history, the first one included: a first call that cannot be
        # witnessed would otherwise leave nothing to continue from, and each
        # later call would render afresh without ever tripping the promise.
        self._refuse_unwitnessable(send_back_reasoning)
        history = ullm.continuation.served_history(conversation)
        if history is None or send_back_reasoning != "all":
            return None
        served, last = history
        # The served images must be the ones the witness saw — the render
        # and the placeholder count cannot tell a same-size swap, and the
        # engine would get the new pixels under the old ids. Images that
        # arrived since the last answer are the delta's; their placeholders
        # expand per image, which only the processor knows.
        served_count = len(last.image_fingerprints)
        served_now = tuple(image.fingerprint() for image in images[:served_count])
        if served_now != last.image_fingerprints:
            if self.exact_history:
                raise ullm.continuation.HistoryDriftError(
                    "exact_history: an image the model was already served has changed (same place, different "
                    "pixels), and no rewrite was declared (Conversation.mark_history_rewritten)"
                )
            return None
        tokenizer = self.engine.get_tokenizer()
        new_render = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **template_kwargs
        )
        if render is not None:
            render.head = str(new_render)
        delta_images = pil_images[served_count:]
        continued = ullm.continuation.continue_prompt(
            tokenizer,
            served,
            last,
            str(new_render),
            encode_delta=self._delta_encoder(delta_images) if pil_images else None,
        )
        if continued is None and self.exact_history:
            raise ullm.continuation.HistoryDriftError(
                "exact_history: the history before the last answer no longer renders as it was served, and "
                "no rewrite was declared (Conversation.mark_history_rewritten)"
            )
        return continued

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        # Lazy tokenizer setup — see __post_init__ for why we defer.
        """Render the conversation through the HF chat template — multimodal via
        the processor, with vision-token placeholders — and return the engine
        prompt alongside the trace context the step builder needs.
        """
        self._ensure_tokenizer_setup()
        if self.require_explicit_sampling:
            _require_explicit_sampling(sampling_params)
        messages, images = self._history_messages(turn, conversation, send_back_reasoning)
        pil_images = _pil_images(images)
        if pil_images:
            # Before any branch attaches them: every path below hands these
            # same images to the engine, and a shrink is silent there.
            self._refuse_images_the_processor_would_shrink(pil_images)
        template_kwargs = self._template_kwargs(conversation)
        # The witness this call leaves renders the history this request is built from.
        render = _PreparedRender(
            conversation=conversation,
            turn=turn,
            turn_size=len(turn.elements),
            send_back_reasoning=send_back_reasoning,
            template_kwargs=template_kwargs,
            messages=messages,
            images=images,
        )
        _PREPARED_RENDER.set(render)
        # Each image's carried digest is its id in the engine's caches, so the
        # engine hashes no pixels to find an image it already holds.
        image_uuids = [image.fingerprint() for image in images]
        # Injected text rides every branch the same way: as ids after the prompt.
        injection = self.resolve_response_injection(sampling_params)
        injection_ids = self._injection_ids(injection)
        # Where the answer starts is read off the prompt's ids, so a call
        # sampling it apart goes as ids on every branch. Settings no different
        # from the call's own sample the whole reply alike: one request.
        answer = self.resolve_after_reasoning(sampling_params)
        if (
            answer is not None
            and not cast(ullm.backend.GenericSamplingParameters, sampling_params).samples_answer_apart()
        ):
            answer = None

        # Dispatch: tokenizer path for text-only (returns token IDs ready for vLLM);
        # processor path when images present (renders text with vision-token placeholders
        # the model expects, paired with PIL images via multi_modal_data).
        # The return value pairs the engine input with the render inputs
        # ("trace_ctx") so _build_trace_step can re-render and report them —
        # _generate/decode_input unwrap via ["engine"].
        trace_ctx = {
            "messages": messages,
            "images": pil_images,
            "image_uuids": image_uuids,
            "template_kwargs": template_kwargs,
            "response_injection": injection,
        }
        if self._continuing:
            continued = self._continuation_input(
                conversation, messages, images, pil_images, template_kwargs, send_back_reasoning, render
            )
            if continued is not None:
                prompt_ids = [*continued.prompt_ids, *injection_ids]
                trace_ctx["continued_prompt_ids"] = prompt_ids
                engine_input: dict[str, Any] = {"type": "token", "prompt_token_ids": prompt_ids}
                if pil_images:
                    # The ids carry the expansions; the engine processes every
                    # image they refer to, served ones included (repeats hit
                    # the mm cache), once `_engine_prompt` hands it the images
                    # in a shape it processes.
                    engine_input["multi_modal_data"] = {"image": pil_images}
                    engine_input["multi_modal_uuids"] = {"image": image_uuids}
                    if self.mm_processor_kwargs:
                        engine_input["mm_processor_kwargs"] = dict(self.mm_processor_kwargs)
                split = self._answer_split(sampling_params, conversation, prompt_ids) if answer is not None else {}
                return {"engine": self._wire_media_input(engine_input), "trace_ctx": trace_ctx, **split}
        if pil_images:
            self._ensure_processor_setup()
            prompt_text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **template_kwargs
            )
            out: dict[str, Any] = {
                "multi_modal_data": {"image": pil_images},
                "multi_modal_uuids": {"image": image_uuids},
            }
            if injection_ids or hasattr(self.engine, "prepare_media") or answer is not None:
                # The injection is ids, so the prompt goes as ids too (the
                # shape a continued multimodal call sends): text appended to
                # text would tokenise across the seam.
                out["type"] = "token"
                out["prompt_token_ids"] = [*self._delta_encoder(pil_images)(str(prompt_text)), *injection_ids]
            else:
                out["prompt"] = prompt_text
            if self.mm_processor_kwargs:
                out["mm_processor_kwargs"] = dict(self.mm_processor_kwargs)
            split = (
                self._answer_split(sampling_params, conversation, out["prompt_token_ids"]) if answer is not None else {}
            )
            return {"engine": self._wire_media_input(out), "trace_ctx": trace_ctx, **split}

        tokenizer = self.engine.get_tokenizer()
        # return_dict=False: transformers v5 flipped this default to True, which returns a
        # BatchEncoding. vLLM's input validator then iterates it as a dict of keys and fails
        # with a type error when it tries to compare a string key against a max token id.
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False, **template_kwargs
        )
        prompt_ids = [*token_ids, *injection_ids] if injection_ids else token_ids
        split = self._answer_split(sampling_params, conversation, prompt_ids) if answer is not None else {}
        return {"engine": {"type": "token", "prompt_token_ids": prompt_ids}, "trace_ctx": trace_ctx, **split}

    def _answer_split(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        conversation: ullm.conv.Conversation,
        prompt_ids: Sequence[int],
    ) -> dict[str, Any]:
        """Where a call sampling its answer apart starts the answer: at the
        first token when the prompt has already closed the reasoning (a
        template rendering no thinking, or an injection closing it), else
        once the model samples the reasoning's end token."""
        parser = self._reasoning_parser_for(conversation)
        if parser is None:
            raise ValueError("after_reasoning needs a reasoning parser to tell the answer from the reasoning")
        end = getattr(parser, "reasoning_end_str", None) or getattr(parser, "end_token", None)
        ids = self.engine.get_tokenizer().encode(end, add_special_tokens=False) if end else []
        # A call that stops at the reasoning's end, as its token or its text,
        # asks for no answer: sampling one apart contradicts it.
        call = _to_vllm_sampling_params(sampling_params, [])
        if end and (end in (call.stop or []) or (len(ids) == 1 and ids[0] in (call.stop_token_ids or []))):
            raise ValueError(f"after_reasoning samples an answer, and this call stops at the reasoning's end {end!r}")
        with self._reasoning_lock:
            closed = parser.is_reasoning_end(list(prompt_ids))
        if closed:
            return {"answer_from": "start"}
        # Two requests cannot hold a whole-reply minimum: vLLM masks every
        # stop token until a request's minimum, the reasoning's end included,
        # and the answer request would start a minimum of its own.
        if cast(ullm.backend.GenericSamplingParameters, sampling_params).extra_params.get("min_tokens"):
            raise ValueError(
                "after_reasoning cannot split a reply that carries min_tokens: the minimum would hold back the "
                "reasoning's end token and start again on the answer"
            )
        if len(ids) != 1:
            raise ValueError(
                f"after_reasoning stops the reasoning at its end token, and {type(parser).__name__}'s end "
                f"{end!r} is not one token ({ids})"
            )
        return {"answer_from": "reasoning_end", "reasoning_end_token_id": int(ids[0])}

    def _wire_media_input(self, prompt: dict[str, Any]) -> dict[str, Any]:
        """Prepare image evidence at the same boundary used for counting and serving."""
        prepare_media = getattr(self.engine, "prepare_media", None)
        if "multi_modal_data" in prompt and prepare_media is not None:
            return {
                "type": "token",
                "prompt_token_ids": prompt["prompt_token_ids"],
                "media": prepare_media(prompt),
            }
        return prompt

    def _stamp_llm_input(self, llm_input: Any, run_id: str) -> Any:
        """Attribute one draw and take application feedback at the same boundary."""
        prepared = {**llm_input, "request_id": f"{run_id}#{uuid.uuid4().hex}", "run_id": run_id}
        if self.request_context_provider is not None:
            prepared["request_context"] = dict(
                self.request_context_provider(run_id, self._completed_requests.get(run_id))
            )
        return prepared

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """Run the request on the engine, yielding its streamed outputs; aborts the
        request engine-side on degenerate output.

        A call sampling its answer apart (``after_reasoning``) whose prompt
        leaves the reasoning open is two requests: the reasoning, stopped at
        its end token, then the answer, sent the same ids under the answer's
        sampling. The context's thread in the prefix cache, with its decode
        tail, carries the first into the second. The answer's outputs stream
        on as the one reply (``_joined_output``).
        """
        prepared = llm_input
        llm_input = prepared["engine"]
        # Admission rechecks mutable parameters after preparation and before serving.
        if self.require_explicit_sampling:
            _require_explicit_sampling(sampling_params)
        answer_from = prepared.get("answer_from")
        first = (
            cast(ullm.backend.GenericSamplingParameters, sampling_params).for_answer()
            if answer_from == "start"
            else sampling_params
        )
        vllm_params = _to_vllm_sampling_params(first, [])
        end_token_id: Optional[int] = prepared.get("reasoning_end_token_id") if answer_from == "reasoning_end" else None
        if end_token_id is not None:
            vllm_params.stop_token_ids = [*(vllm_params.stop_token_ids or []), end_token_id]
        thread = prepared.get("prefix_cache_thread")
        if thread is not None and prepared.get("run_id") is not None:
            vllm_params.extra_args = {
                **(vllm_params.extra_args or {}),
                "prefix_cache_context": f"{prepared['run_id']}/{thread[0]}",
                "prefix_cache_epoch": thread[1],
            }
        if self.verify_engine_output and vllm_params.skip_special_tokens:
            raise ValueError("verify_engine_output requires skip_special_tokens=False to preserve parser markers")
        _id_string = prepared.get("request_id", str(self._request_counter))
        completed = False
        aborted = False
        # The adapter rides as a keyword only when set, so an engine without
        # LoRA support never sees the argument.
        remote = self.engine if isinstance(self.engine, engine_interface.RemoteEngineLike) else None
        adapter = self.lora_request if self.lora_request is not None else (remote.lora_request if remote else None)
        expected_adapter = engine_interface.adapter_identity(adapter)
        context = copy.deepcopy(sampling_params.request_context)
        dynamic_context = prepared.get("request_context", {})
        if set(context) & set(dynamic_context):
            raise ValueError("request context provider duplicates an explicitly supplied field")
        context.update(dynamic_context)
        prepared["served_request_context"] = context
        engine_kwargs: dict[str, Any] = {}
        if adapter is not None:
            engine_kwargs["lora_request"] = adapter
        if context:
            if remote is None:
                raise ValueError("request_context requires an engine implementing RemoteEngineLike")
            engine_kwargs["request_context"] = context
        engine_prompt = self._engine_prompt(llm_input, remote=remote is not None)
        banned = _banned_token_ids(vllm_params)
        # The request in the engine now, what it was sent, and with what: the
        # call's own, then the answer's when the reasoning stops at its end
        # token. ``reasoning`` is that finished first request, whose output
        # the answer's continues.
        serving, served, served_params = _id_string, llm_input, vllm_params
        reasoning: Any = None
        output: Any = None
        end_text = ""
        try:
            while True:
                generation = self.engine.generate(engine_prompt, served_params, serving, **engine_kwargs)
                echo_checked = engine_prompt is served
                verified_final = False
                output = None
                async for output in generation:
                    if not echo_checked:
                        # The engine expanded the collapsed placeholders itself;
                        # the prompt it served must be the one this call built.
                        if list(output.prompt_token_ids or []) != list(served["prompt_token_ids"]):
                            raise ullm.tracing.TraceReconstructionError(
                                "the engine expanded the image placeholders into a different prompt than was served"
                            )
                        echo_checked = True
                    if engine_interface.adapter_identity(getattr(output, "lora_request", None)) != expected_adapter:
                        raise ullm.tracing.TraceReconstructionError(
                            "vLLM returned a different adapter than this request"
                        )
                    if self.verify_engine_output:
                        if len(output.outputs) != 1:
                            raise ullm.tracing.TraceReconstructionError("engine response has unexpected output arity")
                        if getattr(output, "finished", False) is True:
                            self._verify_engine_response(served, served_params, serving, output)
                            verified_final = True
                    reply = output if reasoning is None else _joined_output(reasoning, output, end_text)
                    if getattr(reply, "finished", False) is True:
                        _check_banned_tokens(banned, reply.outputs[0].token_ids)
                    if not self.accept_malformed_output:
                        # A poisoned request is torn down as it degenerates; an
                        # accepting backend keeps repetition as an ordinary,
                        # unmarked policy output and lets it run to its cap.
                        _check_output_degeneracy(reply.outputs[0].text, list(reply.outputs[0].token_ids))
                    yield reply
                if self.verify_engine_output and not verified_final:
                    raise ullm.tracing.TraceReconstructionError("engine stream ended without a verified complete draw")
                if reasoning is not None or end_token_id is None or output is None:
                    break
                completion = output.outputs[0]
                if completion.finish_reason != "stop" or completion.stop_reason != end_token_id:
                    break  # the reasoning ended the reply some other way: its cap, or the turn's end
                # vLLM leaves a stop token out of the text; one request
                # sampling straight through would have it there.
                end_text = (
                    ""
                    if vllm_params.include_stop_str_in_output
                    else self.engine.get_tokenizer().decode(
                        [end_token_id], skip_special_tokens=vllm_params.skip_special_tokens
                    )
                )
                answer = await self._answer_request(prepared, sampling_params, vllm_params, output)
                if answer is None:
                    # The reasoning spent the whole cap reaching its end, which
                    # vLLM reports as the stop it is; one request, for which
                    # the end token is no stop, reports it as the cap.
                    yield _capped(output, end_text)
                    break
                reasoning = output
                served, served_params = answer
                serving = f"{_id_string}:answer"
                engine_prompt = self._engine_prompt(served, remote=remote is not None)
            completed = True
            if prepared.get("run_id") is not None:
                self._completed_requests[prepared["run_id"]] = _id_string
        except ullm.backend.DegenerateOutputError:
            # Tear the poisoned request down immediately, then sleep jittered
            # before re-raising so the whole solver fleet doesn't wake
            # simultaneously and reconverge on the same bad batch composition.
            try:
                await self.engine.abort(serving)
            except Exception:
                pass
            aborted = True
            await asyncio.sleep(random.uniform(15, 20))
            raise
        except (ValueError, vllm.exceptions.VLLMValidationError) as e:
            if "longer than the maximum model length" in str(e) or "context length" in str(e):
                raise ullm.backend.PromptTooLongError(str(e)) from e
            raise
        finally:
            # On exceptional unwind (cancellation, caller break, exception) the
            # vLLM request is still in the scheduler — abort it so the engine
            # doesn't keep processing an orphan. On normal completion we skip
            # the abort: the request is already done, and through the proxy an
            # abort frame serializes on a shared write lock and stretches the
            # finalization window where `_in_flight` stays inflated past vLLM's
            # actual scheduler state.
            if not completed and not aborted:
                try:
                    await self.engine.abort(serving)
                except Exception:
                    pass

    async def _answer_request(
        self,
        prepared: dict[str, Any],
        sampling_params: ullm.backend.SamplingParameters,
        reasoning_params: vllm.SamplingParams,
        reasoning: Any,
    ) -> Optional[tuple[dict[str, Any], vllm.SamplingParams]]:
        """The answer's request after ``reasoning``, stopped at its end token:
        its prompt and reply ids under the answer's sampling, with what is
        left of the reply cap and the same prefix-cache thread. None when the
        reasoning spent the whole cap."""
        completion = reasoning.outputs[0]
        params = _to_vllm_sampling_params(
            cast(ullm.backend.GenericSamplingParameters, sampling_params).for_answer(), []
        )
        if reasoning_params.max_tokens is not None:
            left = reasoning_params.max_tokens - len(completion.token_ids)
            if left <= 0:
                return None
            params.max_tokens = left
        if reasoning_params.extra_args:
            params.extra_args = dict(reasoning_params.extra_args)
        # The shape a continued call takes: ids with every image's placeholder
        # run expanded, the images beside them.
        answer_input: dict[str, Any] = {
            "type": "token",
            "prompt_token_ids": [*reasoning.prompt_token_ids, *completion.token_ids],
        }
        images = prepared["trace_ctx"]["images"]
        if images:
            answer_input["multi_modal_data"] = {"image": list(images)}
            answer_input["multi_modal_uuids"] = {"image": list(prepared["trace_ctx"]["image_uuids"])}
            if self.mm_processor_kwargs:
                answer_input["mm_processor_kwargs"] = dict(self.mm_processor_kwargs)
            if hasattr(self.engine, "prepare_media"):
                # Media evidence is bound to the prompt's length: built anew.
                answer_input = await self._run_counting_work(self._wire_media_input, answer_input)
        return answer_input, params

    async def serve_ids(
        self,
        prompt_ids: Sequence[int],
        images: Sequence[Any],
        sampling_params: ullm.backend.SamplingParameters,
        request_id: str,
        prefix_cache_thread: Optional[tuple[str, int]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Serve a prompt that is already token ids, streaming the engine's
        ``RequestOutput``s: the entry for replaying recorded traffic
        (``ullm.traffic``), with no conversation, parsing or accounting here.
        ``prefix_cache_thread`` (context, history epoch) tags the request
        for the engine's prefix cache as ``_generate`` tags a conversation's
        (vLLM's prefix_cache_context_threads).

        ``prompt_ids`` carries every image's placeholder run expanded, as a
        continued call's ids do, and ``images`` are the prompt's images in
        placeholder order. The prompt reaches the engine in the shape
        ``_generate`` gives a continued call, so image processing, hashing and
        the prefix cache see what a game's call would send. The sampling is
        the caller's alone, with no logit bias added, so a reply may contain
        the image placeholder (``image_token_id``), which a replay must
        replace before appending the reply to a later prompt. Abandoning the
        stream aborts the request engine-side.
        """
        self._ensure_tokenizer_setup()
        # The shape a continued call takes: ids, with the images beside them.
        llm_input: dict[str, Any] = {"type": "token", "prompt_token_ids": [int(t) for t in prompt_ids]}
        if images:
            llm_input["multi_modal_data"] = {"image": list(images)}
            if self.mm_processor_kwargs:
                llm_input["mm_processor_kwargs"] = dict(self.mm_processor_kwargs)
        llm_input = self._wire_media_input(llm_input)
        remote = isinstance(self.engine, engine_interface.RemoteEngineLike)
        engine_prompt = self._engine_prompt(llm_input, remote=remote)
        params = _to_vllm_sampling_params(sampling_params, [])
        if prefix_cache_thread is not None:
            params.extra_args = {
                **(params.extra_args or {}),
                "prefix_cache_context": prefix_cache_thread[0],
                "prefix_cache_epoch": prefix_cache_thread[1],
            }
        generation = self.engine.generate(engine_prompt, params, request_id)
        echo_checked = engine_prompt is llm_input
        completed = False
        try:
            async for output in generation:
                if not echo_checked:
                    # The engine expanded the collapsed placeholders itself; the
                    # prompt it served must be the one asked for.
                    if list(output.prompt_token_ids or []) != llm_input["prompt_token_ids"]:
                        raise ValueError("the engine expanded the image placeholders into a different prompt")
                    echo_checked = True
                yield output
            completed = True
        finally:
            # A request left in the scheduler would keep decoding for nobody.
            if not completed:
                try:
                    await self.engine.abort(request_id)
                except Exception:
                    pass

    def image_token_id(self) -> Optional[int]:
        """The token that stands for an image in an id prompt, or None."""
        return self._image_token_id()

    def _image_token_id(self) -> Optional[int]:
        """The processor's image placeholder token, or None for a processor
        without one."""
        self._ensure_processor_setup()
        token = getattr(self._processor, "image_token", None)
        if token is None:
            return None
        return int(self.engine.get_tokenizer().convert_tokens_to_ids(str(token)))

    def _engine_prompt(self, llm_input: dict[str, Any], *, remote: bool) -> dict[str, Any]:
        """The prompt in the shape the engine consumes. vLLM takes a dict with
        a ``type`` as input it has already processed: a ``token`` input has no
        images, so ``multi_modal_data`` beside it is dropped without a word and
        the model reads bare placeholder tokens. An id prompt with images goes
        untyped instead, each image's expanded placeholder run collapsed to the
        one token the engine's processor expands again; the processor matches
        the image hashes and positions of a rendered prompt, so continued calls
        hit the prefix cache a fresh render left. A remote engine receives the
        prompt as built."""
        images = (llm_input.get("multi_modal_data") or {}).get("image")
        if remote or not images or "prompt_token_ids" not in llm_input:
            return llm_input
        self._ensure_processor_setup()
        image_token_id = int(self.engine.get_tokenizer().convert_tokens_to_ids(str(self._processor.image_token)))
        prompt = {key: value for key, value in llm_input.items() if key != "type"}
        prompt["prompt_token_ids"] = _collapse_image_placeholders(
            llm_input["prompt_token_ids"], image_token_id, len(images)
        )
        return prompt

    def _verify_engine_response(self, prompt: Any, params: Any, request_id: str, output: Any) -> None:
        """Verify raw engine evidence before it can affect parsing or continuation."""
        if output.request_id != request_id or len(output.outputs) != 1:
            raise ullm.tracing.TraceReconstructionError("engine response has different request identity or arity")
        prompt_ids = output.prompt_token_ids
        if not isinstance(prompt_ids, list) or not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
            raise ullm.tracing.TraceReconstructionError("engine response has invalid prompt token IDs")
        expected_prompt = prompt.get("prompt_token_ids")
        if expected_prompt is not None and list(expected_prompt) != prompt_ids:
            raise ullm.tracing.TraceReconstructionError("engine served a different token prompt")
        completion = output.outputs[0]
        ids = completion.token_ids
        if not isinstance(ids, (list, tuple)) or any(type(t) is not int or t < 0 for t in ids):
            raise ullm.tracing.TraceReconstructionError("engine response has invalid sampled token IDs")
        if completion.finish_reason not in ("stop", "length"):
            raise ullm.tracing.TraceReconstructionError("engine response has no completed generation")
        stop = completion.stop_reason
        if type(stop) is int and (not ids or ids[-1] != stop):
            raise ullm.tracing.TraceReconstructionError("engine stop token differs from sampled output")
        if params.logprobs is not None:
            probabilities = ullm.tracing.logprobs_from_vllm(completion, list(ids))
            if (
                probabilities is None
                or len(probabilities) != len(ids)
                or any(
                    p["logprob"] is None or not math.isfinite(p["logprob"]) or p["logprob"] > 0 for p in probabilities
                )
            ):
                raise ullm.tracing.TraceReconstructionError("sampled logprobs must be finite and cover every token")
        ullm.backends.engine_validation.validate_output_text(
            self.engine.get_tokenizer(), prompt_ids, list(ids), completion.text, completion.finish_reason, params
        )

    def decode_input(self, request_id: int) -> str:
        """The recorded request's prompt as text. The multimodal path already has
        one; the text-only path stores token ids and needs decoding.
        """
        inp = self._request_log[request_id]["input"]["engine"]
        # Multimodal path returns a text "prompt" with embedded vision-token markers;
        # text-only path returns "prompt_token_ids" that need decoding.
        if "prompt" in inp:
            return inp["prompt"]
        tokenizer = self.engine.get_tokenizer()
        return tokenizer.decode(inp["prompt_token_ids"])

    def decode_output(self, request_id: int) -> str:
        """The recorded request's generated text."""
        return self._request_log[request_id]["output"].outputs[0].text

    # --- tracing (platypus U5) ---------------------------------------------

    async def trace_capability(self) -> Optional["ullm.tracing.TraceCapability"]:
        """Name the engine identity and the local or remote validation path.

        Both paths use the engine's consumed IDs and a serving-compatible
        tokenizer for the roundtrip. The identity block names the BASE
        weights. LoRA is a
        per-request property, so the adapter that shaped a given output is
        recorded on that step (``extra["adapter"]``, from the engine's own
        echo) rather than here (U7): a host may swap adapters between
        windows without the capability going stale."""
        import ullm.tracing

        await self.probe_continuation()
        model_config = self.engine.model_config
        model = str(model_config.model)
        tokenizer_id = str(getattr(model_config, "tokenizer", None) or model)
        revision = str(getattr(model_config, "revision", None) or "") or _hf_snapshot_commit(model)
        remote = self.engine if isinstance(self.engine, engine_interface.RemoteEngineLike) else None
        metadata = remote.metadata if remote else {}
        return ullm.tracing.TraceCapability(
            provider_name="vllm",
            provider_version=str(metadata.get("vllm_version", vllm.__version__)),
            api_kind="engine_proxy" if remote else "in_process_engine",
            provider_model_id=model,
            validation_method=(
                "platypus.vllmx.engine_proxy_roundtrip.v1" if remote else "platypus.vllmx.inprocess_roundtrip.v1"
            ),
            model_id=model,
            model_revision=revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=str(getattr(model_config, "tokenizer_revision", None) or ""),
            processor_id=tokenizer_id,
            contents=("full", "ids"),
        )

    def _trace_render_ids(
        self, messages: list[dict[str, Any]], template_kwargs: dict[str, Any], image_pngs: list[bytes]
    ) -> list[int]:
        """Re-render stored messages through the serving tokenizer/processor
        to prompt token ids — the input-token roundtrip. The multimodal path
        re-decodes the SAVED PNG bytes, so a passing check also proves the
        stored media reproduce the engine's image-token expansion."""
        if image_pngs:
            import io

            from PIL import Image

            self._ensure_processor_setup()
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **template_kwargs
            )
            pils = [Image.open(io.BytesIO(png)).convert("RGB") for png in image_pngs]
            encoded = self._processor(text=[text], images=pils, **self._local_processor_call_kwargs())
            ids = encoded["input_ids"]
            first = ids[0] if not isinstance(ids[0], int) else ids
            return [int(t) for t in (first.tolist() if hasattr(first, "tolist") else first)]
        tokenizer = self.engine.get_tokenizer()
        return [
            int(t)
            for t in tokenizer.apply_chat_template(
                cast(Any, messages), add_generation_prompt=True, return_dict=False, **template_kwargs
            )
        ]

    async def _build_trace_step(
        self,
        *,
        run_id: str,
        sampling_params: ullm.backend.SamplingParameters,
        llm_input: Any,
        element: ullm.conv.LLMResponse,
        conversation: ullm.conv.Conversation,
    ) -> "ullm.tracing.TraceStepRecord":
        """Render trace evidence on the same serial worker as remote image inputs."""
        arguments: dict[str, Any] = dict(
            run_id=run_id,
            sampling_params=sampling_params,
            llm_input=llm_input,
            element=element,
            conversation=conversation,
        )
        if hasattr(self.engine, "prepare_media"):
            return await self._run_counting_work(self._build_trace_step_sync, **arguments)
        return self._build_trace_step_sync(**arguments)

    def _build_trace_step_sync(
        self,
        *,
        run_id: str,
        sampling_params: ullm.backend.SamplingParameters,
        llm_input: Any,
        element: ullm.conv.LLMResponse,
        conversation: ullm.conv.Conversation,
    ) -> "ullm.tracing.TraceStepRecord":
        """Assemble this call's R5 trace step. The engine is the witness: token ids
        come from it rather than from a re-tokenisation.
        """
        import io

        import ullm.tracing

        ctx = llm_input["trace_ctx"]
        injection: Optional[str] = ctx.get("response_injection")
        request_output = element._model_contents
        prompt_ids = [int(t) for t in (request_output.prompt_token_ids or [])]
        if not prompt_ids:
            raise ullm.tracing.TraceReconstructionError(
                "vLLM returned no prompt_token_ids on the RequestOutput; no witness for the trace"
            )
        output_ids = [int(t) for t in request_output.outputs[0].token_ids]
        remote = self.engine if isinstance(self.engine, engine_interface.RemoteEngineLike) else None
        metadata = remote.metadata if remote else {}
        adapter_identity = (
            request_output.adapter
            if remote
            else engine_interface.adapter_identity(
                getattr(request_output, "lora_request", None), model=str(self.engine.model_config.model)
            )
        )

        image_pngs = list[bytes]()
        for pil in ctx["images"]:
            buffer = io.BytesIO()
            pil.save(buffer, format="PNG")
            image_pngs.append(buffer.getvalue())

        continued_ids = ctx.get("continued_prompt_ids")
        if continued_ids is not None:
            # The prompt WAS token ids: the witness is the engine echoing what
            # was sent. The chat re-render deliberately does not reproduce it
            # — that drift is what continuation removes.
            if [int(t) for t in continued_ids] != prompt_ids:
                raise ullm.tracing.input_mismatch_error(
                    "vllmx token-continuation echo", list(continued_ids), prompt_ids
                )
        # The stored view is the JSON round-trip of what was rendered —
        # re-rendering from IT (not the live objects) is what makes the
        # text-path check non-tautological: it proves the serialized form
        # on disk reproduces the prompt. An ids-only trace (taaf R5.39) stores
        # no messages, so it has nothing to re-render, decode or classify;
        # its input status says whether the echo above was checked.
        stored = json.loads(
            json.dumps(
                {
                    "messages": [] if self.trace_ids_only else ctx["messages"],
                    "template_kwargs": ctx["template_kwargs"],
                },
                default=str,
            )
        )
        output_text = ""
        output_message: Optional[dict[str, Any]] = None
        if self.trace_ids_only:
            input_status = "ok" if continued_ids is not None else "not_applicable"
            output_status = "not_applicable"
        else:
            input_status = "ok"
            if continued_ids is None:
                # An injection is served as ids after the render (the recipe
                # _prepare_input uses), so the roundtrip reproduces it the same way.
                reconstructed = self._trace_render_ids(stored["messages"], stored["template_kwargs"], image_pngs)
                reconstructed.extend(self._injection_ids(injection))
                if reconstructed != prompt_ids:
                    raise ullm.tracing.input_mismatch_error("vllmx input-token roundtrip", reconstructed, prompt_ids)
            tokenizer = self.engine.get_tokenizer()
            output_text = tokenizer.decode(output_ids)
            output_message = ullm.tracing.output_message_from_element(element)
            output_status = "unavailable"
            if output_message is not None:
                # Evidence only, never a gate (R5.38): a structured completion the
                # serving template cannot re-render IS the mismatch signal, so
                # render failures classify rather than crash. The sampled ids
                # start after the injection, so the render is compared without it.
                try:
                    output_status = ullm.tracing.classify_output_render_hf(
                        tokenizer,
                        stored["template_kwargs"],
                        ullm.tracing.output_message_without_injection(output_message, injection),
                        output_ids,
                    )
                except Exception:
                    output_status = "mismatch"

        return ullm.tracing.TraceStepRecord(
            run_id=run_id,
            messages=stored["messages"],
            input_token_ids=prompt_ids,
            output_token_ids=output_ids,
            output_text=output_text,
            output_message=output_message,
            sampling_params=ullm.tracing.sampling_to_dict(sampling_params),
            request_payload={
                "messages": stored["messages"],
                "chat_template_kwargs": stored["template_kwargs"],
                "mm_processor_kwargs": dict(self.mm_processor_kwargs),
                **({"request_id": llm_input["request_id"]} if "request_id" in llm_input else {}),
                **({"media": llm_input["engine"]["media"]} if "media" in llm_input["engine"] else {}),
                **(
                    {"request_context": llm_input["served_request_context"]}
                    if llm_input.get("served_request_context")
                    else {}
                ),
            },
            images=[ullm.tracing.TraceImage(png=png) for png in image_pngs],
            input_token_reconstruction_status=input_status,
            output_token_reconstruction_status=output_status,
            media_reconstruction_status=("ok" if getattr(request_output, "media_evidence", None) else "unavailable")
            if image_pngs
            else "not_applicable",
            uncached_input_tokens=len(prompt_ids) - int(request_output.num_cached_tokens or 0),
            logprobs=ullm.tracing.logprobs_from_vllm(request_output.outputs[0], output_ids),
            extra={
                **ullm.tracing.step_extra(
                    element,
                    continued=continued_ids is not None,
                    history_epoch=conversation.history_epoch,
                    exact_history=self.exact_history,
                    response_injection=injection,
                ),
                "adapter": adapter_identity,
                **(
                    {"media_evidence": request_output.media_evidence}
                    if getattr(request_output, "media_evidence", None)
                    else {}
                ),
                **(
                    {"tokenizer_fingerprint": metadata["tokenizer_fingerprint"]}
                    if metadata.get("tokenizer_fingerprint")
                    else {}
                ),
            },
        )

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Parse the engine's output onto the element, then leave the
        continuation witness (the ids this answer was served after and sampled)."""
        injection = self.resolve_response_injection(sampling_params)
        self._parse_output(llm_output, conversation, element, injection)
        if self._continuing:
            self._leave_witness(
                conversation, turn, element, self.resolve_send_back_reasoning(sampling_params), injection
            )

    def _leave_witness(
        self,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
        send_back_reasoning: str,
        injection: Optional[str] = None,
    ) -> None:
        """Record on ``element`` what the engine served and sampled, with the
        template's render of the answered history, so the next call can
        extend it by tokens (``ullm.continuation``)."""
        request_output = element._model_contents
        prompt_ids = [int(t) for t in (request_output.prompt_token_ids or [])]
        output_ids = [int(t) for t in request_output.outputs[0].token_ids]
        prepared = _PREPARED_RENDER.get()
        _PREPARED_RENDER.set(None)
        self._refuse_unwitnessable(send_back_reasoning)
        if send_back_reasoning != "all":
            return  # no witness: the next call renders afresh
        if not prompt_ids:
            if self.exact_history:
                raise ullm.continuation.HistoryDriftError(
                    "exact_history: the engine returned no prompt ids for this answer, so it cannot be continued"
                )
            return
        tokenizer = self.engine.get_tokenizer()
        template_kwargs = self._template_kwargs(conversation)
        head: Optional[str] = None
        if self._renders_this_answer(prepared, conversation, turn, element, send_back_reasoning, template_kwargs):
            assert prepared is not None
            # The history this answer was sampled after is the one its request
            # was built from, and a turn renders element by element, so the
            # answered history is that render with this answer's own messages.
            messages_before, images, head = prepared.messages, prepared.images, prepared.head
            messages_through = [
                *messages_before,
                *common.to_message_dicts(element, include_reasoning=True, **self._render_kwargs(conversation)),
            ]
        else:
            messages_before, images = self._history_messages(turn, conversation, send_back_reasoning, before=element)
            # The answered history is rendered exactly as the next call will
            # render it (same message builder, raw tool-call text included), so
            # the next render's prefix compares against the same bytes.
            messages_through, _ = self._history_messages(turn, conversation, send_back_reasoning)
        if head is None:
            head = tokenizer.apply_chat_template(
                messages_before, add_generation_prompt=True, tokenize=False, **template_kwargs
            )
        answered = tokenizer.apply_chat_template(
            messages_through, add_generation_prompt=False, tokenize=False, **template_kwargs
        )
        history = ullm.continuation.served_history(conversation, until=element)
        element._served = ullm.continuation.witness(
            tokenizer,
            epoch=conversation.history_epoch,
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            served_before=history[0] if history is not None else None,
            head_render=str(head),
            answered_render=str(answered),
            image_fingerprints=tuple(image.fingerprint() for image in images),
            injected=injection or "",
        )
        if element._served is None and self.exact_history:
            raise ullm.continuation.HistoryDriftError(
                "exact_history: the chat template does not render this answer as a suffix of the prompt it "
                "was sampled after, so the next call could not be continued by tokens"
            )

    @staticmethod
    def _renders_this_answer(
        prepared: Optional[_PreparedRender],
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
        send_back_reasoning: str,
        template_kwargs: dict[str, Any],
    ) -> bool:
        """Whether ``prepared`` is the render ``element``'s request was built
        from: the same conversation, ending in the same turn, which has
        gained only this answer since."""
        return (
            prepared is not None
            and prepared.conversation is conversation
            and prepared.turn is turn
            and bool(conversation.elements)
            and conversation.elements[-1] is turn
            and len(turn.elements) == prepared.turn_size + 1
            and turn.elements[-1] is element
            and prepared.send_back_reasoning == send_back_reasoning
            and prepared.template_kwargs == template_kwargs
        )

    def _shed_served_ids(self, element: ullm.conv.LLMResponse) -> None:
        """The RequestOutput's ``prompt_token_ids`` are the whole prompt; the
        witness and the trace step have them, the history text does not need them."""
        request_output = element._model_contents
        if request_output is not None and getattr(request_output, "prompt_token_ids", None) is not None:
            request_output.prompt_token_ids = None

    def _parse_output(
        self,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        element: ullm.conv.LLMResponse,
        injection: Optional[str] = None,
    ) -> None:
        """The reasoning/content split, tool calls, stop reason and usage."""
        assert len(llm_output.outputs) == 1, "vLLM generator produced unexpected number of outputs"
        element._model_contents = llm_output
        element.generation_request_id = getattr(llm_output, "request_id", None)

        if not self.accept_malformed_output:
            _check_output_degeneracy(llm_output.outputs[0].text, llm_output.outputs[0].token_ids)
        if injection:
            # The response text is what the model was served plus what it
            # sampled: every reader of the raw completion (history re-renders,
            # the diagnostics log) sees the answer whole, while the ids stay
            # the sampled ones for usage and the witness.
            llm_output.outputs[0].text = injection + llm_output.outputs[0].text

        # Token usage from RequestOutput
        element.token_usage = ullm.backend.TokenUsage(
            prompt_tokens=len(llm_output.prompt_token_ids) if llm_output.prompt_token_ids else 0,
            completion_tokens=len(llm_output.outputs[0].token_ids),
            cached_tokens=llm_output.num_cached_tokens or 0,
            num_requests=1,
        )
        element.stop_reason = _normalize_vllm_finish_reason(llm_output.outputs[0].finish_reason)

        # Split reasoning from final answer via vLLM's reasoning parser.
        # Falls back to "all final" when no parser is configured.
        tokenizer = self.engine.get_tokenizer()
        token_ids = [t for t in llm_output.outputs[0].token_ids if t != tokenizer.eos_token_id]
        full_text = (
            llm_output.outputs[0].text if self.verify_engine_output else (injection or "") + tokenizer.decode(token_ids)
        )
        try:
            self._parse_model_content(conversation, element, full_text, token_ids)
        except _ModelContentError as error:
            if not self.accept_malformed_output:
                raise
            element.reasoning_content = ullm.conv.EmptyContent()
            element.final_content = ullm.conv.StringContent(text=full_text)
            was_capped = element.stop_reason == "max_tokens"
            _mark_malformed(element, str(error))
            if self.classify_tool_calls and re.search(r"<tool_call(?:>|\s|$)|<function=", full_text):
                element.malformed_call_count = 1
            if was_capped:
                element.stop_reason = "max_tokens"
            return

    def _parse_model_content(
        self,
        conversation: ullm.conv.Conversation,
        element: ullm.conv.LLMResponse,
        full_text: str,
        token_ids: list[int],
    ) -> None:
        """Parse model-controlled reasoning and tool text after validating engine evidence."""
        assert element.token_usage is not None
        reasoning_parser = self._reasoning_parser_for(conversation)
        if reasoning_parser is not None:
            try:
                with self._reasoning_lock:
                    reasoning_text, content_text = reasoning_parser.extract_reasoning(
                        full_text, request=None
                    )  # qwen3 + base parsers ignore the request arg
            except ValueError as error:
                raise _ModelContentError(str(error)) from error
            if reasoning_text is not None and reasoning_text.strip():
                element.reasoning_content = ullm.conv.StringContent(text=reasoning_text.strip())
                element.token_usage.reasoning_tokens = _count_reasoning_tokens(reasoning_parser, token_ids)
            final_text = (content_text or "").strip()
        else:
            final_text = full_text.strip()

        # Parse tool calls from the final text if a tool parser is configured
        if self._tool_parser and conversation.system_prompt.tools:
            # Some parsers (e.g. qwen3_coder) access request.tools for type conversion
            from vllm.entrypoints.openai.chat_completion.protocol import (
                ChatCompletionRequest,
                ChatCompletionToolsParam,
                FunctionDefinition,  # pyright: ignore[reportPrivateImportUsage] — vLLM exports it from here
            )

            tool_params = [
                ChatCompletionToolsParam(
                    type="function",
                    function=FunctionDefinition(name=t.name, description=t.description, parameters=t.parameters),
                )
                for t in conversation.system_prompt.tools
            ]
            mock_request = ChatCompletionRequest(model="", messages=[], tools=tool_params)  # type: ignore[call-arg]
            try:
                extracted = self._tool_parser.extract_tool_calls(final_text, request=mock_request)  # type: ignore[arg-type]
            except ValueError as error:
                raise _ModelContentError(str(error)) from error
            # Quoted strings and XML parameter bodies carry argument data,
            # including literal tags that are not additional calls.
            marker_text = re.sub(
                r'"(?:\\.|[^"\\])*"|<parameter=[^>]+>.*?</parameter>', " ", final_text, flags=re.DOTALL
            )
            markers = len(re.findall(r"<tool_call(?:>|\s|$)", marker_text))
            if extracted.tools_called:
                contents = list[ullm.conv.Content]()
                if extracted.content:
                    contents.append(ullm.conv.StringContent(text=extracted.content.strip()))
                for i, tc in enumerate(extracted.tool_calls):
                    malformed_error = None
                    try:
                        parsed_args: Any = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as error:
                        parsed_args = {"_raw": tc.function.arguments}
                        if self.classify_tool_calls:
                            malformed_error = f"Invalid tool argument JSON: {error}"
                    # Preserve non-object JSON under _raw so arguments stay
                    # dict-typed without losing their malformed classification.
                    if not isinstance(parsed_args, dict):
                        if self.classify_tool_calls:
                            malformed_error = "Invalid tool arguments: expected a JSON object"
                        parsed_args = {"_raw": parsed_args}
                    contents.append(
                        ullm.conv.ToolCallContent(
                            call_id=f"engine-tool-{i}",
                            tool_name=tc.function.name,
                            arguments=parsed_args,
                            malformed_error=malformed_error,
                        )
                    )
                element.final_content = ullm.conv.ListContent(elements=contents) if len(contents) > 1 else contents[0]
                if self.classify_tool_calls and markers > len(extracted.tool_calls):
                    element.malformed_call_count = 1
                    element.malformed_error = "Malformed or incomplete tool-call syntax after recovered calls"
                element.status = "processed"
                return

            if self.classify_tool_calls and (markers or re.search(r"<function=", marker_text)):
                element.malformed_call_count = 1
                element.malformed_error = (
                    "Malformed or incomplete tool-call syntax; no individual call could be recovered"
                )

        element.final_content = ullm.conv.StringContent(text=final_text)
        element.status = "processed"

    def _usage_from_partial(self, llm_output: Any) -> Optional[ullm.backend.TokenUsage]:
        """Usage from a partial ``RequestOutput``: every field
        ``_process_output`` reads is already final at each streamed chunk
        (the prompt is fixed, the token list grows), so a cancelled call
        can still be counted exactly."""
        if llm_output is None or not getattr(llm_output, "outputs", None):
            return None
        return ullm.backend.TokenUsage(
            prompt_tokens=len(llm_output.prompt_token_ids) if llm_output.prompt_token_ids else 0,
            completion_tokens=len(llm_output.outputs[0].token_ids),
            cached_tokens=llm_output.num_cached_tokens or 0,
            num_requests=1,
        )

    def _process_partial_output(
        self, sampling_params: ullm.backend.SamplingParameters, llm_output: Any, intermediate: ullm.conv.LLMResponse
    ) -> None:
        """For streaming: split the accumulated partial output via the reasoning parser.

        Calls `extract_reasoning` (non-streaming) on the full accumulated text
        each tick — idempotent and simpler than maintaining streaming state, at
        the cost of re-parsing on each partial (``partial_interval_s`` is what
        keeps that from being quadratic). The reasoning parser gracefully
        handles 'no </think> yet' (treated as still-reasoning).

        The text comes from the engine, which detokenizes incrementally in C
        and already drops special tokens (``skip_special_tokens`` defaults
        True and nothing here overrides it), so this path neither decodes nor
        filters: both would be Python work per tick in the process that feeds
        the engine.

        Splits with the backend's default parser: this hook does not see the
        conversation, and a snapshot is display only — the answer's split is
        ``_process_output``'s, which picks the parser per request."""
        if llm_output is None or not hasattr(llm_output, "outputs") or len(llm_output.outputs) == 0:
            return

        full_text = (self.resolve_response_injection(sampling_params) or "") + (llm_output.outputs[0].text or "")
        if self._reasoning_parser is None:
            intermediate.final_content = ullm.conv.StringContent(text=full_text.strip())
            return

        with self._reasoning_lock:
            reasoning_text, content_text = self._reasoning_parser.extract_reasoning(full_text, request=None)
        if reasoning_text is not None and reasoning_text.strip():
            intermediate.reasoning_content = ullm.conv.StringContent(text=reasoning_text.strip())
        if content_text is not None and content_text.strip():
            intermediate.final_content = ullm.conv.StringContent(text=content_text.strip())


@dataclass
class VLLMHarmony(ullm.backend.Backend):
    """A vLLM backend using the Harmony tokenization interface.

    Instead of relying on HuggingFace chat templates, this uses openai_harmony
    to render conversations into token IDs. This is necessary for models that use
    the Harmony protocol (e.g. GPT-OSS)."""

    engine: vllm.engine.async_llm_engine.AsyncLLMEngine = field(
        kw_only=True
    )  # must be provided by the user at construction
    encoding: openai_harmony.HarmonyEncoding = field(
        default_factory=lambda: openai_harmony.load_harmony_encoding(openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
    )
    _stop_token_ids: list[int] = field(init=False, default_factory=lambda: list[int]())

    def __post_init__(self) -> None:
        self._stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        super().__post_init__()

    async def engine_metrics(self) -> Optional["ullm.engine_monitor.EngineMetrics"]:
        """This process's prometheus registry — the engine's own scheduler
        numbers, readable from the frontend while EngineCore runs in its own
        subprocess."""
        import ullm.engine_monitor

        return await ullm.engine_monitor.inprocess_source()

    def status(self) -> str:
        """Extends the base ``status()`` with vLLM engine liveness flags and
        triggers vLLM's periodic stats log line as a side effect."""
        return f"{super().status()} {_vllm_engine_status_str(self.engine)}"

    def __reduce__(self) -> Any:
        raise TypeError(
            f"{type(self).__name__} wraps an in-process AsyncLLMEngine and is not picklable; "
            "serve it through ullm.backends.proxy to reach it from other processes"
        )

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> dict[str, list[int]]:
        # Convert conversation to Harmony messages, respecting send_back_reasoning policy
        """Render the conversation with openai-harmony, which speaks channels
        rather than a chat template.
        """
        if send_back_reasoning == "all":
            messages = _to_harmony_messages(conversation, self.encoding, include_reasoning=True)
        elif send_back_reasoning == "none":
            messages = _to_harmony_messages(conversation, self.encoding, include_reasoning=False)
        else:  # "current_turn"
            # Two-pass: older elements without reasoning, current turn with reasoning
            messages = _to_harmony_messages(conversation.system_prompt, self.encoding)
            for element in conversation.elements:
                is_current_turn = element is turn
                messages += _to_harmony_messages(element, self.encoding, include_reasoning=is_current_turn)
        harmony_conv = openai_harmony.Conversation.from_messages(messages)
        token_ids = self.encoding.render_conversation_for_completion(harmony_conv, openai_harmony.Role.ASSISTANT)
        return {"prompt_token_ids": token_ids}

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """Run the request on the engine, yielding its streamed outputs."""
        vllm_params = _to_vllm_sampling_params(sampling_params, self._stop_token_ids)
        id_string = str(self._request_counter)
        completed = False
        aborted = False
        banned = _banned_token_ids(vllm_params)
        try:
            async for output in self.engine.generate(llm_input, vllm_params, id_string):
                _check_output_degeneracy(output.outputs[0].text, list(output.outputs[0].token_ids))
                if getattr(output, "finished", False) is True:
                    _check_banned_tokens(banned, output.outputs[0].token_ids)
                yield output
            completed = True
        except ullm.backend.DegenerateOutputError:
            # Tear the poisoned request down immediately, then sleep jittered
            # before re-raising so the whole solver fleet doesn't wake
            # simultaneously and reconverge on the same bad batch composition.
            try:
                await self.engine.abort(id_string)
            except Exception:
                pass
            aborted = True
            await asyncio.sleep(random.uniform(15, 20))
            raise
        except (ValueError, vllm.exceptions.VLLMValidationError) as e:
            if "longer than the maximum model length" in str(e) or "context length" in str(e):
                raise ullm.backend.PromptTooLongError(str(e)) from e
            raise
        finally:
            # Abort only on exceptional unwind (cancellation / exception) — on
            # that path the vLLM request is still alive and needs to be torn
            # down. On normal completion the request is already finished and
            # the abort would be a no-op on the engine but still costs a
            # socket write through the proxy's shared write lock, inflating
            # `_in_flight` past vLLM's scheduler state.
            if not completed and not aborted:
                try:
                    await self.engine.abort(id_string)
                except Exception:
                    pass

    def decode_input(self, request_id: int) -> str:
        """The recorded request's prompt, decoded from its token ids."""
        return self.encoding.decode(self._request_log[request_id]["input"]["prompt_token_ids"])

    def decode_output(self, request_id: int) -> str:
        """The recorded request's generated text."""
        return self._request_log[request_id]["output"].outputs[0].text

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Parse harmony's channels onto the element."""
        assert len(llm_output.outputs) == 1, "vLLM generator produced unexpected number of outputs"
        element._model_contents = llm_output

        _check_output_degeneracy(llm_output.outputs[0].text, llm_output.outputs[0].token_ids)

        # Token usage from RequestOutput
        element.token_usage = ullm.backend.TokenUsage(
            prompt_tokens=len(llm_output.prompt_token_ids) if llm_output.prompt_token_ids else 0,
            completion_tokens=len(llm_output.outputs[0].token_ids),
            cached_tokens=llm_output.num_cached_tokens or 0,
            num_requests=1,
        )
        element.stop_reason = _normalize_vllm_finish_reason(llm_output.outputs[0].finish_reason)

        # Parse output tokens into Harmony messages with channel info
        # Prepend <|start|> (200006) and "assistant" tokens since the engine generates
        # starting from <|channel|>, but the parser expects the full message format
        start_tokens = self.encoding.encode("<|start|>assistant", allowed_special="all")
        token_ids = list(start_tokens) + list(llm_output.outputs[0].token_ids)
        try:
            messages = self.encoding.parse_messages_from_completion_tokens(token_ids)
        except openai_harmony.HarmonyError as e:
            raise ullm.backend.MalformedOutputError(f"Harmony parser rejected completion tokens: {e}") from e

        # Split by channel, detecting tool calls by recipient regardless of channel
        analysis_parts = list[ullm.conv.Content]()
        final_parts = list[ullm.conv.Content]()
        _tool_call_counter = 0
        for msg in messages:
            # Tool calls are identified by recipient, which can arrive on any channel
            if msg.recipient and msg.recipient.startswith("functions."):
                tool_name = msg.recipient.split("functions.", 1)[1]
                raw_args = msg.content[0].text  # type: ignore
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {"_raw": raw_args}
                # `json.loads` returns the parsed JSON value, which is
                # only a dict if the LLM emitted a JSON object. If it
                # emitted a bare string / list / number / bool / null,
                # wrap so ToolCallContent.arguments stays dict[str, Any].
                if not isinstance(arguments, dict):
                    arguments = {"_raw": arguments}
                # Double-escape cleanup happens at the tool level (via
                # `ullm.tool_helpers`), not here — see python.py / file.py.
                call_id = f"harmony-tool-{_tool_call_counter}"
                _tool_call_counter += 1
                final_parts.append(ullm.conv.ToolCallContent(call_id=call_id, tool_name=tool_name, arguments=arguments))
                continue
            match msg.channel:
                case "analysis":
                    analysis_parts.append(ullm.conv.StringContent(text=msg.content[0].text))  # type: ignore
                case "final":
                    final_parts.append(ullm.conv.StringContent(text=msg.content[0].text))  # type: ignore
                case "commentary":
                    # Exposed reasoning or other commentary without tool recipient
                    final_parts.append(ullm.conv.StringContent(text=msg.content[0].text))  # type: ignore
                case _:
                    raise ullm.backend.MalformedOutputError(f"Unknown Harmony channel: {msg.channel}")

        if analysis_parts:
            element.reasoning_content = ullm.conv.ListContent(elements=analysis_parts)
            # Count reasoning tokens by encoding the analysis text back
            analysis_text = "\n".join(p.to_string() for p in analysis_parts)
            element.token_usage.reasoning_tokens = len(self.encoding.encode(analysis_text, allowed_special="all"))
        if final_parts:
            element.final_content = (
                ullm.conv.ListContent(elements=final_parts) if len(final_parts) > 1 else final_parts[0]
            )
        # If `final_parts` is empty, `element.final_content` stays `EmptyContent`
        # and `Backend.process_output` raises MalformedOutputError as a backstop.

        element.status = "processed"

    def _usage_from_partial(self, llm_output: Any) -> Optional[ullm.backend.TokenUsage]:
        """Usage from a partial ``RequestOutput`` — see ``VLLM._usage_from_partial``;
        the harmony encoding changes how output is PARSED, not how it is counted."""
        if llm_output is None or not getattr(llm_output, "outputs", None):
            return None
        return ullm.backend.TokenUsage(
            prompt_tokens=len(llm_output.prompt_token_ids) if llm_output.prompt_token_ids else 0,
            completion_tokens=len(llm_output.outputs[0].token_ids),
            cached_tokens=llm_output.num_cached_tokens or 0,
            num_requests=1,
        )

    def _process_partial_output(
        self, sampling_params: ullm.backend.SamplingParameters, llm_output: Any, intermediate: ullm.conv.LLMResponse
    ) -> None:
        """For streaming: parse partial output with channel splitting, falling back to raw text on parse failure."""
        if llm_output is None or not hasattr(llm_output, "outputs") or len(llm_output.outputs) == 0:
            return
        start_tokens = self.encoding.encode("<|start|>assistant", allowed_special="all")
        token_ids = list(start_tokens) + list(llm_output.outputs[0].token_ids)
        try:
            messages = self.encoding.parse_messages_from_completion_tokens(token_ids)
        except openai_harmony.HarmonyError:
            # Any parse failure on a mid-stream snapshot is silently skipped;
            # the next peek (or the final parse in `_process_output`) will
            # either succeed or raise properly.
            return

        analysis_parts = list[ullm.conv.Content]()
        final_parts = list[ullm.conv.Content]()
        for msg in messages:
            match msg.channel:
                case "analysis":
                    analysis_parts.append(ullm.conv.StringContent(text=msg.content[0].text))  # type: ignore
                case "final":
                    final_parts.append(ullm.conv.StringContent(text=msg.content[0].text))  # type: ignore
                case "commentary":
                    final_parts.append(ullm.conv.StringContent(text=msg.content[0].text))  # type: ignore
                case _:
                    pass  # ignore unknown channels in partial output

        if analysis_parts:
            intermediate.reasoning_content = ullm.conv.ListContent(elements=analysis_parts)
        if final_parts:
            intermediate.final_content = ullm.conv.ListContent(elements=final_parts)


def _vllm_engine_status_str(engine: Any) -> str:
    """Return a status suffix for a vLLM engine: liveness flags, and as a side
    effect fire vLLM's ``do_log_stats()`` so the engine emits its standard
    "Avg prompt throughput / Running / Waiting / KV cache" log line. Output
    goes to vLLM's logger, not the returned string."""
    log_stats = getattr(engine, "do_log_stats", None)
    if callable(log_stats):
        try:
            coro: Any = log_stats()
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                asyncio.run(coro)
        except Exception:
            pass

    parts = list[str]()
    for flag in ("is_running", "is_stopped", "errored"):
        attr = getattr(engine, flag, None)
        if callable(attr):
            try:
                attr = attr()
            except Exception:
                attr = "<err>"
        if attr is not None:
            parts.append(f"{flag}={attr}")
    return " ".join(parts)
