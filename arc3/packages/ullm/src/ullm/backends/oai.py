"""Backend for OpenAI-compatible API servers (vllm serve, OpenAI, OpenRouter, etc.)."""

import asyncio
import hashlib
import itertools
import json
import math
import re
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal, Optional

import httpx
import openai
import openai.lib.streaming.chat
import openai.types.chat
import pydantic
import ullm.backend
import ullm.backends.common as common
import ullm.continuation
import ullm.conv
import ullm.engine_monitor
import ullm.tracing

# Note: reasoning in history is controlled by send_back_reasoning ("none"/"current_turn"/"all").
# Passback vehicle: reasoning_history_field (server chat template renders it — vLLM)
# or parse_think_tags re-embedding. Anthropic native API (if added) would use
# "current_turn" and pass back opaque thinking blocks.

# Tool-call markup the model may emit as text when the server-side parser
# misses it (Qwen coder format). Same patterns as ARC3-Inference's
# _recover_tool_calls_from_markup.
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _render_template_kwargs(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The chat-template kwargs a ``/tokenize`` re-render needs to reproduce
    what ``/v1/chat/completions`` rendered for ``payload``.

    Native reasoning effort travels as the top-level ``reasoning_effort``
    API param, which the chat endpoint feeds to the template as a kwarg —
    but ``/tokenize`` has no such param, so a re-render that omits it falls
    back to the template default (xhigh on Qwen3.8) and the roundtrip fails
    for every other effort. Fold it into the kwargs; an explicit template
    kwarg of the same name wins."""
    kwargs = dict((payload.get("extra_body") or {}).get("chat_template_kwargs") or {})
    effort = payload.get("reasoning_effort")
    if effort is not None and "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = effort
    return kwargs or None


def _hf_render_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-wire messages as an HF template wants them: assistant reasoning
    under ``reasoning_content`` (vLLM accepts ``reasoning`` on the wire and
    renders it as that), tool-call arguments as mappings."""
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            rendered.append(message)
            continue
        view = dict(message)
        if "reasoning" in view and "reasoning_content" not in view:
            view["reasoning_content"] = view.pop("reasoning")
        rendered.append(ullm.tracing.template_render_view(view))
    return rendered


def _chat_completion_from_completion(completion: Any, *, think_seeded: bool, eos: str) -> Any:
    """Lift a raw ``/v1/completions`` answer into the ChatCompletion shape
    the rest of this backend parses: the terminator is stripped, a think
    block the generation prompt opened is split into the ``reasoning``
    field, tool-call markup stays in the content for the markup salvage."""
    data = completion.model_dump()
    choice = data["choices"][0]
    text = str(choice.get("text") or "")
    if eos and text.endswith(eos):
        text = text[: -len(eos)]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if think_seeded:
        end = text.find("</think>")
        if end == -1:
            message["reasoning"] = text.strip()
            message["content"] = ""
        else:
            message["reasoning"] = text[:end].strip()
            message["content"] = text[end + len("</think>") :].lstrip("\n")
    return openai.types.chat.ChatCompletion.model_validate(
        {
            "id": data.get("id") or "cmpl",
            "object": "chat.completion",
            "created": int(data.get("created") or 0),
            "model": data.get("model") or "",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": choice.get("finish_reason") or "stop",
                    "message": message,
                    "token_ids": choice.get("token_ids"),
                    "logprobs": None,
                }
            ],
            "usage": data.get("usage"),
            "prompt_token_ids": choice.get("prompt_token_ids"),
        }
    )


def check_continuation_served(sent_ids: list[int], completion: Any) -> None:
    """The token-native invariant, checked on every continued call: the
    engine consumed exactly the ids the client sent. The prompt was a list
    of ids, so there is no rendering in between — the echoed ids and the
    billed ``prompt_tokens`` have no legitimate way to differ from the
    request. A mismatch means the serving stack changed the sequence under
    us (a template applied to an id prompt, a truncation, a proxy rewrite),
    which the append-only ledger exists to make impossible — raised, never
    absorbed."""
    data = completion.model_dump() if hasattr(completion, "model_dump") else dict(completion)
    choices = data.get("choices") or []
    echoed = choices[0].get("prompt_token_ids") if choices else None
    if echoed is not None and [int(t) for t in echoed] != [int(t) for t in sent_ids]:
        raise RuntimeError(
            f"token-native invariant violated: sent a prompt of {len(sent_ids)} ids but the engine "
            f"echoed {len(echoed)} different ids back — the served sequence is not the ledger."
        )
    usage = data.get("usage") or {}
    billed = usage.get("prompt_tokens")
    if billed is not None and int(billed) != len(sent_ids):
        raise RuntimeError(
            f"token-native invariant violated: sent a prompt of {len(sent_ids)} ids but the engine "
            f"billed {billed} prompt tokens — the served sequence is not the ledger."
        )


def _completion_request(llm_input: dict[str, Any]) -> dict[str, Any]:
    """The wire payload of a continued call: private bookkeeping keys and
    the chat-only ``messages`` stay behind."""
    return {k: v for k, v in llm_input.items() if not k.startswith("_") and k != "messages"}


def _recover_tool_calls_from_markup(*chunks: str) -> list[dict[str, Any]]:
    """Salvage tool calls the model emitted as raw ``<tool_call>`` markup that
    the server-side parser failed to extract (they surface in content or
    reasoning text instead of ``message.tool_calls``). Returns entries in the
    same ``{id, name, arguments-json}`` shape as the structured path."""
    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not chunk.strip():
            continue
        for match in _TOOL_CALL_BLOCK_RE.finditer(chunk):
            tool_name = str(match.group(1) or "").strip()
            if not tool_name:
                continue
            arguments = {
                str(name).strip(): value
                for name, value in _TOOL_CALL_PARAMETER_RE.findall(str(match.group(2) or ""))
                if str(name).strip()
            }
            key = (tool_name, json.dumps(arguments, ensure_ascii=True, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            recovered.append(
                {
                    "id": f"markup-call-{len(recovered) + 1}",
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=True),
                }
            )
    return recovered


class UpstreamErrorBodyError(RuntimeError):
    """A 2xx response whose body is an error object, not a completion.

    The sibling of the ``pydantic.ValidationError`` case below. Usually the
    SDK rejects such a body outright; it can also PARSE, leaving ``choices``
    None or empty. Nothing downstream notices until ``choices[0]`` raises
    ``TypeError: 'NoneType' object is not subscriptable`` in
    ``_process_output`` — which runs OUTSIDE the retry wrapper and so kills
    the run. Measured: 5 of 26 games lost their tails to exactly this on
    2026-07-30, one at 3 of 4 levels completed.
    """


# What counts as transient for a chat-completions request. The pydantic
# error belongs here because a provider can answer HTTP 200 with an error
# body carrying no ``choices``: the SDK parses it, finds nothing to build a
# completion from, and raises validation — so nothing keyed on status or on
# ``RateLimitError`` can see a rate limit in that shape.
# ``UpstreamErrorBodyError`` is that same condition in its other shape, where
# the body parses instead of being rejected.
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    pydantic.ValidationError,
    UpstreamErrorBodyError,
)


def _require_choices(response: Any) -> Any:
    """Pass the response through, or raise if it carries no usable choice.

    Deliberately a named error rather than putting ``TypeError`` in
    ``_TRANSIENT_ERRORS``: a blanket catch would silently retry genuine bugs
    anywhere in the generate path. This checks the one condition we mean.
    """
    if getattr(response, "choices", None):
        return response
    err = (getattr(response, "model_extra", None) or {}).get("error")
    detail = f": {err}" if err else ""
    raise UpstreamErrorBodyError(f"response carries no choices — a 2xx error body rather than a completion{detail}")


# BARE httpx transport failures: the connection to the (usually in-process
# vLLM) server dropped mid-stream. The SDK wraps transport errors only around
# the initial request — those arrive as ``openai.APIConnectionError`` and are
# already retried pre-yield by the transient machinery above — but errors
# during STREAM reads escape that wrapping and used to kill the game outright.
# These restart the whole request even after snapshots were yielded: snapshots
# are CUMULATIVE and the consumer rebuilds its view from each whole snapshot
# (``backend.generate`` overwrites ``llm_output`` per item), so a restart
# regresses the live view momentarily but duplicates nothing; only the final
# completion is terminal. Measured: ``httpx.ReadError`` alone cost 183/500
# runs of the 2026-08-10 muse-glimmer benchmark.
_CONNECTION_ERRORS = (
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)
_CONNECTION_RETRY_DELAYS_S = (2.0, 8.0)  # 3 attempts total
_SERVER_ROUTE_RETRY_DELAYS_S = (0.25, 1.0)  # 3 attempts total

_RETRY_BACKOFF_CAP_S = 30.0
# Past this exponent the backoff is the cap whatever the attempt.
_RETRY_BACKOFF_MAX_EXPONENT = math.ceil(math.log2(_RETRY_BACKOFF_CAP_S))
# A pydantic error quotes the input it rejected; the line needs what was
# wrong, not the whole body.
_RETRY_DETAIL_CHARS = 2000


def _retry_detail(error: BaseException) -> str:
    """What a retried failure said, on one line for stdout.log.

    The class name alone cannot tell a provider's limits apart: a 429 is the
    account's concurrency, its requests or tokens a minute, or the
    provider's own overload, and only the body and the headers say which.
    Here are the error's own message (the SDK's carries the status and the
    body), the transport error underneath a connection failure, and the
    response headers but cookies.
    """
    parts = [" ".join(str(error).split())[:_RETRY_DETAIL_CHARS] or type(error).__name__]
    cause = error.__cause__
    if cause is not None:
        parts.append(f"cause {type(cause).__name__}: {' '.join(str(cause).split())[:_RETRY_DETAIL_CHARS]}")
    response = getattr(error, "response", None)
    if isinstance(response, httpx.Response):
        headers = {name: value for name, value in response.headers.items() if name.lower() != "set-cookie"}
        parts.append(f"headers {json.dumps(headers)}")
    return " | ".join(parts)


def _retry_after_seconds(error: BaseException, attempt: int = 0) -> float:
    """The provider's own ``Retry-After`` when it sent one, else exponential
    in the attempt number (1, 2, 4, ... seconds): repeated transient failures
    mean the upstream needs recovery time, and hammering it every second is
    how one bad minute becomes a crashed run.

    Capped: a provider is free to name a delay longer than the run has left,
    and waiting it out would spend a game's budget sleeping.
    """
    # The exponent capped as well: a retry window bounded by time counts past
    # a thousand attempts, where 2**attempt no longer converts to a float.
    backoff = min(2.0 ** min(attempt, _RETRY_BACKOFF_MAX_EXPONENT), _RETRY_BACKOFF_CAP_S)
    response = getattr(error, "response", None)
    header = getattr(response, "headers", {}) or {}
    raw = header.get("retry-after") or header.get("Retry-After")
    if raw is not None:
        try:
            return min(max(float(raw), backoff), _RETRY_BACKOFF_CAP_S)
        except (TypeError, ValueError):
            pass
    return backoff


def _oai_assistant_text(model_contents: Any) -> str:
    """Extract raw assistant text from an OpenAI ChatCompletion stored in
    _model_contents.

    Also serves as the ``assistant_final_text_extractor`` for OpenAIServer
    in ``common.to_message_dicts``: OpenAI's API natively separates
    reasoning tokens into ``message.reasoning`` (output-only), so
    ``message.content`` already excludes reasoning. With
    ``parse_think_tags=True``, decode-time parsing also strips
    ``<think>...</think>`` from ``message.content`` before storage —
    same property holds either way."""
    return model_contents.choices[0].message.content or ""


@dataclass
class OpenAIServer(ullm.backend.Backend):
    """Backend that talks to any OpenAI-compatible API server.
    Works with vllm serve, OpenAI API, and other compatible endpoints."""

    base_url: str = field(kw_only=True)  # API endpoint URL (e.g. "http://localhost:8000/v1")
    api_key: str = field(kw_only=True, default="unused")  # API key; "unused" for local servers
    model: str = field(kw_only=True)  # Model name to use in API requests
    native_reasoning_effort: bool = field(
        default=True
    )  # If True, pass reasoning_effort as API param rather than appending to system prompt
    parse_think_tags: bool = field(
        default=False
    )  # If True, parse <think>...</think> tags to split reasoning from final answer
    # Assistant-message field name that carries past reasoning back to the server
    # ("reasoning" for vLLM; reasoning_content is its deprecated alias). None =
    # the server has no such field, so reasoning passback needs parse_think_tags.
    reasoning_history_field: str | None = None
    # Merged into every request's extra_body, under the per-request sampling
    # extras: fields a router reads from the body, e.g. OpenRouter's
    # {"provider": {"ignore": [...]}}. Empty = nothing added.
    request_extra_body: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    # Sent per-request as extra_body["chat_template_kwargs"] (vLLM merges them
    # over the server's --default-chat-template-kwargs), e.g.
    # {"preserve_thinking": True, "enable_thinking": True} for Qwen thinking
    # models. Empty = not sent.
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    # Streaming rebuilds the completion from accumulated deltas through the
    # SDK's helper, which requires a finish_reason the provider is free to
    # omit; a provider that does gets a MalformedOutputError naming this
    # field. The non-streaming path builds nothing and is unaffected.
    streaming: bool = field(default=True)
    # Retries for a transient provider failure, before anything is yielded.
    # A remote provider under load answers with rate limits, dropped
    # connections, and 200s carrying an error body; without a retry each one
    # costs the whole game it was serving. Retryable only until the first
    # snapshot is yielded — after that the consumer has seen partial output
    # and a replay would duplicate it. 0 disables.
    max_transient_retries: int = field(default=3)
    # Retry for this long since a call's first attempt instead, however many
    # retries that takes: a provider down for hours is waited out, not a
    # count of attempts whose length depends on how long each refusal takes.
    # None bounds by the count.
    transient_retry_window_s: Optional[float] = field(default=None)
    include_cache_control: bool = field(default=True)  # If True, include cache control headers in requests
    # Wire name for the completion-token limit. The ecosystem is split:
    # vLLM's server (>=0.21) 500s on "max_tokens", while OpenRouter and
    # some older OpenAI-compatible endpoints only know "max_tokens".
    token_limit_param: Literal["max_tokens", "max_completion_tokens"] = field(default="max_tokens")
    _client: openai.AsyncOpenAI = field(init=False, repr=False)  # Async OpenAI client, created in __post_init__
    _server_http: Any = field(default=None, init=False, repr=False)  # httpx client for /tokenize etc.
    # Tri-state each: None = unprobed, True/False = whether the server answers
    # /tokenize and /metrics, so a non-vLLM endpoint probes once rather than
    # once per count or once per status interval.
    _server_tokenize: Optional[bool] = field(default=None, init=False, repr=False)
    _server_metrics: Optional[bool] = field(default=None, init=False, repr=False)
    _trace_identity: Any = field(default=None, init=False, repr=False)  # served-policy identity snapshot (drift guard)
    _trace_hf_tokenizer_id: str = field(default="", init=False, repr=False)  # resolved at capability time
    _trace_hf_tokenizer: Any = field(default=None, init=False, repr=False)  # lazy AutoTokenizer (output-status render)
    # One-slot memo for the continuation encode: (content key, ContinuationPrompt).
    # token_count weighs the very request respond() then sends, and the memo is
    # what makes that one delta tokenization, not two.
    _continuation_cache: Any = field(default=None, init=False, repr=False)
    # Set by trace_capability() when the endpoint is a hosted API rather than a
    # vLLM server: traces are still produced, but without the token-id witness.
    _trace_no_token_ids: bool = field(default=False, init=False, repr=False)
    # Requests this backend has open at the provider, for the retry lines.
    _open_requests: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        super().__post_init__()

    def _transient_fields(self) -> tuple[str, ...]:
        # Both clients wrap an ``httpx.AsyncClient`` — its event-loop and
        # connection-pool refs can't cross a process boundary. The trace
        # tokenizer is a heavy loaded object, reloaded on demand. A copy has
        # no requests open.
        """The live client is rebuilt rather than copied."""
        return ("_client", "_server_http", "_trace_hf_tokenizer", "_open_requests")

    def _rebuild_transient(self) -> None:
        """Recreate the client after a copy or a restore."""
        self._client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        self._server_http = None
        self._trace_hf_tokenizer = None
        self._open_requests = 0

    async def _tokenize_route_count(self, payload: dict[str, Any]) -> Optional[int]:
        """POST one ``/tokenize`` request with the probe, guards, and retries
        every counting path shares.

        ``/tokenize`` is idempotent, so dropped connections use the shared
        server-route retry policy. 404/405 = no such route: this
        endpoint cannot count, remembered so the probe costs one request
        rather than one per count; every other status is a real failure and
        is raised -- a count that silently changed method would be a count
        that silently changed answer. A non-JSON answer or a JSON body
        without ``count`` is likewise not this route (OpenRouter's stripped
        base URL answers the unknown path with 200 text/html)."""
        if self._server_tokenize is False:
            return None
        response = await self._server_request("POST", "tokenize", payload={"model": self.model, **payload})
        if response.status_code in (404, 405):
            self._server_tokenize = False
            return None
        response.raise_for_status()
        if "json" not in (response.headers.get("content-type") or "").lower():
            self._server_tokenize = False
            return None
        try:
            count = response.json()["count"]
        except (ValueError, KeyError, TypeError):
            self._server_tokenize = False
            return None
        self._server_tokenize = True
        return int(count)

    async def _text_token_count(self, text: str) -> Optional[int]:
        """Exact token length of raw ``text``, as the server tokenizes it
        (vLLM ``/tokenize`` with a plain prompt). No special tokens: the
        text is a slice inside a larger prompt, never a prompt of its own.
        ``None`` when the endpoint has no ``/tokenize``."""
        return await self._tokenize_route_count({"prompt": text, "add_special_tokens": False})

    async def _token_count(
        self, sampling_params: ullm.backend.SamplingParameters, conversation: ullm.conv.Conversation
    ) -> Optional[int]:
        """Exact prompt length, as the SERVER renders it (vLLM ``/tokenize``).

        Deliberately the only method. A count has to agree with what the
        request will really cost, and that depends on the chat template, the
        processor's image expansion and the shape the template wants
        tool-call arguments in — all three owned by the server. Rendering a
        local copy means reproducing all three from wire-format messages,
        where the wire and the template genuinely disagree (arguments travel
        as a JSON string, the template reads them as a mapping), so a local
        answer is a second implementation that can drift or fail.

        ``None`` when the endpoint has no ``/tokenize`` — a plain
        OpenAI-compatible API. That is the honest answer per ``token_count``'s
        contract: exact or nothing, never an estimate dressed as a count.
        """
        if self._server_tokenize is False:
            return None
        llm_input = self._prepare_input(
            sampling_params,
            conversation.last_turn,
            conversation,
            self.resolve_send_back_reasoning(sampling_params),
        )
        if llm_input.get("_continuation"):
            # A continued call sends token ids: the request weighs exactly
            # that many, and the server bills exactly that many. A /tokenize
            # of the re-rendered messages would describe a different prompt
            # (the drift continuation exists to avoid).
            return len(llm_input["prompt"])
        payload: dict[str, Any] = {"messages": llm_input["messages"], "add_generation_prompt": True}
        if llm_input.get("tools"):
            payload["tools"] = llm_input["tools"]
        template_kwargs = _render_template_kwargs(llm_input)
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        return await self._tokenize_route_count(payload)

    def _trace_hf_tokenizer_get(self) -> Any:
        """The HF tokenizer for the served policy (from the resolved
        identity), loaded once — powers the local output-status render."""
        if self._trace_hf_tokenizer is None:
            from transformers import AutoTokenizer

            assert self._trace_hf_tokenizer_id, "trace_capability resolves the tokenizer id first"
            self._trace_hf_tokenizer = AutoTokenizer.from_pretrained(self._trace_hf_tokenizer_id)
        return self._trace_hf_tokenizer

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> dict[str, Any]:
        # Convert conversation to message dicts, respecting send_back_reasoning policy.
        # Reasoning passback needs a vehicle: reasoning_history_field (the server's
        # chat template renders it — vLLM) or parse_think_tags re-embedding. Without
        # either, past reasoning would be silently dropped — refuse instead.
        """Render the conversation into chat-completion arguments."""
        if send_back_reasoning in ("all", "current_turn") and not (
            self.reasoning_history_field or self.parse_think_tags
        ):
            raise ValueError(
                f"send_back_reasoning='{send_back_reasoning}' requires reasoning_history_field "
                "or parse_think_tags=True for OpenAIServer (native reasoning fields are "
                "output-only and cannot be sent back verbatim)"
            )

        messages, last_user_prompt_idx, last_user_tool_idx = self._chat_messages(
            turn, conversation, send_back_reasoning
        )

        # Build the kwargs dict that will be passed to client.chat.completions.create()
        kwargs = dict[str, Any]()
        kwargs["model"] = self.model
        kwargs["messages"] = messages

        # Map generic sampling parameters to OpenAI API parameters
        if isinstance(sampling_params, ullm.backend.GenericSamplingParameters):
            if sampling_params.temperature is not None:
                kwargs["temperature"] = sampling_params.temperature
            if sampling_params.top_p is not None:
                kwargs["top_p"] = sampling_params.top_p
            if sampling_params.max_tokens is not None:
                kwargs[self.token_limit_param] = sampling_params.max_tokens
            if sampling_params.stop_sequences is not None:
                kwargs["stop"] = sampling_params.stop_sequences
            if sampling_params.seed is not None:
                kwargs["seed"] = sampling_params.seed
            # Sampling extensions (top_k, min_p, repetition_penalty, ...) are
            # not OpenAI top-level params — the SDK rejects unknown kwargs. Send
            # them in extra_body, which the OpenAI-compatible server reads as
            # sampling params.
            extra_body: dict[str, Any] = {**self.request_extra_body, **sampling_params.extra_params}
            template_kwargs = dict(self.chat_template_kwargs)
            if self.native_reasoning_effort and conversation.system_prompt.reasoning_effort is None:
                # None means what it means on every other backend: no thinking
                # at all. Without this, a static ``enable_thinking: True`` in
                # the configured template kwargs (or the template's own
                # default) would keep thinking ON while the effort text goes
                # missing — the exact inversion of what the caller asked for.
                # Decided per request because the same server serves both
                # thinking and non-thinking conversations.
                template_kwargs["enable_thinking"] = False
            if template_kwargs:
                extra_body["chat_template_kwargs"] = template_kwargs
            if extra_body:
                kwargs["extra_body"] = extra_body
        else:
            raise ValueError(f"Unsupported sampling parameters type: {type(sampling_params)}")

        # For native reasoning effort, extract it from the system prompt and
        # add as API param — VERBATIM. This client speaks to real OpenAI and
        # to OpenAI-compatible servers whose vocabularies differ (OpenAI has
        # no "xhigh"; a vLLM server hosting Qwen3.8 requires it), so any
        # client-side remap is a guess that silently runs a different effort
        # than the caller named. A level the server does not have fails loudly
        # on the first call, which is the error surface a misconfiguration
        # deserves — the house rule is raise, not clamp.
        if self.native_reasoning_effort:
            if conversation.system_prompt.reasoning_effort is not None:
                kwargs["reasoning_effort"] = conversation.system_prompt.reasoning_effort

        # Add tool definitions if available, with explicit auto tool choice.
        # "strict": True is a CLIENT-side requirement: the SDK's streaming
        # helper refuses to auto-parse tool calls from non-strict tools
        # ("`python` is not strict"). vLLM's chat path ignores the field.
        if conversation.system_prompt.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "strict": True,
                    },
                }
                for t in conversation.system_prompt.tools
            ]
            kwargs["tool_choice"] = "auto"

        # Request token usage in streaming mode (non-streaming always includes it)
        if self.streaming:
            kwargs["stream_options"] = {"include_usage": True}

        if self._continuing:
            # Before the cache marks: the local template render that locates
            # the delta wants plain message dicts.
            continued = self._continuation_input(kwargs, conversation, send_back_reasoning)
            if continued is not None:
                return continued

        # Add cache_control breakpoints using tracked indices from conversation structure
        if self.include_cache_control:
            cache_control = {"type": "ephemeral"}

            def _mark(msg: dict[str, Any]) -> None:
                if isinstance(msg.get("content"), str):
                    msg["content"] = [{"type": "text", "text": msg["content"]}]
                msg["content"][-1]["cache_control"] = cache_control

            # Breakpoint 1: static prefix (tools > system prompt)
            tools = kwargs.get("tools")
            if tools:
                tools[-1]["function"]["cache_control"] = cache_control
            else:
                for msg in messages:
                    if msg["role"] == "system":
                        _mark(msg)
                        break

            # Breakpoint 2: last real user prompt (cross-turn cache prefix)
            # Breakpoint 3: last user/tool message (current request boundary)
            if last_user_tool_idx is not None:
                _mark(messages[last_user_tool_idx])
            if last_user_prompt_idx is not None and last_user_prompt_idx != last_user_tool_idx:
                _mark(messages[last_user_prompt_idx])

        if (self.trace_recorder is not None or self._continuing) and not self._trace_no_token_ids:
            # Tracing (platypus U5) and continuation: ask the vLLM server for
            # the exact token ids it consumed/produced. Only injected when one
            # of them is on so plain requests stay byte-identical — and never
            # on the weaker grade, where the endpoint is not vLLM: the field is
            # a vLLM extension a hosted API may reject outright, and the ids
            # would go unread anyway.
            kwargs.setdefault("extra_body", {})["return_token_ids"] = True

        return kwargs

    def _chat_messages(
        self,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
        *,
        before: ullm.conv.LLMResponse | None = None,
    ) -> tuple[list[dict[str, Any]], int | None, int | None]:
        """The conversation as chat-wire messages, plus the indices the cache
        breakpoints mark (last real user prompt, last user/tool message).
        ``before`` stops the current turn at that answer (exclusive)."""
        re_prompts = None if self.native_reasoning_effort else self.reasoning_effort_prompts
        common_kwargs: dict[str, Any] = dict(
            reasoning_effort_prompts=re_prompts,
            assistant_text_extractor=_oai_assistant_text,
            assistant_final_text_extractor=_oai_assistant_text,
            reasoning_field=self.reasoning_history_field,
        )
        messages = common.to_message_dicts(conversation.system_prompt, **common_kwargs)
        last_user_prompt_idx: int | None = None  # Last real UserPrompt message index
        last_user_tool_idx: int | None = None  # Last user/tool message index
        for element in conversation.elements:
            start_idx = len(messages)
            include_reasoning = send_back_reasoning == "all" or (
                send_back_reasoning == "current_turn" and element is turn
            )
            rendered_element: ullm.conv.ConversationElement = element
            stop = False
            if before is not None and isinstance(element, ullm.conv.ResponseTurn) and before in element.elements:
                rendered_element = ullm.conv.ResponseTurn(elements=element.elements[: element.elements.index(before)])
                stop = True
            messages += common.to_message_dicts(rendered_element, include_reasoning=include_reasoning, **common_kwargs)
            # Track message indices for cache breakpoints
            if isinstance(element, ullm.conv.UserPrompt):
                last_user_prompt_idx = start_idx
                last_user_tool_idx = start_idx
            elif isinstance(element, ullm.conv.ResponseTurn):
                # Find the last tool result message produced by this turn
                for i in range(len(messages) - 1, start_idx - 1, -1):
                    if messages[i]["role"] == "tool":
                        last_user_tool_idx = i
                        break
            if stop:
                break
        return messages, last_user_prompt_idx, last_user_tool_idx

    def _hf_template_kwargs(self, conversation: ullm.conv.Conversation) -> dict[str, Any]:
        """The kwargs a local HF render needs to reproduce what the server's
        template renders for ``conversation``: the configured template kwargs,
        thinking off when no effort is set, the native effort folded in."""
        template_kwargs = dict(self.chat_template_kwargs)
        if self.native_reasoning_effort:
            effort = conversation.system_prompt.reasoning_effort
            if effort is None:
                template_kwargs["enable_thinking"] = False
            elif "reasoning_effort" not in template_kwargs:
                template_kwargs["reasoning_effort"] = effort
        return template_kwargs

    def _tool_schemas(self, conversation: ullm.conv.Conversation) -> Any:
        """The tool schemas a local HF render passes to the template, or None."""
        if not conversation.system_prompt.tools:
            return None
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in conversation.system_prompt.tools
        ]

    def supports_token_continuation(self) -> bool:
        """Only against a vLLM server whose capability has been probed
        (``trace_capability()``: the ids come back on the response and the
        tokenizer identity is resolved), and only without streaming (the
        SDK's stream accumulator drops the id extras)."""
        return not self._trace_no_token_ids and not self.streaming and bool(self._trace_hf_tokenizer_id)

    def _continuation_input(
        self, chat_kwargs: dict[str, Any], conversation: ullm.conv.Conversation, send_back_reasoning: str
    ) -> dict[str, Any] | None:
        """The ``/v1/completions`` request extending the served history by
        tokens, or None when this call renders afresh (first call, or a
        declared rewrite). Under exact history every other reason to render
        afresh is an error."""
        # Refused on EVERY call under exact history, the first one included:
        # a first call that cannot be witnessed leaves nothing to continue
        # from, and each later call would render afresh without ever
        # tripping the promise.
        if self._unwitnessable(ullm.continuation.has_image(chat_kwargs["messages"]), send_back_reasoning):
            return None
        history = ullm.continuation.served_history(conversation)
        if history is None:
            return None
        served, last = history
        tokenizer = self._trace_hf_tokenizer_get()
        template_kwargs = self._hf_template_kwargs(conversation)
        new_render = tokenizer.apply_chat_template(
            _hf_render_messages(chat_kwargs["messages"]),
            tools=self._tool_schemas(conversation),
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        continued = self._continue_prompt_cached(tokenizer, served, last, str(new_render), conversation)
        if continued is None:
            if self.exact_history:
                raise ullm.continuation.HistoryDriftError(
                    "exact_history: the history before the last answer no longer renders as it was served, and "
                    "no rewrite was declared (Conversation.mark_history_rewritten)"
                )
            return None
        extra_body = dict(chat_kwargs.get("extra_body") or {})
        extra_body.pop("chat_template_kwargs", None)
        extra_body["return_token_ids"] = True
        # The think/tool-call markup must survive detokenisation: the split
        # happens client-side on a continued call.
        extra_body["skip_special_tokens"] = False
        payload: dict[str, Any] = {
            "model": chat_kwargs["model"],
            "prompt": continued.prompt_ids,
            "extra_body": extra_body,
            "messages": chat_kwargs["messages"],
            "_continuation": True,
            "_think_seeded": continued.think_seeded,
            "_eos": str(tokenizer.eos_token or ""),
            "_chat_template_kwargs": template_kwargs,
        }
        for key in ("temperature", "top_p", "stop", "seed"):
            if key in chat_kwargs:
                payload[key] = chat_kwargs[key]
        if self.token_limit_param in chat_kwargs:
            payload["max_tokens"] = chat_kwargs[self.token_limit_param]
        return payload

    def _continue_prompt_cached(
        self,
        tokenizer: Any,
        served: list[int],
        last: ullm.continuation.ServedTurn,
        new_render: str,
        conversation: ullm.conv.Conversation,
    ) -> ullm.continuation.ContinuationPrompt | None:
        """``continue_prompt`` with a one-slot memo, so a segment is
        tokenized once even when the same request is first weighed
        (``token_count``) and then sent (``respond``): both derive the same
        (served, witness, render) triple, and the encode is a pure function
        of it. Keyed by content, not object identity — the proxy hydrates a
        fresh conversation per shipment."""
        # The ledger participates by CONTENT: sampled text can tokenize
        # noncanonically, so two conversations on a shared backend can agree
        # on epoch, render, answer sha, and ledger LENGTH while holding
        # different served ids -- a length key would hand the second
        # conversation the first one's ids, and the served-echo check
        # cannot catch it (the engine faithfully echoes what was sent).
        key = (
            conversation.history_epoch,
            tuple(served),
            last.answered_sha,
            last.head_len,
            len(last.output_ids),
            hashlib.sha256(new_render.encode("utf-8")).hexdigest(),
        )
        cached = self._continuation_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        continued = ullm.continuation.continue_prompt(tokenizer, served, last, new_render)
        if continued is not None:
            self._continuation_cache = (key, continued)
        return continued

    def _unwitnessable(self, has_images: bool, send_back_reasoning: str) -> bool:
        """Whether this call's answer cannot be witnessed — no probed vLLM
        surface, an image in the prompt, past reasoning stripped. Under
        exact history that is an error rather than a quiet fresh render."""
        refusal: str | None = None
        if not self.supports_token_continuation():
            refusal = (
                "token continuation on OpenAIServer needs a probed vLLM server (trace_capability) with "
                "streaming=False (server token ids + the resolved tokenizer identity)"
            )
        elif has_images:
            refusal = "an image in the prompt cannot be continued by tokens on /v1/completions (no image input)"
        elif send_back_reasoning != "all":
            refusal = (
                f"send_back_reasoning={send_back_reasoning!r} re-renders the answers the model was served; "
                "continuation needs 'all'"
            )
        if refusal is not None and self.exact_history:
            raise ullm.continuation.HistoryDriftError(f"exact_history: {refusal}")
        return refusal is not None

    def _leave_witness(
        self,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
        send_back_reasoning: str,
    ) -> None:
        """Record on ``element`` what the server served and sampled, with the
        local template's render of the answered history, so the next call
        can extend it by tokens (``ullm.continuation``)."""
        messages_before, _, _ = self._chat_messages(turn, conversation, send_back_reasoning, before=element)
        if self._unwitnessable(ullm.continuation.has_image(messages_before), send_back_reasoning):
            return
        completion = element._model_contents
        prompt_ids_raw = (completion.model_extra or {}).get("prompt_token_ids")
        choice = completion.choices[0] if completion.choices else None
        output_ids_raw = (choice.model_extra or {}).get("token_ids") if choice is not None else None
        if prompt_ids_raw is None or output_ids_raw is None:
            if self.exact_history:
                raise ullm.continuation.HistoryDriftError(
                    "exact_history: the server returned no token ids for this answer, so it cannot be continued"
                )
            return
        # The answered history is rendered exactly as the next call will
        # render it (same message builder), so the next render's prefix
        # compares against the same bytes.
        messages_through, _, _ = self._chat_messages(turn, conversation, send_back_reasoning)
        tokenizer = self._trace_hf_tokenizer_get()
        template_kwargs = self._hf_template_kwargs(conversation)
        tools = self._tool_schemas(conversation)
        head = tokenizer.apply_chat_template(
            _hf_render_messages(messages_before),
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        answered = tokenizer.apply_chat_template(
            _hf_render_messages(messages_through),
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
            **template_kwargs,
        )
        history = ullm.continuation.served_history(conversation, until=element)
        element._served = ullm.continuation.witness(
            tokenizer,
            epoch=conversation.history_epoch,
            prompt_ids=[int(t) for t in prompt_ids_raw],
            output_ids=[int(t) for t in output_ids_raw],
            served_before=history[0] if history is not None else None,
            head_render=str(head),
            answered_render=str(answered),
        )
        if element._served is None and self.exact_history:
            raise ullm.continuation.HistoryDriftError(
                "exact_history: the chat template does not render this answer as a suffix of the prompt it "
                "was sampled after, so the next call could not be continued by tokens"
            )

    def _stamp_llm_input(self, llm_input: dict[str, Any], run_id: str) -> dict[str, Any]:
        """Stamp traced requests' ``user`` field with ``run_id#nonce``.

        The OpenAI ``user`` field is the sanctioned attribution vehicle —
        vLLM ignores it, so serving behavior is untouched. The nonce makes
        every LOGICAL call unique, letting an edge-capture gateway dedupe
        transport retries (which re-send this identical llm_input). The
        stamp also lands in the trace step's request payload, which is how
        trace consumers join steps to gateway captures.
        """
        if self.trace_recorder is None:
            return llm_input
        return {**llm_input, "user": f"{run_id}#{uuid.uuid4().hex[:12]}"}

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """The transient-retry generation, restarted wholesale on a dropped
        connection (``_CONNECTION_ERRORS``) — even mid-stream, where the
        transient path refuses to retry (see the comment on
        ``_CONNECTION_ERRORS`` for why a mid-stream restart is safe)."""
        attempts = len(_CONNECTION_RETRY_DELAYS_S) + 1
        started = asyncio.get_running_loop().time()
        for attempt in range(attempts):
            sent = asyncio.get_running_loop().time()
            try:
                async for item in self._generate_with_transient_retry(sampling_params, llm_input):
                    yield item
                return
            except _CONNECTION_ERRORS as e:
                if attempt == attempts - 1:
                    # The bare exception names neither the route nor the
                    # request; a systematic cut (a proxy or read-timeout
                    # between client and server killing slow generations
                    # while instant probes pass) is diagnosable only from
                    # exactly that context, so it rides along on the error.
                    continued = isinstance(llm_input, dict) and llm_input.get("_continuation")
                    route = "/v1/completions (id prompt)" if continued else "/v1/chat/completions"
                    size = (
                        f"{len(llm_input['prompt'])} prompt ids"
                        if continued
                        else f"{len(llm_input.get('messages') or [])} messages"
                        if isinstance(llm_input, dict)
                        else "unknown size"
                    )
                    elapsed = asyncio.get_running_loop().time() - started
                    e.add_note(
                        f"[OpenAIServer] all {attempts} attempts on {route} ({size}) failed with connection "
                        f"errors over {elapsed:.0f}s against {self.base_url}. The server closing without a "
                        f"response while probes and /tokenize succeed points at a read-timeout between "
                        f"client and server cutting non-streaming generations (check proxies on the path, "
                        f"including HTTP(S)_PROXY in the client environment), not at the request content."
                    )
                    raise
                delay = _CONNECTION_RETRY_DELAYS_S[attempt]
                print(
                    f"[OpenAIServer] connection {type(e).__name__} on attempt "
                    f"{attempt + 1}/{attempts}, restarting request in {delay:.1f}s | "
                    f"{self._retry_context(llm_input, sent=sent, started=started)} | {_retry_detail(e)}",
                    flush=True,
                )
                await asyncio.sleep(delay)

    async def _generate_with_transient_retry(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """Call the chat-completions API, yielding snapshots when streaming,
        retrying a transient provider failure while nothing has been yielded."""
        started = asyncio.get_running_loop().time()
        for attempt in itertools.count():
            yielded = False
            sent = asyncio.get_running_loop().time()
            try:
                self._open_requests += 1
                try:
                    async for item in self._generate_once(sampling_params, llm_input):
                        yielded = True
                        yield item
                finally:
                    self._open_requests -= 1
                return
            except _TRANSIENT_ERRORS as e:
                # Past the first yield the consumer holds a snapshot, so a
                # replay would duplicate content: fail instead.
                if yielded or self._out_of_retries(attempt, started):
                    raise
                delay = _retry_after_seconds(e, attempt)
                print(
                    f"[OpenAIServer] transient {type(e).__name__} on attempt "
                    f"{self._attempt_label(attempt)}, retrying in {delay:.1f}s | "
                    f"{self._retry_context(llm_input, sent=sent, started=started)} | {_retry_detail(e)}",
                    flush=True,
                )
                await asyncio.sleep(delay)

    def _out_of_retries(self, attempt: int, started: float) -> bool:
        """Whether a call whose attempt ``attempt`` (from 0) just failed gives
        up: once its window has passed when it has one, else at the count."""
        if self.transient_retry_window_s is not None:
            return asyncio.get_running_loop().time() - started >= self.transient_retry_window_s
        return attempt >= self.max_transient_retries

    def _attempt_label(self, attempt: int) -> str:
        """The failed attempt's number against what bounds the call."""
        if self.transient_retry_window_s is not None:
            return f"{attempt + 1} of a {self.transient_retry_window_s / 3600:g} h window"
        return f"{attempt + 1}/{self.max_transient_retries + 1}"

    def _retry_context(self, llm_input: Any, *, sent: float, started: float) -> str:
        """Where a retry stands: how long the failed attempt was open, how
        long the call has been failing, what else this backend had open at
        the provider then (one backend serves a whole host, so that is the
        run's own concurrency), and which call it is when traced."""
        now = asyncio.get_running_loop().time()
        others = self._open_requests
        context = (
            f"failed after {now - sent:.1f}s, call failing for {now - started:.0f}s, "
            f"{others} other request{'' if others == 1 else 's'} open"
        )
        user = llm_input.get("user") if isinstance(llm_input, dict) else None
        return f"{context}, call {user}" if user else context

    async def _generate_once(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """One attempt at the request, with the response-shape defences."""
        last_snapshot: Any = None
        try:
            if isinstance(llm_input, dict) and llm_input.get("_continuation"):
                # A continued call: an id prompt on /v1/completions, lifted
                # into the chat shape the parser below expects.
                completion = await self._client.completions.create(**_completion_request(llm_input))
                check_continuation_served(list(llm_input["prompt"]), completion)
                response = _chat_completion_from_completion(
                    completion, think_seeded=bool(llm_input["_think_seeded"]), eos=str(llm_input["_eos"])
                )
                yield _require_choices(response)
            elif not self.streaming:
                # Non-streaming: single request, yields a ChatCompletion
                response = await self._client.chat.completions.create(**llm_input)
                # Checked BEFORE the yield: past the first yield ``_generate``
                # will not retry, and a choice-less body is exactly the case
                # that should be retried rather than propagated.
                yield _require_choices(response)
            else:
                # Streaming: the SDK's accumulator yields snapshots, then the
                # final ChatCompletion. Driven here over the raw stream rather
                # than through the SDK's stream helper, which raises at a
                # length finish when a tool is strict (ours all are) before it
                # reads the chunk that follows — the usage — and a truncated
                # reply must still be billed its prompt. The accumulator
                # raises the same way; the stream is read on past it.
                state = openai.lib.streaming.chat.ChatCompletionStreamState(
                    input_tools=llm_input.get("tools", openai.omit)
                )
                stream = await self._client.chat.completions.create(stream=True, **llm_input)
                async with stream:
                    async for chunk in stream:
                        try:
                            list(state.handle_chunk(chunk))
                        except openai.LengthFinishReasonError:
                            continue
                        last_snapshot = state.current_completion_snapshot
                        yield last_snapshot
                final = state.current_completion_snapshot
                if any(choice.finish_reason == "length" for choice in final.choices):
                    # The SDK will not parse a truncated completion, tool or
                    # not; its snapshot is the reply, and _process_output reads
                    # it as a max_tokens stop (the in-process backend also
                    # returns the truncated content) instead of crashing the
                    # game.
                    yield final
                else:
                    # Only the FINAL completion is checked: early snapshots
                    # legitimately have no choices yet. If snapshots were
                    # already yielded this propagates instead of retrying,
                    # which is the intended contract.
                    yield _require_choices(state.get_final_completion())
        except openai.BadRequestError as e:
            if (
                "context length" in str(e)
                or "prompt is too long" in str(e)
                or "max_tokens must be at least 1" in str(e)
            ):
                raise ullm.backend.PromptTooLongError(str(e)) from e
            raise
        except pydantic.ValidationError as e:
            # A provider may close a stream without ever sending
            # finish_reason, and the SDK's streaming accumulator requires
            # one to rebuild the completion. Substituting a value would be
            # worse than failing: finish_reason is how a truncated reply is
            # told from a complete one, and callers act on that distinction.
            if "finish_reason" not in str(e):
                raise
            raise ullm.backend.MalformedOutputError(
                "the provider closed the stream without a finish_reason, which the SDK's "
                "streaming reconstruction requires. The non-streaming path does no such "
                "reconstruction and is unaffected: construct this backend with "
                f"streaming=False. Underlying error: {e}"
            ) from e
        except json.JSONDecodeError as e:
            # The SDK's streaming accumulator eagerly json.loads each tool
            # call's ``arguments`` at its done-event. The snapshot has already
            # accumulated the raw arguments string, so recover it and let
            # ``_process_output``'s ``{"_raw": ...}`` fallback handle the bad
            # JSON (same tolerance as the non-streaming path). Only when no
            # snapshot exists is a retryable MalformedOutputError raised.
            if last_snapshot is not None:
                yield last_snapshot
            else:
                raise ullm.backend.MalformedOutputError(f"tool-call arguments were not valid JSON: {e}") from e

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Parse the response onto the element, then leave the continuation
        witness (the ids this answer was served after and sampled)."""
        self._parse_output(llm_output, element)
        if self._continuing:
            self._leave_witness(conversation, turn, element, self.resolve_send_back_reasoning(sampling_params))

    def _shed_served_ids(self, element: ullm.conv.LLMResponse) -> None:
        """The completion's ``prompt_token_ids`` extra is the whole prompt; the
        witness and the trace step have it, the history text does not need it."""
        completion = element._model_contents
        extra = getattr(completion, "model_extra", None)
        if isinstance(extra, dict):
            extra.pop("prompt_token_ids", None)

    def _parse_output(self, llm_output: Any, element: ullm.conv.LLMResponse) -> None:
        # llm_output is a ChatCompletion or ParsedChatCompletion (streaming returns the latter,
        # which has an unpicklable generic type parameter). Convert to plain ChatCompletion.
        """Parse the response onto the element, including the defences: tool calls
        salvaged from markup the server's parser missed, and bad-JSON tool
        arguments recovered from the last good streamed snapshot.
        """
        if type(llm_output) is not openai.types.chat.ChatCompletion:
            data = llm_output.model_dump()
            # Streaming snapshots may accumulate repeated role strings; fix them
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                if msg.get("role", "").startswith("assistant"):
                    msg["role"] = "assistant"
            llm_output = openai.types.chat.ChatCompletion.model_validate(data)
        element._model_contents = llm_output

        # Token usage from the API response
        usage = llm_output.usage
        if usage is not None:
            prompt_details = usage.prompt_tokens_details
            completion_details = usage.completion_tokens_details
            element.token_usage = ullm.backend.TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=prompt_details.cached_tokens if prompt_details and prompt_details.cached_tokens else 0,
                reasoning_tokens=completion_details.reasoning_tokens
                if completion_details and completion_details.reasoning_tokens
                else 0,
                num_requests=1,
            )
        else:
            element.token_usage = ullm.backend.TokenUsage(num_requests=1)

        choice = llm_output.choices[0]
        # Normalize finish_reason into ullm's StopReason.
        # OpenAI values: "stop", "length", "tool_calls", "function_call", "content_filter".
        _oai_finish = getattr(choice, "finish_reason", None)
        if _oai_finish is None:
            element.stop_reason = None
        else:
            stop_map: dict[str, ullm.conv.StopReason] = {
                "stop": "stop",
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "function_call": "tool_use",
            }
            element.stop_reason = stop_map.get(_oai_finish, "other")
        text = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning", None)
        # getattr: custom tool-call variants in newer SDKs have no .function.
        raw_tool_calls = [
            {"id": tc.id, "name": fn.name, "arguments": fn.arguments}
            for tc in (choice.message.tool_calls or [])
            if (fn := getattr(tc, "function", None)) is not None
        ]
        if not raw_tool_calls:
            # Server parser missed the call but the markup may survive in
            # content or reasoning text — salvage it (and strip it from the
            # content so the model doesn't see its own broken markup echoed).
            raw_tool_calls = _recover_tool_calls_from_markup(text, reasoning or "")
            if raw_tool_calls:
                text = _TOOL_CALL_BLOCK_RE.sub("", text).strip()

        # Set reasoning content if present (native reasoning field from Harmony/OpenAI)
        if reasoning:
            element.reasoning_content = ullm.conv.StringContent(text=reasoning)

        # Tool call response — build ToolCallContent elements
        if raw_tool_calls:
            contents = list[ullm.conv.Content]()
            if text:
                contents.append(ullm.conv.StringContent(text=text))

            for tc in raw_tool_calls:
                # Parse arguments; fall back to wrapping the raw text under
                # `_raw`. `Tool._maybe_remap_single_arg` handles the rename
                # to the tool's declared parameter name at run time.
                try:
                    args: Any = json.loads(tc["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": tc["arguments"]}
                if not isinstance(args, dict):
                    args = {"_raw": tc["arguments"]}
                contents.append(
                    ullm.conv.ToolCallContent(
                        call_id=tc["id"],
                        tool_name=tc["name"],
                        arguments=args,
                    )
                )
            element.final_content = ullm.conv.ListContent(elements=contents) if len(contents) > 1 else contents[0]
            element.status = "processed"
            return

        # Optionally parse <think>...</think> tags for reasoning models
        if self.parse_think_tags:
            think_start = text.find("<think>")
            think_end = text.find("</think>")
            if think_start != -1 and think_end != -1:
                # Validate exactly one pair of tags
                think_count = text.count("<think>")
                end_think_count = text.count("</think>")
                malformed_error: str | None = None
                if think_count != 1 or end_think_count != 1:
                    malformed_error = f"Unexpected <think>/</think> tag counts: {think_count}/{end_think_count}"
                elif think_start > think_end:
                    malformed_error = "<think> must appear before </think>"
                if malformed_error is not None:
                    if not self.accept_malformed_output:
                        raise ullm.backend.MalformedOutputError(malformed_error)
                    # Keep the sampled reply intact for history, tracing and RL credit.
                    element.final_content = ullm.conv.StringContent(text=text)
                    element.stop_reason = "malformed"
                    element.malformed_error = malformed_error
                    element.status = "processed"
                    return
                # Split into reasoning and final content
                reasoning_text = text[think_start + len("<think>") : think_end].strip()
                final = text[think_end + len("</think>") :].strip()
                element.reasoning_content = ullm.conv.StringContent(text=reasoning_text)
                element.final_content = ullm.conv.StringContent(text=final)
                element.status = "processed"
                return

        # No think tags found (or not parsing); entire text is final content
        element.final_content = ullm.conv.StringContent(text=text)
        element.status = "processed"

    async def engine_metrics(self) -> Optional["ullm.engine_monitor.EngineMetrics"]:
        """The served engine's scheduler numbers, from vLLM's ``/metrics``.

        ``None`` for an endpoint with no such route, or one that answers with
        nothing vLLM-shaped: a hosted API has no scheduler to report.
        Remembered, so a non-vLLM endpoint costs one request rather than one
        per status interval.
        """
        import ullm.engine_monitor

        if self._server_metrics is False:
            return None
        response = await self._server_client().get(f"{self._server_root()}/metrics")
        if response.status_code in (404, 405):
            self._server_metrics = False
            return None
        response.raise_for_status()
        # A 200 is not proof of a metrics endpoint: a hosted API's root serves
        # its own web page on any unclaimed path, and feeding that HTML to the
        # prometheus parser raises rather than coming back empty. The exposition
        # content type is what actually distinguishes the two.
        if "text/plain" not in response.headers.get("content-type", ""):
            self._server_metrics = False
            return None
        metrics = ullm.engine_monitor.metrics_from_prometheus_text(response.text)
        if not metrics.scalars:
            self._server_metrics = False  # prometheus, but nothing vLLM-shaped
            return None
        self._server_metrics = True
        return metrics

    # --- tracing (platypus U5) ---------------------------------------------

    def _server_root(self) -> str:
        """vLLM serves ``/tokenize``, ``/detokenize`` and ``/version`` at the
        app ROOT; only chat/completions lives under ``/v1``."""
        root = self.base_url.rstrip("/")
        return root[: -len("/v1")] if root.endswith("/v1") else root

    def _server_client(self) -> Any:
        """The HTTP client for the server's non-OpenAI routes, made on first use."""
        if self._server_http is None:
            import httpx

            self._server_http = httpx.AsyncClient(headers={"Authorization": f"Bearer {self.api_key}"}, timeout=120.0)
        return self._server_http

    async def _server_request(
        self, method: str, route: str, *, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Call an idempotent route with bounded transport retries.

        The caller owns HTTP-status handling because capability probes need
        to inspect expected 404/405 responses before raising other failures.
        """
        attempts = len(_SERVER_ROUTE_RETRY_DELAYS_S) + 1
        url = f"{self._server_root()}/{route.lstrip('/')}"
        for attempt in range(attempts):
            try:
                response = await self._server_client().request(method, url, json=payload)
                return response
            except httpx.TransportError as error:
                if attempt == attempts - 1:
                    raise
                delay = _SERVER_ROUTE_RETRY_DELAYS_S[attempt]
                print(
                    f"[OpenAIServer] server route {route} connection {type(error).__name__} on attempt "
                    f"{attempt + 1}/{attempts}, retrying in {delay:.2f}s",
                    flush=True,
                )
                # httpcore removes a failed connection from the shared pool;
                # retaining the client avoids closing unrelated concurrent calls.
                await asyncio.sleep(delay)
        raise AssertionError("server route retry loop exhausted without returning or raising")

    async def _server_get_json(self, route: str) -> dict[str, Any]:
        """GET one idempotent server-control route and decode its JSON body."""
        response = await self._server_request("GET", route)
        response.raise_for_status()
        return response.json()

    async def _server_post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to one idempotent server-control route and decode its JSON body."""
        response = await self._server_request("POST", route, payload={"model": self.model, **payload})
        response.raise_for_status()
        return response.json()

    async def _trace_tokenize(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: Any = None,
        chat_template_kwargs: Any = None,
    ) -> list[int]:
        """Ask the server for the exact token ids of a rendered prompt — the
        witness a trace step needs, which only the server can produce.
        """
        payload: dict[str, Any] = {"messages": messages, "add_generation_prompt": add_generation_prompt}
        if tools:
            payload["tools"] = tools
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        body = await self._server_post("tokenize", payload)
        if "tokens" not in body:
            raise ullm.tracing.TraceReconstructionError(f"/tokenize response missing 'tokens': {str(body)[:200]}")
        return [int(t) for t in body["tokens"]]

    async def trace_capability(self) -> Any:
        """Runtime probe, not a class property. Two grades of capability:

        * **vLLM server** (``/tokenize`` + honored ``return_token_ids``):
          full R5 records, policy identity resolved from ``/v1/models`` —
          the server is the authority on what it serves (adapter lineage
          included).
        * **hosted API** (OpenAI/OpenRouter/…): no token-id witness is
          obtainable, so records carry empty ``input_token_ids`` /
          ``output_token_ids`` and an ``unavailable`` input status, which
          taaf accepts by design (``_ACCEPTED_STATUSES_BY_FIELD``).

        The second grade exists because SFT on image steps cannot use token
        ids at all: ARC3-Training's ``token_exact`` mode rejects media steps,
        so vision training runs ``retemplate``, which reads ``messages`` +
        ``output_text`` + ``output_message.tool_calls`` + ``media/``. Refusing
        to write a trace whose only missing field the trainer discards would
        block API-provider distillation for no gain.
        """
        import ullm.tracing

        if self.streaming:
            raise ValueError(
                "tracing with OpenAIServer requires streaming=False: the SDK's stream "
                "accumulator drops the extra return_token_ids fields from the final completion. "
                "Construct the backend with streaming=False for trace runs."
            )
        client = self._server_client()
        root = self._server_root()
        if not await self._probe_serving_surface():
            return self._trace_capability_no_token_ids()
        # From here the server IS trace-capable, so an unidentifiable policy
        # is an error, not a None: a trace whose model cannot be identified
        # is unusable training provenance (ARC3-Inference posture).
        identity = await self._trace_fetch_identity()
        self._trace_identity = identity
        try:
            version = str((await client.get(f"{root}/version")).json().get("version", "unknown"))
        except Exception:
            version = "unknown"
        tokenizer_id = identity["parent_model_revision"] or identity["model_revision"] or self.model
        self._trace_hf_tokenizer_id = tokenizer_id
        return ullm.tracing.TraceCapability(
            provider_name="vllm",
            provider_version=version,
            api_kind="openai_chat_completions",
            provider_model_id=self.model,
            validation_method="platypus.oai.server_roundtrip.v1",
            model_id=self.model,
            model_revision=identity["model_revision"],
            model_weight_hash=identity["model_weight_hash"],
            parent_model_revision=identity["parent_model_revision"],
            tokenizer_id=tokenizer_id,
            processor_id=tokenizer_id,
        )

    async def _probe_serving_surface(self) -> bool:
        """Whether the endpoint is a vLLM server that returns token ids —
        and, if so, which tokenizer it serves (from ``/v1/models``: the
        adapter's base model, else the served model). This is everything
        continuation needs, and it touches no filesystem; the adapter's
        weight identity (a runtime-export file, U7) is tracing's concern
        and resolved by ``trace_capability`` alone. Returns False for a
        hosted API, arming the weaker trace grade."""
        try:
            # Discriminator 1: /tokenize exists (vLLM-only route).
            await self._trace_tokenize([{"role": "user", "content": "ping"}], add_generation_prompt=True)
            # Discriminator 2: return_token_ids is honored end to end.
            probe_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                self.token_limit_param: 1,
                "extra_body": {"return_token_ids": True},
            }
            probe = await self._client.chat.completions.create(**probe_kwargs)
            if (probe.model_extra or {}).get("prompt_token_ids") is None:
                self._trace_no_token_ids = True
                return False
            if (probe.choices[0].model_extra or {}).get("token_ids") is None:
                self._trace_no_token_ids = True
                return False
        except Exception as error:
            if self._probe_shows_no_vllm_surface(error):
                # Positively not a vLLM surface.
                self._trace_no_token_ids = True
                return False
            # Anything else — unreachable, unauthenticated, rate-limited, a
            # 5xx from a real vLLM server — is a setup failure, not evidence
            # about the route. Downgrading would permanently arm the weak
            # grade and silently record weaker training data; the fail-loud
            # posture (U5/U7) is to surface it instead.
            raise
        models = await self._server_get_json("v1/models")
        entry = next((m for m in models.get("data") or [] if str(m.get("id")) == self.model), None)
        if entry is None:
            raise RuntimeError(f"served model {self.model!r} not in /v1/models")
        self._trace_hf_tokenizer_id = str(entry.get("parent") or entry.get("root") or self.model)
        return True

    async def probe_continuation(self) -> None:
        """The serving-surface probe alone: continuation needs the server's
        ids and tokenizer identity, not the adapter's export file."""
        if self.streaming:
            return  # never continuable; supports_token_continuation says so
        await self._probe_serving_surface()

    @staticmethod
    def _probe_shows_no_vllm_surface(error: BaseException) -> bool:
        """Whether a capability-probe failure positively identifies an endpoint
        WITHOUT the vLLM trace surface, as opposed to one that could not be
        asked. Only the former may downgrade the grade: a transient failure
        classified as "hosted" would arm the weak grade for the whole run.
        """
        import ullm.tracing

        # The route answered, but not with the route's shape: a web app's
        # catch-all page or a body without 'tokens' — not vLLM.
        if isinstance(error, ullm.tracing.TraceReconstructionError):
            return True
        if isinstance(error, json.JSONDecodeError):
            return True
        # The route (or the vLLM-only request field) is refused as such.
        # Auth, rate limits and server errors are absent deliberately: they
        # say nothing about whether the surface exists.
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in (400, 404, 405, 415)
        if isinstance(error, openai.APIStatusError):
            return error.status_code in (400, 404, 405, 415)
        return False

    def _trace_capability_no_token_ids(self) -> Any:
        """Capability for a hosted API: provenance is the endpoint + model id
        the request was pinned to, since there is no served-weights identity to
        resolve. ``_trace_identity`` stays None so the drift guard no-ops (a
        hosted model cannot be hot-swapped under us mid-run the way the
        training server's adapters can — and if the provider does swap, no
        query we can make would reveal it)."""
        import ullm.tracing

        self._trace_no_token_ids = True
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        provider = "openrouter" if host.endswith("openrouter.ai") else (host or "openai_compatible")
        return ullm.tracing.TraceCapability(
            provider_name=provider,
            provider_version="unknown",  # hosted APIs expose no build id
            api_kind="openai_chat_completions",
            provider_model_id=self.model,
            validation_method="platypus.oai.no_token_witness.v1",
            model_id=self.model,
            # Identity fields stay empty: "empty means not applicable / unknown"
            # per TraceCapability. Inventing a revision would be worse than
            # admitting the provider does not publish one.
        )

    async def _trace_fetch_identity(self) -> dict[str, str]:
        """Resolve the served policy's identity — the server is the authority
        (`/v1/models` root/parent; adapters additionally pinned by their
        runtime export's `adapter_sha256`). Raises when the policy cannot be
        identified: unusable training provenance."""
        models = await self._server_get_json("v1/models")
        entries = {str(m.get("id")): m for m in models.get("data") or []}
        entry = entries.get(self.model)
        if entry is None:
            raise RuntimeError(f"served model {self.model!r} not in /v1/models (served: {sorted(entries)})")
        weights_path = str(entry.get("root") or "")
        parent = str(entry.get("parent") or "")
        weight_hash = ""
        if parent:
            from pathlib import Path

            summary_path = Path(weights_path) / "runtime_adapter_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"served adapter {weights_path} has no runtime_adapter_summary.json — "
                    "serve the runtime export, not a raw checkpoint"
                )
            summary = json.loads(summary_path.read_text())
            weight_hash = str(summary.get("adapter_sha256") or "")
            parent = str(summary.get("base_model_name") or parent)
        return {"model_revision": weights_path, "model_weight_hash": weight_hash, "parent_model_revision": parent}

    async def _trace_check_identity_drift(self) -> None:
        """The permanent training server hot-swaps adapters (checkpoint
        schedule); a swap mid-run would silently attribute steps generated
        under new weights to the trace's identity block. Re-resolve on every
        step (one tiny GET + a stat next to a model call) and stop the run on
        drift — same proof-at-capture posture as the token roundtrips."""
        if self._trace_identity is None:
            return
        current = await self._trace_fetch_identity()
        if current != self._trace_identity:
            raise ullm.tracing.TraceReconstructionError(
                f"served policy identity changed mid-run (adapter hot-swap?): "
                f"{self._trace_identity} -> {current}. Steps after a swap cannot be "
                "attributed to this trace's identity block; tracing refuses to continue."
            )

    @staticmethod
    def _trace_output_message(message: Any) -> dict[str, Any]:
        """Structured completion straight off the wire message (R5.37):
        reasoning keyed ``reasoning_content`` (the template contract), tool
        calls preserved. This is the channel that carries the ACTION for
        tool-calling agents, so consumers can re-render it through another
        model's template — ARC3-Training's retemplate depends on it."""
        out: dict[str, Any] = {"role": "assistant", "content": message.content}
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            out["reasoning_content"] = reasoning
        if message.tool_calls:
            out["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": fn.name, "arguments": fn.arguments}}
                for tc in message.tool_calls
                if (fn := getattr(tc, "function", None)) is not None
            ]
        return out

    def _build_trace_step_no_token_ids(
        self,
        *,
        run_id: str,
        sampling_params: ullm.backend.SamplingParameters,
        llm_input: Any,
        element: ullm.conv.LLMResponse,
    ) -> Any:
        """Trace step for a hosted API, where no token-id witness exists.

        Everything the trainer's retemplate path reads is present and exact:
        ``messages`` (the request as sent), ``output_text`` (the prose channel),
        ``output_message.tool_calls`` (the action), and the inline images the
        collector splits into ``media/``. What is absent is only the proof that
        those messages re-render to the exact prompt the provider tokenised,
        which is why the input status is ``unavailable`` rather than ``ok``.

        No synthetic text is put in ``output_text``: a step whose prose is empty
        but which carries a tool call is a real action, and ARC3-Training treats
        it as one. Faking a rendering here would train the model on markup no
        provider ever emitted. Measured on the 2026-07-30 26-game run: 79% of
        steps have empty prose and would lose their action without this.
        """
        import ullm.tracing

        completion = element._model_contents
        choice = completion.choices[0]
        # Stored view = JSON round-trip of the request kwargs actually sent.
        payload = json.loads(json.dumps(llm_input, default=str))
        messages = payload["messages"]
        usage = completion.usage
        prompt_tokens, cached = 0, 0
        if usage is not None:
            prompt_tokens = usage.prompt_tokens or 0
            if usage.prompt_tokens_details is not None:
                cached = usage.prompt_tokens_details.cached_tokens or 0
        has_media = any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for msg in messages
            for part in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        )
        return ullm.tracing.TraceStepRecord(
            run_id=run_id,
            messages=messages,
            input_token_ids=[],
            output_token_ids=[],
            output_text=str(choice.message.content or ""),
            output_message=self._trace_output_message(choice.message),
            sampling_params=ullm.tracing.sampling_to_dict(sampling_params),
            request_payload=payload,
            images=[],  # OAI media live inline as data URLs; the collector splits them
            input_token_reconstruction_status="unavailable",
            output_token_reconstruction_status="unavailable",
            media_reconstruction_status="unavailable" if has_media else "not_applicable",
            # From usage, not len(prompt_ids): there are no ids to measure.
            uncached_input_tokens=max(0, int(prompt_tokens) - int(cached)),
            # The hosted-API record carries the same malformed marker as the
            # token-witness path: a consumer penalizing accepted empty
            # replies must see it on every record kind.
            extra={
                "stop_reason": element.stop_reason,
                "malformed": getattr(element, "malformed_error", None),
            },
        )

    async def _build_trace_step(
        self,
        *,
        run_id: str,
        sampling_params: ullm.backend.SamplingParameters,
        llm_input: Any,
        element: ullm.conv.LLMResponse,
        conversation: ullm.conv.Conversation,
    ) -> Any:
        """Assemble this call's R5 trace step, with the server's own tokenisation
        as the witness.
        """
        import ullm.tracing

        if self._trace_no_token_ids:
            return self._build_trace_step_no_token_ids(
                run_id=run_id, sampling_params=sampling_params, llm_input=llm_input, element=element
            )
        await self._trace_check_identity_drift()
        completion = element._model_contents
        prompt_ids_raw = (completion.model_extra or {}).get("prompt_token_ids")
        choice = completion.choices[0]
        output_ids_raw = (choice.model_extra or {}).get("token_ids")
        if prompt_ids_raw is None or output_ids_raw is None:
            raise ullm.tracing.TraceReconstructionError(
                "server response carries no return_token_ids fields although tracing is enabled "
                "(prompt_token_ids/token_ids missing) — server too old or flag dropped mid-run"
            )
        prompt_ids = [int(t) for t in prompt_ids_raw]
        output_ids = [int(t) for t in output_ids_raw]

        # Stored view = JSON round-trip of the request kwargs actually sent.
        payload = json.loads(json.dumps(llm_input, default=str))
        messages = payload["messages"]
        continued = bool(payload.get("_continuation"))
        if continued:
            # The prompt WAS token ids: the witness is the server echoing
            # exactly what was sent. (The chat re-render deliberately does
            # not reproduce it — that drift is what continuation removes.)
            sent = [int(t) for t in payload["prompt"]]
            if sent != prompt_ids:
                raise ullm.tracing.input_mismatch_error(
                    "OpenAIServer token-continuation echo (/v1/completions)", sent, prompt_ids
                )
            chat_template_kwargs = dict(payload.get("_chat_template_kwargs") or {}) or None
            # The wire request (ids included) plus the messages view the
            # collector splits media out of and requires to round-trip.
            payload = {**_completion_request(payload), "messages": messages}
        else:
            chat_template_kwargs = _render_template_kwargs(payload)
            reconstructed = await self._trace_tokenize(
                messages,
                add_generation_prompt=True,
                tools=payload.get("tools"),
                chat_template_kwargs=chat_template_kwargs,
            )
            if reconstructed != prompt_ids:
                raise ullm.tracing.input_mismatch_error(
                    "OpenAIServer input-token roundtrip (/tokenize)", reconstructed, prompt_ids
                )

        detok = await self._server_post("detokenize", {"tokens": output_ids})
        if "prompt" not in detok:
            raise ullm.tracing.TraceReconstructionError(f"/detokenize response missing 'prompt': {str(detok)[:200]}")
        output_text = str(detok["prompt"])

        # A continued call answered as plain text on /v1/completions; the
        # lifted choice.message is a wire adapter, not the provider's
        # structured completion. The parsed element is the semantic view
        # the trace records for it (R5.37) — what the chat route's server
        # parser would have returned.
        output_message = (
            ullm.tracing.output_message_from_element(element)
            if continued
            else self._trace_output_message(choice.message)
        )

        # Output-reconstruction evidence (R5.38, non-gating): computed with a
        # LOCAL HF tokenizer loaded from the resolved policy identity — the
        # shared implementation with vllmx. (A server-side re-render proved
        # dialect-fragile: /tokenize's wire schema remaps reasoning/argument
        # fields before the template sees them.) Render failures classify as
        # mismatch; the input roundtrip above stays hard and server-side.
        output_status = "mismatch" if output_message is not None else "unavailable"
        if output_message is not None:
            try:
                output_status = ullm.tracing.classify_output_render_hf(
                    self._trace_hf_tokenizer_get(), chat_template_kwargs or {}, output_message, output_ids
                )
            except Exception:
                pass

        usage = completion.usage
        cached = 0
        if usage is not None and usage.prompt_tokens_details is not None:
            cached = usage.prompt_tokens_details.cached_tokens or 0
        has_media = any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for msg in messages
            for part in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        )
        return ullm.tracing.TraceStepRecord(
            run_id=run_id,
            messages=messages,
            input_token_ids=prompt_ids,
            output_token_ids=output_ids,
            output_text=output_text,
            output_message=output_message,
            sampling_params=ullm.tracing.sampling_to_dict(sampling_params),
            request_payload=payload,
            images=[],  # OAI media live inline as data URLs; the collector splits them
            output_token_reconstruction_status=output_status,
            media_reconstruction_status="unavailable" if has_media else "not_applicable",
            uncached_input_tokens=len(prompt_ids) - int(cached),
            extra=ullm.tracing.step_extra(
                element,
                continued=continued,
                history_epoch=conversation.history_epoch,
                exact_history=self.exact_history,
            ),
        )

    def _process_partial_output(
        self, sampling_params: ullm.backend.SamplingParameters, llm_output: Any, intermediate: ullm.conv.LLMResponse
    ) -> None:
        """For streaming: update intermediate output from the SDK's accumulated snapshot.
        Tool calls are not shown during streaming — they only appear in the final output.
        Any unexpected shape or parse hiccup is silently skipped; the next peek or
        the final `_process_output` will surface real errors."""
        try:
            if llm_output is None or not hasattr(llm_output, "choices") or not llm_output.choices:
                return
            message = llm_output.choices[0].message
            text = message.content or ""
            reasoning = getattr(message, "reasoning", None)

            if reasoning:
                intermediate.reasoning_content = ullm.conv.StringContent(text=reasoning)

            # Handle partial think tag parsing
            if self.parse_think_tags:
                think_start = text.find("<think>")
                think_end = text.find("</think>")
                if think_start != -1 and think_end == -1:
                    # Mid-reasoning: <think> seen but no </think> yet
                    intermediate.reasoning_content = ullm.conv.StringContent(
                        text=text[think_start + len("<think>") :].strip()
                    )
                    return
                elif think_start != -1 and think_end != -1:
                    # Reasoning complete
                    intermediate.reasoning_content = ullm.conv.StringContent(
                        text=text[think_start + len("<think>") : think_end].strip()
                    )
                    intermediate.final_content = ullm.conv.StringContent(
                        text=text[think_end + len("</think>") :].strip()
                    )
                    return

            if text.strip():
                intermediate.final_content = ullm.conv.StringContent(text=text)
            else:
                intermediate.final_content = ullm.conv.EmptyContent()
        except Exception:
            # Silently absorb any mid-stream parse failure.
            return

    async def close(self) -> None:
        """Close the underlying HTTP clients to avoid async cleanup warnings."""
        if self._server_http is not None:
            try:
                await self._server_http.aclose()
            except Exception:
                pass
            self._server_http = None
        await self._client.close()

    def decode_input(self, request_id: int) -> str:
        """Format the stored input as a human-readable string."""
        kwargs = self._request_log[request_id]["input"]
        messages = kwargs.get("messages", [])
        # Separate messages from other params for cleaner display
        params = {k: v for k, v in kwargs.items() if k not in ("messages", "model", "tools")}
        parts = [f"model: {kwargs.get('model', '?')}"]
        if "tools" in kwargs:
            tool_names = [t["function"]["name"] for t in kwargs["tools"]]
            parts.append(f"tools: [{', '.join(tool_names)}]")
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                parts.append(f"[tool:{msg.get('tool_call_id', '?')}] {msg['content']}")
            elif "tool_calls" in msg:
                tc_strs = [f"{tc['function']['name']}({tc['function']['arguments']})" for tc in msg["tool_calls"]]
                parts.append(f"[assistant] {msg.get('content') or ''} | tool_calls: {', '.join(tc_strs)}")
            else:
                parts.append(f"[{role}] {msg['content']}")
        if params:
            parts.append(f"params: {json.dumps(params, default=str)}")
        return "\n".join(parts)

    def decode_output(self, request_id: int) -> str:
        """Extract the response text from the stored ChatCompletion, including reasoning and tool calls."""
        output = self._request_log[request_id]["output"]
        message = output.choices[0].message
        parts = list[str]()
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            parts.append(f"[reasoning] {reasoning}")
        parts.append(f"[content] {message.content or ''}")
        if message.tool_calls:
            for tc in message.tool_calls:
                parts.append(f"[tool_call] {tc.function.name}({tc.function.arguments})")
        return "\n".join(parts)
