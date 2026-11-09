"""Moonshot's OpenAI-compatible Kimi API (``https://api.moonshot.ai/v1``),
behind ``OpenAIServer``.

The API speaks OpenAI's chat-completions shape with these differences, read
off live calls to ``kimi-k3.1-pro`` (2026-10-27):

- Reasoning comes back as ``reasoning_content``, on the message and on stream
  deltas, and goes back the same way on past assistant messages, which the
  API replays by default (turn 2's prompt grows by the reasoning, and a
  number picked only in it is recalled).
- ``reasoning_effort`` ``none`` answers directly (as does ``thinking: {type:
  disabled}``); ``minimal``, ``low``, ``medium`` and ``high`` are rendered
  into the prompt and reason less; any other value, ``xhigh``, ``max`` or
  nonsense alike, is taken silently as the API's default, which reasons
  most. The backend sends its own ``reasoning_effort`` when one is set, and
  none for the default; a conversation's level is left to the system prompt
  (``reasoning_effort_prompts``), as an engine teacher's play level is.
- Sampling is fixed for the model: temperature 1, top_p 0.95 and
  presence_penalty 0 are the only values taken, anything else refused (400).
  The backend sends neither temperature nor top_p, and refuses any other
  value rather than let a caller believe it sampled differently.
- Unknown request fields are ignored, not refused, so a vLLM extension
  (``chat_template_kwargs``, ``top_k``) would vanish silently: those are
  refused here, ``return_token_ids`` dropped. ``logit_bias`` is refused by
  the API.
- The prompt cache is the provider's own (automatic), reported as
  ``usage.prompt_tokens_details.cached_tokens``: a call hits the previous
  prompt rounded down to 2,048 tokens, never the previous reply; images keep
  it. No ``cache_control`` marks are sent. An entry expires once unused for
  its lifetime, which ``prompt_cache_options`` (``mode`` ``implicit``,
  ``ttl`` ``5m`` or ``1h``, ``5m`` when omitted) sets at its first write;
  a hit renews it for that lifetime. ``1h`` writes cost twice (the pricing
  page lists $3 against $6 a million for ``kimi-k3``, nothing for
  ``kimi-k3.1-pro``).
- No rate-limit headers; 200 base64 images in one request are taken. Past
  the account's concurrency (50) a 429 says so (``rate_limit_reached_error``).
  An account that has sent heavily for a while, ~100M prompt tokens an hour
  for an hour and a half, is then refused for hours with a 429 that reads as
  the API's own overload (``engine_overloaded_error``, "The engine is
  currently overloaded"), each refusal after the request was held about a
  minute, while the same model serves another account (2026-10-29): the
  backend retries for twelve hours.
- A past assistant message with no text and no tool call is refused (400),
  reasoning alone not counting: one is sent with ``(no text)`` (2026-10-28).
- ``logprobs`` / ``top_logprobs`` (at most 20) return per-token
  alternatives at ``choices[0].logprobs.content``, reasoning tokens included,
  section markers not (2026-10-28). A traced call asks for them.

There is no ``/tokenize`` and no token-id witness: counts are ``None`` and
continuation is unsupported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import openai
import ullm.backend
import ullm.backends.oai
import ullm.conv

# The only sampling values the API takes for the model; it refuses others.
_FIXED_SAMPLING = {"temperature": 1.0, "top_p": 0.95}
# extra_body fields the API reads; it ignores every other one silently.
_KNOWN_EXTRA_FIELDS = ("presence_penalty", "frequency_penalty")
# What a past assistant message with no text and no tool call carries
# instead: the API refuses one (400, "the message at position N with role
# 'assistant' must not be empty"), its reasoning alone not counting. The
# ARC2 duck's word for a reply with no text (``DuckAgent.audit_no_reply``).
_EMPTY_REPLY_TEXT = "(no text)"


def _with_reasoning(llm_output: Any) -> Any:
    """The response with its ``reasoning_content`` also under ``reasoning``,
    the attribute ``OpenAIServer`` reads; unchanged when there is none."""
    message = llm_output.choices[0].message if getattr(llm_output, "choices", None) else None
    reasoning = getattr(message, "reasoning_content", None) if message is not None else None
    if not reasoning:
        return llm_output
    out = llm_output.model_copy(deep=True)
    setattr(out.choices[0].message, "reasoning", reasoning)
    return out


@dataclass
class KimiServer(ullm.backends.oai.OpenAIServer):
    """``OpenAIServer`` on Moonshot's Kimi API: ``reasoning_content`` parsed and
    replayed, the API's reasoning parameter, its fixed sampling, no vLLM
    extensions."""

    base_url: str = field(kw_only=True, default="https://api.moonshot.ai/v1")
    # Where the key is read when the client is built, so a pickled backend
    # (an agent in a run's files) never carries it.
    api_key_env: str = field(kw_only=True, default="KIMI_API_KEY")
    # Sent as the API's reasoning_effort when set: "none" answers directly,
    # minimal to high reason less than the default. None sends no level: the
    # API's default, which reasons most.
    reasoning_effort: Optional[str] = field(kw_only=True, default=None)
    # How long one call may take before the client gives up.
    request_timeout_s: float = field(kw_only=True, default=7200.0)
    send_back_reasoning: Any = field(kw_only=True, default="all")
    native_reasoning_effort: bool = field(default=False)
    reasoning_history_field: str | None = "reasoning_content"
    include_cache_control: bool = field(default=False)
    # The API's refusals of a heavily used account last hours (the module
    # docstring). A call waits them out for twelve hours, however many
    # retries that takes, so a run hibernates through them and plays on
    # after (2026-10-29). Its caller's own deadlines must allow that
    # long: the teacher runs turn the host's inactivity watchdog off.
    transient_retry_window_s: Optional[float] = field(default=12 * 3600.0)
    # Per-output-token alternatives asked of the API on a traced call (the
    # top 20, its most; reasoning tokens included) and kept as returned in
    # the trace step's ``extra["teacher_logprobs"]``: TAAF's own ``logprobs``
    # aligns to ``output_token_ids``, which an API trace has none of
    # (2026-10-28, lossless, for whatever the SFT side can make of them).
    # None asks for none.
    top_logprobs: Optional[int] = field(kw_only=True, default=20)
    # The prompt cache's lifetime, sent as prompt_cache_options' ttl: "5m" or
    # "1h". An hour, at twice the price of a write: on a busy afternoon
    # (2026-10-29) retries held calls 20-40 minutes, and under five minutes
    # each one lost its game's cache, worth $0.13-0.35 a call against about
    # $2 a million more for the writes (2026-10-29). None sends no
    # option: the API's five minutes.
    prompt_cache_ttl: Optional[str] = field(kw_only=True, default="1h")

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = self._new_client()
        # No /tokenize and no /metrics behind the API: not probed.
        self._server_tokenize = False
        self._server_metrics = False

    def _new_client(self) -> openai.AsyncOpenAI:
        """The client, with the key from ``api_key`` when given, else from
        the environment."""
        key = self.api_key if self.api_key != "unused" else os.environ[self.api_key_env]
        return openai.AsyncOpenAI(base_url=self.base_url, api_key=key, timeout=self.request_timeout_s)

    def _rebuild_transient(self) -> None:
        """The parent's rebuild, with the client keyed from the environment."""
        super()._rebuild_transient()
        self._client = self._new_client()

    def supports_token_continuation(self) -> bool:
        """The API returns no token ids to continue from."""
        return False

    async def engine_metrics(self) -> Optional[Any]:
        """A hosted API publishes no engine metrics."""
        return None

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> dict[str, Any]:
        """The parent's arguments, with the API's reasoning parameter, its
        fixed sampling left to the API, nothing it would drop unannounced,
        no past assistant message it would refuse as empty, the prompt
        cache's lifetime, and on a traced call the per-token alternatives."""
        kwargs = super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)
        for message in kwargs.get("messages", []):
            if message.get("role") == "assistant" and not message.get("tool_calls") and not message.get("content"):
                message["content"] = _EMPTY_REPLY_TEXT
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.top_logprobs is not None and self.trace_recorder is not None:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.top_logprobs
        for name, value in _FIXED_SAMPLING.items():
            asked = kwargs.pop(name, None)
            if asked is not None and asked != value:
                raise ValueError(f"Kimi's API samples at {name}={value} only; asked for {asked}")
        extra_body = dict(kwargs.pop("extra_body", {}))
        extra_body.pop("return_token_ids", None)
        ignored = sorted(set(extra_body) - set(_KNOWN_EXTRA_FIELDS))
        if ignored:
            raise ValueError(f"Kimi's API ignores {ignored} silently")
        if self.prompt_cache_ttl is not None:
            extra_body["prompt_cache_options"] = {"mode": "implicit", "ttl": self.prompt_cache_ttl}
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _parse_output(self, llm_output: Any, element: ullm.conv.LLMResponse) -> None:
        """The parent's parse, reading ``reasoning_content`` as reasoning."""
        super()._parse_output(_with_reasoning(llm_output), element)

    def _build_trace_step_no_token_ids(
        self,
        *,
        run_id: str,
        sampling_params: ullm.backend.SamplingParameters,
        llm_input: Any,
        element: ullm.conv.LLMResponse,
    ) -> Any:
        """The parent's hosted-API step, with the API's per-token alternatives
        under ``extra["teacher_logprobs"]`` as returned (token, bytes,
        logprob and the top alternatives' same) when the call asked for
        them, packed (``ullm.tracing.PackedJson``) while the collector holds
        the step; the written trace carries them unpacked."""
        import ullm.tracing

        record = super()._build_trace_step_no_token_ids(
            run_id=run_id, sampling_params=sampling_params, llm_input=llm_input, element=element
        )
        logprobs = getattr(element._model_contents.choices[0], "logprobs", None)
        content = getattr(logprobs, "content", None)
        if content is not None:
            record.extra["teacher_logprobs"] = ullm.tracing.PackedJson.pack(
                [entry.model_dump(mode="json") for entry in content]
            )
        return record

    def _shed_served_ids(self, element: ullm.conv.LLMResponse) -> None:
        """The parent's shedding, and the per-token alternatives too: the
        trace step has them, and the conversation, which keeps the completion
        for its history and goes to the game whole, never reads them."""
        super()._shed_served_ids(element)
        for choice in getattr(element._model_contents, "choices", None) or []:
            if getattr(choice, "logprobs", None) is not None:
                choice.logprobs = None

    def _process_partial_output(
        self, sampling_params: ullm.backend.SamplingParameters, llm_output: Any, intermediate: ullm.conv.LLMResponse
    ) -> None:
        """The parent's streaming update, reading ``reasoning_content`` as
        reasoning."""
        try:
            llm_output = _with_reasoning(llm_output)
        except Exception:
            return  # the parent's contract: a mid-stream hiccup is skipped
        super()._process_partial_output(sampling_params, llm_output, intermediate)
