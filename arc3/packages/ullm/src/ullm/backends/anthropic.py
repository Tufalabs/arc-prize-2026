"""Backend for Anthropic's native Messages API."""

import base64
import io
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import anthropic
import ullm.backend
import ullm.backends.common as common
import ullm.conv


def _encode_image_for_anthropic(img: ullm.conv.ImageContent) -> dict[str, Any]:
    """Encode an ImageContent (numpy H×W×3 uint8) as an Anthropic image content block.
    Always emits PNG; base64-encodes the bytes for inline transport."""
    from PIL import Image

    pil = Image.fromarray(img.data, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64},
    }


def _user_content_to_anthropic(content: ullm.conv.Content) -> str | list[dict[str, Any]]:
    """Convert a UserPrompt's Content into the Anthropic ``content`` field for a user message.
    Returns a plain string when there are no images (preserving the existing single-string
    shape); otherwise returns a list of text/image content blocks.

    Walking is order-preserving: text accumulated since the last image/flush becomes one
    text block, then the image block, and so on. Empty/tool-call subtrees are skipped."""
    blocks = list[dict[str, Any]]()
    pending_text = list[str]()
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
            blocks.append(_encode_image_for_anthropic(c))
            has_image = True
        elif isinstance(c, ullm.conv.EmptyContent):
            pass
        elif isinstance(c, ullm.conv.ToolCallContent):
            # ToolCallContent should never appear in a UserPrompt; if it does, skip.
            pass
        else:
            raise ValueError(f"Unsupported content type for Anthropic user message: {type(c)}")

    walk(content)
    if not has_image:
        return "\n".join(pending_text)
    flush_text()
    return blocks


def _to_anthropic_messages(
    conversation: ullm.conv.Conversation,
    current_turn: ullm.conv.ResponseTurn | None,
    send_back_reasoning: str,
    include_cache_control: bool = False,
) -> list[dict[str, Any]]:
    """Convert ullm conversation to Anthropic message format.
    Returns the messages list (system prompt is handled separately).

    Key differences from OpenAI format:
    - Tool results are content blocks in a user message, not standalone role=tool messages
    - Thinking blocks are content blocks in assistant messages with a signature field
    - Consecutive same-role messages must be merged

    If include_cache_control is True, adds cache_control breakpoints on:
    - The last real UserPrompt before the current turn (cross-turn cache prefix)
    - The last user message overall (current request boundary)"""
    messages = list[dict[str, Any]]()
    cache_control = {"type": "ephemeral"}

    # Track indices for cache breakpoint placement
    last_user_prompt_idx: int | None = None  # Last real UserPrompt message index
    last_user_msg_idx: int | None = None  # Last user-role message index (could be tool_result)

    for element in conversation.elements:
        if isinstance(element, ullm.conv.UserPrompt):
            messages.append({"role": "user", "content": _user_content_to_anthropic(element.content)})
            last_user_prompt_idx = len(messages) - 1
            last_user_msg_idx = len(messages) - 1

        elif isinstance(element, ullm.conv.ResponseTurn):
            is_current_turn = element is current_turn
            include_reasoning = send_back_reasoning == "all" or (
                send_back_reasoning == "current_turn" and is_current_turn
            )

            # Walk the turn's elements, grouping LLMResponses as assistant messages
            # and consecutive ToolResults as a single user message
            for sub in element.elements:
                if isinstance(sub, ullm.conv.LLMResponse):
                    if sub.do_not_send_to_agent:
                        continue
                    content_blocks = list[dict[str, Any]]()

                    # Include thinking blocks if reasoning should be sent back
                    if include_reasoning and sub._model_contents is not None:
                        for block in sub._model_contents.content:
                            if block.type == "thinking":
                                content_blocks.append(
                                    {
                                        "type": "thinking",
                                        "thinking": block.thinking,
                                        "signature": block.signature,
                                    }
                                )
                            elif block.type == "redacted_thinking":
                                content_blocks.append(
                                    {
                                        "type": "redacted_thinking",
                                        "data": block.data,
                                    }
                                )

                    # Add text content
                    text = sub.final_content.to_string()
                    tool_calls = sub.final_content.tool_calls()

                    if tool_calls:
                        # Text before tool calls (if any)
                        non_tool_text = common._extract_text(sub.final_content)
                        if non_tool_text:
                            content_blocks.append({"type": "text", "text": non_tool_text})
                        # Tool use blocks
                        for tc in tool_calls:
                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tc.call_id,
                                    "name": tc.tool_name,
                                    "input": tc.arguments,
                                }
                            )
                    elif text:
                        content_blocks.append({"type": "text", "text": text})

                    if content_blocks:
                        messages.append({"role": "assistant", "content": content_blocks})

                elif isinstance(sub, ullm.conv.ToolResult):
                    # Tool results go in user messages; merge with previous if also a tool result
                    result_block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": sub.call_id,
                        "content": common.content_to_string(sub.content),
                    }
                    if sub.is_error:
                        result_block["is_error"] = True

                    # Merge consecutive tool results into the same user message
                    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
                        messages[-1]["content"].append(result_block)
                    else:
                        messages.append({"role": "user", "content": [result_block]})
                    last_user_msg_idx = len(messages) - 1

    # Place cache breakpoints
    if include_cache_control:

        def _add_breakpoint(idx: int) -> None:
            msg = messages[idx]
            if isinstance(msg["content"], str):
                text_block: dict[str, Any] = {"type": "text", "text": msg["content"]}
                msg["content"] = [text_block]
            msg["content"][-1]["cache_control"] = cache_control

        if last_user_msg_idx is not None:
            _add_breakpoint(last_user_msg_idx)
        # Only add cross-turn breakpoint if it's a different message
        if last_user_prompt_idx is not None and last_user_prompt_idx != last_user_msg_idx:
            _add_breakpoint(last_user_prompt_idx)

    return messages


@dataclass
class Anthropic(ullm.backend.Backend):
    """Backend for Anthropic's native Messages API.
    Uses the Anthropic Python SDK directly, supporting extended thinking,
    native tool use, and prompt caching."""

    api_key: str = field(kw_only=True)
    model: str = field(kw_only=True)  # e.g. "claude-haiku-4-5-20251001", "claude-sonnet-4-6-20250514"
    streaming: bool = field(default=True)
    include_cache_control: bool = field(default=True)  # Add cache_control breakpoints for prompt caching
    set_adaptive_thinking: bool = field(default=True)  # If False, reasoning_effort is not passed in any way
    _client: anthropic.AsyncAnthropic = field(init=False, repr=False)

    def _transient_fields(self) -> tuple[str, ...]:
        """The live client is rebuilt rather than copied."""
        return ("_client",)

    def _rebuild_transient(self) -> None:
        """Recreate the client after a copy or a restore."""
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

    def __post_init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        # Auto-set pricing from known Anthropic pricing tables
        for prefix, pricing in ullm.backend.ANTHROPIC_PRICING.items():
            if self.model.startswith(prefix):
                self.pricing = pricing
                break
        super().__post_init__()

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> dict[str, Any]:
        # Build messages using Anthropic format (cache breakpoints placed during construction)
        """Render the conversation into Anthropic's message shape, cache
        breakpoints included.
        """
        messages = _to_anthropic_messages(
            conversation, turn, send_back_reasoning, include_cache_control=self.include_cache_control
        )

        # Build kwargs for messages.create() / .stream()
        kwargs = dict[str, Any]()
        kwargs["model"] = self.model
        kwargs["messages"] = messages

        # System prompt as separate parameter
        system_text = common.content_to_string(conversation.system_prompt.content)
        if system_text:
            kwargs["system"] = system_text

        # Sampling parameters
        if isinstance(sampling_params, ullm.backend.GenericSamplingParameters):
            if sampling_params.max_tokens is not None:
                kwargs["max_tokens"] = sampling_params.max_tokens
            else:
                kwargs["max_tokens"] = 128_000  # Anthropic requires max_tokens; 128k is the Opus 4.6/4.7 output cap
            if sampling_params.temperature is not None:
                kwargs["temperature"] = sampling_params.temperature
            if sampling_params.top_p is not None:
                kwargs["top_p"] = sampling_params.top_p
            if sampling_params.stop_sequences is not None:
                kwargs["stop_sequences"] = sampling_params.stop_sequences
            for k, v in sampling_params.extra_params.items():
                kwargs[k] = v
        else:
            raise ValueError(f"Unsupported sampling parameters type: {type(sampling_params)}")

        # Thinking / reasoning effort via output config
        # Anthropic requires temperature=1 when thinking is enabled
        effort = conversation.system_prompt.reasoning_effort
        if effort is not None and self.set_adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}
            temp = kwargs.get("temperature")
            if temp is not None:
                if 0.7 <= temp <= 1.3:
                    kwargs["temperature"] = 1.0
                else:
                    raise ValueError(
                        f"Anthropic requires temperature=1 when thinking is enabled, got temperature={temp}"
                    )

        # Tool definitions
        if conversation.system_prompt.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in conversation.system_prompt.tools
            ]

        # Prompt caching: system/tools breakpoint (message breakpoints handled in _to_anthropic_messages)
        if self.include_cache_control:
            cache_control = {"type": "ephemeral"}
            tools = kwargs.get("tools")
            if tools:
                tools[-1]["cache_control"] = cache_control
            elif "system" in kwargs:
                if isinstance(kwargs["system"], str):
                    system_block: dict[str, Any] = {"type": "text", "text": kwargs["system"]}
                    kwargs["system"] = [system_block]
                kwargs["system"][-1]["cache_control"] = cache_control

        return kwargs

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """Call the Messages API — snapshots then the final message when streaming
        — translating a too-long prompt into ``PromptTooLongError``.
        """
        try:
            if not self.streaming:
                response = await self._client.messages.create(**llm_input)
                yield response
            else:
                # Streaming: yields snapshots, then the final Message
                async with self._client.messages.stream(**llm_input) as stream:
                    async for _event in stream:
                        yield stream.current_message_snapshot
                    yield await stream.get_final_message()
        except anthropic.BadRequestError as e:
            if "prompt is too long" in str(e):
                raise ullm.backend.PromptTooLongError(str(e)) from e
            raise

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        # llm_output is an anthropic.types.Message (streaming and non-streaming unified)
        """Parse Anthropic's response onto the element: text, thinking blocks with
        their signatures, tool calls, stop reason and usage.
        """
        element._model_contents = llm_output

        # Token usage — Anthropic reports non-cached input tokens separately from cached ones,
        # so prompt_tokens = input_tokens + cache_read + cache_creation (total input)
        usage = llm_output.usage
        cached_read = usage.cache_read_input_tokens or 0
        cached_write = usage.cache_creation_input_tokens or 0
        element.token_usage = ullm.backend.TokenUsage(
            prompt_tokens=usage.input_tokens + cached_read + cached_write,
            completion_tokens=usage.output_tokens,
            cached_tokens=cached_read,
            cache_write_tokens=cached_write,
            num_requests=1,
        )

        # Normalize Anthropic stop_reason into ullm's StopReason.
        # Anthropic values: "end_turn", "max_tokens", "stop_sequence", "tool_use",
        # "pause_turn", "refusal".
        _ant_stop = getattr(llm_output, "stop_reason", None)
        if _ant_stop is None:
            element.stop_reason = None
        else:
            stop_map: dict[str, ullm.conv.StopReason] = {
                "end_turn": "stop",
                "max_tokens": "max_tokens",
                "stop_sequence": "stop_sequence",
                "tool_use": "tool_use",
            }
            element.stop_reason = stop_map.get(_ant_stop, "other")

        # Parse content blocks into reasoning/final content
        reasoning_parts = list[ullm.conv.Content]()
        final_parts = list[ullm.conv.Content]()

        for block in llm_output.content:
            if block.type == "thinking":
                reasoning_parts.append(ullm.conv.StringContent(text=block.thinking))
                # Approximate reasoning tokens by encoding; exact count not provided by API
            elif block.type == "redacted_thinking":
                pass  # Can't show redacted thinking, but it's preserved in _model_contents
            elif block.type == "text":
                final_parts.append(ullm.conv.StringContent(text=block.text))
            elif block.type == "tool_use":
                final_parts.append(
                    ullm.conv.ToolCallContent(
                        call_id=block.id,
                        tool_name=block.name,
                        arguments=dict(block.input) if block.input else {},
                    )
                )

        if reasoning_parts:
            element.reasoning_content = (
                ullm.conv.ListContent(elements=reasoning_parts) if len(reasoning_parts) > 1 else reasoning_parts[0]
            )

        if final_parts:
            element.final_content = (
                ullm.conv.ListContent(elements=final_parts) if len(final_parts) > 1 else final_parts[0]
            )

        element.status = "processed"

    def _process_partial_output(
        self, sampling_params: ullm.backend.SamplingParameters, llm_output: Any, intermediate: ullm.conv.LLMResponse
    ) -> None:
        """Parse a streaming snapshot into reasoning/final content for peeking.
        Any unexpected shape or parse hiccup is silently skipped; the next peek
        or the final `_process_output` will surface real errors."""
        try:
            if llm_output is None or not hasattr(llm_output, "content"):
                return

            reasoning_parts = list[ullm.conv.Content]()
            final_parts = list[ullm.conv.Content]()

            for block in llm_output.content:
                if hasattr(block, "type"):
                    if block.type == "thinking" and hasattr(block, "thinking") and block.thinking:
                        reasoning_parts.append(ullm.conv.StringContent(text=block.thinking))
                    elif block.type == "text" and hasattr(block, "text") and block.text:
                        final_parts.append(ullm.conv.StringContent(text=block.text))

            if reasoning_parts:
                intermediate.reasoning_content = (
                    ullm.conv.ListContent(elements=reasoning_parts) if len(reasoning_parts) > 1 else reasoning_parts[0]
                )
            if final_parts:
                intermediate.final_content = (
                    ullm.conv.ListContent(elements=final_parts) if len(final_parts) > 1 else final_parts[0]
                )
        except Exception:
            # Silently absorb any mid-stream parse failure.
            return

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    def decode_input(self, request_id: int) -> str:
        """Format the stored input as a human-readable string."""
        kwargs = self._request_log[request_id]["input"]
        parts = [f"model: {kwargs.get('model', '?')}"]
        if "system" in kwargs:
            system = kwargs["system"]
            if isinstance(system, list):
                system = " ".join(b.get("text", "") for b in system)
            parts.append(f"[system] {system}")
        if "tools" in kwargs:
            tool_names = [t["name"] for t in kwargs["tools"]]
            parts.append(f"tools: [{', '.join(tool_names)}]")
        for msg in kwargs.get("messages", []):
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                parts.append(f"[{role}] {content}")
            elif isinstance(content, list):
                block_strs = list[str]()
                for block in content:
                    btype = block.get("type", "?")
                    if btype == "text":
                        block_strs.append(block.get("text", ""))
                    elif btype == "tool_use":
                        block_strs.append(f"{block['name']}({block['input']})")
                    elif btype == "tool_result":
                        block_strs.append(f"[result:{block.get('tool_use_id', '?')}] {block.get('content', '')}")
                    elif btype == "thinking":
                        block_strs.append(f"[thinking] {block.get('thinking', '')[:100]}...")
                    else:
                        block_strs.append(f"[{btype}]")
                parts.append(f"[{role}] {' | '.join(block_strs)}")
        return "\n".join(parts)

    def decode_output(self, request_id: int) -> str:
        """Format the stored output as a human-readable string."""
        output = self._request_log[request_id]["output"]
        parts = list[str]()
        for block in output.content:
            if block.type == "thinking":
                parts.append(f"[thinking] {block.thinking[:200]}...")
            elif block.type == "redacted_thinking":
                parts.append("[redacted_thinking]")
            elif block.type == "text":
                parts.append(f"[text] {block.text}")
            elif block.type == "tool_use":
                parts.append(f"[tool_use] {block.name}({block.input})")
            else:
                parts.append(f"[{block.type}]")
        parts.append(f"stop_reason: {output.stop_reason}")
        return "\n".join(parts)
