"""Common utilities shared across backends that use OpenAI-style message dicts."""

import base64
import io
import json
from typing import Any, Callable, Optional

import ullm.conv


def content_to_string(content: ullm.conv.Content) -> str:
    """Convert a Content object to its string representation for message dicts.

    Raises ``NotImplementedError`` on ``ImageContent``: backends that don't
    implement multimodal must fail loudly at request-construction time
    rather than silently dropping image payloads."""
    match content:
        case ullm.conv.StringContent():
            return content.text
        case ullm.conv.EmptyContent():
            # A legal Content — the declared default of Tool.result and
            # ToolResult.content, so a tool that produces no output reaches
            # request construction as EmptyContent. Rendering it as "" keeps
            # the message well-formed; refusing it kills the game on the NEXT
            # request, when history is re-serialized.
            return ""
        case ullm.conv.ListContent():
            return "\n".join(content_to_string(e) for e in content.elements)
        case ullm.conv.ImageContent():
            raise NotImplementedError(
                "This backend does not support ImageContent. Use a multimodal-capable "
                "backend (e.g. Anthropic) or remove image content from the conversation."
            )
        case _:
            raise ValueError(f"Unsupported content type: {type(content)}")


def _encode_image_for_oai(img: ullm.conv.ImageContent) -> dict[str, Any]:
    """Encode an ImageContent (numpy H×W×3 uint8) as an OpenAI ``image_url``
    content part. Always emits a base64 data URL with PNG payload."""
    from PIL import Image

    pil = Image.fromarray(img.data, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _user_content_to_oai(content: ullm.conv.Content) -> str | list[dict[str, Any]]:
    """Convert a UserPrompt's Content into the OpenAI-style ``content`` field for
    a user message. Returns a plain string when there are no images (preserves
    the existing single-string wire shape); otherwise returns a list of
    text/image_url content parts. Walking is order-preserving."""
    parts = list[dict[str, Any]]()
    pending_text = list[str]()
    has_image = False

    def flush_text() -> None:
        if pending_text:
            parts.append({"type": "text", "text": "\n".join(pending_text)})
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
            parts.append(_encode_image_for_oai(c))
            has_image = True
        elif isinstance(c, ullm.conv.EmptyContent):
            pass
        elif isinstance(c, ullm.conv.ToolCallContent):
            # ToolCallContent should never appear in a UserPrompt; if it does, skip.
            pass
        else:
            raise ValueError(f"Unsupported content type for OpenAI user message: {type(c)}")

    walk(content)
    if not has_image:
        return "\n".join(pending_text)
    flush_text()
    return parts


def _extract_text(content: ullm.conv.Content) -> str:
    """Extract text from Content, excluding ToolCallContent and ImageContent elements."""
    if isinstance(content, ullm.conv.ToolCallContent):
        return ""
    elif isinstance(content, ullm.conv.ImageContent):
        return ""
    elif isinstance(content, ullm.conv.StringContent):
        return content.text
    elif isinstance(content, ullm.conv.ListContent):
        parts = [_extract_text(e) for e in content.elements]
        return "\n".join(p for p in parts if p)
    return ""


def to_message_dicts(
    element: ullm.conv.ConversationElement,
    reasoning_effort_prompts: Optional[dict[str, str]] = None,
    *,
    assistant_text_extractor: Callable[[Any], str],
    assistant_final_text_extractor: Callable[[Any], str],
    include_reasoning: bool = True,
    raw_tool_calls: bool = False,
    reasoning_field: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Convert a conversation element to the standard message dict format
    expected by chat templates: [{"role": "...", "content": "..."}].
    Recursive: containers delegate to their children.

    reasoning_effort_prompts: if provided, the appropriate fallback prompt is appended to the
        system message content. If None, reasoning_effort is handled natively by the backend.
    assistant_text_extractor: converts ``LLMResponse._model_contents`` to the raw assistant text
        (with reasoning still in it).
    assistant_final_text_extractor: same shape but returns assistant text with reasoning
        STRIPPED, in the model's native output format (any raw tool-call markup the model
        originally produced — e.g. Qwen3.5 ``<tool_call>...</tool_call>`` XML — must still be
        present). Required because the only safe alternative is ``final_content.to_string()``,
        which re-renders ``ToolCallContent`` via its Python repr
        (``[tool call <id>: <name>({args})]``); that string ends up in the assistant message and
        the model parrots it on its next turn under ``raw_tool_calls=True``.
    include_reasoning: if False, reasoning content is stripped from LLMResponse assistant text
        via ``assistant_final_text_extractor``.
    raw_tool_calls: if True, tool calls are kept as raw text in the assistant message instead
        of being converted to OpenAI-style structured tool_calls. Use for models whose chat
        templates embed tool calls as text (e.g. Qwen3.5 XML format). ToolResult elements are
        also rendered as user messages with text content instead of role=tool.
    reasoning_field: if set, past reasoning rides as this extra field on assistant message
        dicts (``LLMResponse.reasoning_content``, requiring ``include_reasoning``) and the
        assistant ``content`` stays reasoning-free — for OpenAI-compatible servers whose chat
        template renders a structured reasoning field (vLLM accepts ``"reasoning"``). The
        text extractors are then never expected to embed reasoning in the content."""
    match element:
        case ullm.conv.SystemPrompt():
            content = content_to_string(element.content)
            if (
                reasoning_effort_prompts is not None
                and element.reasoning_effort is not None
                and element.reasoning_effort in reasoning_effort_prompts
            ):
                content += "\n" + reasoning_effort_prompts[element.reasoning_effort]
            return [{"role": "system", "content": content}]
        case ullm.conv.UserPrompt():
            return [{"role": "user", "content": _user_content_to_oai(element.content)}]
        case ullm.conv.LLMResponse():
            if element.do_not_send_to_agent:
                return []
            assert element._model_contents is not None, (
                "LLMResponse must have _model_contents for history reconstruction"
            )
            reasoning_kv: Optional[tuple[str, str]] = None
            if (
                reasoning_field is not None
                and include_reasoning
                and not isinstance(element.reasoning_content, ullm.conv.EmptyContent)
            ):
                reasoning_text = content_to_string(element.reasoning_content).strip()
                if reasoning_text:
                    reasoning_kv = (reasoning_field, reasoning_text)
            # Check if response contains tool calls — reconstruct from user format (lossless)
            tool_calls = element.final_content.tool_calls()
            if tool_calls and not raw_tool_calls:
                if include_reasoning and reasoning_field is None:
                    # Include reasoning (e.g. <think> tags) before tool calls
                    text = assistant_text_extractor(element._model_contents)
                else:
                    text = _extract_text(element.final_content)
                msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                if reasoning_kv is not None:
                    msg[reasoning_kv[0]] = reasoning_kv[1]
                msg["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ]
                return [msg]
            if include_reasoning and reasoning_field is None:
                text = assistant_text_extractor(element._model_contents)
            else:
                text = assistant_final_text_extractor(element._model_contents)
            msg = {"role": "assistant", "content": text}
            if reasoning_kv is not None:
                msg[reasoning_kv[0]] = reasoning_kv[1]
            return [msg]
        case ullm.conv.ToolResult():
            if raw_tool_calls:
                # Render as user message with text content for templates that don't support role=tool
                error_tag = " [ERROR]" if element.is_error else ""
                return [
                    {
                        "role": "user",
                        "content": f"[Tool result for {element.tool_name}{error_tag}]\n{content_to_string(element.content)}",
                    }
                ]
            return [{"role": "tool", "tool_call_id": element.call_id, "content": content_to_string(element.content)}]
        case ullm.conv.ResponseTurn():
            result = list[dict[str, Any]]()
            for e in element.elements:
                result += to_message_dicts(
                    e,
                    reasoning_effort_prompts,
                    assistant_text_extractor=assistant_text_extractor,
                    assistant_final_text_extractor=assistant_final_text_extractor,
                    include_reasoning=include_reasoning,
                    raw_tool_calls=raw_tool_calls,
                    reasoning_field=reasoning_field,
                )
            return result
        case ullm.conv.Conversation():
            result = to_message_dicts(
                element.system_prompt,
                reasoning_effort_prompts,
                assistant_text_extractor=assistant_text_extractor,
                assistant_final_text_extractor=assistant_final_text_extractor,
                include_reasoning=include_reasoning,
                raw_tool_calls=raw_tool_calls,
                reasoning_field=reasoning_field,
            )
            for e in element.elements:
                result += to_message_dicts(
                    e,
                    reasoning_effort_prompts,
                    assistant_text_extractor=assistant_text_extractor,
                    assistant_final_text_extractor=assistant_final_text_extractor,
                    include_reasoning=include_reasoning,
                    raw_tool_calls=raw_tool_calls,
                    reasoning_field=reasoning_field,
                )
            return result
        case _:
            raise ValueError(f"Unsupported conversation element type: {type(element)}")
