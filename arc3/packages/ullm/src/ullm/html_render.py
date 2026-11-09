"""HTML rendering of serialized conversations.

Renders the structure produced by ``ullm.capture.serialize_conversation``
(and, being a superset, ``canonicalize_request``) — so any host that
stored a conversation in the canonical format (diagnostics, capture
files) renders it with this one piece of code. Presentation only; no
knowledge of where blocks live: the caller supplies ``text_of`` (resolve
a text field that may be a ``{"$block": key}`` reference), ``image_src``
(an ``<img src>`` value for an image block key — a viewer route, a
relative file, or a data URI) and optionally ``image_label`` (a
caption for an image block, e.g. its pixel dimensions and stored size).

Collapse policy: the model's ACTIONS stay visible, bulky payloads
collapse. Tool calls render inline (the decision skeleton, usually
short); tool results collapse behind a summary carrying size + first
line (errors start open); reasoning collapses with its token count;
the system prompt (tool definitions included) and everything before the
turn's last user prompt ("earlier conversation") collapse.

Output is self-contained semantic HTML (`<details>`-based collapsing, no
scripts) styled via the ``ullm-conv`` class family; the host page brings
the CSS.
"""

from __future__ import annotations

import functools
import html
import json
from typing import Any, Callable, Optional

import mistune

_PREVIEW_CHARS = 90
# Tool-call argument names whose value is code/text, rendered as a real
# multi-line block instead of an escaped-JSON one-liner. Any other arg
# whose value is a long/multi-line string is auto-detected the same way.
_CODE_ARG_NAMES = frozenset({"code", "content", "new_str", "old_str", "new_string", "old_string", "text", "source"})

# escape=True: model/tool text can't inject HTML. Rendered markdown lets
# the model's ``` fences and formatting show; tool OUTPUT stays raw <pre>.
if hasattr(mistune, "create_markdown"):
    _render_markdown = mistune.create_markdown(escape=True, plugins=["strikethrough", "table"])
else:
    # mistune < 2.0 (Kaggle's base image, where the duck rides as a source
    # snapshot against the image's site-packages and its old nbconvert needs
    # this same ancient mistune). The 0.8.x markdown() renders tables and
    # strikethrough by default; same escape semantics, no plugins arg.
    _render_markdown = functools.partial(mistune.markdown, escape=True)


def _esc(value: Any) -> str:
    """HTML-escape a value, attribute-safe."""
    return html.escape(str(value), quote=True)


@functools.lru_cache(maxsize=512)
def _markdown(text: str) -> str:
    """Render markdown text into a wrapped block, or nothing if it is empty."""
    return f'<div class="ullm-md">{_render_markdown(text)}</div>' if text else ""


def _is_code_arg(name: str, value: Any) -> bool:
    """Whether an argument should render as code rather than prose."""
    return isinstance(value, str) and (name in _CODE_ARG_NAMES or "\n" in value or len(value) > 80)


def _content_html(
    content: Any,
    text_of: Callable[[Any], str],
    image_src: Callable[[str], str],
    image_label: Optional[Callable[[str], str]],
    *,
    markdown: bool = False,
) -> str:
    """Render one content value, resolving text and images through the
    caller's accessors so the same walk serves diagnostics and capture.
    """
    kind = content["c"]
    if kind == "str":
        text = text_of(content["text"])
        return _markdown(text) if markdown else f'<pre class="ullm-text">{_esc(text)}</pre>'
    if kind == "list":
        return "".join(
            _content_html(element, text_of, image_src, image_label, markdown=markdown)
            for element in content["elements"]
        )
    if kind == "image":
        # The alt text is the image's textual stand-in — for the whale, an ASCII
        # rendering of the whole grid — so it belongs *on* the image, in `alt` and
        # `title`, where a reader gets it by hovering. Not in the caption: there it
        # is the image's own content a second time, a wall of digits between one
        # frame and the next.
        alt = _esc(content.get("alt", ""))
        img = f'<img class="ullm-image" src="{_esc(image_src(content["block"]))}" alt="{alt}" title="{alt}">'
        # The caption keeps what is *about* the image rather than what is in
        # it: pixel dimensions and stored size, which nothing else shows.
        caption = image_label(content["block"]) if image_label is not None else ""
        if not caption:
            return img
        return f'<figure class="ullm-image-box">{img}<figcaption>{_esc(caption)}</figcaption></figure>'
    if kind == "tool_call":
        return _tool_call_html(content)
    if kind == "empty":
        return ""
    return f'<pre class="ullm-text">[unknown content kind {_esc(kind)}]</pre>'


def _tool_call_html(content: Any) -> str:
    """A tool call: code/text arguments as real multi-line blocks (so the
    python tool's ``code`` and file tools' ``content`` read as code, not
    escaped JSON), other arguments as a compact key-value line."""
    args = content.get("arguments") or {}
    parts = [f'<div class="ullm-tool-call"><span class="ullm-label">tool call: {_esc(content["tool_name"])}</span>']
    if isinstance(args, dict):
        scalars: list[str] = []
        for name, value in args.items():
            if _is_code_arg(name, value):
                parts.append(f'<div class="ullm-arg-name">{_esc(name)}</div><pre class="ullm-code">{_esc(value)}</pre>')
            else:
                scalars.append(f"<b>{_esc(name)}</b>={_esc(json.dumps(value, ensure_ascii=False))}")
        if scalars:
            parts.append(f'<div class="ullm-args">{" · ".join(scalars)}</div>')
    else:
        parts.append(f'<pre class="ullm-text">{_esc(json.dumps(args, ensure_ascii=False))}</pre>')
    parts.append("</div>")
    return "".join(parts)


def _content_preview(content: Any, text_of: Callable[[Any], str]) -> str:
    """Flatten content to plain text for summary previews."""
    kind = content["c"]
    if kind == "str":
        return text_of(content["text"])
    if kind == "list":
        return " ".join(_content_preview(element, text_of) for element in content["elements"])
    if kind == "image":
        return "[image]"
    if kind == "tool_call":
        return f"[tool call {content['tool_name']}]"
    return ""


def _try_json_dict(text: str) -> Optional[dict[str, Any]]:
    """Parse ``text`` as a JSON object, or ``None`` if it is not one."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        value = json.loads(stripped)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _tool_result_html(
    content: Any,
    text_of: Callable[[Any], str],
    image_src: Callable[[str], str],
    image_label: Optional[Callable[[str], str]],
) -> tuple[str, str]:
    """(body html, plain preview). Tool output is often a JSON envelope
    (``{stdout, result, error, ...}`` — the duck's python tool): unwrap it
    so ``stdout`` shows with real newlines instead of an escaped-JSON
    one-liner. Non-JSON output renders raw."""
    if content.get("c") != "str":
        return _content_html(content, text_of, image_src, image_label), _content_preview(content, text_of)
    text = text_of(content["text"])
    parsed = _try_json_dict(text)
    if parsed is None or not any(k in parsed for k in ("stdout", "output", "result", "error")):
        return f'<pre class="ullm-text">{_esc(text)}</pre>', text
    parts: list[str] = []
    preview: list[str] = []
    meta = {
        key: value
        for key, value in parsed.items()
        if key not in ("stdout", "output", "result", "error")
        and isinstance(value, (str, int, float, bool))
        and not (isinstance(value, str) and "\n" in value)
    }
    if meta:
        parts.append(
            '<div class="ullm-args">' + " · ".join(f"<b>{_esc(k)}</b>={_esc(v)}" for k, v in meta.items()) + "</div>"
        )
    for key in ("stdout", "output"):
        value = parsed.get(key)
        if isinstance(value, str) and value != "":
            parts.append(f'<pre class="ullm-text">{_esc(value)}</pre>')
            preview.append(value)
    result = parsed.get("result")
    if result not in (None, "", [], {}):
        rendered = json.dumps(result, indent=1, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
        parts.append(f'<div class="ullm-arg-name">result</div><pre class="ullm-text">{_esc(rendered)}</pre>')
        preview.append(str(result))
    error = parsed.get("error")
    if isinstance(error, str) and error.strip():
        parts.append(f'<div class="ullm-arg-name">error</div><pre class="ullm-code">{_esc(error)}</pre>')
        preview.append(error)
    body = "".join(parts) or f'<pre class="ullm-text">{_esc(text)}</pre>'
    return body, " ".join(preview) or text


def _usage_line(response: dict[str, Any]) -> str:
    """The token-usage line for a response, omitting what it does not carry."""
    parts: list[str] = []
    usage = response.get("usage")
    if usage:
        parts.append(f"ctx {usage['prompt_tokens'] + usage['completion_tokens']:,}")
        parts.append(f"{usage['completion_tokens']} generated")
        if usage.get("reasoning_tokens"):
            parts.append(f"{usage['reasoning_tokens']} reasoning")
        parts.append(f"{usage['prompt_tokens'] - usage['cached_tokens']} uncached input")
        if usage.get("cached_tokens"):
            parts.append(f"{usage['cached_tokens']} cached input")
    if response.get("stop_reason"):
        parts.append(f"stop: {response['stop_reason']}")
    if response.get("malformed_retries"):
        parts.append(f"{response['malformed_retries']} malformed retries")
    if not parts:
        return ""
    return f'<div class="ullm-usage">{_esc(" · ".join(parts))}</div>'


def _element_html(
    element: dict[str, Any],
    text_of: Callable[[Any], str],
    image_src: Callable[[str], str],
    image_label: Optional[Callable[[str], str]],
    *,
    open_final: bool,
) -> str:
    """Render one conversation element."""
    kind = element["kind"]
    if kind == "user":
        return (
            '<div class="ullm-element ullm-user"><span class="ullm-label">user</span>'
            + _content_html(element["content"], text_of, image_src, image_label)
            + "</div>"
        )
    if kind == "tool_result":
        body, plain = _tool_result_html(element["content"], text_of, image_src, image_label)
        preview = " ".join(plain.split())
        summary = f"tool result: {_esc(element['tool_name'])}"
        if element["is_error"]:
            summary += " (error)"
        summary += f" — {len(preview):,} chars"
        if preview:
            summary += f" — <span class='ullm-preview'>{_esc(preview[:_PREVIEW_CHARS])}</span>"
        flag = " ullm-error" if element["is_error"] else ""
        return (
            f'<details class="ullm-element ullm-tool-result{flag}"{" open" if element["is_error"] else ""}>'
            f"<summary>{summary}</summary>{body}</details>"
        )
    if kind == "response":
        # Model prose renders as markdown (its ``` fences and formatting
        # show); tool OUTPUT (results, below) stays raw.
        reasoning = _content_html(element["reasoning"], text_of, image_src, image_label, markdown=True)
        parts = ['<div class="ullm-element ullm-response"><span class="ullm-label">assistant</span>']
        injected = element.get("response_injection")
        if injected:
            # The answer's text already starts with it; this names which
            # part of that start was served rather than sampled.
            parts.append(
                f'<div class="ullm-injected"><span class="ullm-label">injected</span> <code>{_esc(injected)}</code></div>'
            )
        if reasoning:
            usage = element.get("usage")
            tokens = f" ({usage['reasoning_tokens']} tokens)" if usage and usage.get("reasoning_tokens") else ""
            parts.append(
                f'<details class="ullm-reasoning"><summary>reasoning{_esc(tokens)}</summary>{reasoning}</details>'
            )
        final = _content_html(element["final"], text_of, image_src, image_label, markdown=True)
        if element.get("do_not_send_to_agent"):
            parts.append(f'<div class="ullm-synthetic">synthetic placeholder: {final}</div>')
        elif open_final:
            parts.append(final)
        else:
            parts.append(f'<details class="ullm-final"><summary>response</summary>{final}</details>')
        parts.append(_usage_line(element))
        parts.append("</div>")
        return "".join(parts)
    if kind == "turn":
        inner = list(element["elements"])
        rendered: list[str] = []
        last_response_index = max((i for i, sub in enumerate(inner) if sub["kind"] == "response"), default=-1)
        for i, sub in enumerate(inner):
            rendered.append(_element_html(sub, text_of, image_src, image_label, open_final=(i == last_response_index)))
        return '<div class="ullm-turn">' + "".join(rendered) + "</div>"
    return f'<pre class="ullm-text">[unknown element kind {_esc(kind)}]</pre>'


def _system_html(
    system: dict[str, Any],
    text_of: Callable[[Any], str],
    image_src: Callable[[str], str],
    image_label: Optional[Callable[[str], str]],
) -> str:
    """Render the system prompt, tools included."""
    tools = system.get("tools", [])
    tool_names = ", ".join(_esc(tool["name"]) for tool in tools)
    parts = [
        '<details class="ullm-element ullm-system"><summary>system prompt'
        + (f" (tools: {tool_names})" if tool_names else "")
        + "</summary>",
        # System prompts are prose (backtick-code, dash-lists) — markdown
        # like the model's own text. User prompts stay raw (they carry
        # ASCII boards that markdown would mangle).
        _content_html(system["content"], text_of, image_src, image_label, markdown=True),
    ]
    for tool in tools:
        parameters = json.dumps(tool.get("parameters", {}), indent=1, ensure_ascii=False)
        parts.append(
            f'<details class="ullm-tool-def"><summary>tool: {_esc(tool["name"])}</summary>'
            f'<pre class="ullm-text">{_esc(tool.get("description", ""))}</pre>'
            f"<details><summary>parameters</summary><pre class='ullm-text'>{_esc(parameters)}</pre></details>"
            "</details>"
        )
    parts.append("</details>")
    return "".join(parts)


def conversation_html(
    structure: dict[str, Any],
    *,
    text_of: Callable[[Any], str],
    image_src: Callable[[str], str],
    image_label: Optional[Callable[[str], str]] = None,
) -> str:
    """Render one serialized conversation. Everything before the LAST
    user prompt is the pre-history of the current turn and collapses into
    one block; within the visible turn, the last response renders open
    (the model's closing answer) and intermediate ones collapse."""
    elements = list(structure["elements"])
    last_user = max((i for i, element in enumerate(elements) if element["kind"] == "user"), default=0)
    parts = ['<div class="ullm-conv">', _system_html(structure["system"], text_of, image_src, image_label)]
    if last_user > 0:
        earlier = elements[:last_user]
        n_prompts = sum(1 for element in earlier if element["kind"] == "user")
        n_turns = sum(1 for element in earlier if element["kind"] == "turn")
        rendered = "".join(
            _element_html(element, text_of, image_src, image_label, open_final=False) for element in earlier
        )
        parts.append(
            f'<details class="ullm-prehistory"><summary>earlier conversation '
            f"({n_prompts} prompts, {n_turns} response turns)</summary>{rendered}</details>"
        )
    for element in elements[last_user:]:
        parts.append(_element_html(element, text_of, image_src, image_label, open_final=True))
    parts.append("</div>")
    return "".join(parts)
