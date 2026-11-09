"""conversation_html: markdown for model prose, real code blocks for
tool-call code args (not escaped JSON), raw for tool output."""

from __future__ import annotations

from typing import Any

import ullm.html_render


def _render(structure: dict[str, Any]) -> str:
    return ullm.html_render.conversation_html(
        structure,
        text_of=lambda field: str(field),
        image_src=lambda key: f"png/{key}.png",
    )


def _str(text: str) -> dict[str, Any]:
    return {"c": "str", "text": text}


def test_tool_code_arg_renders_as_block_not_escaped_json() -> None:
    code = "def f():\n    return 1 + 2\n"
    structure = {
        "system": {"content": _str("sys"), "tools": []},
        "elements": [
            {
                "kind": "turn",
                "elements": [
                    {
                        "kind": "response",
                        "reasoning": {"c": "empty"},
                        "final": {"c": "tool_call", "tool_name": "python", "arguments": {"code": code}},
                    }
                ],
            }
        ],
    }
    html = _render(structure)
    assert 'class="ullm-code"' in html
    # Real newlines preserved (not the JSON-escaped "\\n" one-liner).
    assert "def f():\n    return 1 + 2" in html
    assert "\\n" not in html


def test_scalar_tool_args_stay_compact() -> None:
    structure = {
        "system": {"content": _str("sys"), "tools": []},
        "elements": [
            {
                "kind": "turn",
                "elements": [
                    {
                        "kind": "response",
                        "reasoning": {"c": "empty"},
                        "final": {"c": "tool_call", "tool_name": "reset", "arguments": {"hard": True, "n": 3}},
                    }
                ],
            }
        ],
    }
    html = _render(structure)
    assert "ullm-code" not in html  # short scalars aren't code blocks
    assert "hard" in html and "true" in html.lower()


def test_assistant_markdown_fence_renders() -> None:
    structure = {
        "system": {"content": _str("sys"), "tools": []},
        "elements": [
            {"kind": "user", "content": _str("go **now**")},
            {
                "kind": "turn",
                "elements": [
                    {
                        "kind": "response",
                        "reasoning": {"c": "empty"},
                        "final": _str("Here is code:\n\n```python\nx = 1\n```\n"),
                    }
                ],
            },
        ],
    }
    html = _render(structure)
    # Assistant prose is markdown-rendered: the fence becomes a code block.
    assert 'class="ullm-md"' in html and "<pre>" in html and "x = 1" in html
    # The user prompt stays raw (agent-structured text isn't markdown-mangled).
    assert '<pre class="ullm-text">go **now**</pre>' in html


def test_tool_result_json_envelope_unwrapped() -> None:
    """The duck's {stdout, returncode, ...} envelope shows stdout with
    real newlines, not an escaped-JSON one-liner."""
    stdout = "Level: 1 Step: 0\nShape: (64, 64)\nNodes:\n id=0 color=B\n"
    envelope = '{"tool": "python", "returncode": 0, "stdout": "Level: 1 Step: 0\\nShape: (64, 64)\\nNodes:\\n id=0 color=B\\n"}'
    structure = {
        "system": {"content": _str("sys"), "tools": []},
        "elements": [
            {"kind": "user", "content": _str("go")},
            {
                "kind": "turn",
                "elements": [
                    {"kind": "response", "reasoning": {"c": "empty"}, "final": _str("done")},
                    {"kind": "tool_result", "tool_name": "python", "is_error": False, "content": _str(envelope)},
                ],
            },
        ],
    }
    html = _render(structure)
    assert stdout.rstrip() in html  # real newlines
    assert "\\n" not in html  # not the escaped-JSON blob
    assert "returncode" in html  # scalar meta preserved


def test_system_prompt_is_markdown() -> None:
    structure = {
        "system": {"content": _str("Use `current_frame` to observe.\n- step one\n- step two"), "tools": []},
        "elements": [],
    }
    html = _render(structure)
    assert "ullm-md" in html and "<code>current_frame</code>" in html and "<li>" in html


def test_markdown_escapes_injected_html() -> None:
    structure = {
        "system": {"content": _str("sys"), "tools": []},
        "elements": [
            {
                "kind": "turn",
                "elements": [
                    {"kind": "response", "reasoning": {"c": "empty"}, "final": _str("<script>evil()</script>")},
                ],
            }
        ],
    }
    html = _render(structure)
    assert "<script>" not in html and "evil()" in html
