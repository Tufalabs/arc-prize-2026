# type: ignore
"""Tests for ``tools/trace_page.py``: the page claims to be verbatim and to need
nothing beside it, so what it does to the trace's text is pinned here."""

import html
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from platypus.viewer_data import _tool_call_text

_PATH = Path(__file__).resolve().parents[1] / "tools" / "trace_page.py"
_SPEC = importlib.util.spec_from_file_location("trace_page", _PATH)
trace_page = importlib.util.module_from_spec(_SPEC)
sys.modules["trace_page"] = trace_page
_SPEC.loader.exec_module(trace_page)

_NOTES = """=== title
A title

=== intro
Intro text.

{{levels}}

{{legend}}

=== epilogue
Epilogue text.

{{result}}

=== provenance
Provenance text.
"""


def _text_of(rendered: str) -> str:
    """Rendered HTML with its tags gone."""
    return re.sub(r"<[^>]+>", "", rendered)


def _call(code: str) -> dict:
    """A python call as the replay adapter renders it."""
    return {
        "label": "TOOL CALL: python",
        "kind": "tool",
        "content": _tool_call_text({"tool_name": "python", "arguments": {"code": code}}),
    }


def _game(sections: list[dict]) -> dict:
    """A one-level game and its trace, under one turn as the replay's timeline holds them."""
    return {
        "events": [
            {"type": "initial", "level": 1, "board": [[5, 0]]},
            {"type": "analysis", "level": 1, "board": [[5, 0]], "transcript_sections": sections},
            {"type": "action", "level": 1, "board": [[0, 5]]},
        ]
    }


def test_tool_call_reads_back_what_the_adapter_wrote():
    text = _tool_call_text({"tool_name": "create_file", "arguments": {"content": "x = 1\n", "path": "a.py"}})
    assert trace_page.parse_tool_call(text) == ("create_file", {"content": "x = 1\n", "path": "a.py"})


def test_a_body_that_is_not_a_call_is_refused():
    with pytest.raises(ValueError, match="not a tool call"):
        trace_page.parse_tool_call("print(1)")


class _Template:
    """A tokenizer's chat template, recording what it was asked to render."""

    def apply_chat_template(self, messages, *, tools, tokenize, reasoning_effort):
        assert tokenize is False
        self.tools = tools
        return f"<|im_start|>system\nEffort {reasoning_effort}.\n# Tools\n{tools[0]['function']['name']}\n\n{messages[0]['content']}<|im_end|>"


def test_the_preamble_is_what_the_template_puts_ahead_of_the_system_text():
    tools = [{"name": "python", "description": "Run it."}]
    template = _Template()
    assert trace_page.system_preamble(template, tools, "xhigh") == "Effort xhigh.\n# Tools\npython"
    assert template.tools == [{"type": "function", "function": tools[0]}]


_PARAMETERS = {"type": "object", "properties": {"code": {"type": "string"}}}


class _Agent:
    def _build_turn_tools(self, ctx):
        assert ctx is None
        return [SimpleNamespace(name="python", description="Run it.", parameters=_PARAMETERS)]


def test_tool_definitions_keep_the_agents_key_order_where_the_recording_sorts_them():
    recorded = [
        {
            "description": "Run it.",
            "name": "python",
            "parameters": {"properties": {"code": {"type": "string"}}, "type": "object"},
        }
    ]
    tools = trace_page.tool_definitions(_Agent(), recorded)
    assert list(tools[0]) == ["name", "description", "parameters"]
    assert list(tools[0]["parameters"]) == ["type", "properties"]


def test_tool_definitions_refuse_an_agent_that_is_not_the_one_recorded():
    recorded = [{"name": "python", "description": "Something else.", "parameters": _PARAMETERS}]
    with pytest.raises(ValueError, match="not the ones"):
        trace_page.tool_definitions(_Agent(), recorded)


def test_spread_json_chooses_only_whitespace():
    value = {
        "type": "function",
        "function": {
            "name": "python",
            "description": "long " * 40,
            "parameters": {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}},
        },
    }
    spread = trace_page.spread_json(value)
    assert json.dumps(json.loads(spread)) == json.dumps(value)  # the keys' order as well
    assert '    "name": "python",\n' in spread
    assert '"required": ["code"]' in spread


def test_preamble_html_spreads_only_the_tools_section():
    definition = {"name": "python", "description": "<long> " * 40}
    line = json.dumps(definition)
    preamble = f"Effort  <high>.\n\n<tools>\n{line}\n{line}\n</tools>\n\nThe format:  <tool_call>"
    page = trace_page.preamble_html(preamble)
    assert "Effort  &lt;high&gt;.\n\n&lt;tools&gt;\n" in page
    assert "&lt;/tools&gt;\n\nThe format:  &lt;tool_call&gt;" in page
    assert page.count('class="tool-definition"') == 2
    spread = html.unescape(re.sub(r"<[^>]+>", "", page)).partition("<tools>\n")[2].partition("</tools>")[0]
    assert spread == trace_page.spread_json(definition).replace("\n", "") * 2


def test_preamble_html_leaves_a_preamble_without_tools_as_written():
    assert trace_page.preamble_html("Effort  <high>.") == '<pre class="output">Effort  &lt;high&gt;.</pre>'


@pytest.mark.parametrize(
    ("label", "direction"),
    [
        ("SYSTEM PROMPT", "input"),
        ("USER PROMPT", "input"),
        ("TOOL RESULT: python", "input"),
        ("THINKING", "output"),
        ("TOOL CALL: python", "output"),
        ("ASSISTANT", "output"),
    ],
)
def test_direction_of_a_block(label, direction):
    assert trace_page.direction_of(label) == direction


def test_python_in_a_markdown_fence_is_highlighted():
    assert 'class="code"' in trace_page.markdown("```python\nimport numpy\n```\n")


def test_a_harness_message_keeps_its_lines_and_names_its_pictures():
    key = "a" * 64
    rendered = trace_page.prompt_html(
        f"Line one.\nLine two.\n\ntrain_in[0]  (1x2)\n[image block {key}]\ntrain_out[0]  (1x2)\n[image block {key}]\n"
        "Train pair 0:\n0\t44\nThe end.",
        lambda block: f"png/{block}",
    )
    assert "Line one.<br" in rendered
    assert rendered.count('<div class="gallery">') == 1
    assert rendered.count(f'src="png/{key}"') == 2
    assert '<span class="word">train_in[0]</span>' in rendered
    assert '<pre class="output">0\t44</pre>' in rendered
    assert (
        rendered.index("gallery") < rendered.index("Train pair 0") < rendered.index("0\t44") < rendered.index("The end")
    )


_PALETTE = [f"rgb({value}, 0, 0)" for value in range(16)]


def test_grid_svg_draws_each_colour_once_in_runs():
    rendered = trace_page.grid_svg([[1, 1, 4], [4, 4, 4]], _PALETTE, "a grid")
    assert rendered.count("<path fill=") == 2
    assert 'fill="rgb(4, 0, 0)"' in rendered
    assert "M0 0h2v1h-2z" in rendered
    assert 'aria-label="a grid"' in rendered


def test_a_level_opens_on_the_frame_of_its_first_turn():
    game = {
        "events": [
            {"type": "initial", "level": 1, "board": [[1]]},
            {"type": "analysis", "level": 1, "board": [[2]]},
            {"type": "action", "level": 1, "board": [[3]]},
            {"type": "analysis", "level": 2, "board": [[4]]},
            {"type": "analysis", "level": 2, "board": [[5]]},
            {"type": "action", "level": 2, "board": [[6]]},
        ]
    }
    assert trace_page.first_frames(game) == {1: [[1]], 2: [[4]]}
    assert trace_page.levels_figure(game, _PALETTE).count("<figure>") == 2


def test_a_run_is_named_once():
    runs = [SimpleNamespace(run_id="abc1"), SimpleNamespace(run_id="abd2")]
    assert trace_page.find_run(runs, "abd") == 1
    with pytest.raises(SystemExit, match="names 2 runs"):
        trace_page.find_run(runs, "ab")
    with pytest.raises(SystemExit, match="names 0 runs"):
        trace_page.find_run(runs, "x")


def test_notes_need_their_sections():
    notes = trace_page.parse_notes(_NOTES)
    assert notes.title == "A title"
    assert notes.provenance == "Provenance text."
    with pytest.raises(ValueError, match="epilogue"):
        trace_page.parse_notes("=== title\nA\n=== intro\nB\n=== provenance\nC\n")


def _run(**overrides) -> SimpleNamespace:
    """A run of a game of three levels that won the first two, the first over its baseline."""
    run = {
        "game_id": "abc",
        "base_actions_per_level": [2, 16, 60],
        "actions_per_level": [4, 16, 9],
        "levels_completed": 2,
        "number_of_levels": 3,
        "final_score": (25.0 * 1 + 100.0 * 2) / 6,
    }
    return SimpleNamespace(**{**run, **overrides})


def test_the_result_is_scored_from_the_record():
    rendered = trace_page.result_table(_run())
    assert "<td>1</td><td>1</td><td>2</td><td>4</td><td>25.0</td>" in rendered
    assert "<td>2</td><td>2</td><td>16</td><td>16</td><td>100.0</td>" in rendered
    # A level not won scores nothing, whatever it spent.
    assert "<td>3</td><td>3</td><td>60</td><td>9</td><td>0.0</td>" in rendered
    assert "<td>37.5</td>" in rendered


def test_a_result_the_record_contradicts_is_refused():
    with pytest.raises(ValueError, match="recorded as"):
        trace_page.result_table(_run(final_score=95.0))


def test_the_page_is_a_document_of_the_trace_alone():
    sections = [
        {"label": "SYSTEM PROMPT", "kind": "meta", "content": "You are an agent."},
        {"label": "USER PROMPT", "kind": "meta", "content": f"Puzzle.\n[image block {'b' * 64}]"},
        {"label": "THINKING", "kind": "reasoning", "content": "Hmm, `x` < y.", "generated_tokens": 7},
        _call("print(1)"),
        _call("print(2)"),
        {"label": "ANSWER #1", "kind": "boundary", "content": "Answer 1 stands from here."},
        {"label": "TOOL RESULT: python", "kind": "tool", "content": "check: ALL TRAIN PAIRS OK\n0\t4411"},
        {"label": "TOOL RESULT: python", "kind": "tool", "content": "Traceback (most recent call last):"},
        {"label": "ASSISTANT", "kind": "reasoning", "content": "**Done**", "generated_tokens": 3},
    ]
    page = trace_page.render_page(
        trace_page.parse_notes(_NOTES),
        _game(sections),
        _PALETTE,
        "<p>the result</p>",
        lambda _: "data:image/png;base64,AA",
        lambda section: f"PREAMBLE of {section['label']}",
    )
    # The system prompt opens as the chat template rendered it, ahead of the harness's text.
    assert page.index("PREAMBLE of SYSTEM PROMPT") < page.index("You are an agent.")
    # Needs nothing beside it, and nothing on it moves.
    assert not re.search(r'(src|href)="(?!data:)', page)
    assert "https://" not in page and "http://" not in page
    for interactive in ("<script", "<details", "<summary", "<button", "<a "):
        assert interactive not in page
    # Input, then output, then input and output again: the viewer's marker neither appears nor breaks a bar.
    assert re.findall(r'class="stretch (\w+)"', page) == ["input", "output", "input", "output"]
    assert "stands from here" not in page
    # What the harness and the model wrote is left as it was: no counts, no tints, no verdict marks.
    assert "Hmm, `x` &lt; y." in page
    assert '<pre class="output">check: ALL TRAIN PAIRS OK\n0\t4411</pre>' in page
    assert '<pre class="output">Traceback (most recent call last):</pre>' in page
    assert "tokens" not in _text_of(page.split("</style>")[1])
    assert page.index("Output of call 1") < page.index("Output of call 2")
    assert "<strong>Done</strong>" in page
    assert "the result" in page
