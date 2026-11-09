"""One game of a run as a single self-contained HTML page: everything that passed
between the harness and the model, in order, for a reader who was not there.

    uv run python tools/trace_page.py JOB_DIR --game GAME_ID --tokenizer CHECKPOINT --notes NOTES.md --out PAGE.html

The transcript is the replay viewer's (``platypus.viewer_data``) and the pictures
are the exact ones the model was shown (``platypus.viewer.block_png_bytes``),
embedded in the page, so the file needs nothing beside it: no server, no network,
no script. It is a document: nothing collapses and nothing responds.

The system prompt opens as the model read it: its chat template (``--tokenizer``)
renders the request's reasoning effort and tool definitions ahead of the harness's
text. The definitions come from the job's agent, which holds their key order, and must
match the recording, which holds their content but sorts the keys.

The trace is only the trace. A bar in the left margin says which way each block
went, what the model read and what it generated, and Python is highlighted; nothing
else is added to what the harness and the model wrote, so that nothing on the page
can be taken for a harness feature. Markers the viewer places for its own reader
(``kind == "boundary"``) stay out.

Everything that is not the trace comes from the notes file, sections headed by a
line ``=== name``: ``title``, ``intro``, ``epilogue`` and ``provenance``, in markdown.

Three placeholders, each alone on a line, place what the tool draws itself:
``{{task}}`` the task's grids, ``{{legend}}`` the key to the margin bars, and
``{{answers}}`` the grids the game ended on beside the expected ones, scored again
here from the run's own record (``taaf.game.score_submission``).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import pickle
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer  # pyright: ignore[reportUnknownVariableType]

import taaf.benchmark
import taaf.game
from platypus import viewer_data
from platypus.viewer import block_png_bytes
from taaf.grid import ARC_COLORS

Direction = Literal["input", "output"]

#: The arguments of a file tool that hold Python when the file they name is Python.
_FILE_BODY_ARGUMENTS = ("content", "old_string", "new_string", "text")
#: Stands in for the system text while the chat template renders what comes before it.
_SYSTEM_TEXT_MARK = "<<THE SYSTEM TEXT>>"
#: How long a line of a tool definition may run before its object or array is spread over
#: several: what the page's fixed-width block holds unwrapped at the page's full width.
_JSON_WIDTH = 104
#: How wide a grid's cell is drawn, in the task's figure and the answers'.
_CELL_PX = 8

_CALL = re.compile(r"\A<tool_call>\n<function=(?P<name>[^>\n]+)>\n(?P<body>.*)\n</function>\n</tool_call>\Z", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=(?P<key>[^>\n]+)>\n(?P<value>.*?)\n</parameter>", re.DOTALL)
_IMAGE_LINE = re.compile(r"^\[image block (?P<key>[0-9a-f]{64})\]$")
_GRID_ROW = re.compile(r"^\d+\t\d+$")
_NOTES_HEADING = re.compile(r"^=== (?P<name>.+)$", re.MULTILINE)
_PLACEHOLDER = re.compile(r"^\{\{(?P<name>\w+)\}\}$", re.MULTILINE)

#: The code's colours. pygments has no type stubs, so these two are ``Any``.
_FORMATTER: Any = HtmlFormatter(style="github-dark", nowrap=True)  # pyright: ignore[reportUnknownVariableType]
_LEXER: Any = PythonLexer()  # pyright: ignore[reportUnknownVariableType]


def highlight_python(code: str) -> str:
    """``code`` as a highlighted block."""
    return f'<pre class="code"><code>{highlight(code, _LEXER, _FORMATTER).rstrip()}</code></pre>'


def _fence_highlight(content: str, language: str, _attributes: str) -> str:
    """A markdown code fence: highlighted where it is Python, plain otherwise."""
    if language == "python":
        return highlight_python(content)
    return f'<pre class="code"><code>{html.escape(content)}</code></pre>'


_MARKDOWN = MarkdownIt("commonmark", {"html": False, "highlight": _fence_highlight})
#: A harness message's line breaks are its own: a report's lines stay lines.
_MESSAGE_MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True, "highlight": _fence_highlight})


def markdown(text: str) -> str:
    """``text``, which the page's notes and the model's replies write in markdown, as HTML."""
    return _MARKDOWN.render(text)


# --- the trace ---------------------------------------------------------------


def parse_tool_call(text: str) -> tuple[str, dict[str, str]]:
    """A tool call's name and arguments, from the text ``viewer_data`` renders it to.

    A body that does not read back is an error rather than a block drawn wrongly: this
    page claims to be verbatim.
    """
    call = _CALL.match(text)
    if call is None:
        raise ValueError(f"not a tool call: {text[:80]!r}")
    return call["name"], {found["key"]: found["value"] for found in _PARAMETER.finditer(call["body"])}


def tool_definitions(agent: Any, recorded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tools as the request sent them. The recording sorts every JSON object's keys and
    the template prints keys as they come, so the order is read off the agent that played;
    the recording vouches for the content."""
    tools: list[dict[str, Any]] = [
        {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
        for tool in agent._build_turn_tools(None)
    ]
    if json.dumps(tools, sort_keys=True) != json.dumps(recorded, sort_keys=True):
        raise ValueError("the agent's tools are not the ones the game's request recorded")
    return tools


def system_preamble(tokenizer: Any, tools: list[dict[str, Any]], reasoning_effort: str | None) -> str:
    """What the model's chat template puts ahead of the system text."""
    rendered: str = tokenizer.apply_chat_template(
        [{"role": "system", "content": _SYSTEM_TEXT_MARK}, {"role": "user", "content": "-"}],
        tools=[{"type": "function", "function": tool} for tool in tools],
        tokenize=False,
        reasoning_effort=reasoning_effort,
    )
    head, found, _ = rendered.partition(_SYSTEM_TEXT_MARK)
    if not found:
        raise ValueError("the chat template did not render the system text")
    # The role's own header line is not text the page has to show.
    return head.removeprefix("<|im_start|>system\n").rstrip("\n")


def direction_of(label: str) -> Direction:
    """Which way a block went: what the model read, or what it generated."""
    if label in ("SYSTEM PROMPT", "USER PROMPT", "INJECTED PREFIX", "COMPACTION REQUEST") or label.startswith(
        "TOOL RESULT"
    ):
        return "input"
    return "output"


def _verbatim(text: str) -> str:
    """Text exactly as it was written, in a fixed-width block."""
    return f'<pre class="output">{html.escape(text)}</pre>'


def spread_json(value: Any, *, indent: int = 0, lead: int = 0) -> str:
    """JSON with each object or array on one line where that fits, ``lead`` being what
    stands ahead of it on its line. Keys keep their order; only whitespace is chosen."""
    flat = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, dict | list) or indent + lead + len(flat) <= _JSON_WIDTH:
        return flat
    inner = " " * (indent + 2)
    if isinstance(value, dict):
        rows: list[str] = []
        for key, item in cast(dict[str, Any], value).items():
            name = f"{json.dumps(key, ensure_ascii=False)}: "
            rows.append(inner + name + spread_json(item, indent=indent + 2, lead=len(name)))
        opening, closing = "{", "}"
    else:
        rows = [inner + spread_json(item, indent=indent + 2) for item in cast(list[Any], value)]
        opening, closing = "[", "]"
    return f"{opening}\n" + ",\n".join(rows) + f"\n{' ' * indent}{closing}"


def _hanging_line(line: str) -> str:
    """A line of a tool definition, whose wrap stays under its own key."""
    hang = len(line) - len(line.lstrip(" ")) + 2
    return f'<span class="json-line" style="padding-left:{hang}ch;text-indent:-{hang}ch">{html.escape(line)}</span>'


def preamble_html(preamble: str) -> str:
    """What the template renders ahead of the system text, as written, but for the
    ``<tools>`` section: the template puts each definition on one long line."""
    head, tools_open, rest = preamble.partition("<tools>\n")
    if not tools_open:
        return _verbatim(preamble)
    definitions, tools_close, tail = rest.partition("</tools>")
    spread = "".join(
        '<span class="tool-definition">'
        + "".join(_hanging_line(line) for line in spread_json(json.loads(each)).split("\n"))
        + "</span>"
        for each in definitions.strip().split("\n")
    )
    return f'<pre class="output">{html.escape(head + tools_open)}{spread}{html.escape(tools_close + tail)}</pre>'


def prompt_html(text: str, image_src: Callable[[str], str]) -> str:
    """A message the harness sent: its pictures in a row, each under the line that names
    it; printed grid rows verbatim; the rest as the markdown it is written in."""
    blocks: list[str] = []
    prose: list[str] = []
    gallery: list[str] = []
    rows: list[str] = []

    def flush() -> None:
        """Place what has gathered, in the order it came."""
        if gallery:
            blocks.append(f'<div class="gallery">{"".join(gallery)}</div>')
            gallery.clear()
        if rows:
            blocks.append(_verbatim("\n".join(rows)))
            rows.clear()
        if any(line.strip() for line in prose):
            blocks.append(_MESSAGE_MARKDOWN.render("\n".join(prose)))
        prose.clear()

    for line in text.split("\n"):
        image = _IMAGE_LINE.match(line)
        if image is not None:
            caption = prose.pop() if prose and prose[-1].strip() else ""
            if any(entry.strip() for entry in prose):
                flush()
            prose.clear()
            words = "".join(f'<span class="word">{html.escape(word)}</span> ' for word in caption.split())
            gallery.append(
                f'<figure><img src="{image_src(image["key"])}" alt="{html.escape(caption)}">'
                f"<figcaption>{words}</figcaption></figure>"
            )
        elif _GRID_ROW.match(line):
            if gallery or any(entry.strip() for entry in prose):
                flush()
            prose.clear()
            rows.append(line)
        else:
            if rows:
                flush()
            prose.append(line)
    flush()
    return "".join(blocks)


def _block(kind: str, label: str, body: str) -> str:
    """One block of the trace under its heading."""
    return f'<section class="step {kind}"><header><b>{label}</b></header>{body}</section>'


def tool_call_html(section: dict[str, Any], number: int) -> str:
    """A tool call: its Python highlighted, whichever argument holds it."""
    name, arguments = parse_tool_call(str(section["content"]))
    path = arguments.get("path", "")
    is_python = name == "python" or path.endswith(".py")
    body: list[str] = []
    for key, value in arguments.items():
        if key == "path":
            continue
        if is_python and (key == "code" or key in _FILE_BODY_ARGUMENTS):
            body.append(highlight_python(value.rstrip("\n")))
        else:
            body.append(_verbatim(value))
    title = f"Call {number} · <code>{html.escape(name)}</code>"
    if path:
        title += f" · <code>{html.escape(path)}</code>"
    return _block("call", title, "".join(body))


@dataclass
class TraceWriter:
    """The trace's blocks in order, those going the same way under one margin bar."""

    image_src: Callable[[str], str]
    #: What the chat template renders ahead of a system prompt section's text.
    preamble: Callable[[dict[str, Any]], str]
    _out: list[str] = field(default_factory=list[str])
    _stretch: list[str] = field(default_factory=list[str])
    _direction: Direction | None = None
    _calls: int = 0
    _unanswered: list[int] = field(default_factory=list[int])

    def _close_stretch(self) -> None:
        """End the current bar."""
        if self._stretch:
            self._out.append(f'<div class="stretch {self._direction}">{"".join(self._stretch)}</div>')
            self._stretch = []
        self._direction = None

    def _render(self, section: dict[str, Any]) -> str:
        """One block of the trace as HTML."""
        label, text = str(section["label"]), str(section.get("content") or "")
        if label == "SYSTEM PROMPT":
            return _block(
                "prompt",
                "System prompt",
                f'{preamble_html(self.preamble(section))}<div class="prose">{markdown(text)}</div>',
            )
        if label == "THINKING":
            return _block("thinking", "Thinking", f'<div class="thinking-body">{html.escape(text)}</div>')
        if label.startswith("TOOL CALL"):
            self._calls += 1
            self._unanswered.append(self._calls)
            return tool_call_html(section, self._calls)
        if label.startswith("TOOL RESULT"):
            number = self._unanswered.pop(0) if self._unanswered else None
            return _block("result", f"Output of call {number}", _verbatim(text))
        if label == "ASSISTANT":
            return _block("reply", "Reply", f'<div class="prose">{markdown(text)}</div>')
        return _block("prompt", "Harness message", f'<div class="prose">{prompt_html(text, self.image_src)}</div>')

    def add(self, section: dict[str, Any]) -> None:
        """The trace's next block, unless it is a marker the viewer placed for its own reader."""
        if section.get("kind") == "boundary":
            return
        direction = direction_of(str(section["label"]))
        rendered = self._render(section)
        if direction != self._direction:
            self._close_stretch()
            self._direction = direction
        self._stretch.append(rendered)

    def finish(self) -> str:
        """The trace as HTML."""
        self._close_stretch()
        return "".join(self._out)


# --- the grids the tool draws -------------------------------------------------


def grid_svg(rows: Sequence[Sequence[int]], label: str = "") -> str:
    """A grid as an inline SVG: a path a colour, cell by cell runs along each row, and the
    lines between cells over them."""
    height, width = len(rows), len(rows[0])
    paths: dict[int, list[str]] = {}
    for y, row in enumerate(rows):
        x = 0
        while x < width:
            end = x
            while end < width and row[end] == row[x]:
                end += 1
            paths.setdefault(int(row[x]), []).append(f"M{x} {y}h{end - x}v1h{x - end}z")
            x = end
    fills = "".join(
        f'<path fill="{ARC_COLORS[value]}" d="{"".join(cells)}"/>' for value, cells in sorted(paths.items())
    )
    verticals = "".join(f"M{x} 0V{height}" for x in range(1, width))
    horizontals = "".join(f"M0 {y}H{width}" for y in range(1, height))
    return (
        f'<svg class="grid" width="{width * _CELL_PX}" height="{height * _CELL_PX}" '
        f'viewBox="0 0 {width} {height}" shape-rendering="crispEdges" role="img" aria-label="{html.escape(label)}">'
        f'{fills}<path class="lines" d="{verticals}{horizontals}"/></svg>'
    )


def _grid_figure(rows: Sequence[Sequence[int]], caption: str, *, note: str = "") -> str:
    """A grid under its caption, and its size."""
    size = f"{len(rows)}×{len(rows[0])}"
    note_html = f'<span class="grid-note">{note}</span>' if note else ""
    return (
        f"<figure>{grid_svg(rows, caption)}<figcaption>{html.escape(caption)} "
        f'<span class="size">{size}</span>{note_html}</figcaption></figure>'
    )


def task_figure(puzzle: dict[str, Any]) -> str:
    """The task as it was given: each example's input and output, and the test inputs."""
    pairs: list[str] = []
    for index, pair in enumerate(puzzle["train"]):
        pairs.append(
            f'<div class="pair"><span class="pair-name">Example {index + 1}</span>'
            f'{_grid_figure(pair["input"], "input")}<span class="arrow">→</span>'
            f"{_grid_figure(pair['output'], 'output')}</div>"
        )
    for index, pair in enumerate(puzzle["test"]):
        pairs.append(
            f'<div class="pair test"><span class="pair-name">Test {index + 1}</span>'
            f'{_grid_figure(pair["input"], "input")}<span class="arrow">→</span>'
            '<span class="unknown">?</span></div>'
        )
    return f'<div class="task">{"".join(pairs)}</div>'


LEGEND = (
    '<div class="legend"><span class="key input"><b>Model input</b> what the harness sends: the prompts, and what '
    'each tool returns</span><span class="key output"><b>Model output</b> what the model generates: its thinking, '
    "its code and its replies</span></div>"
)


def answers_figure(run: taaf.game.GameRun, spec: taaf.game.GameSpec) -> str:
    """The grids the game ended on beside the expected ones, scored again from the record."""
    submission = run.history[-1].submission
    result = taaf.game.score_submission(spec, submission)
    rows: list[str] = []
    for index, (pair, attempts, scored) in enumerate(
        zip(spec.to_answer, submission.attempts, result.pairs, strict=True)
    ):
        expected = pair.answer
        if expected is None:
            raise ValueError(f"{spec.game_id}: test {index} has no recorded answer")
        figures = [_grid_figure(pair.question.data.tolist(), f"test_in[{index}]")]
        for number, attempt in enumerate(attempts):
            matches = attempt == expected
            mark = (
                '<span class="verdict-ok">✓ correct</span>'
                if matches
                else '<span class="verdict-failed">✗ wrong</span>'
            )
            same = number > 0 and attempt == attempts[0]
            if same:
                figures.append(
                    f'<div class="same-attempt">Attempt {number + 1}: the same grid. The model left its second slot '
                    "empty, and the submission repeats the first.</div>"
                )
            else:
                figures.append(_grid_figure(attempt.data.tolist(), f"Attempt {number + 1}", note=mark))
        figures.append(_grid_figure(expected.data.tolist(), "Expected"))
        verdict = (
            f'<p class="verdict-line"><span class="verdict-ok">Test output {index + 1} is correct</span>: attempt '
            f"{scored.matched_attempt + 1} equals the expected grid in all {expected.data.size} cells.</p>"
            if scored.correct and scored.matched_attempt is not None
            else f'<p class="verdict-line"><span class="verdict-failed">Test output {index + 1} is wrong</span>.</p>'
        )
        rows.append(f'<div class="pair answers">{"".join(figures)}</div>{verdict}')
    return f'<div class="task">{"".join(rows)}</div>'


# --- the page -----------------------------------------------------------------


@dataclass
class Notes:
    """What a page says beside its trace, from its notes file."""

    title: str
    intro: str
    epilogue: str
    provenance: str


def parse_notes(text: str) -> Notes:
    """A notes file's sections (the module's docstring has the format)."""
    headings = list(_NOTES_HEADING.finditer(text))
    sections: dict[str, str] = {}
    for found, following in zip(headings, [*headings[1:], None], strict=True):
        sections[found["name"]] = text[found.end() : following.start() if following else len(text)].strip()
    missing = {"title", "intro", "epilogue", "provenance"} - set(sections)
    if missing:
        raise ValueError(f"notes without {sorted(missing)}")
    return Notes(sections["title"], sections["intro"], sections["epilogue"], sections["provenance"])


def with_placeholders(text: str, drawn: dict[str, str]) -> str:
    """Markdown ``text`` with each placeholder line replaced by what the tool drew."""
    parts: list[str] = []
    position = 0
    for found in _PLACEHOLDER.finditer(text):
        parts.append(markdown(text[position : found.start()]))
        parts.append(drawn[found["name"]])
        position = found.end()
    parts.append(markdown(text[position:]))
    return "".join(parts)


def _css() -> str:
    """The page's styles: its own, and the code's."""
    style_lines: list[str] = _FORMATTER.get_style_defs(".code").splitlines()
    code = "\n".join(line for line in style_lines if line.startswith(".code"))
    return _PAGE_CSS + code


_PAGE_CSS = """
:root{--ink:#1c2530;--muted:#5d6b7a;--line:#dfe5ec;--soft:#f5f7fa;--input:#e8912d;--input-ink:#a8620f;
--output:#2f6fde;--output-ink:#2158b6;--ok:#14804a;--bad:#c23a2b}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:44px 32px 90px 56px}
h1{font-size:2rem;line-height:1.2;margin:0 0 14px}
h2{font-size:1.32rem;line-height:1.3;margin:40px 0 6px;padding-top:6px}
code{font:.87em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#eef1f5;border-radius:4px;padding:.08em .32em}
a{color:var(--output-ink)}
.stretch{margin:12px 0 12px -24px;padding:2px 0 2px 19px;border-left:5px solid;border-radius:3px}
.stretch.input{border-color:var(--input)}
.stretch.output{border-color:var(--output)}
.step{margin:0 0 14px}
.step:last-child{margin-bottom:2px}
.step>header{display:flex;justify-content:space-between;gap:12px;font-size:.76rem;letter-spacing:.05em;text-transform:uppercase;margin-bottom:4px;color:var(--muted)}
.step>header code{text-transform:none;letter-spacing:0;font-size:1.05em}
.stretch.input b{color:var(--input-ink)}
.stretch.output b{color:var(--output-ink)}
.prose>:first-child{margin-top:0}
.prose>:last-child{margin-bottom:0}
.prose{font-size:.95rem}
.prose blockquote{margin:8px 0;padding:2px 14px;border-left:3px solid var(--line);color:var(--muted)}
.thinking-body{white-space:pre-wrap;overflow-wrap:anywhere;color:#46525f;font-size:.88rem;line-height:1.55;background:var(--soft);border-radius:6px;padding:10px 14px}
pre{margin:0 0 6px}
pre.code{background:#0d1117;color:#e6edf3;border-radius:6px;padding:12px 15px;overflow-x:auto;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre.code code{background:none;padding:0;font:inherit;color:inherit}
pre.output{background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:9px 13px;white-space:pre-wrap;overflow-wrap:anywhere;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.tool-definition{display:block;margin:.9em 0}
.json-line{display:block}
.verdict-ok{color:var(--ok);font-weight:650}
.verdict-failed{color:var(--bad);font-weight:650}
.gallery{display:flex;flex-wrap:wrap;gap:12px 16px;margin:8px 0}
figure{margin:0}
.gallery figure{width:min-content}
.word{white-space:nowrap}
.gallery img{display:block;max-height:215px;height:auto;image-rendering:pixelated;border:1px solid var(--line)}
figcaption{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--muted);margin-top:3px;white-space:pre-wrap}
.legend{display:flex;flex-wrap:wrap;align-items:center;gap:10px 24px;margin:14px 0;font-size:.9rem}
.key{border-left:5px solid;padding-left:9px;flex:1 1 300px}
.key b{display:block}
.key.input{border-color:var(--input)}
.key.input b{color:var(--input-ink)}
.key.output{border-color:var(--output)}
.key.output b{color:var(--output-ink)}
.task{margin:14px 0 18px;display:flex;flex-direction:column;gap:14px}
.pair{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px}
.pair.answers{align-items:flex-start}
.pair-name{width:76px;white-space:nowrap;font-size:.84rem;font-weight:600;color:var(--muted)}
.arrow{font-size:1.4rem;color:var(--muted)}
.unknown{display:flex;align-items:center;justify-content:center;width:120px;height:120px;border:2px dashed var(--line);border-radius:6px;font-size:2rem;color:var(--muted)}
svg.grid{display:block;background:#000}
svg.grid .lines{stroke:#7f7f7f;stroke-opacity:.38;stroke-width:1px;vector-effect:non-scaling-stroke;fill:none}
.size{opacity:.75}
.grid-note{display:block;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px}
.same-attempt{max-width:150px;font-size:.84rem;color:var(--muted);align-self:center}
.verdict-line{margin:0;font-size:.95rem}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);font-size:.88rem;color:var(--muted)}
"""


def render_page(
    notes: Notes,
    game: dict[str, Any],
    answers: str,
    image_src: Callable[[str], str],
    preamble: Callable[[dict[str, Any]], str],
) -> str:
    """The whole page."""
    drawn = {"task": task_figure(game["puzzle"]), "legend": LEGEND, "answers": answers}
    writer = TraceWriter(image_src=image_src, preamble=preamble)
    for section in game["sections"]:
        writer.add(section)
    title = html.escape(notes.title)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<style>{_css()}</style></head><body><main><h1>{title}</h1>"
        f"{with_placeholders(notes.intro, drawn)}"
        f"{writer.finish()}"
        f"<h2>Epilogue: the answer</h2>{with_placeholders(notes.epilogue, drawn)}"
        f"<footer>{markdown(notes.provenance)}</footer>"
        "</main></body></html>\n"
    )


def _find_game(job: Path, game_id: str) -> dict[str, Any]:
    """The replay viewer's payload for one game of the run."""
    summary = viewer_data.load_run_summary(job, None)
    for index, listed in enumerate(summary["games"]):
        if listed["game_id"] == game_id:
            return viewer_data.load_game_full(job, None, index)
    raise SystemExit(f"{job} has no game {game_id}")


def main() -> None:
    """Write one game's page."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", type=Path)
    parser.add_argument("--game", required=True)
    parser.add_argument("--tokenizer", required=True, help="the checkpoint whose chat template the run served")
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    game = _find_game(args.job, args.game)
    benchmark = taaf.benchmark.Benchmark.from_json(args.job / "benchmark.json", with_game_specs=True)
    run = next(run for run in benchmark.game_runs if run.game_id == args.game)
    if run.spec is None:
        raise SystemExit(f"{args.job} recorded no task spec for {args.game}")

    images: dict[str, str] = {}

    def image_src(key: str) -> str:
        """An image block as a data URI, read once."""
        if key not in images:
            images[key] = "data:image/png;base64," + base64.b64encode(block_png_bytes(args.job, key)).decode()
        return images[key]

    from transformers import AutoTokenizer  # the slowest import here, and only this path needs it

    tokenizer: Any = AutoTokenizer.from_pretrained(args.tokenizer)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    agent: Any = pickle.loads((args.job / "solver.pkl").read_bytes()).agent
    page = render_page(
        parse_notes(args.notes.read_text()),
        game,
        answers_figure(run, run.spec),
        image_src,
        lambda section: system_preamble(
            tokenizer, tool_definitions(agent, section["tools"]), section["reasoning_effort"]
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"{args.out}: {len(page) / 1024:.0f} KiB, {len(game['sections'])} blocks, {len(images)} pictures")


if __name__ == "__main__":
    main()
