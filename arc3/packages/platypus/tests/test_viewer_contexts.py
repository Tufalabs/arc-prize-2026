"""The replay viewer's context view: a compaction request's cards, the
boundary card between contexts, what a rebuilt context opens with, the
context numbering, and prompt images on the page."""

import json
import re
import shutil
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from platypus.diagnostics import BlockStore, SpanLogger, SpanNode, build_span_tree, read_events
from platypus.viewer_data import (
    _boundary_section,
    _compaction_request_sections,
    _ContextTracker,
    _move_transcript_sections,
    _summary_compaction_outcomes,
)


def _str(text: str) -> dict[str, Any]:
    return {"c": "str", "text": text}


def _turn(*responses: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "turn", "elements": list(responses)}


def _response(reasoning: str, final: str) -> dict[str, Any]:
    return {"kind": "response", "reasoning": _str(reasoning), "final": _str(final), "usage": {"completion_tokens": 10}}


def _conversation(elements: list[dict[str, Any]]) -> dict[str, Any]:
    return {"system": {"content": _str("SYSTEM v1")}, "elements": elements}


def test_compaction_request_cards_cover_the_folded_rounds_and_the_reply(tmp_path: Path) -> None:
    store = BlockStore(tmp_path / "blocks")
    elements = [
        {"kind": "user", "content": _str("status move 3")},
        _turn(_response("think 3", "call 3")),
        {"kind": "user", "content": _str("status move 4")},
        _turn(_response("think 4 (folded into the request)", "call 4")),
        {"kind": "user", "content": _str("Your context is full.")},
        _turn(_response("dump thinking", "memory call")),
        {"kind": "tool_result", "tool_name": "memory", "content": _str("Notes saved.")},
    ]
    with_previous = _compaction_request_sections(_conversation(elements), store, include_previous_block=True)
    labels = [section["label"] for section in with_previous]
    assert labels[0] == "USER PROMPT" and with_previous[0]["content"] == "status move 4"
    assert "COMPACTION REQUEST" in labels
    assert labels[-1] == "ASSISTANT" and with_previous[-1]["content"] == "memory call"
    assert "TOOL RESULT: memory" not in labels  # the model never sees the acknowledgement
    assert "status move 3" not in [section["content"] for section in with_previous]
    without_previous = _compaction_request_sections(_conversation(elements), store, include_previous_block=False)
    assert without_previous[0]["label"] == "COMPACTION REQUEST"


def test_boundary_card_reports_the_compaction_outcome() -> None:
    saved = _boundary_section(1, {"compaction_dump_ok": True, "context_before": 20873, "context_after": 3635})
    assert saved["label"] == "COMPACTION #1" and saved["kind"] == "boundary"
    assert "notes saved" in saved["content"] and "20,873 → 3,635" in saved["content"]
    assert saved["compaction"]["dump_ok"] is True
    failed = _boundary_section(2, {"compaction_dump_ok": False, "compaction_dump_reason": "no memory call"})
    assert "dump FAILED (no memory call); previous notes kept" in failed["content"]


@pytest.mark.parametrize("before", [None, 20000])
def test_summary_boundary_does_not_require_a_memory_dump(before: int | None) -> None:
    boundary = _boundary_section(
        4,
        {"compacted": True, "compaction_strategy": "summarize", "context_before": before, "context_after": 8376},
    )
    assert boundary["compaction"]["mode"] == "summarize"
    assert "dump_ok" not in boundary["compaction"]
    assert "summary installed" in boundary["content"]
    assert "summary and retained messages" in boundary["content"]
    assert "8,376" in boundary["content"]
    assert "FAILED" not in boundary["content"] and "notes kept" not in boundary["content"]


def test_context_tracker_numbers_contexts_and_records_closures() -> None:
    tracker = _ContextTracker()
    first = tracker.stamp([{"label": "THINKING"}, {"label": "TOOL CALL: python"}])
    assert [section["context"] for section in first] == [0, 0]
    boundary = _boundary_section(1, {"compaction_dump_ok": True})
    tracker.close(reason="compaction", move_index=12, level=1, boundary=boundary)
    assert tracker.index == 1 and tracker.last_system is None
    assert tracker.contexts[0]["closed_by"] == "compaction" and tracker.contexts[0]["compaction"]["number"] == 1
    assert tracker.contexts[1] == {"index": 1, "opened_by": "compaction", "opened_at_move": 12, "level": 1}
    later = tracker.stamp([{"label": "SYSTEM PROMPT"}])
    assert later[0]["context"] == 1


@pytest.mark.parametrize("next_level,expected", [(1, "CONTEXT REWRITE"), (2, "NEW LEVEL")])
def test_system_prompt_rewrites_only_mark_a_new_level_when_the_level_changed(
    tmp_path: Path,
    next_level: int,
    expected: str,
) -> None:
    store = BlockStore(tmp_path / "blocks")
    tracker = _ContextTracker()
    sections = []
    for move, level, system in [(1, 1, "notes before trim"), (2, next_level, "notes after trim")]:
        conversation = _conversation(
            [{"kind": "user", "content": _str("status")}, _turn(_response("reason", "answer"))]
        )
        conversation["system"]["content"] = _str(system)
        block = store.store(conversation, "json")
        turn = SpanNode(
            span_id=move,
            kind="llm/turn",
            name="analyze",
            t_open=float(move),
            artifacts=[
                {
                    "name": "conversation",
                    "codec": "json",
                    "block": block,
                }
            ],
        )
        sections = _move_transcript_sections(
            tracker, move_index=move, level=level, turns=[turn], store=store, turn_elapsed={}
        )
    assert [section["label"] for section in sections if section["kind"] == "boundary"] == [expected]
    assert tracker.contexts[0]["closed_by"] == ("new level" if next_level == 2 else "context rewrite")
    assert tracker.contexts[1]["level"] == next_level


def test_nested_compactions_keep_their_individual_outcomes(tmp_path: Path) -> None:
    store = BlockStore(tmp_path / "blocks")
    tracker = _ContextTracker()
    compactions = [
        SpanNode(
            span_id=2,
            kind="llm/turn",
            name="compaction",
            t_open=2.0,
            kv={"compaction_dump_ok": True, "context_before": 20000, "context_after": 3000},
        ),
        SpanNode(
            span_id=3,
            kind="llm/turn",
            name="compaction",
            t_open=3.0,
            kv={
                "compaction_dump_ok": False,
                "compaction_dump_reason": "reply cap",
                "context_before": 22000,
                "context_after": 3500,
            },
        ),
    ]
    turn = SpanNode(
        span_id=1, kind="llm/turn", name="analyze", t_open=1.0, children=compactions, kv=dict(compactions[1].kv)
    )
    sections = _move_transcript_sections(
        tracker, move_index=1, level=1, turns=[turn, *compactions], store=store, turn_elapsed={}
    )
    boundaries = [section for section in sections if section["kind"] == "boundary"]
    assert len(boundaries) == 2
    assert boundaries[0]["compaction"]["dump_ok"] is True
    assert boundaries[1]["compaction"]["dump_ok"] is False
    assert "20,000 → 3,000" in boundaries[0]["content"]
    assert "22,000 → 3,500" in boundaries[1]["content"]
    assert tracker.contexts[0]["compaction"]["dump_ok"] is True


def test_summary_retries_and_multiple_installs_use_individual_stream_outcomes(tmp_path: Path) -> None:
    store = BlockStore(tmp_path / "blocks")
    stream = tmp_path / "spans.jsonl"
    logger = SpanLogger(stream, store)
    with closing(logger), logger.span("llm/turn", "analyzer") as analyzer:
        for after in (None, 3000, None, 4000, None):
            with logger.span("llm/turn", "compaction"):
                pass
            if after is not None:
                analyzer.kv(compacted=True, compaction_strategy="summarize", context_after=after)
    records = read_events(stream)
    (turn,) = build_span_tree(records)
    outcomes = _summary_compaction_outcomes(records)
    tracker = _ContextTracker()
    sections = _move_transcript_sections(
        tracker,
        move_index=1,
        level=1,
        turns=[turn, *turn.children],
        store=store,
        turn_elapsed={},
        summary_outcomes=outcomes,
    )
    boundaries = [section for section in sections if "compaction" in section]
    attempts = [section for section in sections if section["label"] == "SUMMARY ATTEMPT"]
    assert [section["compaction"]["tokens_after"] for section in boundaries] == [3000, 4000]
    assert [section["compaction"]["number"] for section in boundaries] == [1, 2]
    assert [section["context"] for section in attempts] == [0, 1, 2]
    assert len(tracker.contexts) == 3


def test_first_request_after_a_summary_shows_what_the_rebuild_carried_in(tmp_path: Path) -> None:
    store = BlockStore(tmp_path / "blocks")
    tracker = _ContextTracker()
    rebuilt = [
        {"kind": "user", "content": _str("summary of the work so far")},
        _turn(_response("", "call 17"), {"kind": "tool_result", "tool_name": "python", "content": _str("result 17")}),
        {"kind": "user", "content": _str("status move 18")},
        {"kind": "user", "content": _str("context compacted, continue")},
        _turn(_response("think 18", "call 18")),
    ]
    later = [*rebuilt, {"kind": "user", "content": _str("status move 19")}, _turn(_response("think 19", "call 19"))]

    def analyzer(span_id: int, elements: list[dict[str, Any]], **kwargs: Any) -> SpanNode:
        block = store.store(_conversation(elements), "json")
        artifacts = [{"name": "conversation", "codec": "json", "block": block}]
        return SpanNode(span_id=span_id, kind="llm/turn", name="analyze", t_open=1.0, artifacts=artifacts, **kwargs)

    compaction = SpanNode(span_id=2, kind="llm/turn", name="compaction", t_open=1.5)
    installed = {"compacted": True, "compaction_strategy": "summarize", "context_after": 9000}
    first = _move_transcript_sections(
        tracker,
        move_index=18,
        level=1,
        turns=[analyzer(1, rebuilt, children=[compaction], kv=dict(installed)), compaction],
        store=store,
        turn_elapsed={},
        summary_outcomes={2: installed},
    )
    after = first[[section["kind"] for section in first].index("boundary") + 1 :]
    assert [(section["content"], bool(section.get("retained"))) for section in after[1:]] == [
        ("summary of the work so far", True),
        ("call 17", True),
        ("result 17", True),
        ("status move 18", True),
        ("context compacted, continue", False),
        ("think 18", False),
        ("call 18", False),
    ]
    assert after[0]["label"] == "SYSTEM PROMPT"
    assert all("generated_tokens" not in section for section in after if section.get("retained"))
    assert after[-1]["generated_tokens"] > 0
    second = _move_transcript_sections(
        tracker, move_index=19, level=1, turns=[analyzer(3, later)], store=store, turn_elapsed={}
    )
    assert [section["content"] for section in second] == ["status move 19", "think 19", "call 19"]


def test_a_second_summary_in_one_turn_opens_its_context_with_what_the_first_carried_in(tmp_path: Path) -> None:
    store = BlockStore(tmp_path / "blocks")

    def carried(summary: str) -> list[dict[str, Any]]:
        return [
            {"kind": "user", "content": _str(summary)},
            {"kind": "user", "content": _str("status move 18")},
            {"kind": "user", "content": _str("context compacted, continue")},
        ]

    def request(elements: list[dict[str, Any]], span_id: int) -> SpanNode:
        block = store.store(_conversation(elements), "json")
        artifacts = [{"name": "conversation", "codec": "json", "block": block}]
        return SpanNode(span_id=span_id, kind="llm/turn", name="compaction", t_open=1.0, artifacts=artifacts)

    ask = {"kind": "user", "content": _str("Your context window is nearly full.")}
    compactions = [
        request(
            [
                {"kind": "user", "content": _str("status move 18")},
                _turn(_response("think a", "call a")),
                ask,
                _turn(_response("", "first summary")),
            ],
            2,
        ),
        request(
            [
                *carried("first summary"),
                _turn(_response("think b", "call b")),
                ask,
                _turn(_response("", "second summary")),
            ],
            3,
        ),
    ]
    block = store.store(_conversation([*carried("second summary"), _turn(_response("think c", "call c"))]), "json")
    # The seal records every round a summary folds in as discarded.
    folded: dict[str, Any] = {"records": [{"reason": "context_compaction"}] * 2, "conversation": {"elements": []}}
    installed = {"compacted": True, "compaction_strategy": "summarize", "context_after": 9000}
    analyzer = SpanNode(
        span_id=1,
        kind="llm/turn",
        name="analyze",
        t_open=0.5,
        artifacts=[
            {"name": "conversation", "codec": "json", "block": block},
            {"name": "discarded_rounds", "codec": "json", "block": store.store(folded, "json")},
        ],
        children=compactions,
        kv=dict(installed),
    )
    sections = _move_transcript_sections(
        _ContextTracker(),
        move_index=18,
        level=1,
        turns=[analyzer, *compactions],
        store=store,
        turn_elapsed={},
        summary_outcomes={2: installed, 3: installed},
    )
    second_context = [
        (section["content"], bool(section.get("retained"))) for section in sections if section["context"] == 1
    ]
    assert second_context[1:5] == [
        ("first summary", True),
        ("status move 18", True),
        ("context compacted, continue", False),
        ("think b", False),
    ]
    third_context = [section["content"] for section in sections if section["context"] == 2]
    assert third_context[1:3] == ["second summary", "status move 18"]


def test_browser_shows_prompt_images_where_they_were_inserted() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the viewer JavaScript")
    app = (Path(__file__).parents[1] / "src/platypus/viewer_app.html").read_text()
    placeholder = re.search(r"^    const IMAGE_BLOCK_PLACEHOLDER = .*$", app, re.MULTILINE)
    assert placeholder is not None
    functions = [placeholder.group()]
    for name in ("renderTranscriptContent", "transcriptImageUrl", "escapeHtml"):
        match = re.search(rf"^    function {name}\(.*?^    }}", app, re.MULTILINE | re.DOTALL)
        assert match is not None
        functions.append(match.group())
    key = "ab" * 32
    content = f"Current grid image:\n[image block {key}]\n<done>"
    script = (
        "const window = {location: {origin: 'http://viewer'}};\n"
        "const currentRunName = 'experiment/seed_00';\n"
        + "\n".join(functions)
        + f"\nconsole.log(JSON.stringify(renderTranscriptContent({json.dumps(content)})));"
    )
    result = subprocess.run([node, "-e", script], check=True, text=True, capture_output=True, timeout=10)
    html = json.loads(result.stdout)
    src = f"/_api/image?run=experiment%2Fseed_00&amp;block={key}"
    assert html.startswith("Current grid image:\n<a href=")
    assert f'<img class="transcript-image" src="{src}"' in html
    assert html.endswith("</a>\n&lt;done&gt;")


def test_browser_indexes_a_turn_past_the_argument_limit() -> None:
    """A turn can hold hundreds of thousands of generated pieces (seven
    compactions in one move), more than one call takes as arguments: they
    are indexed without passing them all to a single call."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the viewer JavaScript")
    app = (Path(__file__).parents[1] / "src/platypus/viewer_app.html").read_text()
    functions: list[str] = []
    for name in ("prepareGeneratedIndex", "transcriptPiecesFromBlocks", "tokenizeGeneratedText"):
        match = re.search(rf"^    function {name}\(.*?^    }}", app, re.MULTILINE | re.DOTALL)
        assert match is not None
        functions.append(match.group())
    script = (
        "let generatedTokens = [];\n"
        + "\n".join(functions)
        + """
const tokens = transcriptPiecesFromBlocks([{content: "word ".repeat(400000), generated: true}]);
const index = prepareGeneratedIndex([{tokenCount: tokens.length, tokens, responseCount: 1}]);
console.log(JSON.stringify({pieces: tokens.length, indexed: generatedTokens.length, tokens: index.tokens}));
"""
    )
    result = subprocess.run([node, "-e", script], check=True, text=True, capture_output=True, timeout=30)
    assert json.loads(result.stdout) == {"pieces": 400000, "indexed": 400000, "tokens": 400000}


def test_browser_renders_data_boundaries_and_context_outcomes(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the viewer JavaScript")
    app = (Path(__file__).parents[1] / "src/platypus/viewer_app.html").read_text()
    functions: list[str] = []
    for name in (
        "finiteNumber",
        "normalizePositiveInt",
        "transcriptBlocksFromSections",
        "isGeneratedSection",
        "isPromptSection",
        "isToolResponseSection",
        "isBoundarySection",
        "displayTranscriptLabel",
        "transcriptBlockKind",
        "contextOutcome",
        "isGeneratedBlock",
    ):
        match = re.search(rf"^    function {name}\(.*?^    }}", app, re.MULTILINE | re.DOTALL)
        assert match is not None
        functions.append(match.group())
    sections = [
        _boundary_section(1, {"compaction_dump_ok": True}),
        _boundary_section(2, {"compacted": True, "compaction_strategy": "summarize", "context_after": 8376}),
        _boundary_section(3, {"compaction_dump_ok": False, "compaction_dump_reason": "reply cap"}),
    ]
    contexts = [{"closed_by": "compaction", "compaction": section["compaction"]} for section in sections]
    contexts.extend(
        [
            {"closed_by": "context rewrite"},
            {"closed_by": "new level"},
            {"closed_by": "compaction", "compaction": {"mode": "journal"}},
        ]
    )
    for label in ("NEW LEVEL", "CONTEXT REWRITE"):
        sections.append({"label": label, "content": "Context rebuilt.", "kind": "boundary"})
    sections.append({"label": "TOOL CALL: python", "content": "call 17", "kind": "tool", "retained": True})
    script = (
        "\n".join(functions)
        + "\n"
        + f"const sections = {json.dumps(sections)};"
        + f"const currentGame = {{contexts: {json.dumps(contexts)}}};"
        + """
const contextStats = currentGame.contexts.map(() => ({}));
const blocks = transcriptBlocksFromSections(sections);
console.log(JSON.stringify({blocks, counted: blocks.map(isGeneratedBlock), outcomes: currentGame.contexts.map((_, i) => contextOutcome(i))}));
"""
    )
    result = subprocess.run([node, "-e", script], check=True, text=True, capture_output=True, timeout=10)
    data = json.loads(result.stdout)
    assert [block["label"] for block in data["blocks"]] == [
        "Compaction 1",
        "Compaction 2",
        "Compaction 3",
        "New level",
        "Context rewrite",
        "Retained · TOOL CALL PYTHON",
    ]
    assert all(block["kind"] == "boundary" and not block["generated"] for block in data["blocks"][:-1])
    # A retained call is shown, but neither animated nor counted as this turn's output.
    assert data["blocks"][-1]["kind"] == "tool-call" and not data["blocks"][-1]["generated"]
    assert data["counted"] == [False] * len(data["blocks"])
    assert data["outcomes"] == [
        "compacted",
        "compacted",
        "compaction FAILED",
        "context rewrite",
        "new level",
        "compacted",
    ]
