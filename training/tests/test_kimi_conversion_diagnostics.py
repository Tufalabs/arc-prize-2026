"""Keep recorded tool failures intact across live and zipped diagnostic formats."""

import json
import zipfile

import pytest

from scripts.flash_mixed_cosine120.conversion.kimi_steps import (
    SUMMARY_THINK_CLOSE,
    G,
    diagnostics_errors,
    summary_injection,
    to_e307,
)


def _step():
    return [{"messages": [
        {"role": "assistant", "tool_calls": [
            {"id": "call-1", "function": {"name": "python", "arguments": "{}"}},
            {"id": "call-2", "function": {"name": "memory", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": "failed"},
        {"role": "tool", "tool_call_id": "call-2", "content": "saved"},
    ]}]


def _spans(names=("python", "memory"), flags=(True, False)):
    return "".join(json.dumps({"kind": "tool/result", "t": i,
                              "values": {"tool_name": name, "is_error": flag}}) + "\n"
                   for i, (name, flag) in enumerate(zip(names, flags)))


@pytest.mark.parametrize("storage", ["live", "zip", "both"])
def test_live_and_zipped_spans_preserve_tool_error_flags(tmp_path, storage):
    steps = _step()
    if storage in {"live", "both"}:
        spans = tmp_path / "games/run/spans.jsonl"
        spans.parent.mkdir(parents=True)
        spans.write_text(_spans())
    if storage in {"zip", "both"}:
        with zipfile.ZipFile(tmp_path / "diagnostics-0.zip", "w") as archive:
            archive.writestr("games/run/spans.jsonl", _spans())
    # The same history appears in subsequent steps; each result is aligned once.
    errors = diagnostics_errors(tmp_path, "run", steps + steps)
    assert errors == {"call-1": True, "call-2": False}
    messages, images = to_e307(steps[0]["messages"], tmp_path, errors)
    assert images == []
    assert messages[1]["content"] == "[Tool result for python [ERROR]]\nfailed"
    assert messages[2]["content"] == "[Tool result for memory]\nsaved"


@pytest.mark.parametrize("names", [(), ("memory", "python")])
def test_missing_or_misaligned_spans_still_fail(tmp_path, names):
    spans = tmp_path / "games/run/spans.jsonl"
    spans.parent.mkdir(parents=True)
    spans.write_text(_spans(names))
    with pytest.raises(AssertionError, match="spans misaligned"):
        diagnostics_errors(tmp_path, "run", _step())


def test_conflicting_live_and_archived_flags_fail(tmp_path):
    spans = tmp_path / "games/run/spans.jsonl"
    spans.parent.mkdir(parents=True)
    spans.write_text(_spans())
    with zipfile.ZipFile(tmp_path / "diagnostics-0.zip", "w") as archive:
        archive.writestr("games/run/spans.jsonl", _spans(flags=(False, False)))
    with pytest.raises(ValueError, match="conflicting tool error flags"):
        diagnostics_errors(tmp_path, "run", _step())


@pytest.mark.parametrize("thinking", [True, False])
def test_summary_prefill_follows_explicit_student_profile(monkeypatch, thinking):
    monkeypatch.setitem(G, "summary_thinking", thinking)
    step = {"messages": [{"role": "user", "content": "Your context window is nearly full. Write a summary."}]}
    assert summary_injection(step) == (None if thinking else SUMMARY_THINK_CLOSE)
