"""Replay viewer: a recorded game read as puzzle, transcript and answers, the
order of its contexts when summaries fall inside a turn, where an answer comes
to stand, and the API routes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from platypus.diagnostics import BlockStore, DiagnosticsArchiver, blocks_dir, game_stream_path
from platypus.viewer import resolve_interactive_request, resolve_request
from platypus.viewer_data import load_game_full, load_run_summary, resolve_run_dir

# Game time zero of every recorded game here, as a wall-clock stamp.
T0 = 1_000.0
RUN = "experiment/seed_00"
TASK = "Here is the puzzle."
SUMMARY = "[context compacted] Summary: mirror the grid."
REQUEST = "Your context window is nearly full. Write a compact summary."


class _Stream:
    """One game's span stream written record by record, each at a game time
    the test chooses: where an answer stands is read from those times."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        """A stream for ``run_id`` under ``run_dir``, with the run's block store."""
        self.store = BlockStore(blocks_dir(run_dir))
        self.path = game_stream_path(run_dir / "games" / run_id)
        self.records: list[dict[str, Any]] = []
        self._opened: dict[int, float] = {}

    def open(self, kind: str, name: str, at: float, parent: int | None = None) -> int:
        """Open a span and return its id."""
        span = len(self._opened)
        self._opened[span] = at
        self.records.append({"e": "open", "id": span, "parent": parent, "kind": kind, "name": name, "t": T0 + at})
        return span

    def close(self, span: int, at: float, *, outcome: str = "ok", generated: int = 0) -> None:
        """Close a span that generated ``generated`` tokens, its children's included."""
        self.records.append(
            {
                "e": "close",
                "id": span,
                "outcome": outcome,
                "t": T0 + at,
                "duration_s": at - self._opened[span],
                "self_generated": generated,
                "self_uncached": 0,
                "incl_generated": generated,
                "incl_uncached": 0,
            }
        )

    def artifact(self, span: int, name: str, payload: Any, codec: str = "json") -> None:
        """Attach a stored payload to a span."""
        block = self.store.store(payload, codec)
        self.records.append({"e": "artifact", "span": span, "name": name, "codec": codec, "block": block})

    def event(self, span: int, kind: str, at: float, **values: Any) -> None:
        """Record an event on a span."""
        self.records.append({"e": "event", "span": span, "kind": kind, "t": T0 + at, "values": values})

    def kv(self, span: int, **values: Any) -> None:
        """Record key-values on a span."""
        self.records.append({"e": "kv", "span": span, "values": values})

    def write(self) -> None:
        """Write the stream where a reader of the run finds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8")


def _user(text: str) -> dict[str, Any]:
    """A user message."""
    return {"kind": "user", "content": {"c": "str", "text": text}}


def _round(thought: str, tokens: int, result: str) -> list[dict[str, Any]]:
    """One round: a reply that thinks and calls ``python``, and the call's result."""
    return [
        {
            "kind": "response",
            "reasoning": {"c": "str", "text": thought},
            "final": {"c": "tool_call", "call_id": "call", "tool_name": "python", "arguments": {"code": thought}},
            "do_not_send_to_agent": False,
            "usage": {"completion_tokens": tokens},
        },
        {
            "kind": "tool_result",
            "call_id": "call",
            "tool_name": "python",
            "is_error": False,
            "content": {"c": "str", "text": result},
        },
    ]


def _reply(text: str, tokens: int) -> dict[str, Any]:
    """A reply that calls nothing."""
    return {
        "kind": "response",
        "reasoning": {"c": "str", "text": ""},
        "final": {"c": "str", "text": text},
        "do_not_send_to_agent": False,
        "usage": {"completion_tokens": tokens},
    }


def _turn(*children: dict[str, Any]) -> dict[str, Any]:
    """A turn element holding rounds."""
    return {"kind": "turn", "elements": list(children)}


def _conversation(*elements: dict[str, Any], system: str = "You solve puzzles.") -> dict[str, Any]:
    """A captured request."""
    return {"system": {"content": {"c": "str", "text": system}, "tools": []}, "elements": list(elements)}


def _present_puzzle(stream: _Stream, move: int) -> None:
    """The grids a move is given: two example pairs and one input to answer."""
    for index in range(2):
        stream.artifact(move, f"given_{index}_question", np.full((2, 3), index, dtype=np.uint8), "png-grid")
        stream.artifact(move, f"given_{index}_answer", np.full((3, 2), index + 4, dtype=np.uint8), "png-grid")
    stream.artifact(move, "ask_0", np.full((2, 2), 7, dtype=np.uint8), "png-grid")


def _submit(
    stream: _Stream, turn: int, number: int, at: float, first: int, second: int, *, placeholders: int = 0
) -> None:
    """An answer submitted from inside a turn: one grid per attempt."""
    span = stream.open("infra/submit", f"answer {number}", at, parent=turn)
    stream.artifact(span, "ask_0_attempt_1", np.full((2, 2), first, dtype=np.uint8), "png-grid")
    stream.artifact(span, "ask_0_attempt_2", np.full((2, 2), second, dtype=np.uint8), "png-grid")
    stream.close(span, at)
    stream.event(turn, "answer", at, n=number, placeholders=placeholders)


def _ran(stream: _Stream, span: int, at: float) -> None:
    """The kernel's two executions of one tool call: the code, then the reading of the answer."""
    stream.event(span, "kernel/exec", at, duration_s=0.01, is_error=False, fresh_kernel=False)
    stream.event(span, "kernel/exec", at, duration_s=0.01, is_error=False, fresh_kernel=False)


def _write_benchmark(run_dir: Path, **record: Any) -> None:
    """The run's benchmark summary with one game record."""
    game: dict[str, Any] = {"game_id": "task0001", "run_id": "run123", "pass_index": 0, "state": "won", **record}
    (run_dir / "benchmark.json").write_text(json.dumps({"end_time": "now", "game_runs": [game]}), encoding="utf-8")


def _record_run(root: Path, *, write_benchmark: bool = True) -> Path:
    """A game of one turn: three calls, an answer from the second and a
    changed one from the third, then a closing reply."""
    run_dir = root / RUN
    stream = _Stream(run_dir, "run123")
    game = stream.open("infra/game", "task0001", 0.0)
    move = stream.open("infra/move", "play", 0.0, parent=game)
    _present_puzzle(stream, move)
    turn = stream.open("llm/turn", "analyzer", 1.0, parent=move)
    _submit(stream, turn, 1, 20.01, first=1, second=1)
    _submit(stream, turn, 2, 30.01, first=3, second=1)
    # The run-level stream is what marks a run that has no benchmark yet.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_spans.jsonl").write_text(
        json.dumps({"e": "open", "id": 0, "parent": None, "kind": "infra/run", "name": "test", "t": T0}) + "\n",
        encoding="utf-8",
    )
    # A turn's responses and results are written together when it closes; the
    # kernel's executions keep the time they ran at.
    for tokens in (11, 13, 17, 5):
        stream.event(turn, "llm/response", 35.0, completion_tokens=tokens, stop_reason="stop")
    for _ in range(3):
        stream.event(turn, "tool/result", 35.0, tool_name="python", is_error=False)
    stream.event(turn, "kernel/exec", 0.5, duration_s=0.2, is_error=False, fresh_kernel=True)
    for at in (10.0, 20.0, 30.0):
        _ran(stream, turn, at)
    conversation = _conversation(
        _user(TASK),
        _turn(*_round("look", 11, "seen"), *_round("try", 13, "set"), *_round("fix", 17, "reset"), _reply("Done.", 5)),
    )
    stream.artifact(turn, "conversation", conversation)
    stream.close(turn, 35.0, generated=46)
    stream.close(move, 35.0, generated=46)
    stream.close(game, 36.0, outcome="submitted 2", generated=46)
    stream.write()
    if write_benchmark:
        _write_benchmark(
            run_dir,
            final_score=100.0,
            final_generated_tokens=5,
            history=[{"generated_tokens": 24}, {"generated_tokens": 17}],
            result={"pairs": [{"correct": True, "matched_attempt": 0}]},
        )
    return run_dir


def _labels(game: dict[str, Any]) -> list[str]:
    """The labels of a game's transcript cards, in order."""
    return [str(section["label"]) for section in game["sections"]]


def test_a_game_reads_as_its_puzzle_its_transcript_and_its_answers(tmp_path: Path) -> None:
    run_dir = _record_run(tmp_path / "runs")

    summary = load_run_summary(tmp_path / "runs")
    assert summary["available_runs"] == [RUN]
    assert summary["image_base"] == f"/{RUN}/viewer/png/"
    assert len(summary["arc_palette"]) == 10
    (listed,) = summary["games"]
    assert (listed["game_id"], listed["status"], listed["final_score"]) == ("task0001", "won", 100.0)
    assert listed["display_name"] == "task0001 · pass 1 · won 100"
    assert (listed["generated_tokens"], listed["answer_count"]) == (46, 2)

    game = load_game_full(tmp_path / "runs", RUN, 0)
    assert game["puzzle"]["train"] == [
        {"input": [[0, 0, 0]] * 2, "output": [[4, 4]] * 3},
        {"input": [[1, 1, 1]] * 2, "output": [[5, 5]] * 3},
    ]
    assert game["puzzle"]["test"] == [{"input": [[7, 7]] * 2}]
    assert [answer["predictions"] for answer in game["answers"]] == [
        [{"attempt_1": [[1, 1]] * 2, "attempt_2": [[1, 1]] * 2}],
        [{"attempt_1": [[3, 3]] * 2, "attempt_2": [[1, 1]] * 2}],
    ]
    assert [answer["elapsed_s"] for answer in game["answers"]] == pytest.approx([20.01, 30.01])
    assert game["result"] == {"pairs": [{"correct": True, "matched_attempt": 0}]}
    assert (game["response_count"], game["elapsed_s"]) == (4, 36.0)
    assert sum(section.get("generated_tokens", 0) for section in game["sections"]) == 46
    # The placement marks are the adapter's own and do not leave it.
    assert not {key for section in game["sections"] for key in section if key.startswith("_")}
    assert resolve_run_dir(tmp_path / "runs", RUN) == run_dir.resolve()


def test_an_answer_stands_from_the_tool_call_it_was_submitted_from(tmp_path: Path) -> None:
    """Submitting says nothing to the model, so no card announces it: the
    marker goes between the call whose code set the answer and that call's
    result, and every card knows which answer stood when it was written."""
    _record_run(tmp_path / "runs")

    game = load_game_full(tmp_path / "runs", RUN, 0)
    assert _labels(game) == [
        "SYSTEM PROMPT",
        "USER PROMPT",
        "THINKING",
        "TOOL CALL: python",
        "TOOL RESULT: python",
        "THINKING",
        "TOOL CALL: python",
        "ANSWER #1",
        "TOOL RESULT: python",
        "THINKING",
        "TOOL CALL: python",
        "ANSWER #2",
        "TOOL RESULT: python",
        "ASSISTANT",
    ]
    assert [section["answer"] for section in game["sections"]] == [0] * 7 + [1] * 4 + [2] * 3
    # A card's time: the turn's opening for what it was handed, the kernel's
    # for a round, the answer's own for a marker and for the result handed
    # back after it, the turn's close for the reply no call followed.
    assert [section["elapsed_s"] for section in game["sections"]] == pytest.approx(
        [1.0, 1.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.01, 20.01, 30.0, 30.0, 30.01, 30.01, 35.0]
    )
    assert "test 1: 2x2 and 2x2" in game["sections"][11]["content"]


def _record_summary_inside_a_turn(root: Path) -> Path:
    """One turn that was summarised after its second round: the duck leaves
    the request as served and the round the summary dropped, and the turn's
    own conversation is what followed."""
    run_dir = root / RUN
    stream = _Stream(run_dir, "run123")
    game = stream.open("infra/game", "task0001", 0.0)
    move = stream.open("infra/move", "play", 0.0, parent=game)
    _present_puzzle(stream, move)
    turn = stream.open("llm/turn", "analyzer", 1.0, parent=move)
    _submit(stream, turn, 1, 20.01, first=1, second=1)
    first, second, third = _round("look", 10, "seen"), _round("try", 20, "set"), _round("fix", 30, "reset")

    request = stream.open("llm/turn", "compaction", 22.0, parent=turn)
    stream.event(request, "llm/response", 25.0, completion_tokens=7, stop_reason="stop")
    # The executions so far are written out with the span open at the time.
    _ran(stream, request, 10.0)
    _ran(stream, request, 20.0)
    served = _conversation(_user(TASK), _turn(*first, *second), _user(REQUEST), _turn(_reply("Mirror the grid.", 7)))
    stream.artifact(request, "conversation", served)
    stream.close(request, 25.0, generated=7)
    stream.kv(turn, compacted=True, compaction_strategy="summarize", compaction_number=1, context_after=120)

    dropped = stream.open("llm/turn", "compacted", 25.1, parent=turn)
    stream.kv(dropped, rounds_dropped=1, dropped_after_round=2)
    stream.event(dropped, "llm/response", 25.1, completion_tokens=10, stop_reason="stop")
    stream.event(dropped, "tool/result", 25.1, tool_name="python", is_error=False)
    stream.artifact(dropped, "conversation", _conversation(_user(TASK), _turn(*first)))
    stream.close(dropped, 25.1, generated=10)

    for tokens in (20, 30):
        stream.event(turn, "llm/response", 40.0, completion_tokens=tokens, stop_reason="stop")
    for _ in range(2):
        stream.event(turn, "tool/result", 40.0, tool_name="python", is_error=False)
    _ran(stream, turn, 33.0)
    stream.artifact(turn, "conversation", _conversation(_user(TASK), _user(SUMMARY), _turn(*second, *third)))
    stream.close(turn, 40.0, generated=67)
    stream.close(move, 40.0, generated=67)
    stream.close(game, 41.0, outcome="submitted 1", generated=67)
    stream.write()
    _write_benchmark(run_dir, final_score=0.0, final_generated_tokens=47, history=[{"generated_tokens": 20}])
    return run_dir


def test_summaries_inside_a_turn_read_in_the_order_they_happened(tmp_path: Path) -> None:
    """A game summarised mid-turn opens on its system prompt and its first
    rounds, not on the summary request: each context is read from the request
    that closed it, and the round the summary kept opens the next one marked
    as kept, counted where it was generated."""
    _record_summary_inside_a_turn(tmp_path / "runs")

    game = load_game_full(tmp_path / "runs", RUN, 0)
    round_cards = ["THINKING", "TOOL CALL: python", "TOOL RESULT: python"]
    assert _labels(game) == [
        "SYSTEM PROMPT",
        "USER PROMPT",
        *round_cards,
        "THINKING",
        "TOOL CALL: python",
        "ANSWER #1",
        "TOOL RESULT: python",
        "COMPACTION REQUEST",
        "ASSISTANT",
        "COMPACTION #1",
        "SYSTEM PROMPT",
        "USER PROMPT",
        "USER PROMPT",
        *round_cards,
        *round_cards,
    ]
    sections = game["sections"]
    assert [section["context"] for section in sections] == [0] * 12 + [1] * 9
    assert [bool(section.get("retained")) for section in sections] == [False] * 15 + [True] * 3 + [False] * 3
    assert sections[10]["content"] == "Mirror the grid." and sections[14]["content"] == SUMMARY
    assert all("generated_tokens" not in section for section in sections[15:18])
    # Every reply is counted once, where it was generated.
    assert sum(section.get("generated_tokens", 0) for section in sections) == 67 == game["generated_tokens"]
    assert [sum(s.get("generated_tokens", 0) for s in sections if s["context"] == c) for c in (0, 1)] == [37, 30]
    assert [context.get("closed_by") for context in game["contexts"]] == ["compaction", None]
    assert game["contexts"][0]["compaction"]["tokens_after"] == 120
    # The answer stood before the summary and still stands after it.
    assert [section["answer"] for section in sections] == [0] * 7 + [1] * 14
    times = [section["elapsed_s"] for section in sections]
    assert times == sorted(times) and times[-1] == pytest.approx(33.0)


def test_a_summary_at_the_start_of_a_turn_rebuilds_from_the_turns_own_conversation(tmp_path: Path) -> None:
    """A summary requested before the turn has run a round drops nothing from
    it: what the fresh context kept is an earlier turn's round, shown again
    as kept, and the message that opened the turn follows it."""
    run_dir = tmp_path / "runs" / RUN
    stream = _Stream(run_dir, "run123")
    game = stream.open("infra/game", "task0001", 0.0)
    move = stream.open("infra/move", "play", 0.0, parent=game)
    _present_puzzle(stream, move)
    first, second = _round("look", 10, "seen"), _round("try", 20, "set")

    opening = stream.open("llm/turn", "analyzer", 1.0, parent=move)
    stream.event(opening, "llm/response", 12.0, completion_tokens=10, stop_reason="stop")
    stream.event(opening, "tool/result", 12.0, tool_name="python", is_error=False)
    _ran(stream, opening, 10.0)
    stream.artifact(opening, "conversation", _conversation(_user(TASK), _turn(*first)))
    stream.close(opening, 12.0, generated=10)

    turn = stream.open("llm/turn", "analyzer", 13.0, parent=move)
    request = stream.open("llm/turn", "compaction", 13.1, parent=turn)
    stream.event(request, "llm/response", 15.0, completion_tokens=7, stop_reason="stop")
    served = _conversation(
        _user(TASK), _turn(*first), _user("check failed"), _user(REQUEST), _turn(_reply("Mirror the grid.", 7))
    )
    stream.artifact(request, "conversation", served)
    stream.close(request, 15.0, generated=7)
    stream.kv(turn, compacted=True, compaction_strategy="summarize", compaction_number=1, context_after=90)
    stream.event(turn, "llm/response", 30.0, completion_tokens=20, stop_reason="stop")
    stream.event(turn, "tool/result", 30.0, tool_name="python", is_error=False)
    _ran(stream, turn, 28.0)
    rebuilt = _conversation(_user(TASK), _user(SUMMARY), _turn(*first), _user("check failed"), _turn(*second))
    stream.artifact(turn, "conversation", rebuilt)
    stream.close(turn, 30.0, generated=27)
    stream.close(move, 30.0, generated=37)
    stream.close(game, 31.0, outcome="gave_up", generated=37)
    stream.write()
    _write_benchmark(run_dir, state="gave_up", final_score=0.0, final_generated_tokens=37, history=[])

    game_payload = load_game_full(tmp_path / "runs", RUN, 0)
    round_cards = ["THINKING", "TOOL CALL: python", "TOOL RESULT: python"]
    assert _labels(game_payload) == [
        "SYSTEM PROMPT",
        "USER PROMPT",
        *round_cards,
        "COMPACTION REQUEST",
        "ASSISTANT",
        "COMPACTION #1",
        "SYSTEM PROMPT",
        "USER PROMPT",
        "USER PROMPT",
        *round_cards,
        "USER PROMPT",
        *round_cards,
    ]
    sections = game_payload["sections"]
    assert [bool(section.get("retained")) for section in sections] == [False] * 11 + [True] * 3 + [False] * 4
    assert sections[14]["content"] == "check failed"
    assert sum(section.get("generated_tokens", 0) for section in sections) == 37
    assert game_payload["answers"] == [] and {section["answer"] for section in sections} == {0}


def test_a_new_take_opens_a_context_of_its_own(tmp_path: Path) -> None:
    """A game played in takes: the second take's conversation knows nothing of
    the first, so it starts a context and shows its system prompt again."""
    run_dir = tmp_path / "runs" / RUN
    stream = _Stream(run_dir, "run123")
    game = stream.open("infra/game", "task0001", 0.0)
    move = stream.open("infra/move", "play", 0.0, parent=game)
    _present_puzzle(stream, move)
    for number, (opened, thought, tokens) in enumerate(((1.0, "look", 10), (20.0, "look again", 15)), start=1):
        take = stream.open("take", f"take {number}", opened, parent=move)
        turn = stream.open("llm/turn", "analyzer", opened, parent=take)
        stream.event(turn, "llm/response", opened + 9, completion_tokens=tokens, stop_reason="stop")
        stream.event(turn, "tool/result", opened + 9, tool_name="python", is_error=False)
        _ran(stream, turn, opened + 5)
        stream.artifact(turn, "conversation", _conversation(_user(TASK), _turn(*_round(thought, tokens, "seen"))))
        stream.close(turn, opened + 9, generated=tokens)
        stream.close(take, opened + 9, generated=tokens)
    stream.close(move, 30.0, generated=25)
    stream.close(game, 30.0, outcome="gave_up", generated=25)
    stream.write()
    _write_benchmark(run_dir, state="gave_up", final_score=0.0, final_generated_tokens=25, history=[])

    game_payload = load_game_full(tmp_path / "runs", RUN, 0)
    round_cards = ["THINKING", "TOOL CALL: python", "TOOL RESULT: python"]
    opening = ["SYSTEM PROMPT", "USER PROMPT", *round_cards]
    assert _labels(game_payload) == [*opening, "NEW TAKE", *opening]
    assert [section["context"] for section in game_payload["sections"]] == [0] * 6 + [1] * 5
    assert [context.get("closed_by") for context in game_payload["contexts"]] == ["new take", None]
    assert "take 2" in game_payload["sections"][5]["content"]


def test_a_takes_answer_carries_its_own_placeholder_count(tmp_path: Path) -> None:
    """Each take numbers its answers from one, so the game's second answer is
    the second take's first: it reports that answer's placeholders, found by
    when it was submitted, not the count of the first take's answer."""
    run_dir = tmp_path / "runs" / RUN
    stream = _Stream(run_dir, "run123")
    game = stream.open("infra/game", "task0001", 0.0)
    move = stream.open("infra/move", "play", 0.0, parent=game)
    _present_puzzle(stream, move)
    for number, (opened, filled) in enumerate(((1.0, 0), (20.0, 1)), start=1):
        take = stream.open("take", f"take {number}", opened, parent=move)
        turn = stream.open("llm/turn", "analyzer", opened, parent=take)
        _ran(stream, turn, opened + 5)
        _submit(stream, turn, 1, opened + 5, number, number, placeholders=filled)
        stream.event(turn, "llm/response", opened + 9, completion_tokens=10, stop_reason="stop")
        stream.event(turn, "tool/result", opened + 9, tool_name="python", is_error=False)
        stream.artifact(turn, "conversation", _conversation(_user(TASK), _turn(*_round("look", 10, "seen"))))
        stream.close(turn, opened + 9, generated=10)
        stream.close(take, opened + 9, generated=10)
    stream.close(move, 30.0, generated=20)
    stream.close(game, 30.0, outcome="lost", generated=20)
    stream.write()
    _write_benchmark(run_dir, state="lost", final_score=0.0, final_generated_tokens=20, history=[])

    game_payload = load_game_full(tmp_path / "runs", RUN, 0)
    assert [(answer["number"], answer["placeholders"]) for answer in game_payload["answers"]] == [(1, 0), (2, 1)]
    assert "1 placeholder grids" in next(
        section["content"] for section in game_payload["sections"] if section["label"] == "ANSWER #2"
    )


def test_rounds_pruned_with_no_summary_are_shown_where_they_left(tmp_path: Path) -> None:
    """Under the prune strategy the oldest rounds leave with no request: the
    record of them is the only place they are, and the turn's own conversation
    holds what stayed."""
    run_dir = tmp_path / "runs" / RUN
    stream = _Stream(run_dir, "run123")
    game = stream.open("infra/game", "task0001", 0.0)
    move = stream.open("infra/move", "play", 0.0, parent=game)
    _present_puzzle(stream, move)
    first, second = _round("look", 10, "seen"), _round("try", 20, "set")
    turn = stream.open("llm/turn", "analyzer", 1.0, parent=move)
    dropped = stream.open("llm/turn", "compacted", 15.0, parent=turn)
    stream.kv(dropped, rounds_dropped=1, dropped_after_round=1)
    stream.event(dropped, "llm/response", 15.0, completion_tokens=10, stop_reason="stop")
    stream.event(dropped, "tool/result", 15.0, tool_name="python", is_error=False)
    _ran(stream, dropped, 10.0)
    stream.artifact(dropped, "conversation", _conversation(_user(TASK), _turn(*first)))
    stream.close(dropped, 15.0, generated=10)
    stream.event(turn, "llm/response", 30.0, completion_tokens=20, stop_reason="stop")
    stream.event(turn, "tool/result", 30.0, tool_name="python", is_error=False)
    _ran(stream, turn, 25.0)
    stream.artifact(turn, "conversation", _conversation(_user(TASK + " [1 round pruned]"), _turn(*second)))
    stream.close(turn, 30.0, generated=30)
    stream.close(move, 30.0, generated=30)
    stream.close(game, 30.0, outcome="gave_up", generated=30)
    stream.write()
    _write_benchmark(run_dir, state="gave_up", final_score=0.0, final_generated_tokens=30, history=[])

    game_payload = load_game_full(tmp_path / "runs", RUN, 0)
    round_cards = ["THINKING", "TOOL CALL: python", "TOOL RESULT: python"]
    opening = ["SYSTEM PROMPT", "USER PROMPT", *round_cards]
    assert _labels(game_payload) == [*opening, "ROUNDS PRUNED", *opening]
    sections = game_payload["sections"]
    assert "1 oldest rounds" in sections[5]["content"]
    assert not any(section.get("retained") for section in sections)
    assert sum(section.get("generated_tokens", 0) for section in sections) == 30
    assert [section["elapsed_s"] for section in sections] == pytest.approx(
        [1.0, 1.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 25.0, 25.0, 25.0]
    )


def test_a_live_run_is_visible_before_its_benchmark_is_written(tmp_path: Path) -> None:
    """Append-only spans make an active run viewable before the final bookkeeping."""
    _record_run(tmp_path / "runs", write_benchmark=False)

    summary = load_run_summary(tmp_path / "runs", RUN)
    (listed,) = summary["games"]
    assert (listed["game_id"], listed["status"], listed["final_score"]) == ("task0001", "submitted 2", None)
    assert (listed["generated_tokens"], listed["answer_count"]) == (46, 2)

    game = load_game_full(tmp_path / "runs", RUN, 0)
    assert len(game["answers"]) == 2 and game["result"] is None
    assert _labels(game).count("ANSWER #2") == 1


@pytest.mark.parametrize("write_benchmark", [True, False])
def test_an_archived_run_replays_as_the_loose_one_does(tmp_path: Path, write_benchmark: bool) -> None:
    """A run on scratch leaves its streams and blocks in the job dir's zip
    parts and no ``games/`` dir: the summary and the replay read the same
    from the parts as from loose files, finished or still live."""
    loose = _record_run(tmp_path / "loose", write_benchmark=write_benchmark)
    archived = tmp_path / "archived" / RUN
    scratch = tmp_path / "scratch"
    shutil.copytree(loose, scratch)
    archived.mkdir(parents=True)
    if write_benchmark:
        shutil.move(scratch / "benchmark.json", archived / "benchmark.json")
    assert DiagnosticsArchiver(scratch, archived).finish() is not None
    assert not (archived / "games").exists() and not blocks_dir(archived).exists()

    assert load_run_summary(tmp_path / "archived", RUN) == load_run_summary(tmp_path / "loose", RUN)
    assert load_game_full(tmp_path / "archived", RUN, 0) == load_game_full(tmp_path / "loose", RUN, 0)


def test_interactive_routes_are_separate_and_confined(tmp_path: Path) -> None:
    _record_run(tmp_path / "runs")

    app = resolve_interactive_request(tmp_path / "runs", "/_replay/")
    assert app is not None and app[0] == 200 and b"ARC2 Replay Viewer" in app[1]
    # The shell speaks to the reserved API prefix, not the file space, and
    # honours the ?run= parameter the listing's replay links carry.
    assert b'"/_api/run"' in app[1] and b'"/_api/game"' in app[1]
    assert b'URLSearchParams(window.location.search).get("run")' in app[1]

    api = resolve_interactive_request(tmp_path / "runs", "/_api/run", "run=experiment%2Fseed_00")
    assert api is not None and api[0] == 200
    assert json.loads(api[1])["selected_run"] == RUN

    game_api = resolve_interactive_request(tmp_path / "runs", "/_api/game", "run=experiment%2Fseed_00&index=0")
    assert game_api is not None and game_api[0] == 200
    assert len(json.loads(game_api[1])["answers"]) == 2

    missing = resolve_interactive_request(tmp_path / "runs", "/_api/game", "run=experiment%2Fseed_00")
    assert missing is not None and missing[0] == 400
    beyond = resolve_interactive_request(tmp_path / "runs", "/_api/game", "run=experiment%2Fseed_00&index=5")
    assert beyond is not None and beyond[0] == 404
    escaped = resolve_interactive_request(tmp_path / "runs", "/_api/run", "run=..%2Foutside")
    assert escaped is not None and escaped[0] == 404

    # Everything outside the two reserved prefixes falls through to the
    # canonical resolver: the server root, run paths, and viewer pages.
    assert resolve_interactive_request(tmp_path / "runs", "/") is None
    assert resolve_interactive_request(tmp_path / "runs", "/index.html") is None
    assert resolve_interactive_request(tmp_path / "runs", f"/{RUN}/viewer/index.html") is None


def test_the_listing_links_a_run_to_its_replay(tmp_path: Path) -> None:
    _record_run(tmp_path / "runs")

    root_listing = resolve_request(tmp_path / "runs", "/")
    assert root_listing is not None and b'href="/_replay/"' in root_listing[0]
    listing = resolve_request(tmp_path / "runs", "/experiment/")
    assert listing is not None and b'href="/_replay/?run=experiment%2Fseed_00"' in listing[0]


def test_an_image_the_model_was_shown_is_served_where_the_transcript_points(tmp_path: Path) -> None:
    """A transcript card names an image by its block, and the run's own
    viewer path serves that block as PNG: the app joins the two."""
    run_dir = _record_run(tmp_path / "runs")
    store = BlockStore(blocks_dir(run_dir))
    key = store.store(np.full((2, 2), 3, dtype=np.uint8), "png-grid")
    stream = _Stream(run_dir, "run456")
    game = stream.open("infra/game", "task0002", 0.0)
    move = stream.open("infra/move", "play", 0.0, parent=game)
    turn = stream.open("llm/turn", "analyzer", 1.0, parent=move)
    shown = {
        "kind": "user",
        "content": {"c": "list", "elements": [{"c": "str", "text": TASK}, {"c": "image", "block": key}]},
    }
    stream.artifact(turn, "conversation", _conversation(shown, _turn(_reply("Seen.", 3))))
    stream.event(turn, "llm/response", 2.0, completion_tokens=3, stop_reason="stop")
    stream.close(turn, 2.0, generated=3)
    stream.close(move, 2.0, generated=3)
    stream.close(game, 2.0, outcome="gave_up", generated=3)
    stream.write()
    games: list[dict[str, Any]] = [
        {"game_id": "task0001", "run_id": "run123", "state": "won", "final_generated_tokens": 46, "history": []},
        {"game_id": "task0002", "run_id": "run456", "state": "gave_up", "final_generated_tokens": 3, "history": []},
    ]
    (run_dir / "benchmark.json").write_text(json.dumps({"end_time": "now", "game_runs": games}), encoding="utf-8")

    summary = load_run_summary(tmp_path / "runs", RUN)
    game_payload = load_game_full(tmp_path / "runs", RUN, 1)
    assert game_payload["sections"][1]["content"] == f"{TASK}\n[image block {key}]"
    served = resolve_request(tmp_path / "runs", f"{summary['image_base']}{key}.png")
    assert served is not None and served[1] == "image/png" and served[0].startswith(b"\x89PNG")
