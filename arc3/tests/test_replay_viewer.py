"""Replay viewer: nested run discovery, span adaptation, and API routes
(the prompt-image route included)."""

from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from platypus.diagnostics import BlockStore, DiagnosticsArchiver, SpanLogger, blocks_dir, game_stream_path
from platypus.viewer import resolve_interactive_request
from platypus.viewer_data import load_game_full, load_run_summary, resolve_run_dir


def _conversation(
    store: BlockStore, *, thought: str, generated_tokens: int, action_number: int | None = None
) -> dict[str, object]:
    """One captured LLM round with exact completion-token usage."""
    system_key = store.store("Inspect the game and act.", "json")
    tool_result = (
        "1"
        if action_number is None
        else json.dumps({"tool": "python", "stdout": {"executed": True, "action_num": action_number}})
    )
    return {
        "system": {"content": {"c": "str", "text": {"$block": system_key}}, "tools": []},
        "elements": [
            {"kind": "user", "content": {"c": "str", "text": "Current board"}},
            {
                "kind": "turn",
                "elements": [
                    {
                        "kind": "response",
                        "reasoning": {"c": "str", "text": thought},
                        "final": {
                            "c": "tool_call",
                            "call_id": "call-1",
                            "tool_name": "python",
                            "arguments": {"code": "print(current_frame.level)"},
                        },
                        "do_not_send_to_agent": False,
                        "usage": {"completion_tokens": generated_tokens},
                    },
                    {
                        "kind": "tool_result",
                        "call_id": "call-1",
                        "tool_name": "python",
                        "is_error": False,
                        "content": {"c": "str", "text": tool_result},
                    },
                ],
            },
        ],
    }


def _record_run(
    root: Path,
    *,
    write_benchmark: bool = True,
    trailing_actionless_move: bool = False,
) -> Path:
    """Create one recorded run with optional live, pre-benchmark state."""
    run_dir = root / "experiment" / "seed_00"
    store = BlockStore(blocks_dir(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_spans.jsonl").write_text(
        json.dumps({"e": "open", "id": 0, "parent": None, "kind": "infra/run", "name": "test", "t": 1.0}) + "\n",
        encoding="utf-8",
    )
    stream = game_stream_path(run_dir / "games" / "run123")
    logger = SpanLogger(stream, store)
    with logger.span("infra/game", "ar25-test") as game:
        game.artifact(
            "game_metadata",
            {"number_of_levels": 2, "base_actions_per_level": [2, 2]},
            codec="json",
        )
        with logger.span("infra/move", "move 0") as move:
            move.artifact("frame_at", np.zeros((3, 3), dtype=np.uint8), codec="png-grid")
            with move.span("llm/turn", "analyzer") as turn:
                turn.artifact(
                    "conversation",
                    _conversation(store, thought="I should inspect it.", generated_tokens=17),
                    codec="json",
                )
                turn.add_usage(17, 4)
                turn.event("llm/response", completion_tokens=17)
            with move.span("llm/turn", "analyzer retry") as turn:
                turn.artifact(
                    "conversation",
                    _conversation(store, thought="I should act now.", generated_tokens=23, action_number=1),
                    codec="json",
                )
                turn.add_usage(23, 5)
                turn.event("llm/response", completion_tokens=23)
                # A cancelled request leaves this diagnostic shell behind but
                # produced no backend response and no transcript card.
                turn.event(
                    "llm/response",
                    completion_tokens=0,
                    uncached_input_tokens=0,
                    cached_input_tokens=0,
                    stop_reason=None,
                    n_tool_calls=0,
                    synthetic=False,
                )
            move.event(
                "infra/action", name="ACTION1", row=None, col=None, outcome="playing", level=0, levels_completed=0
            )
            move.artifact("frame_after_1", np.ones((3, 3), dtype=np.uint8), codec="png-grid")
        with logger.span("infra/move", "move 1") as move:
            move.artifact("frame_at", np.full((3, 3), 2, dtype=np.uint8), codec="png-grid")
            with move.span("llm/turn", "analyzer") as turn:
                turn.artifact(
                    "conversation",
                    _conversation(store, thought="Finish the level.", generated_tokens=31, action_number=2),
                    codec="json",
                )
                turn.add_usage(31, 6)
                turn.event("llm/response", completion_tokens=31)
            move.event(
                "infra/action", name="ACTION4", row=None, col=None, outcome="level_won", level=0, levels_completed=1
            )
            move.artifact("frame_after_1", np.full((3, 3), 3, dtype=np.uint8), codec="png-grid")
        if trailing_actionless_move:
            with logger.span("infra/move", "terminal bookkeeping") as move:
                move.artifact("frame_at", np.full((3, 3), 4, dtype=np.uint8), codec="png-grid")
    logger.close()
    benchmark: dict[str, Any] = {
        "label": "test",
        "game_runs": [
            {
                "game_id": "ar25-test",
                "run_id": "run123",
                "state": "gave_up",
                "number_of_levels": 2,
                "levels_completed": 1,
                "actions_per_level": [2],
                "base_actions_per_level": [2, 2],
                "final_score": 37.5,
                "history": [
                    {"action": {}, "generated_tokens": 40},
                    {"action": {}, "generated_tokens": 31},
                ],
                "final_generated_tokens": 0,
            }
        ],
    }
    if write_benchmark:
        (run_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    return run_dir


def test_nested_run_layout_and_replay(tmp_path: Path) -> None:
    run_dir = _record_run(tmp_path / "runs")

    summary = load_run_summary(tmp_path / "runs", "experiment/seed_00")
    assert summary["available_runs"] == ["experiment/seed_00"]
    game = summary["games"][0]
    assert (game["game_id"], game["status"], game["levels_completed"], game["total_levels"]) == (
        "ar25-test",
        "gave_up",
        1,
        2,
    )
    assert game["final_score"] == 37.5
    assert game["final_arc_score"] == 37.5
    assert game["rollout_return"] == 0.5
    assert game["generated_tokens"] == 71

    full = load_game_full(tmp_path / "runs", "experiment/seed_00", 0)
    assert full["elapsed_s"] is not None and full["elapsed_s"] >= 0
    assert [event["type"] for event in full["events"]] == ["initial", "analysis", "action", "analysis", "action"]
    elapsed = [event["elapsed_s"] for event in full["events"]]
    assert elapsed[0] == 0.0
    assert elapsed == sorted(elapsed)
    sections = full["events"][1]["transcript_sections"]
    assert [section["label"] for section in sections] == [
        "SYSTEM PROMPT",
        "USER PROMPT",
        "THINKING",
        "TOOL CALL: python",
        "TOOL RESULT: python",
        "USER PROMPT",
        "THINKING",
        "TOOL CALL: python",
        "TOOL RESULT: python",
    ]
    assert "current_frame.level" in sections[3]["content"]
    assert sum(section.get("generated_tokens", 0) for section in sections) == 40
    assert all(section.get("elapsed_s", 0) > 0 for section in sections)
    assert full["events"][1]["generated_tokens"] == 40
    assert full["events"][1]["response_count"] == 2
    assert sections[-1]["action_num"] == 1
    assert full["events"][2]["action_display"] == "UP"
    assert full["events"][2]["board"] == [[1, 1, 1]] * 3
    assert full["events"][3]["board"] == [[2, 2, 2]] * 3
    actions = [event for event in full["events"] if event["type"] == "action"]
    assert [event["level"] for event in actions] == [1, 1]
    assert [event["reward"] for event in actions] == [0.0, 0.5]
    assert [event["arc_agi_score"] for event in actions] == [0.0, pytest.approx(100 / 3)]
    assert actions[-1]["final_arc_score"] == 37.5
    assert resolve_run_dir(tmp_path / "runs", "experiment/seed_00") == run_dir.resolve()


def test_live_run_is_visible_before_benchmark_is_written(tmp_path: Path) -> None:
    """Append-only spans make an active run viewable before final bookkeeping."""
    _record_run(tmp_path / "runs", write_benchmark=False)

    summary = load_run_summary(tmp_path / "runs", "experiment/seed_00")
    assert summary["available_runs"] == ["experiment/seed_00"]
    game = summary["games"][0]
    assert (game["game_id"], game["status"], game["levels_completed"], game["total_levels"]) == (
        "ar25-test",
        "level_won",
        1,
        2,
    )
    assert game["generated_tokens"] == 71
    assert game["final_arc_score"] is None
    assert game["arc_agi_score"] == pytest.approx(100 / 3)
    assert game["lastEvent"]["score"] == pytest.approx(100 / 3)

    full = load_game_full(tmp_path / "runs", "experiment/seed_00", 0)
    assert full["elapsed_s"] is not None and full["elapsed_s"] >= 0
    assert [event["type"] for event in full["events"]] == ["initial", "analysis", "action", "analysis", "action"]
    assert full["events"][-1]["arc_agi_score"] == pytest.approx(100 / 3)


@pytest.mark.parametrize("write_benchmark", [True, False])
def test_an_archived_run_replays_as_the_loose_one_does(tmp_path: Path, write_benchmark: bool) -> None:
    """A run on scratch leaves its streams and blocks in the job dir's zip
    parts and no ``games/`` dir: discovery, the summary and the full replay
    read the same from the parts as from loose files, finished or still live
    (every part holds the run stream, so a live run is found by its parts)."""
    loose = _record_run(tmp_path / "loose", write_benchmark=write_benchmark)
    archived = tmp_path / "archived" / "experiment" / "seed_00"
    scratch = tmp_path / "scratch"
    shutil.copytree(loose, scratch)
    archived.mkdir(parents=True)
    if write_benchmark:
        shutil.move(scratch / "benchmark.json", archived / "benchmark.json")
    assert DiagnosticsArchiver(scratch, archived).finish() is not None
    assert not (archived / "games").exists() and not blocks_dir(archived).exists()

    run = "experiment/seed_00"
    assert load_run_summary(tmp_path / "archived", run) == load_run_summary(tmp_path / "loose", run)
    assert load_game_full(tmp_path / "archived", run, 0) == load_game_full(tmp_path / "loose", run, 0)


def test_live_score_is_computed_at_each_replay_block(tmp_path: Path) -> None:
    """A running level exposes its score without borrowing the eventual final score."""
    run_dir = _record_run(tmp_path / "runs")
    benchmark_path = run_dir / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    game_record = benchmark["game_runs"][0]
    game_record["state"] = "playing"
    game_record["final_score"] = None
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

    summary = load_run_summary(tmp_path / "runs", "experiment/seed_00")
    game = summary["games"][0]
    assert game["final_arc_score"] is None
    assert game["arc_agi_score"] == pytest.approx(100 / 3)
    assert game["lastEvent"]["score"] == pytest.approx(100 / 3)

    full = load_game_full(tmp_path / "runs", "experiment/seed_00", 0)
    assert [event["arc_agi_score"] for event in full["events"]] == [
        0.0,
        0.0,
        0.0,
        0.0,
        pytest.approx(100 / 3),
    ]
    assert "final_arc_score" not in full["events"][-1] or full["events"][-1]["final_arc_score"] is None


def test_terminal_actionless_move_stays_on_the_last_played_level(tmp_path: Path) -> None:
    """Final bookkeeping after a win must not label the final board as level one."""
    _record_run(tmp_path / "runs", trailing_actionless_move=True)

    full = load_game_full(tmp_path / "runs", "experiment/seed_00", 0)
    final_event = full["events"][-1]
    assert final_event["type"] == "analysis"
    assert final_event["level"] == 2
    assert final_event["board"] == [[4, 4, 4]] * 3


def test_interactive_routes_are_separate_and_confined(tmp_path: Path) -> None:
    _record_run(tmp_path / "runs")

    app = resolve_interactive_request(tmp_path / "runs", "/_replay/")
    assert app is not None and app[0] == 200 and b"ARC3 Replay Viewer" in app[1]
    # The shell must speak to the reserved API prefix, not the file space,
    # and honor the ?run= parameter the listing's replay links carry.
    assert b'"/_api/run"' in app[1] and b'"/_api/game"' in app[1]
    assert b'URLSearchParams(window.location.search).get("run")' in app[1]

    api = resolve_interactive_request(tmp_path / "runs", "/_api/run", "run=experiment%2Fseed_00")
    assert api is not None and api[0] == 200
    assert json.loads(api[1])["selected_run"] == "experiment/seed_00"

    game_api = resolve_interactive_request(tmp_path / "runs", "/_api/game", "run=experiment%2Fseed_00&index=0")
    assert game_api is not None and game_api[0] == 200
    assert any(event["type"] == "analysis" for event in json.loads(game_api[1])["events"])

    escaped = resolve_interactive_request(tmp_path / "runs", "/_api/run", "run=..%2Foutside")
    assert escaped is not None and escaped[0] == 404

    # Everything outside the two reserved prefixes falls through to the
    # canonical resolver: the server root, run paths, and viewer pages.
    assert resolve_interactive_request(tmp_path / "runs", "/") is None
    assert resolve_interactive_request(tmp_path / "runs", "/index.html") is None
    assert resolve_interactive_request(tmp_path / "runs", "/experiment/seed_00/viewer/index.html") is None


def test_the_image_route_serves_a_prompt_image_as_png(tmp_path: Path) -> None:
    run_dir = _record_run(tmp_path / "runs")
    store = BlockStore(blocks_dir(run_dir))
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    image = store.store({"image_b64": base64.b64encode(pixels.tobytes()).decode(), "shape": [2, 2, 3]}, "json")
    text = store.store("not an image", "json")

    def get(query: str) -> tuple[int, bytes, str]:
        response = resolve_interactive_request(tmp_path / "runs", "/_api/image", query)
        assert response is not None
        return response

    status, body, content_type = get(f"run=experiment%2Fseed_00&block={image}")
    assert status == 200 and content_type == "image/png"
    assert (np.asarray(Image.open(io.BytesIO(body))) == pixels).all()
    assert get(f"run=experiment%2Fseed_00&block={text}")[0] == 404
    assert get(f"run=experiment%2Fseed_00&block={'0' * 64}")[0] == 404
    assert get("run=experiment%2Fseed_00&block=../benchmark.json")[0] == 400
    assert get(f"run=..%2Foutside&block={image}")[0] == 404


def test_discovery_survives_a_permission_denied_sibling(tmp_path: Path) -> None:
    """Shared trees hold other users' unreadable dirs: not replayable,
    never an API error."""
    _record_run(tmp_path / "runs")
    locked = tmp_path / "runs" / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        api = resolve_interactive_request(tmp_path / "runs", "/_api/run", "run=experiment%2Fseed_00")
        assert api is not None and api[0] == 200
        assert json.loads(api[1])["available_runs"] == ["experiment/seed_00"]
    finally:
        locked.chmod(0o700)
