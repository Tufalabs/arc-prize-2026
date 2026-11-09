from __future__ import annotations

import json
import socket
import threading
import urllib.request
from types import SimpleNamespace

import pytest

from re_arc.cli import (
    _action_name_from_id,
    _print_recorded_actions,
    _register_worker_repo_packages,
    _replay_index_html,
    _ReplayHandler,
    _ReplaySession,
    _run_dsl_policy,
    _run_replay_ui,
    _serialize_replay_observation,
    _start_http_server,
    _unwrap_attr,
    _WebSession,
)


def _args(*, gif: str, replay: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        config="config.env",
        game="identify_the_agent-0002",
        policy="dsl",
        host=None,
        port=None,
        replay=replay,
        replay_dir=None,
        max_actions=500,
        seed=3,
        gif=gif,
        fps=4,
        scale=2,
        separator=1,
        list_games=False,
    )


def test_dsl_gif_run_writes_sibling_replay_trace(tmp_path):
    gif_path = tmp_path / "identify.gif"

    exit_code = _run_dsl_policy(_args(gif=str(gif_path)))

    replay_path = gif_path.with_suffix(".replay.json")
    assert exit_code == 0
    assert gif_path.exists()
    assert replay_path.exists()

    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["metadata"]["game_id"] == "identify_the_agent-0002"
    assert payload["metadata"]["policy"] == "dsl"
    assert payload["metadata"]["actions"] == len(payload["steps"])
    assert payload["initial_observation"]["state"] == "NOT_FINISHED"
    assert "rows" in payload["initial_observation"]["layers"][0]
    assert "cells" not in payload["initial_observation"]["layers"][0]
    assert payload["steps"][-1]["observation"]["state"] == "WIN"
    assert payload["steps"][-1]["action"]["name"]


def test_replay_observation_serializes_animation_frames_compactly():
    observation = SimpleNamespace(
        state=SimpleNamespace(name="NOT_FINISHED"),
        levels_completed=0,
        win_levels=1,
        frame=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    )

    payload = _serialize_replay_observation(observation)

    assert payload["layers"] == [{"width": 2, "height": 2, "rows": ["56", "78"]}]
    assert payload["animation_frames"] == [
        {"width": 2, "height": 2, "rows": ["12", "34"]},
        {"width": 2, "height": 2, "rows": ["56", "78"]},
    ]


def test_replay_session_loads_trace_and_html_template_mentions_viewer(tmp_path):
    replay_path = tmp_path / "sample.replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "version": 2,
                "metadata": {"game_id": "identify_the_agent-0002", "seed": 3, "policy": "dsl"},
                "initial_observation": {"state": "NOT_FINISHED", "levels_completed": 0, "win_levels": 1, "layers": []},
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    session = _ReplaySession(replay_path)

    assert session.payload()["metadata"]["game_id"] == "identify_the_agent-0002"
    assert "ARC Replay Viewer" in _replay_index_html()


def test_replay_session_lists_directory_and_switches_between_replays(tmp_path):
    first = tmp_path / "a.replay.json"
    second = tmp_path / "b.replay.json"
    first.write_text(
        json.dumps(
            {"version": 2, "metadata": {"game_id": "alpha-0001"}, "initial_observation": {"layers": []}, "steps": []}
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {"version": 2, "metadata": {"game_id": "beta-0001"}, "initial_observation": {"layers": []}, "steps": []}
        ),
        encoding="utf-8",
    )

    session = _ReplaySession(replay_dir=tmp_path)
    options = session.replay_options()

    assert [item["label"] for item in options["replays"]] == ["alpha-0001", "beta-0001"]
    assert session.payload("b.replay.json")["metadata"]["game_id"] == "beta-0001"


def test_replay_session_exposes_review_idea_context(tmp_path):
    review_dir = tmp_path / "review"
    replay_dir = review_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / "a.replay.json"
    replay_path.write_text(
        json.dumps(
            {"version": 2, "metadata": {"game_id": "alpha-0001"}, "initial_observation": {"layers": []}, "steps": []}
        ),
        encoding="utf-8",
    )
    (review_dir / "manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "review_replay_path": str(replay_path),
                        "idea_id": "idea-a",
                        "idea_title": "Alpha Idea",
                        "idea_description": "Match switches in sequence.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "decisions.json").write_text(
        json.dumps(
            {
                "submitted": False,
                "entries": [
                    {
                        "review_replay_name": replay_path.name,
                        "idea_id": "idea-a",
                        "game_id": "alpha-0001",
                        "decision": "pending",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    session = _ReplaySession(replay_dir=replay_dir)
    replay_payload = session.payload()

    assert replay_payload["review_context"] == {
        "idea_id": "idea-a",
        "idea_title": "Alpha Idea",
        "idea_description": "Match switches in sequence.",
    }
    assert replay_payload["play_url"] == ""
    assert session.replay_options()["replays"][0]["idea_title"] == "Alpha Idea"


def test_replay_session_exposes_play_url_for_review_game(tmp_path):
    review_dir = tmp_path / "review"
    replay_dir = review_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / "fresh.replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "version": 2,
                "metadata": {"game_id": "fresh_game-0001"},
                "initial_observation": {"layers": []},
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    worker_repo = tmp_path / "worker" / "repo"
    (worker_repo / "re_arc" / "environment_files" / "fresh_game" / "0001").mkdir(parents=True, exist_ok=True)
    (review_dir / "manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "review_replay_path": str(replay_path),
                        "idea_id": "idea-a",
                        "game_id": "fresh_game-0001",
                        "worker_repo_path": str(worker_repo),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "decisions.json").write_text(
        json.dumps(
            {
                "submitted": False,
                "entries": [
                    {
                        "review_replay_name": replay_path.name,
                        "idea_id": "idea-a",
                        "game_id": "fresh_game-0001",
                        "decision": "pending",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    session = _ReplaySession(replay_dir=replay_dir)

    assert session.payload()["play_url"] == "/play?name=fresh.replay.json"


def test_replay_session_submit_review_uses_config_root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    review_dir = tmp_path / "review"
    replay_dir = review_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / "fresh.replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "version": 2,
                "metadata": {"game_id": "fresh_game-0001"},
                "initial_observation": {"layers": []},
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "manifest.json").write_text(
        json.dumps({"entries": [{"review_replay_path": str(replay_path), "idea_id": "idea-a"}]}), encoding="utf-8"
    )
    (review_dir / "decisions.json").write_text(
        json.dumps(
            {
                "submitted": False,
                "entries": [
                    {
                        "review_replay_name": replay_path.name,
                        "idea_id": "idea-a",
                        "game_id": "fresh_game-0001",
                        "decision": "accepted",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config_path = root / "config.env"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_submit_review(*, review_dir, root=None):
        captured["review_dir"] = review_dir
        captured["root"] = root
        return {"promoted_count": 1, "promoted": []}

    monkeypatch.setattr("pipeline.review.submit_review", fake_submit_review)

    session = _ReplaySession(replay_dir=replay_dir, config={"__CONFIG_PATH__": str(config_path)})
    payload = session.submit_review()

    assert captured["review_dir"] == review_dir
    assert captured["root"] == root
    assert payload["submission"]["promoted_count"] == 1


def test_action_name_mapping_uses_human_labels_for_standard_actions():
    assert _action_name_from_id(1) == "UP"
    assert _action_name_from_id(2) == "DOWN"
    assert _action_name_from_id(3) == "LEFT"
    assert _action_name_from_id(4) == "RIGHT"
    assert _action_name_from_id(5) == "SPACE"
    assert _action_name_from_id(6) == "CLICK"


def test_replay_template_mentions_click_highlight():
    html = _replay_index_html()

    assert "highlightClick" in html
    assert "clickOutlineColor" in html
    assert "animation_frames" in html
    assert "renderAnimatedObservation" in html
    assert "#00e5ff" in html
    assert "REPLAY_DIGITS" in html
    assert "layer.rows" in html
    assert "layer.cells" not in html


def test_web_player_blurs_game_select_after_switch():
    from re_arc.cli import _web_index_html

    html = _web_index_html()

    assert "gameSelectEl.blur();" in html


def test_web_player_can_target_custom_api_base():
    from re_arc.cli import _web_index_html

    html = _web_index_html(api_base="/play/api")

    assert 'const API_BASE = "/play/api";' in html
    assert 'apiUrl("/state", { resample_random: 1 })' in html


def test_web_player_uses_turn_actions_to_gate_controls_and_mouse_status():
    from re_arc.cli import _web_index_html

    html = _web_index_html()

    assert "<label>Mouse</label>" not in html
    assert 'id="mouseLabel"' in html
    assert "Mouse: unavailable" in html
    assert "function syncControls()" in html
    assert "button.disabled = disableInteractive || !available;" in html
    assert "const avail = state ? state.available_actions : null;" in html
    assert ".kbd-btn:disabled" in html
    assert "resetLevelBtn.disabled = disableInteractive || !resetAvailable();" in html
    assert 'mouseLabel.textContent = `Mouse: ${pointerEnabled ? "available" : "unavailable"}`;' in html
    assert 'canvas.style.cursor = pointerEnabled ? "crosshair" : "not-allowed";' in html


def test_web_player_template_polls_hot_reload_and_renders_level_buttons():
    from re_arc.cli import _web_index_html

    html = _web_index_html()

    assert 'id="levelButtons"' in html
    assert 'apiUrl("/reload-status")' in html
    assert 'apiUrl("/reload")' in html
    assert "reload_available" in html
    assert 'apiUrl("/jump-level")' in html
    assert "function renderLevelButtons" in html


def test_web_player_template_can_record_actions():
    from re_arc.cli import _web_index_html

    html = _web_index_html()

    assert 'id="recordBtn"' in html
    assert 'apiUrl("/recording")' in html
    assert "function recordAction" in html
    assert "recordAction(actionId, extra);" in html
    assert "levels: recordedLevels" in html
    assert "Stop recording" in html


def test_web_player_template_can_copy_ascii_frame():
    from re_arc.cli import _web_index_html

    html = _web_index_html()

    assert 'id="copyAsciiBtn"' in html
    assert 'id="asciiFrame"' not in html
    assert "function layerToAscii" in html
    assert "function asciiToLayer" not in html
    assert "function copyAsciiFrame" in html
    assert "navigator.clipboard.writeText(currentAsciiFrame)" in html


def test_print_recorded_actions_normalizes_and_prints(capsys):
    payload = _print_recorded_actions("paint_snakes-0001", [[4, {}], [6, {"x": "12", "y": 7}]], [2, "2"])

    assert payload == {"printed": True, "actions": 2}
    out = capsys.readouterr().out
    assert "Recorded solution: game=paint_snakes-0001 level=2 actions=2" in out
    assert out.rstrip().endswith("4 6(x=12,y=7)")


def test_web_session_reload_status_does_not_reload(monkeypatch):
    import re_arc.cli as cli

    session = _WebSession(
        {"ENVIRONMENTS_DIR": "re_arc/environment_files", "OPERATION_MODE": "OFFLINE"}, "identify_the_agent-0002", seed=0
    )
    original_token = session.reload_status()["reload_token"]

    monkeypatch.setattr(cli, "_environment_watch_signature", lambda _config: (999, 1))

    status = session.reload_status()

    assert status["reload_available"] is True
    assert status["reloaded"] is False
    assert status["reload_token"] == original_token

    state = session.state()

    assert state["reload_token"] == original_token

    reloaded_state = session.reload()

    assert reloaded_state["reload_token"] == [999, 1]


def test_replay_handler_exposes_play_reload_route():
    class FakePlaySession:
        def reload(self):
            return {"reloaded": True, "reload_token": [1, 2], "game_id": "toy-0001"}

    class FakeReplaySession:
        def __init__(self):
            self.names: list[str] = []

        def _play_session_for_replay(self, name):
            self.names.append(name)
            return FakePlaySession()

    session = FakeReplaySession()
    server = _start_http_server(host="127.0.0.1", port=0, handler=_ReplayHandler, server_name="Replay UI")
    server.session = session
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = urllib.request.Request(
            f"http://{host}:{port}/play/api/reload?name=sample.replay.json",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status = response.status
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert status == 200
    assert payload == {"reloaded": True, "reload_token": [1, 2], "game_id": "toy-0001"}
    assert session.names == ["sample.replay.json"]


def test_web_session_can_jump_to_level():
    session = _WebSession(
        {"ENVIRONMENTS_DIR": "re_arc/environment_files", "OPERATION_MODE": "OFFLINE"}, "identify_the_agent-0002", seed=0
    )

    payload = session.jump_level(1)

    assert payload["levels_completed"] == 1
    assert payload["win_levels"] == 3
    assert payload["episode_reward"] == pytest.approx(1 / 3)


def test_web_session_undo_after_level_jump_stays_on_selected_level():
    session = _WebSession(
        {"ENVIRONMENTS_DIR": "re_arc/environment_files", "OPERATION_MODE": "OFFLINE"}, "identify_the_agent-0002", seed=0
    )

    jumped = session.jump_level(1)
    moved = session.step(3, {})
    undone = session.undo()

    assert jumped["levels_completed"] == 1
    assert moved["levels_completed"] == 1
    assert moved["steps"] == 1
    assert undone["levels_completed"] == 1
    assert undone["steps"] == 0
    assert undone["episode_reward"] == pytest.approx(1 / 3)


def test_web_session_undo_uses_paint_snakes_game_undo_action():
    session = _WebSession(
        {"ENVIRONMENTS_DIR": "re_arc/environment_files", "OPERATION_MODE": "OFFLINE"}, "paint_snakes-0001", seed=0
    )
    game = _unwrap_attr(session._env, "_game")
    start_energy = game.steps_left
    start_pos = tuple(game.snakes[game.active_snake]["pos"])

    session.step(4, {})
    assert tuple(game.snakes[game.active_snake]["pos"]) != start_pos

    undone = session.undo()

    assert tuple(game.snakes[game.active_snake]["pos"]) == start_pos
    assert game.steps_left == start_energy - 2
    assert undone["steps"] == 2

    session.undo()

    assert tuple(game.snakes[game.active_snake]["pos"]) == start_pos
    assert game.steps_left == start_energy - 2


def test_web_session_undo_uses_anchor_drift_game_undo_action():
    session = _WebSession(
        {"ENVIRONMENTS_DIR": "re_arc/environment_files", "OPERATION_MODE": "OFFLINE"}, "anchor_drift-0001", seed=0
    )
    game = _unwrap_attr(session._env, "_game")
    start_energy = game.remaining_steps
    start_sailor = game.sailor

    session.step(4, {})
    assert game.sailor != start_sailor

    undone = session.undo()

    assert game.sailor == start_sailor
    assert game.remaining_steps == start_energy - 2
    assert undone["steps"] == 2

    session.undo()

    assert game.sailor == start_sailor
    assert game.remaining_steps == start_energy - 2


def test_replay_template_mentions_play_button():
    html = _replay_index_html()

    assert 'id="playGameBtn"' in html
    assert 'window.open(replay.play_url, "_blank", "noopener")' in html


def test_register_worker_repo_packages_extends_loaded_package_paths(tmp_path):
    import re_arc
    import re_arc.dsl
    import re_arc.dsl.agents

    worker_repo = tmp_path / "worker" / "repo"
    (worker_repo / "re_arc" / "dsl" / "agents").mkdir(parents=True, exist_ok=True)

    _register_worker_repo_packages(worker_repo)

    assert str(worker_repo / "re_arc") in list(re_arc.__path__)
    assert str(worker_repo / "re_arc" / "dsl") in list(re_arc.dsl.__path__)
    assert str(worker_repo / "re_arc" / "dsl" / "agents") in list(re_arc.dsl.agents.__path__)


def test_replay_ui_reports_address_in_use_with_actionable_message(tmp_path):
    replay_path = tmp_path / "sample.replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "version": 2,
                "metadata": {"game_id": "identify_the_agent-0002"},
                "initial_observation": {"layers": []},
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.env"
    config_path.write_text("", encoding="utf-8")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = int(occupied.getsockname()[1])
        args = SimpleNamespace(
            config=str(config_path), host="127.0.0.1", port=port, replay=str(replay_path), replay_dir=None
        )

        with pytest.raises(RuntimeError) as exc_info:
            _run_replay_ui(args)

        message = str(exc_info.value)
        assert f"Replay UI could not bind to http://127.0.0.1:{port}/" in message
        assert "already in use" in message
        assert "--port" in message
