from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import GameAction, GameState

from re_arc.prolong_player import (
    ActionQueue,
    FullInteractionLog,
    _build_parser,
    build_prolong_codex_command,
    parse_actions_json,
    run_prolong_player,
)
from re_arc.prolong_runtime import ProlongGameSession


def _observation(*, x: int, level: int, state: GameState, animated: bool = True):
    before = np.zeros((64, 64), dtype=np.int8)
    before[10, max(0, x - 1)] = 9
    after = np.zeros((64, 64), dtype=np.int8)
    after[10, x] = 9
    return SimpleNamespace(
        frame=[before, after] if animated else [after],
        levels_completed=level,
        win_levels=2,
        state=state,
        full_reset=False,
        available_actions=[GameAction.ACTION4],
    )


class _TwoLevelEnvironment:
    action_space = (GameAction.ACTION4,)

    def __init__(self):
        self.level = 0

    def reset(self):
        self.level = 0
        return _observation(x=1, level=0, state=GameState.NOT_FINISHED, animated=False)

    def step(self, action, data=None):
        assert action == GameAction.ACTION4
        assert data == {}
        self.level += 1
        won = self.level == 2
        return (
            _observation(x=self.level + 1, level=self.level, state=GameState.WIN if won else GameState.NOT_FINISHED),
            1.0,
            won,
            {
                "level_transition": True,
                "level_reset": False,
                "current_level": self.level,
                "num_levels": 2,
                "actions_in_level": 0,
            },
        )


class _ResettableEnvironment:
    action_space = (GameAction.RESET, GameAction.ACTION4)

    def reset(self):
        return _observation(x=1, level=0, state=GameState.NOT_FINISHED, animated=False)

    def step(self, action, data=None):
        assert data == {}
        if action == GameAction.RESET:
            return self.reset(), 0.0, False, {"level_reset": True}
        assert action == GameAction.ACTION4
        return _observation(x=2, level=0, state=GameState.GAME_OVER), -1.0, False, {}


def test_parse_actions_json_accepts_reference_syntax_and_caps_plan(tmp_path: Path):
    actions_path = tmp_path / "actions.json"
    actions_path.write_text(json.dumps({"actions": ["ACTION4", "ACTION6(30,40)", "ACTION1", "BAD"]}), encoding="utf-8")

    assert parse_actions_json(actions_path, action_cap=3) == [
        {"name": "RIGHT"},
        {"name": "CLICK", "x": 30, "y": 40},
        {"name": "UP"},
    ]


def test_action_queue_tracks_plan_position_and_can_invalidate():
    queue = ActionQueue()
    queue.load([{"name": "RIGHT"}, {"name": "DOWN"}])

    assert queue.pop() == {"name": "RIGHT"}
    assert queue.plan_index == 1
    assert queue.plan_size == 2
    assert len(queue) == 1

    queue.clear()

    assert not queue
    assert queue.plan_index == 0
    assert queue.plan_size == 0


def test_full_log_preserves_animation_frames_and_outcome(tmp_path: Path):
    initial = np.zeros((64, 64), dtype=np.int8)
    moved = initial.copy()
    moved[4, 5] = 9
    log = FullInteractionLog(tmp_path / "logs.txt")
    log.write_initial(grid=initial, level=0, state="NOT_FINISHED")
    log.write_action(
        action_number=1,
        level_before=0,
        level_after=0,
        state_before="NOT_FINISHED",
        state_after="NOT_FINISHED",
        action={"name": "RIGHT"},
        analysis="[PLAN] Test one move.",
        plan_index=1,
        plan_size=1,
        stopped_reason="sequence_completed",
        automatic_resets=0,
        grids=[initial, moved],
    )

    text = (tmp_path / "logs.txt").read_text(encoding="utf-8")
    assert "[INITIAL BOARD STATE]" in text
    assert "[ANALYSIS]\n[PLAN] Test one move." in text
    assert "Tool Call: ACTION4" in text
    assert "[frame 1/1]" in text
    assert "[settled]" in text
    assert text.endswith("\n\n")


def test_prolong_command_uses_local_coding_workspace_without_image(tmp_path: Path):
    command = build_prolong_codex_command(
        codex_bin="codex",
        workspace=tmp_path,
        prompt="Read logs.txt.",
        model="gpt-test",
        reasoning_effort="medium",
        sandbox_mode="workspace-write",
    )

    assert "--image" not in command
    assert "sandbox_workspace_write.network_access=false" in command
    assert command[-1] == "Read logs.txt."


def test_prolong_runtime_automatically_resets_game_over(tmp_path: Path):
    session = ProlongGameSession(
        env=_ResettableEnvironment(),
        trace_path=tmp_path / "trace.jsonl",
        game_id="resettable-0001",
        seed=0,
        max_batch_actions=1,
        max_episode_actions=3,
    )

    result = session.execute({"reason": "Probe the only move.", "actions": [{"name": "RIGHT"}]})

    assert result["stopped_reason"] == "game_over_reset"
    assert result["automatic_resets"] == 1
    assert session.primitive_actions == 2
    assert session.finished is False
    assert len(session.last_transition_grids) == 3


def test_prolong_cli_defaults():
    args = _build_parser().parse_args(["fake-0001"])

    assert args.action_cap == 20
    assert args.max_episode_actions == 500
    assert args.timeout_seconds == 1200.0
    assert args.resume_from is None


@pytest.mark.skipif(os.name == "nt", reason="The fake Codex executable uses a POSIX shebang.")
def test_prolong_runner_flushes_remaining_plan_on_level_transition(tmp_path: Path):
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        """#!/usr/bin/env python3
import json
from pathlib import Path

counter = Path("scratch/calls.txt")
call = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(call))
actions = ["ACTION4", "ACTION4"] if call == 1 else ["ACTION4"]
Path("actions.json").write_text(json.dumps({"actions": actions}))
print(json.dumps({"type": "thread.started", "thread_id": "fake-prolong-thread"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": f"[PLAN] call {call}"}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10 * call, "output_tokens": 2 * call}}))
""",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    result = run_prolong_player(
        game_id="fake-0001",
        seed=0,
        output_root=tmp_path / "runs",
        run_id="prolong-run",
        codex_bin=str(fake_codex),
        timeout_seconds=10,
        env_factory=_TwoLevelEnvironment,
        fps=8,
    )

    run_dir = Path(result["run_dir"])
    assert result["final_state"] == "WIN"
    assert result["primitive_actions"] == 2
    assert result["plans"] == 2
    assert result["invalidated_plans"] == 1
    assert result["analyzer_calls"] == 2
    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["harness"] == "pro-long-v1"
    log_text = (run_dir / "workspace" / "logs.txt").read_text(encoding="utf-8")
    assert "[PLAN] call 1" in log_text
    assert "[PLAN] call 2" in log_text
    assert "Levels completed: 2" in log_text
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["harness"] == "pro-long-v1"
    assert manifest["action_cap"] == 20


@pytest.mark.skipif(os.name == "nt", reason="The fake Codex executable uses a POSIX shebang.")
def test_prolong_runner_resumes_by_replaying_prior_actions(tmp_path: Path):
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        """#!/usr/bin/env python3
import json
from pathlib import Path

counter = Path("scratch/calls.txt")
call = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(call))
Path("actions.json").write_text(json.dumps({"actions": ["ACTION4"]}))
print(json.dumps({"type": "thread.started", "thread_id": "fake-resume-thread"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": f"[PLAN] call {call}"}}))
""",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    first = run_prolong_player(
        game_id="fake-0001",
        output_root=tmp_path / "runs",
        run_id="first",
        codex_bin=str(fake_codex),
        max_episode_actions=1,
        timeout_seconds=10,
        env_factory=_TwoLevelEnvironment,
        fps=8,
    )
    resumed = run_prolong_player(
        game_id="fake-0001",
        output_root=tmp_path / "runs",
        run_id="resumed",
        resume_from=Path(first["run_dir"]),
        codex_bin=str(fake_codex),
        max_episode_actions=2,
        timeout_seconds=10,
        env_factory=_TwoLevelEnvironment,
        fps=8,
    )

    assert resumed["final_state"] == "WIN"
    assert resumed["primitive_actions"] == 2
    assert resumed["replayed_actions"] == 1
    assert resumed["resume_from"] == first["run_dir"]
    resumed_dir = Path(resumed["run_dir"])
    assert (resumed_dir / "workspace" / "scratch" / "calls.txt").read_text() == "2"
    assert "resume fake-resume-thread" in " ".join(
        json.loads((resumed_dir / "manifest.json").read_text())["last_command"]
    )
