from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import prolong_panel


def test_default_panel_contains_all_short_action7_games() -> None:
    panel = prolong_panel.load_panel(prolong_panel.DEFAULT_PANEL)

    assert len(panel["games"]) == 27
    assert panel["games"][0] == "balanced_reservoirs-0001"
    assert panel["model"] == "gpt-5.6-sol"
    assert panel["concurrency"] == 8
    assert panel["timeout_seconds"] == 1800


def test_archived_panel_is_not_runnable(tmp_path: Path) -> None:
    panel = tmp_path / "panel.json"
    panel.write_text('{"archived": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="archived provenance"):
        prolong_panel.load_panel(panel)


def test_first_action_written_parses_compact_jsonl(tmp_path: Path) -> None:
    game_id = "beam_compass-0001"
    game = tmp_path / game_id
    game.mkdir()
    trace = game / "trace.jsonl"
    trace.write_text(
        '{"type":"run_started"}\nnot-yet-valid-json\n{"type":"decision","decision_id":1}\n', encoding="utf-8"
    )

    assert prolong_panel.first_action_written(tmp_path, game_id)


def test_first_action_written_rejects_nondecision_events(tmp_path: Path) -> None:
    game_id = "beam_compass-0001"
    game = tmp_path / game_id
    game.mkdir()
    (game / "trace.jsonl").write_text('{"type":"run_started"}\n', encoding="utf-8")

    assert not prolong_panel.first_action_written(tmp_path, game_id)


def test_package_writes_markdown_report_and_records_missing_result(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    submission = tmp_path / "submission"
    game = raw / "balanced_reservoirs-0001"
    (game / "workspace").mkdir(parents=True)
    (game / "workspace/logs.txt").write_text("[INITIAL BOARD STATE]\n", encoding="utf-8")
    panel = {"protocol": "test-panel", "description": "Test panel", "games": ["balanced_reservoirs-0001"]}
    metadata = {
        "evaluator_branch": "test-branch",
        "evaluator_commit": "abc123",
        "codex_version": "codex-test",
        "model": "gpt-test",
        "reasoning_effort": "medium",
    }

    prolong_panel.package(panel, raw, submission, {"balanced_reservoirs-0001": 130}, metadata)

    assert (submission / "games/balanced_reservoirs-0001/logs.txt").is_file()
    failure = json.loads((submission / "games/balanced_reservoirs-0001/failure.json").read_text(encoding="utf-8"))
    assert failure["termination_reason"] == "infrastructure_failure"
    report = (submission / "README.md").read_text(encoding="utf-8")
    assert "# PRO-LONG results: Test panel" in report
    assert "[logs.txt](games/balanced_reservoirs-0001/logs.txt)" in report


def test_verify_bundle_rejects_credential_markers(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text('{"games": []}\n', encoding="utf-8")
    (tmp_path / "codex.stderr.log").write_text("CODEX_ACCESS_TOKEN=secret\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Credential marker"):
        prolong_panel.verify_bundle(tmp_path)


def test_verify_bundle_rejects_unlabeled_tokens(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text('{"games": []}\n', encoding="utf-8")
    (tmp_path / "codex.stderr.log").write_text("sk-abcdefghijklmnop\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Sensitive value"):
        prolong_panel.verify_bundle(tmp_path)


def test_package_redacts_sensitive_values_and_omits_raw_artifacts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    submission = tmp_path / "submission"
    game = raw / "balanced_reservoirs-0001"
    (game / "workspace").mkdir(parents=True)
    (game / "workspace/logs.txt").write_text(
        "tester@example.com /Users/tester/repo HOME=/Users/tester PATH=/secret/bin sk-abcdefghijklmnop\n",
        encoding="utf-8",
    )
    (game / "codex.events.jsonl").write_text('{"private": true}\n', encoding="utf-8")
    (game / "manifest.json").write_text('{"HOME": "/Users/tester"}\n', encoding="utf-8")
    panel = {"protocol": "test-panel", "description": "Test panel", "games": ["balanced_reservoirs-0001"]}
    metadata = {
        "evaluator_branch": "test-branch",
        "evaluator_commit": "abc123",
        "codex_version": "codex-test",
        "model": "gpt-test",
        "reasoning_effort": "medium",
    }

    prolong_panel.package(panel, raw, submission, {"balanced_reservoirs-0001": 130}, metadata)

    published = submission / "games/balanced_reservoirs-0001"
    text = (published / "logs.txt").read_text(encoding="utf-8")
    assert "tester@example.com" not in text
    assert "/Users/tester" not in text
    assert "sk-abcdefghijklmnop" not in text
    assert not (published / "codex.events.jsonl").exists()
    assert not (published / "manifest.json").exists()


def test_verify_bundle_rejects_raw_evaluator_artifacts(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text('{"games": []}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Raw evaluator artifact"):
        prolong_panel.verify_bundle(tmp_path)


def test_evaluate_automatically_submits_packaged_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    panel = {
        "protocol": "test-panel",
        "description": "Test panel",
        "games": ["balanced_reservoirs-0001"],
        "concurrency": 1,
        "model": "gpt-test",
        "reasoning_effort": "medium",
        "seed": 0,
        "timeout_seconds": 1,
        "action_cap": 1,
        "max_episode_actions": 1,
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text("{}\n", encoding="utf-8")
    submitted: list[argparse.Namespace] = []

    def fake_run_text(*command: str, check: bool = True) -> str:
        del check
        return "test-branch" if "branch" in command else "abc"

    def fake_codex_preflight(*, allow_login: bool) -> tuple[str, str]:
        assert allow_login
        return "codex", "codex-test"

    monkeypatch.setattr(prolong_panel, "load_panel", lambda _path: panel)
    monkeypatch.setattr(prolong_panel, "run_text", fake_run_text)
    monkeypatch.setattr(prolong_panel, "codex_preflight", fake_codex_preflight)
    monkeypatch.setattr(prolong_panel, "result_root", lambda _panel: tmp_path / "results")
    monkeypatch.setattr(prolong_panel, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(prolong_panel, "supervise", lambda *_args: {"balanced_reservoirs-0001": 0})
    monkeypatch.setattr(prolong_panel, "package", lambda *_args: None)
    monkeypatch.setattr(prolong_panel, "submit", lambda args: submitted.append(args) or 0)
    args = argparse.Namespace(
        panel=str(panel_path), concurrency=None, tester="tester", remote="origin", no_submit=False, dry_run=False
    )

    assert prolong_panel.evaluate(args) == 0
    assert len(submitted) == 1
    assert submitted[0].remote == "origin"
    assert submitted[0].submission.startswith(str(tmp_path / "results/tester-"))
