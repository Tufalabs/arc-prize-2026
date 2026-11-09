from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pipeline.implementation_stage import prepare_and_launch_worker_package, prepare_worker_package


def _write_reference_games(source_root: Path, slugs: list[str]) -> None:
    for slug in slugs:
        env_dir = source_root / "pipeline" / "reference_games" / "re_arc" / "environment_files" / slug / "0001"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / f"{slug}.py").write_text(f"# env {slug}\n", encoding="utf-8")
        (env_dir / "metadata.json").write_text(json.dumps({"game_id": f"{slug}-0001"}), encoding="utf-8")
        agent_path = source_root / "pipeline" / "reference_games" / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text(f"# agent {slug}\n", encoding="utf-8")


def _write_worker_context(source_root: Path) -> None:
    docs_dir = source_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "arcengine_authoring_guide.md").write_text("# ArcEngine Authoring Guide\n", encoding="utf-8")
    screenshot_dir = source_root / "pipeline" / "reference_games" / "screenshots" / "ls20_close"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    (screenshot_dir / "dsl_replay.gif").write_bytes(b"gif")


def _write_generated_game(worker_repo: Path, *, slug: str, game_id: str) -> None:
    env_dir = worker_repo / "re_arc" / "environment_files" / slug / "0001"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / f"{slug}.py").write_text(f"# generated env {slug}\n", encoding="utf-8")
    (env_dir / "metadata.json").write_text(json.dumps({"game_id": game_id}), encoding="utf-8")
    agent_path = worker_repo / "re_arc" / "dsl" / "agents" / f"{slug}.py"
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.write_text(f"# generated agent {slug}\n", encoding="utf-8")


def _assert_generated_worker_tests(worker_repo: Path) -> None:
    tests_root = worker_repo / "tests"
    test_files = sorted(path.name for path in tests_root.iterdir() if path.is_file())
    assert test_files == [
        "_worker_game.py",
        "test_generated_game_color_contract.py",
        "test_generated_game_reset_contract.py",
        "test_generated_game_score_contract.py",
    ]

    helper_text = (tests_root / "_worker_game.py").read_text(encoding="utf-8")
    expected_reference_slugs = 'REFERENCE_GAME_SLUGS = frozenset(json.loads("""["ft09_close", "ls20_close"]"""))'
    assert expected_reference_slugs in helper_text

    score_text = (tests_root / "test_generated_game_score_contract.py").read_text(encoding="utf-8")
    assert "ARCBaseGame._score" in score_text
    assert "expected total reward 1.0" in score_text
    assert "MAX_ENV_STEP_MS = 5.0" in score_text
    assert "env.step exceeded" in score_text

    color_text = (tests_root / "test_generated_game_color_contract.py").read_text(encoding="utf-8")
    assert "MIN_COLOR = 0" in color_text
    assert "MAX_COLOR = 15" in color_text

    reset_text = (tests_root / "test_generated_game_reset_contract.py").read_text(encoding="utf-8")
    assert "post_reset_frame" in reset_text
    assert "np.array_equal(initial_frame, post_reset_frame)" in reset_text
    assert "create_dsl_agent" in reset_text


def _assert_worker_validation_tool(worker_repo: Path) -> None:
    assert (worker_repo / "pipeline" / "__init__.py").exists()
    validator_path = worker_repo / "pipeline" / "validate_generated_game.py"
    assert validator_path.exists()
    validator_text = validator_path.read_text(encoding="utf-8")
    assert "baseline_actions must be a non-empty list" in validator_text
    assert '"_win_score"' in validator_text
    assert "test_generated_game_score_contract.py" in validator_text


def _assert_worker_agents(worker_repo: Path) -> None:
    agents_path = worker_repo / "AGENTS.md"
    assert agents_path.exists()
    agents_text = agents_path.read_text(encoding="utf-8")
    assert "isolated pipeline worker repo" in agents_text
    assert "Do not run `make prepare`" in agents_text
    assert "Do not run `pytest -q tests/test_environment_score_contract.py`" in agents_text
    assert "$PY -m pipeline.validate_generated_game --game <game_id> --files-only" in agents_text


def test_codex_visual_prompt_mentions_arcengine_reference_and_skips_make_prepare():
    prompt_path = Path(__file__).resolve().parents[1] / "pipeline" / "prompts" / "codex_check_visual_prompt.md"
    prompt_text = prompt_path.read_text(encoding="utf-8")

    assert "vendor/arcengine/" in prompt_text
    assert "Do not run `make prepare`" in prompt_text
    assert "Do not use `git rev-parse --show-toplevel`" in prompt_text


def test_codex_visual_prompt_requires_game_native_action7_undo():
    prompt_path = Path(__file__).resolve().parents[1] / "pipeline" / "prompts" / "codex_check_visual_prompt.md"
    prompt_text = prompt_path.read_text(encoding="utf-8")

    assert "`GameAction.ACTION7` / id `7` = UNDO / Z" in prompt_text
    assert "Every generated game must provide game-native single-step undo" in prompt_text
    assert "Include action id `7` in both the environment's `available_actions` and `manifest.json`" in prompt_text
    assert "Use only valid ArcEngine actions `1..7`" in prompt_text
    assert "Use only valid ArcEngine actions `1..6`" not in prompt_text


def test_prepare_worker_package_creates_worker_repo_prompt_and_status(tmp_path: Path):
    source_root = tmp_path / "source"
    (source_root / "re_arc").mkdir(parents=True)
    (source_root / "re_arc" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    _write_reference_games(source_root, ["ft09_close", "ls20_close", "spare_clone"])
    _write_worker_context(source_root)
    (source_root / "tests").mkdir(parents=True)
    (source_root / "tests" / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        (source_root / filename).write_text(f"{filename}\n", encoding="utf-8")
    template_path = source_root / "implementation_codex_prompt.md"
    template_path.write_text("Implement one game.", encoding="utf-8")

    spec_path = tmp_path / "shape-matcher-survival.md"
    spec_path.write_text("# Shape Matcher Survival\n\nLevel count: 6\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    worker_root = run_dir / "workers" / "shape-matcher-survival"
    prompts_dir = run_dir / "prompts"
    result = prepare_worker_package(
        run_id="run-123",
        idea_id="shape-matcher-survival",
        idea_title="Shape Matcher Survival",
        idea_description="Match shapes while surviving hazards.",
        spec_path=spec_path,
        worker_root=worker_root,
        prompts_dir=prompts_dir,
        source_root=source_root,
        implementation_prompt_template_path=template_path,
    )

    worker_repo = worker_root / "repo"
    assert (worker_repo / "re_arc" / "__init__.py").exists()
    assert (worker_repo / "pyproject.toml").exists()
    assert (worker_repo / "docs" / "arcengine_authoring_guide.md").exists()
    assert (worker_repo / "vendor" / "arcengine" / "base_game.py").exists()
    assert (worker_repo / "pipeline" / "reference_games" / "screenshots" / "ls20_close" / "dsl_replay.gif").exists()
    _assert_generated_worker_tests(worker_repo)
    _assert_worker_validation_tool(worker_repo)
    _assert_worker_agents(worker_repo)
    kept_slugs = list(result["reference_game_slugs"])
    assert kept_slugs == ["ft09_close", "ls20_close"]

    copied_env_slugs = sorted(
        path.name for path in (worker_repo / "re_arc" / "environment_files").iterdir() if path.is_dir()
    )
    copied_agent_slugs = sorted(
        path.stem for path in (worker_repo / "re_arc" / "dsl" / "agents").glob("*.py") if path.stem != "__init__"
    )
    assert copied_env_slugs == sorted(kept_slugs)
    assert copied_agent_slugs == sorted(kept_slugs)

    worker_prompt_path = prompts_dir / "shape-matcher-survival.implementation_prompt.txt"
    worker_status_path = worker_root / "status.json"
    worker_idea_path = worker_root / "idea.json"

    assert Path(str(result["worker_repo_path"])) == worker_repo
    assert Path(str(result["worker_spec_path"])) == spec_path
    assert Path(str(result["worker_idea_path"])) == worker_idea_path
    assert Path(str(result["worker_prompt_path"])) == worker_prompt_path
    assert Path(str(result["worker_status_path"])) == worker_status_path
    assert result["idea_title"] == "Shape Matcher Survival"
    assert result["idea_description"] == "Match shapes while surviving hazards."

    prompt_text = worker_prompt_path.read_text(encoding="utf-8")
    assert prompt_text.startswith("Implement one game.")
    assert "Game spec:" in prompt_text
    assert "# Shape Matcher Survival" in prompt_text

    status_payload = json.loads(worker_status_path.read_text(encoding="utf-8"))
    assert status_payload["run_id"] == "run-123"
    assert status_payload["idea_id"] == "shape-matcher-survival"
    assert status_payload["status"] == "queued"
    idea_payload = json.loads(worker_idea_path.read_text(encoding="utf-8"))
    assert idea_payload == {
        "idea_id": "shape-matcher-survival",
        "title": "Shape Matcher Survival",
        "description": "Match shapes while surviving hazards.",
    }


def test_prepare_worker_package_keeps_only_default_reference_games(tmp_path: Path):
    source_root = tmp_path / "source"
    (source_root / "re_arc").mkdir(parents=True)
    (source_root / "re_arc" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    _write_reference_games(source_root, ["ft09_close", "ls20_close"])
    for filename in ("README.md", "config.env", "pyproject.toml", "uv.lock"):
        (source_root / filename).write_text(f"{filename}\n", encoding="utf-8")
    template_path = source_root / "implementation_codex_prompt.md"
    template_path.write_text("Implement one game.", encoding="utf-8")
    spec_path = tmp_path / "shape-matcher-survival.md"
    spec_path.write_text("# Shape Matcher Survival\n\nLevel count: 6\n", encoding="utf-8")

    worker_root = tmp_path / "run" / "workers" / "shape-matcher-survival"
    prompts_dir = tmp_path / "run" / "prompts"
    result = prepare_worker_package(
        run_id="run-123",
        idea_id="shape-matcher-survival",
        spec_path=spec_path,
        worker_root=worker_root,
        prompts_dir=prompts_dir,
        source_root=source_root,
        implementation_prompt_template_path=template_path,
    )

    worker_repo = Path(str(result["worker_repo_path"]))
    kept_env_slugs = sorted(
        path.name for path in (worker_repo / "re_arc" / "environment_files").iterdir() if path.is_dir()
    )
    kept_agent_slugs = sorted(
        path.stem for path in (worker_repo / "re_arc" / "dsl" / "agents").glob("*.py") if path.stem != "__init__"
    )
    assert kept_env_slugs == ["ft09_close", "ls20_close"]
    assert kept_agent_slugs == ["ft09_close", "ls20_close"]
    _assert_generated_worker_tests(worker_repo)
    _assert_worker_validation_tool(worker_repo)


def test_prepare_and_launch_worker_package_runs_codex_and_updates_status(tmp_path: Path):
    source_root = tmp_path / "source"
    (source_root / "re_arc").mkdir(parents=True)
    (source_root / "re_arc" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    _write_reference_games(source_root, ["ft09_close", "ls20_close", "spare_clone"])
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        (source_root / filename).write_text(f"{filename}\n", encoding="utf-8")
    template_path = source_root / "implementation_codex_prompt.md"
    template_path.write_text("Implement one game.", encoding="utf-8")

    spec_path = tmp_path / "shape-matcher-survival.md"
    spec_path.write_text("# Shape Matcher Survival\n\nLevel count: 6\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    worker_root = run_dir / "workers" / "shape-matcher-survival"
    prompts_dir = run_dir / "prompts"

    def fake_codex_runner(cmd, _cwd, log_path):
        assert list(cmd[:8]) == [
            "codex",
            "exec",
            "--cd",
            str(worker_root / "repo"),
            "--dangerously-bypass-approvals-and-sandbox",
            "-c",
            'model_reasoning_effort="high"',
            "-o",
        ]
        output_path = Path(str(cmd[8]))
        assert cmd[9].startswith("Implement one game.")
        assert _cwd == worker_root / "repo"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("implementation complete\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("codex log\n", encoding="utf-8")
        _write_generated_game(
            worker_root / "repo", slug="shape_matcher_survival", game_id="shape_matcher_survival-0001"
        )
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    def fake_artifact_runner(cmd, cwd, log_path):
        assert cwd == worker_root / "repo"
        assert "--policy" in cmd
        gif_path = Path(str(cmd[cmd.index("--gif") + 1]))
        replay_path = Path(str(cmd[cmd.index("--replay") + 1]))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.write_text("gif\n", encoding="utf-8")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text('{"version":2}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("artifact generation log\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    prepare_and_launch_worker_package(
        run_id="run-123",
        idea_id="shape-matcher-survival",
        spec_path=spec_path,
        worker_root=worker_root,
        prompts_dir=prompts_dir,
        source_root=source_root,
        implementation_prompt_template_path=template_path,
        codex_runner=fake_codex_runner,
        codex_reasoning_effort="high",
        artifact_runner=fake_artifact_runner,
    )

    worker_status_path = worker_root / "status.json"
    status_payload = json.loads(worker_status_path.read_text(encoding="utf-8"))
    assert status_payload["status"] == "codex finished"
    assert status_payload["run_id"] == "run-123"
    assert status_payload["idea_id"] == "shape-matcher-survival"
    assert isinstance(status_payload["codex_started_at"], str)
    _assert_generated_worker_tests(worker_root / "repo")


def test_prepare_and_launch_worker_package_runs_claude_and_updates_status(tmp_path: Path):
    source_root = tmp_path / "source"
    (source_root / "re_arc").mkdir(parents=True)
    (source_root / "re_arc" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    _write_reference_games(source_root, ["ft09_close", "ls20_close", "spare_clone"])
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        (source_root / filename).write_text(f"{filename}\n", encoding="utf-8")
    template_path = source_root / "implementation_codex_prompt.md"
    template_path.write_text("Implement one game.", encoding="utf-8")

    spec_path = tmp_path / "shape-matcher-survival.md"
    spec_path.write_text("# Shape Matcher Survival\n\nLevel count: 6\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    worker_root = run_dir / "workers" / "shape-matcher-survival"
    prompts_dir = run_dir / "prompts"

    def fake_claude_runner(cmd, cwd, log_path, output_path):
        assert cmd[:3] == ["claude", "--dangerously-skip-permissions", "-p"]
        assert cmd[3].startswith("Implement one game.")
        assert cwd == worker_root / "repo"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("claude log\n", encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("claude response\n", encoding="utf-8")
        _write_generated_game(
            worker_root / "repo", slug="shape_matcher_survival", game_id="shape_matcher_survival-0001"
        )
        return subprocess.CompletedProcess(list(cmd), 0, "claude response\n", "")

    def fake_artifact_runner(cmd, cwd, log_path):
        assert cwd == worker_root / "repo"
        gif_path = Path(str(cmd[cmd.index("--gif") + 1]))
        replay_path = Path(str(cmd[cmd.index("--replay") + 1]))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.write_text("gif\n", encoding="utf-8")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text('{"version":2}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("artifact generation log\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    result = prepare_and_launch_worker_package(
        run_id="run-123",
        idea_id="shape-matcher-survival",
        spec_path=spec_path,
        worker_root=worker_root,
        prompts_dir=prompts_dir,
        source_root=source_root,
        implementation_prompt_template_path=template_path,
        implementation_backend="claude",
        claude_runner=fake_claude_runner,
        artifact_runner=fake_artifact_runner,
    )

    worker_status_path = worker_root / "status.json"
    status_payload = json.loads(worker_status_path.read_text(encoding="utf-8"))
    assert status_payload["status"] == "claude finished"
    assert status_payload["codex_exit_code"] == "0"
    assert result["implementation_backend"] == "claude"
    assert Path(str(result["implementation_output_path"])).name == "last_message.txt"
    assert Path(str(result["claude_output_path"])).read_text(encoding="utf-8") == "claude response\n"
    assert Path(str(result["claude_log_path"])).read_text(encoding="utf-8") == "claude log\n"


def test_prepare_and_launch_worker_package_marks_artifact_failure(tmp_path: Path):
    source_root = tmp_path / "source"
    (source_root / "re_arc").mkdir(parents=True)
    (source_root / "re_arc" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "re_arc" / "dsl" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    _write_reference_games(source_root, ["ft09_close", "ls20_close", "spare_clone"])
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        (source_root / filename).write_text(f"{filename}\n", encoding="utf-8")
    template_path = source_root / "implementation_codex_prompt.md"
    template_path.write_text("Implement one game.", encoding="utf-8")

    spec_path = tmp_path / "shape-matcher-survival.md"
    spec_path.write_text("# Shape Matcher Survival\n\nLevel count: 6\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    worker_root = run_dir / "workers" / "shape-matcher-survival"
    prompts_dir = run_dir / "prompts"

    def fake_codex_runner(cmd, _cwd, log_path):
        output_path = Path(str(cmd[6]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("implementation complete\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("codex log\n", encoding="utf-8")
        _write_generated_game(
            worker_root / "repo", slug="shape_matcher_survival", game_id="shape_matcher_survival-0001"
        )
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    def fake_artifact_runner(cmd, cwd, log_path):
        _ = cmd, cwd, log_path
        return subprocess.CompletedProcess(["python"], 1, "", "")

    with pytest.raises(RuntimeError, match="artifact generation failed with exit code 1"):
        prepare_and_launch_worker_package(
            run_id="run-123",
            idea_id="shape-matcher-survival",
            spec_path=spec_path,
            worker_root=worker_root,
            prompts_dir=prompts_dir,
            source_root=source_root,
            implementation_prompt_template_path=template_path,
            codex_runner=fake_codex_runner,
            artifact_runner=fake_artifact_runner,
        )

    worker_status_path = worker_root / "status.json"
    status_payload = json.loads(worker_status_path.read_text(encoding="utf-8"))
    assert status_payload["status"] == "artifact failed"
    assert status_payload["error"] == "artifact generation failed with exit code 1"
    assert isinstance(status_payload["codex_finished_at"], str)
    assert isinstance(status_payload["codex_duration_seconds"], float)
    assert status_payload["codex_duration_seconds"] >= 0.0
    assert status_payload["codex_exit_code"] == "0"
