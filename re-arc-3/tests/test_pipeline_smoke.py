from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pipeline.cli import run_pipeline
from pipeline.config import PipelineRunConfig
from pipeline.slurm import python_bin_for_root


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_reference_games(root: Path) -> None:
    for slug in ("ft09_close", "ls20_close", "spare_clone"):
        _write_text(
            root / "pipeline" / "reference_games" / "re_arc" / "environment_files" / slug / "0001" / f"{slug}.py",
            f"# env {slug}\n",
        )
        _write_text(
            root / "pipeline" / "reference_games" / "re_arc" / "environment_files" / slug / "0001" / "metadata.json",
            json.dumps({"game_id": f"{slug}-0001"}),
        )
        _write_text(
            root / "pipeline" / "reference_games" / "re_arc" / "dsl" / "agents" / f"{slug}.py", f"# agent {slug}\n"
        )


def _assert_generated_worker_tests(worker_repo_path: Path) -> None:
    tests_root = worker_repo_path / "tests"
    test_files = sorted(path.name for path in tests_root.iterdir() if path.is_file())
    assert test_files == [
        "_worker_game.py",
        "test_generated_game_color_contract.py",
        "test_generated_game_reset_contract.py",
        "test_generated_game_score_contract.py",
    ]
    assert (worker_repo_path / "pipeline" / "__init__.py").exists()
    validator_path = worker_repo_path / "pipeline" / "validate_generated_game.py"
    assert validator_path.exists()
    assert "validate_generated_game" in validator_path.read_text(encoding="utf-8")


def test_pipeline_end_to_end_smoke(tmp_path: Path):
    root = tmp_path
    _write_text(root / "re_arc" / "__init__.py", "")
    _write_text(root / "re_arc" / "dsl" / "agents" / "__init__.py", "")
    _write_reference_games(root)
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        _write_text(root / filename, f"{filename}\n")

    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "spec_generation_instructions.md", "Write a structured game spec."
    )
    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "implementation_codex_prompt.md",
        "Implement one game from the provided spec.",
    )

    ideas_path = root / "pipeline" / "game_ideas" / "debugging_ideas.json"
    ideas_payload = [
        {
            "title": "Toy Switches",
            "description": "A tiny switch puzzle with a player, a switch, and a goal.",
            "num_levels": 3,
        },
        {
            "title": "Toy Paths",
            "description": "A tiny path puzzle where the player navigates obstacles to reach a goal.",
            "num_levels": 5,
        },
    ]
    _write_text(ideas_path, json.dumps(ideas_payload, indent=2))

    captured_calls: list[dict[str, str]] = []
    captured_codex_calls: list[dict[str, str]] = []
    captured_artifact_calls: list[dict[str, str]] = []

    def fake_spec_provider(*, api_key: str, model: str, prompt: str, reasoning_effort: str):
        captured_calls.append(
            {"api_key": api_key, "model": model, "reasoning_effort": reasoning_effort, "prompt": prompt}
        )
        raw_idea = json.loads(prompt.split("Game idea JSON:\n", 1)[1])
        return f"# {raw_idea['title']}\n\nOne sentence pitch: Spec for {raw_idea['title']}\n\nLevel count: 3"

    def fake_codex_runner(cmd, cwd, log_path):
        captured_codex_calls.append(
            {"cwd": str(cwd), "log_path": str(log_path), "cmd0": str(cmd[0]), "cmd1": str(cmd[1])}
        )
        output_path = Path(str(cmd[6]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("implementation complete\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("codex log\n", encoding="utf-8")
        slug = cwd.parent.name.replace("-", "_")
        env_dir = cwd / "re_arc" / "environment_files" / slug / "0001"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / f"{slug}.py").write_text("# env\n", encoding="utf-8")
        (env_dir / "metadata.json").write_text(json.dumps({"game_id": f"{slug}-0001"}), encoding="utf-8")
        agent_path = cwd / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text("# agent\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    def fake_artifact_runner(cmd, _cwd, log_path):
        captured_artifact_calls.append(
            {"cwd": str(_cwd), "log_path": str(log_path), "cmd0": str(cmd[0]), "cmd1": str(cmd[1])}
        )
        gif_path = Path(str(cmd[cmd.index("--gif") + 1]))
        replay_path = Path(str(cmd[cmd.index("--replay") + 1]))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.write_text("gif\n", encoding="utf-8")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text('{"version":2}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("artifact generation log\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    run_id = "20260324-170000"
    result = run_pipeline(
        api_key="test-key",
        config=PipelineRunConfig(
            idea_json="pipeline/game_ideas/debugging_ideas.json",
            implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
            spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
            model="gpt-5.4",
            reasoning_effort="high",
        ),
        run_id=run_id,
        root=root,
        spec_provider=fake_spec_provider,
        codex_runner=fake_codex_runner,
        artifact_runner=fake_artifact_runner,
    )

    assert len(captured_calls) == 2
    assert len(captured_codex_calls) == 2
    assert len(captured_artifact_calls) == 2
    assert [call["model"] for call in captured_calls] == ["gpt-5.4", "gpt-5.4"]
    assert [call["reasoning_effort"] for call in captured_calls] == ["high", "high"]
    assert all(call["api_key"] == "test-key" for call in captured_calls)

    run_dir = root / "pipeline" / "runs" / run_id
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    status_log_lines = (run_dir / "status_log.jsonl").read_text(encoding="utf-8").splitlines()
    status_log_entries = [json.loads(line) for line in status_log_lines]
    assert run_json == result
    assert run_json["idea_count"] == 2
    assert len(run_json["outputs"]) == 2
    assert len(run_json["worker_packages"]) == 2
    assert len(status_log_entries) == 10
    assert {entry["idea_id"] for entry in status_log_entries} == {"toy-switches", "toy-paths"}
    assert {entry["status"] for entry in status_log_entries} >= {
        "now writing specs",
        "done",
        "queued",
        "launching codex",
        "codex finished",
    }

    for idea in ideas_payload:
        idea_id = idea["title"].lower().replace(" ", "-")
        spec_path = run_dir / "specs" / f"{idea_id}.md"
        prompt_path = run_dir / "prompts" / f"{idea_id}.prompt.txt"
        implementation_prompt_path = run_dir / "prompts" / f"{idea_id}.implementation_prompt.txt"
        worker_root = run_dir / "workers" / idea_id
        worker_status_path = worker_root / "status.json"
        worker_repo_path = worker_root / "repo"
        codex_output_path = worker_root / "codex" / "last_message.txt"
        codex_log_path = worker_root / "codex" / "stdout.log"
        artifacts_dir = worker_root / "repo" / "artifacts"

        spec_text = spec_path.read_text(encoding="utf-8")
        assert f"# {idea['title']}" in spec_text
        assert "Level count: 3" in spec_text

        spec_prompt = prompt_path.read_text(encoding="utf-8")
        assert "Game idea JSON:" in spec_prompt
        assert idea["title"] in spec_prompt
        assert idea["description"] in spec_prompt
        assert f'"num_levels": {idea["num_levels"]}' in spec_prompt

        implementation_prompt = implementation_prompt_path.read_text(encoding="utf-8")
        assert implementation_prompt.startswith("Implement one game from the provided spec.")
        assert "Game spec:" in implementation_prompt
        assert f"# {idea['title']}" in implementation_prompt

        worker_status = json.loads(worker_status_path.read_text(encoding="utf-8"))
        assert worker_status["run_id"] == run_id
        assert worker_status["idea_id"] == idea_id
        assert worker_status["status"] == "codex finished"
        assert isinstance(worker_status["spec_started_at"], str)
        assert isinstance(worker_status["spec_finished_at"], str)
        assert isinstance(worker_status["spec_duration_seconds"], float)
        assert worker_status["spec_duration_seconds"] >= 0.0
        assert worker_status["codex_exit_code"] == "0"

        assert (worker_repo_path / "re_arc" / "__init__.py").exists()
        assert (worker_repo_path / "pyproject.toml").exists()
        _assert_generated_worker_tests(worker_repo_path)
        copied_env_slugs = sorted(
            path.name for path in (worker_repo_path / "re_arc" / "environment_files").iterdir() if path.is_dir()
        )
        copied_agent_slugs = sorted(
            path.stem
            for path in (worker_repo_path / "re_arc" / "dsl" / "agents").glob("*.py")
            if path.stem != "__init__"
        )
        assert {"ft09_close", "ls20_close"} <= set(copied_env_slugs)
        assert {"ft09_close", "ls20_close"} <= set(copied_agent_slugs)
        assert codex_output_path.read_text(encoding="utf-8") == "implementation complete\n"
        assert codex_log_path.read_text(encoding="utf-8") == "codex log\n"
        assert (artifacts_dir / "replays" / f"{idea_id.replace('-', '_')}-0001.gif").exists()
        assert (artifacts_dir / "replays" / f"{idea_id.replace('-', '_')}-0001.replay.json").exists()


def test_pipeline_claude_end_to_end_smoke(tmp_path: Path):
    root = tmp_path
    _write_text(root / "re_arc" / "__init__.py", "")
    _write_text(root / "re_arc" / "dsl" / "agents" / "__init__.py", "")
    _write_reference_games(root)
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        _write_text(root / filename, f"{filename}\n")

    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "spec_generation_instructions.md", "Write a structured game spec."
    )
    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "implementation_codex_prompt.md",
        "Implement one game from the provided spec.",
    )

    ideas_path = root / "pipeline" / "game_ideas" / "debugging_ideas.json"
    _write_text(
        ideas_path,
        json.dumps(
            [
                {
                    "title": "Toy Switches",
                    "description": "A tiny switch puzzle with a player, a switch, and a goal.",
                    "num_levels": 3,
                }
            ],
            indent=2,
        ),
    )

    captured_claude_calls: list[dict[str, str]] = []

    def fake_spec_provider(*, api_key: str, model: str, prompt: str, reasoning_effort: str):
        _ = api_key, model, prompt, reasoning_effort
        return "# Toy Switches\n\nLevel count: 3"

    def fake_claude_runner(cmd, cwd, log_path, output_path):
        captured_claude_calls.append({"cwd": str(cwd), "log_path": str(log_path), "output_path": str(output_path)})
        assert cmd[:3] == ["claude", "--dangerously-skip-permissions", "-p"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("claude response\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("claude log\n", encoding="utf-8")
        slug = cwd.parent.name.replace("-", "_")
        env_dir = cwd / "re_arc" / "environment_files" / slug / "0001"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / f"{slug}.py").write_text("# env\n", encoding="utf-8")
        (env_dir / "metadata.json").write_text(json.dumps({"game_id": f"{slug}-0001"}), encoding="utf-8")
        agent_path = cwd / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text("# agent\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "claude response\n", "")

    def fake_artifact_runner(cmd, _cwd, log_path):
        gif_path = Path(str(cmd[cmd.index("--gif") + 1]))
        replay_path = Path(str(cmd[cmd.index("--replay") + 1]))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.write_text("gif\n", encoding="utf-8")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text('{"version":2}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("artifact generation log\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    run_id = "20260401-130000"
    result = run_pipeline(
        api_key="test-key",
        config=PipelineRunConfig(
            idea_json="pipeline/game_ideas/debugging_ideas.json",
            implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
            spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
            implementation_backend="claude",
            model="gpt-5.4",
            reasoning_effort="high",
        ),
        run_id=run_id,
        root=root,
        spec_provider=fake_spec_provider,
        claude_runner=fake_claude_runner,
        artifact_runner=fake_artifact_runner,
    )

    worker_package = result["worker_packages"][0]
    worker_status = json.loads((root / worker_package["worker_status_path"]).read_text(encoding="utf-8"))
    assert worker_status["status"] == "claude finished"
    assert worker_package["implementation_backend"] == "claude"
    assert Path(str(worker_package["claude_output_path"])).read_text(encoding="utf-8") == "claude response\n"
    assert len(captured_claude_calls) == 1


def test_pipeline_can_generate_specs_from_official_description_pairs(tmp_path: Path):
    root = tmp_path
    _write_text(root / "re_arc" / "__init__.py", "")
    _write_text(root / "re_arc" / "dsl" / "agents" / "__init__.py", "")
    _write_reference_games(root)
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        _write_text(root / filename, f"{filename}\n")

    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "spec_generation_instructions.md", "Write a structured game spec."
    )
    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "implementation_codex_prompt.md",
        "Implement one game from the provided spec.",
    )
    first_description = root / "arc_official_description" / "ft09_flip_tiles.md"
    second_description = root / "arc_official_description" / "ls20_locksmith.md"
    _write_text(first_description, "# FT09 Flip Tiles\n\nClick tiles to satisfy local color constraints.\n")
    _write_text(second_description, "# LS20 Locksmith\n\nCarry a key through lock tutorials.\n")

    captured_prompts: list[str] = []

    def fake_spec_provider(*, api_key: str, model: str, prompt: str, reasoning_effort: str):
        del api_key, model, reasoning_effort
        captured_prompts.append(prompt)
        assert "Official-description combination request:" in prompt
        assert "FT09 Flip Tiles" in prompt
        assert "LS20 Locksmith" in prompt
        assert "source_official_descriptions" in prompt
        return "# Hybrid Spec\n\nLevel count: 4\n"

    def fake_codex_runner(cmd, cwd, log_path):
        output_path = Path(str(cmd[6]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("implementation complete\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("codex log\n", encoding="utf-8")
        slug = cwd.parent.name.replace("-", "_")
        env_dir = cwd / "re_arc" / "environment_files" / slug / "0001"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / f"{slug}.py").write_text("# env\n", encoding="utf-8")
        (env_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "game_id": f"{slug}-0001",
                    "source_official_descriptions": [
                        {"name": "ft09 flip tiles", "path": "arc_official_description/ft09_flip_tiles.md"},
                        {"name": "ls20 locksmith", "path": "arc_official_description/ls20_locksmith.md"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        agent_path = cwd / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text("# agent\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    def fake_artifact_runner(cmd, _cwd, log_path):
        gif_path = Path(str(cmd[cmd.index("--gif") + 1]))
        replay_path = Path(str(cmd[cmd.index("--replay") + 1]))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.write_text("gif\n", encoding="utf-8")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text('{"version":2}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("artifact generation log\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    run_id = "20260401-140000"
    result = run_pipeline(
        api_key="test-key",
        config=PipelineRunConfig(
            official_description_paths=(
                "arc_official_description/ft09_flip_tiles.md",
                "arc_official_description/ls20_locksmith.md",
            ),
            official_description_num_levels=4,
            implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
            spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
            model="gpt-5.4",
            reasoning_effort="high",
        ),
        run_id=run_id,
        root=root,
        spec_provider=fake_spec_provider,
        codex_runner=fake_codex_runner,
        artifact_runner=fake_artifact_runner,
    )

    assert len(captured_prompts) == 1
    assert result["idea_count"] == 1
    assert result["idea_json"] is None
    assert result["official_description_paths"] == [str(first_description), str(second_description)]
    spec_path = root / "pipeline" / "runs" / run_id / "specs" / "ft09-flip-tiles-ls20-locksmith.md"
    spec_text = spec_path.read_text(encoding="utf-8")
    assert "## Source official descriptions" in spec_text
    assert "arc_official_description/ft09_flip_tiles.md" in spec_text


def test_pipeline_interrupt_run_json_preserves_official_description_sources(tmp_path: Path):
    root = tmp_path
    _write_text(root / "re_arc" / "__init__.py", "")
    _write_text(root / "re_arc" / "dsl" / "agents" / "__init__.py", "")
    _write_reference_games(root)
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        _write_text(root / filename, f"{filename}\n")

    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "spec_generation_instructions.md", "Write a structured game spec."
    )
    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "implementation_codex_prompt.md",
        "Implement one game from the provided spec.",
    )
    first_description = root / "arc_official_description" / "ft09_flip_tiles.md"
    second_description = root / "arc_official_description" / "ls20_locksmith.md"
    _write_text(first_description, "# FT09 Flip Tiles\n\nClick tiles to satisfy local color constraints.\n")
    _write_text(second_description, "# LS20 Locksmith\n\nCarry a key through lock tutorials.\n")

    def fake_spec_provider(*, api_key: str, model: str, prompt: str, reasoning_effort: str):
        del api_key, model, prompt, reasoning_effort
        return "# Hybrid Spec\n\nLevel count: 4\n"

    def interrupting_codex_runner(cmd, cwd, log_path):
        del cmd, cwd, log_path
        raise KeyboardInterrupt

    run_id = "20260401-141000"
    try:
        run_pipeline(
            api_key="test-key",
            config=PipelineRunConfig(
                official_description_paths=(
                    "arc_official_description/ft09_flip_tiles.md",
                    "arc_official_description/ls20_locksmith.md",
                ),
                official_description_num_levels=4,
                implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
                model="gpt-5.4",
                reasoning_effort="high",
            ),
            run_id=run_id,
            root=root,
            spec_provider=fake_spec_provider,
            codex_runner=interrupting_codex_runner,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Expected KeyboardInterrupt")

    run_json = json.loads((root / "pipeline" / "runs" / run_id / "run.json").read_text(encoding="utf-8"))
    assert run_json["idea_json"] is None
    assert run_json["official_description_paths"] == [str(first_description), str(second_description)]
    assert run_json["interrupted"] is True


def test_pipeline_slurm_prepare_smoke(tmp_path: Path):
    root = tmp_path
    _write_text(root / "re_arc" / "__init__.py", "")
    _write_text(root / "re_arc" / "dsl" / "agents" / "__init__.py", "")
    _write_reference_games(root)
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        _write_text(root / filename, f"{filename}\n")

    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "spec_generation_instructions.md", "Write a structured game spec."
    )
    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "implementation_codex_prompt.md",
        "Implement one game from the provided spec.",
    )

    ideas_path = root / "pipeline" / "game_ideas" / "debugging_ideas.json"
    ideas_payload = [
        {
            "title": "Toy Switches",
            "description": "A tiny switch puzzle with a player, a switch, and a goal.",
            "num_levels": 3,
        },
        {
            "title": "Toy Paths",
            "description": "A tiny path puzzle where the player navigates obstacles to reach a goal.",
            "num_levels": 5,
        },
    ]
    _write_text(ideas_path, json.dumps(ideas_payload, indent=2))

    slurm_calls: list[dict[str, object]] = []
    squeue_calls = 0
    run_dir = root / "pipeline" / "runs" / "20260331-120000"

    def fake_slurm_runner(cmd, cwd):
        nonlocal squeue_calls
        slurm_calls.append({"cmd": list(cmd), "cwd": str(cwd)})
        if cmd[0] == "squeue":
            squeue_calls += 1
            workers_dir = run_dir / "workers"
            if squeue_calls == 1:
                (workers_dir / "toy-switches").mkdir(parents=True, exist_ok=True)
                (workers_dir / "toy-switches" / "status.json").write_text(
                    json.dumps({"run_id": "20260331-120000", "idea_id": "toy-switches", "status": "launching codex"}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(list(cmd), 0, "RUNNING\nPENDING\n", "")
            (workers_dir / "toy-paths").mkdir(parents=True, exist_ok=True)
            (workers_dir / "toy-switches" / "status.json").write_text(
                json.dumps({"run_id": "20260331-120000", "idea_id": "toy-switches", "status": "codex finished"}),
                encoding="utf-8",
            )
            (workers_dir / "toy-paths" / "status.json").write_text(
                json.dumps({"run_id": "20260331-120000", "idea_id": "toy-paths", "status": "artifact failed"}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(list(cmd), 0, "", "")
        if cmd[0] == "scontrol":
            return subprocess.CompletedProcess(list(cmd), 0, "JobState=COMPLETED\n", "")
        return subprocess.CompletedProcess(list(cmd), 0, "Submitted batch job 4242\n", "")

    run_id = "20260331-120000"
    run_payload = run_pipeline(
        api_key="test-key",
        config=PipelineRunConfig(
            idea_json="pipeline/game_ideas/debugging_ideas.json",
            implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
            spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
            executor="slurm",
            model="gpt-5.4",
            reasoning_effort="high",
        ),
        run_id=run_id,
        root=root,
        slurm_runner=fake_slurm_runner,
    )

    run_meta_path = run_dir / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    run_status_history = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    run_status = run_status_history[-1]

    assert run_payload["execution_backend"] == "slurm"
    assert run_payload["slurm_job_id"] == "4242"
    assert run_payload["run_status_path"] == f"pipeline/runs/{run_id}/run_status.json"
    assert run_payload["outputs"] == []
    assert run_payload["worker_packages"] == []
    assert run_meta == {"run_id": run_id, "idea_count": 2, "source_mode": "idea_json"}
    assert isinstance(run_status_history, list)
    assert len(run_status_history) == 2
    assert run_status_history[0]["active"] is True
    assert run_status_history[0]["worker_state_counts"] == {"launching codex": 1, "pending": 1}
    assert run_status["slurm_job_id"] == "4242"
    assert run_status["active"] is False
    assert run_status["slurm_terminal_state"] == "COMPLETED"
    assert run_status["worker_state_counts"] == {"artifact failed": 1, "codex finished": 1}
    assert [worker["idea_id"] for worker in run_status["workers"]] == ["toy-switches", "toy-paths"]
    assert run_status["workers"][0]["slurm_log_path"].endswith("/pipeline/runs/20260331-120000/slurm/4242_0.out")
    assert run_status["workers"][0]["slurm_error_log_path"].endswith("/pipeline/runs/20260331-120000/slurm/4242_0.err")
    assert run_payload["run_status"] == run_status
    assert slurm_calls == [
        {
            "cmd": [
                "sbatch",
                "--partition=llm",
                "--time=01:30:00",
                "--mem=8G",
                "--ntasks=1",
                "--cpus-per-task=2",
                "--array=0-1",
                f"--output={root}/pipeline/runs/{run_id}/slurm/%A_%a.out",
                f"--error={root}/pipeline/runs/{run_id}/slurm/%A_%a.err",
                f'--wrap=cd "{root}" && exec "{python_bin_for_root(root)}" -m pipeline.slurm --run-id "{run_id}"',
            ],
            "cwd": str(root),
        },
        {"cmd": ["squeue", "-h", "-j", "4242", "-o", "%T"], "cwd": str(root)},
        {"cmd": ["squeue", "-h", "-j", "4242", "-o", "%T"], "cwd": str(root)},
        {"cmd": ["scontrol", "show", "job", "4242"], "cwd": str(root)},
    ]


def test_pipeline_slurm_interrupt_cancels_job(tmp_path: Path):
    root = tmp_path
    _write_text(root / "re_arc" / "__init__.py", "")
    _write_text(root / "re_arc" / "dsl" / "agents" / "__init__.py", "")
    _write_reference_games(root)
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        _write_text(root / filename, f"{filename}\n")

    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "spec_generation_instructions.md", "Write a structured game spec."
    )
    _write_text(
        root / "pipeline" / "prompts_and_schemas" / "implementation_codex_prompt.md",
        "Implement one game from the provided spec.",
    )

    ideas_path = root / "pipeline" / "game_ideas" / "debugging_ideas.json"
    _write_text(
        ideas_path,
        json.dumps(
            [
                {
                    "title": "Toy Switches",
                    "description": "A tiny switch puzzle with a player, a switch, and a goal.",
                    "num_levels": 3,
                }
            ],
            indent=2,
        ),
    )

    slurm_calls: list[dict[str, object]] = []

    def fake_slurm_runner(cmd, cwd):
        slurm_calls.append({"cmd": list(cmd), "cwd": str(cwd)})
        if cmd[0] == "squeue":
            raise KeyboardInterrupt
        if cmd[0] == "scancel":
            return subprocess.CompletedProcess(list(cmd), 0, "", "")
        return subprocess.CompletedProcess(list(cmd), 0, "Submitted batch job 4242\n", "")

    try:
        run_pipeline(
            api_key="test-key",
            config=PipelineRunConfig(
                idea_json="pipeline/game_ideas/debugging_ideas.json",
                implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
                executor="slurm",
                model="gpt-5.4",
                reasoning_effort="high",
            ),
            run_id="20260401-120000",
            root=root,
            slurm_runner=fake_slurm_runner,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Expected KeyboardInterrupt")

    assert slurm_calls == [
        {
            "cmd": [
                "sbatch",
                "--partition=llm",
                "--time=01:30:00",
                "--mem=8G",
                "--ntasks=1",
                "--cpus-per-task=2",
                "--array=0-0",
                f"--output={root}/pipeline/runs/20260401-120000/slurm/%A_%a.out",
                f"--error={root}/pipeline/runs/20260401-120000/slurm/%A_%a.err",
                (
                    f'--wrap=cd "{root}" && exec "{python_bin_for_root(root)}" '
                    '-m pipeline.slurm --run-id "20260401-120000"'
                ),
            ],
            "cwd": str(root),
        },
        {"cmd": ["squeue", "-h", "-j", "4242", "-o", "%T"], "cwd": str(root)},
        {"cmd": ["scancel", "4242"], "cwd": str(root)},
    ]
