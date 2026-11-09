from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from pipeline.cli import main, run_implementation_from_specs, run_pipeline
from pipeline.config import PipelineRunConfig
from pipeline.game_catalog import build_game_catalog


def test_pipeline_cli_delegates_to_run_pipeline(tmp_path: Path, monkeypatch):
    root = tmp_path
    config_path = root / "pipeline" / "configs" / "run.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_id = "20260324-160000"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("pipeline.cli.repo_root", lambda: root)
    monkeypatch.setattr("pipeline.cli._timestamp_run_id", lambda: run_id)
    captured: dict[str, object] = {}

    def fake_run_pipeline(*, api_key: str, config, run_id: str, root: Path | None, spec_provider) -> dict[str, object]:
        captured["api_key"] = api_key
        captured["config"] = config
        captured["run_id"] = run_id
        captured["root"] = root
        captured["spec_provider"] = spec_provider
        return {"workers_dir": f"pipeline/runs/{run_id}/workers"}

    monkeypatch.setattr("pipeline.cli.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(sys, "argv", ["pipeline.cli", "--config", str(config_path.relative_to(root))])

    assert main() == 0
    assert captured == {
        "api_key": "test-key",
        "config": captured["config"],
        "run_id": run_id,
        "root": root,
        "spec_provider": captured["spec_provider"],
    }
    assert captured["config"].idea_json == "pipeline/game_ideas/ideas.json"
    assert captured["config"].model == "gpt-5.4"
    assert captured["config"].reasoning_effort == "high"
    assert captured["config"].executor == "local"
    assert (
        captured["config"].implementation_prompt_path == "pipeline/prompts_and_schemas/implementation_codex_prompt.md"
    )


def test_pipeline_cli_run_supports_run_name(tmp_path: Path, monkeypatch):
    root = tmp_path
    config_path = root / "pipeline" / "configs" / "run.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/custom_implementation_prompt.md",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
                "executor: slurm",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("pipeline.cli.repo_root", lambda: root)
    monkeypatch.setattr("pipeline.cli._timestamp_run_id", lambda: "20260324-190000")
    captured: dict[str, object] = {}

    def fake_run_pipeline(*, api_key: str, config, run_id: str, root: Path | None, spec_provider) -> dict[str, object]:
        captured["api_key"] = api_key
        captured["config"] = config
        captured["run_id"] = run_id
        captured["root"] = root
        captured["spec_provider"] = spec_provider
        return {"workers_dir": f"pipeline/runs/{run_id}/workers"}

    monkeypatch.setattr("pipeline.cli.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys, "argv", ["pipeline.cli", "--config", str(config_path.relative_to(root)), "--run-name", "debug batch"]
    )

    assert main() == 0
    assert captured == {
        "api_key": "test-key",
        "config": captured["config"],
        "run_id": "20260324-190000-debug-batch",
        "root": root,
        "spec_provider": captured["spec_provider"],
    }
    assert captured["config"].idea_json == "pipeline/game_ideas/ideas.json"
    assert captured["config"].model == "gpt-5.4"
    assert captured["config"].reasoning_effort == "high"
    assert captured["config"].executor == "slurm"
    assert (
        captured["config"].implementation_prompt_path == "pipeline/prompts_and_schemas/custom_implementation_prompt.md"
    )


def test_pipeline_cli_run_supports_yaml_config(tmp_path: Path, monkeypatch):
    root = tmp_path
    config_path = root / "pipeline" / "configs" / "simple.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "model: gpt-5.4",
                "reasoning_effort: medium",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("pipeline.cli.repo_root", lambda: root)
    monkeypatch.setattr("pipeline.cli._timestamp_run_id", lambda: "20260324-191500")
    captured: dict[str, object] = {}

    def fake_run_pipeline(*, api_key: str, config, run_id: str, root: Path | None, spec_provider) -> dict[str, object]:
        captured["api_key"] = api_key
        captured["config"] = config
        captured["run_id"] = run_id
        captured["root"] = root
        captured["spec_provider"] = spec_provider
        return {"workers_dir": f"pipeline/runs/{run_id}/workers"}

    monkeypatch.setattr("pipeline.cli.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(sys, "argv", ["pipeline.cli", "--config", str(config_path.relative_to(root))])

    assert main() == 0
    assert captured["api_key"] == "test-key"
    assert captured["run_id"] == "20260324-191500"
    assert captured["root"] == root
    assert captured["config"] is not None
    assert captured["config"].idea_json == "pipeline/game_ideas/ideas.json"
    assert captured["config"].model == "gpt-5.4"
    assert captured["config"].reasoning_effort == "medium"
    assert captured["config"].executor == "local"
    assert (
        captured["config"].implementation_prompt_path == "pipeline/prompts_and_schemas/implementation_codex_prompt.md"
    )


def test_pipeline_cli_run_requires_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["pipeline.cli"])

    with pytest.raises(RuntimeError, match="--config is required for `run`\\."):
        main()


def test_pipeline_cli_human_model_does_not_require_openai_key(tmp_path: Path, monkeypatch):
    root = tmp_path
    config_path = root / "pipeline" / "configs" / "human.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/ideas.json",
                "implementation_prompt_path: pipeline/prompts/codex_check_visual_prompt.md",
                "spec_instructions_path: pipeline/prompts/spec_prompt.md",
                "model: Human",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "pipeline" / "game_ideas").mkdir(parents=True, exist_ok=True)
    (root / "pipeline" / "game_ideas" / "ideas.json").write_text(
        '[{"title":"Manual Game","description":"Write a prompt only.","num_levels":3}]\n', encoding="utf-8"
    )
    (root / "pipeline" / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "pipeline" / "prompts" / "spec_prompt.md").write_text("Write a structured spec.\n", encoding="utf-8")
    (root / "pipeline" / "prompts" / "codex_check_visual_prompt.md").write_text(
        "Implement this spec.\n", encoding="utf-8"
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("pipeline.cli.repo_root", lambda: root)
    monkeypatch.setattr("pipeline.cli._timestamp_run_id", lambda: "20260608-130000")
    monkeypatch.setattr(sys, "argv", ["pipeline.cli", "--config", str(config_path.relative_to(root))])

    assert main() == 0

    run_json = json.loads((root / "pipeline" / "runs" / "20260608-130000" / "run.json").read_text(encoding="utf-8"))
    assert run_json["run_status"] == "waiting_for_human_specs"
    assert run_json["source_mode"] == "human_spec_prompts"
    assert run_json["outputs"][0]["prompt_path"] == "pipeline/runs/20260608-130000/prompts/manual-game.prompt.txt"
    assert run_json["outputs"][0]["suggested_spec_path"] == "pipeline/runs/20260608-130000/specs/manual-game.md"


def test_run_pipeline_human_model_writes_prompts_without_calling_spec_provider(tmp_path: Path):
    root = tmp_path
    idea_path = root / "pipeline" / "game_ideas" / "ideas.json"
    idea_path.parent.mkdir(parents=True, exist_ok=True)
    idea_path.write_text('[{"title":"Manual Game","description":"Prompt only.","num_levels":4}]\n', encoding="utf-8")
    prompts_dir = root / "pipeline" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "spec_prompt.md").write_text("Write a structured spec.\n", encoding="utf-8")
    (prompts_dir / "codex_check_visual_prompt.md").write_text("Implement this spec.\n", encoding="utf-8")

    def forbidden_spec_provider(**_kwargs):
        raise AssertionError("spec provider should not be called for model: Human")

    result = run_pipeline(
        api_key="",
        config=PipelineRunConfig(
            idea_json="pipeline/game_ideas/ideas.json",
            implementation_prompt_path="pipeline/prompts/codex_check_visual_prompt.md",
            spec_instructions_path="pipeline/prompts/spec_prompt.md",
            model="Human",
        ),
        run_id="20260608-human",
        root=root,
        spec_provider=forbidden_spec_provider,
    )

    prompt_path = root / "pipeline" / "runs" / "20260608-human" / "prompts" / "manual-game.prompt.txt"
    assert result["run_status"] == "waiting_for_human_specs"
    assert result["specs_dir"] == "pipeline/runs/20260608-human/specs"
    assert prompt_path.read_text(encoding="utf-8").startswith("Write a structured spec.\n\nGame idea JSON:\n")


def test_pipeline_cli_implement_specs_delegates_to_spec_runner(tmp_path: Path, monkeypatch):
    root = tmp_path
    spec_dir = root / "pipeline" / "specs" / "my_batch"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "one.md").write_text("# One\n", encoding="utf-8")
    prompt_path = root / "pipeline" / "prompts" / "codex_check_visual_prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("Implement this spec.\n", encoding="utf-8")

    monkeypatch.setattr("pipeline.cli.repo_root", lambda: root)
    monkeypatch.setattr("pipeline.cli._timestamp_run_id", lambda: "20260608-120000")
    captured: dict[str, object] = {}

    def fake_run_implementation_from_specs(**kwargs):
        captured.update(kwargs)
        return {"workers_dir": "pipeline/runs/20260608-120000-my-impl-run/workers"}

    monkeypatch.setattr("pipeline.cli.run_implementation_from_specs", fake_run_implementation_from_specs)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pipeline.cli", "implement-specs", "--spec-dir", "pipeline/specs/my_batch", "--run-name", "my impl run"],
    )

    assert main() == 0
    assert captured["run_id"] == "20260608-120000-my-impl-run"
    assert captured["spec_dir"] == spec_dir
    assert captured["root"] == root
    assert captured["implementation_prompt_path"] == prompt_path
    assert captured["implementation_backend"] == "codex"


def test_run_implementation_from_specs_launches_workers_without_spec_generation(tmp_path: Path, monkeypatch):
    root = tmp_path
    spec_dir = root / "pipeline" / "specs" / "batch"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "zeta_game.md").write_text("# Zeta Game\n", encoding="utf-8")
    (spec_dir / "alpha-game.md").write_text("# Alpha Game\n", encoding="utf-8")
    prompt_path = root / "pipeline" / "prompts" / "codex_check_visual_prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("Implement this spec.\n", encoding="utf-8")

    launched: list[tuple[str, Path]] = []

    def fake_prepare_and_launch_worker_package(**kwargs):
        launched.append((str(kwargs["idea_id"]), Path(kwargs["spec_path"])))
        worker_root = Path(kwargs["worker_root"])
        return {
            "idea_id": str(kwargs["idea_id"]),
            "worker_root": str(worker_root),
            "worker_spec_path": str(kwargs["spec_path"]),
            "status": "done",
            "game_id": f"{kwargs['idea_id']}-0001",
        }

    monkeypatch.setattr("pipeline.cli.prepare_and_launch_worker_package", fake_prepare_and_launch_worker_package)

    result = run_implementation_from_specs(
        run_id="20260608-impl", spec_dir=spec_dir, root=root, implementation_prompt_path=prompt_path
    )

    assert sorted(launched) == [("alpha-game", spec_dir / "alpha-game.md"), ("zeta-game", spec_dir / "zeta_game.md")]
    assert result["source_mode"] == "spec_dir"
    assert result["run_status"] == "completed"
    assert result["succeeded_count"] == 2
    assert result["failed_count"] == 0
    assert [entry["idea_id"] for entry in result["outputs"]] == ["alpha-game", "zeta-game"]
    run_json = json.loads((root / "pipeline" / "runs" / "20260608-impl" / "run.json").read_text(encoding="utf-8"))
    assert run_json == result


def test_build_game_catalog_collects_only_ideas(tmp_path: Path):
    root = tmp_path
    native_dir = root / "re_arc" / "environment_files" / "native_game" / "0001"
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "metadata.json").write_text(
        (
            "{\n"
            '  "game_id": "native-game-0001",\n'
            '  "title": "Native Game",\n'
            '  "idea": "Collect every token.",\n'
            '  "default_fps": 8,\n'
            '  "tags": ["keys-only", "timed"],\n'
            '  "class_name": "NativeGame"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    imported_dir = root / "re_arc" / "environment_files" / "imported_game" / "abcd1234"
    imported_dir.mkdir(parents=True, exist_ok=True)
    (imported_dir / "metadata.json").write_text(
        (
            "{\n"
            '  "game_id": "imported-game-abcd1234",\n'
            '  "title": "Imported Game",\n'
            '  "idea": "Reference game.",\n'
            '  "default_fps": 6,\n'
            '  "tags": ["timed"],\n'
            '  "class_name": "ImportedGame",\n'
            '  "game_dir_url": "https://example.com/game"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    catalog = build_game_catalog(root=root)

    assert catalog == [{"idea": "Reference game."}, {"idea": "Collect every token."}]


def test_pipeline_cli_catalog_games_writes_default_output(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path
    metadata_dir = root / "re_arc" / "environment_files" / "sample_game" / "0001"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "metadata.json").write_text(
        (
            "{\n"
            '  "game_id": "sample-game-0001",\n'
            '  "title": "Sample Game",\n'
            '  "idea": "Reach the exit.",\n'
            '  "default_fps": 8,\n'
            '  "tags": ["keys-only"],\n'
            '  "class_name": "SampleGame"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("pipeline.cli.repo_root", lambda: root)
    monkeypatch.setattr(sys, "argv", ["pipeline.cli", "catalog-games"])

    assert main() == 0

    output = capsys.readouterr().out.strip()
    assert output == "pipeline/game_ideas/catalog.json"
    catalog_path = root / output
    assert catalog_path.exists()
    assert catalog_path.read_text(encoding="utf-8") == '[\n  {\n    "idea": "Reach the exit."\n  }\n]\n'


def test_run_pipeline_resolves_legacy_prompt_paths_to_pipeline_prompts(tmp_path: Path):
    root = tmp_path
    idea_path = root / "pipeline" / "game_ideas" / "ideas.json"
    idea_path.parent.mkdir(parents=True, exist_ok=True)
    idea_path.write_text('[{"title":"One","description":"Desc","num_levels":3}]\n', encoding="utf-8")

    prompts_dir = root / "pipeline" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "spec_prompt.md").write_text("Write a structured game spec.\n", encoding="utf-8")
    (prompts_dir / "codex_check_visual_prompt.md").write_text(
        "Implement one game from the provided spec.\n", encoding="utf-8"
    )

    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        (root / filename).write_text(f"{filename}\n", encoding="utf-8")
    (root / "re_arc" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "re_arc" / "__init__.py").write_text("", encoding="utf-8")
    agents_root = root / "re_arc" / "dsl" / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)
    (agents_root / "__init__.py").write_text("", encoding="utf-8")
    for slug in ("ft09", "ls20", "vc33"):
        source_env_dir = root / "re_arc" / "environment_files" / slug / "0001"
        source_env_dir.mkdir(parents=True, exist_ok=True)
        (source_env_dir / f"{slug}.py").write_text(f"# env {slug}\n", encoding="utf-8")
        (source_env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        (agents_root / f"{slug}.py").write_text(f"# agent {slug}\n", encoding="utf-8")
    for slug in ("ft09_close", "ls20_close"):
        reference_env_dir = root / "pipeline" / "reference_games" / "re_arc" / "environment_files" / slug / "0001"
        reference_env_dir.mkdir(parents=True, exist_ok=True)
        (reference_env_dir / f"{slug}.py").write_text(f"# env {slug}\n", encoding="utf-8")
        (reference_env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        reference_agent_path = root / "pipeline" / "reference_games" / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        reference_agent_path.parent.mkdir(parents=True, exist_ok=True)
        reference_agent_path.write_text(f"# agent {slug}\n", encoding="utf-8")

    captured_prompts: list[str] = []

    def fake_spec_provider(*, api_key: str, model: str, prompt: str, reasoning_effort: str) -> str:
        _ = api_key, model, reasoning_effort
        captured_prompts.append(prompt)
        return "# One\n"

    def fake_codex_runner(cmd, cwd, log_path):
        output_path = Path(str(cmd[cmd.index("-o") + 1]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("implementation complete\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("codex log\n", encoding="utf-8")
        slug = cwd.parent.name.replace("-", "_")
        env_dir = cwd / "re_arc" / "environment_files" / slug / "0001"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / f"{slug}.py").write_text("# env\n", encoding="utf-8")
        (env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        agent_path = cwd / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        agent_path.write_text("# agent\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    def fake_artifact_runner(cmd, cwd, log_path):
        _ = cwd
        gif_path = Path(str(cmd[cmd.index("--gif") + 1]))
        replay_path = Path(str(cmd[cmd.index("--replay") + 1]))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.write_text("gif\n", encoding="utf-8")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text('{"version":2}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("artifact generation log\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    run_pipeline(
        api_key="test-key",
        config=PipelineRunConfig(
            idea_json="pipeline/game_ideas/ideas.json",
            implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
            spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
        ),
        run_id="20260401-180000",
        root=root,
        spec_provider=fake_spec_provider,
        codex_runner=fake_codex_runner,
        artifact_runner=fake_artifact_runner,
    )

    assert len(captured_prompts) == 1
    assert captured_prompts[0].startswith("Write a structured game spec.\n\nGame idea JSON:\n")
    assert '"title": "One"' in captured_prompts[0]
    assert '"description": "Desc"' in captured_prompts[0]
    assert '"num_levels": 3' in captured_prompts[0]


def test_pipeline_cli_review_delegates_to_review_run(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path
    run_id = "20260324-180010"
    review_dir = root / "pipeline" / "runs" / run_id / "review"

    monkeypatch.setattr("pipeline.cli.repo_root", lambda: root)
    captured: dict[str, object] = {}

    def fake_review_run(*, run_id: str, root: Path | None, host: str | None, port: int | None):
        captured["run_id"] = run_id
        captured["root"] = root
        captured["host"] = host
        captured["port"] = port
        return {"review_dir": str(review_dir)}

    monkeypatch.setattr("pipeline.cli.review_run", fake_review_run)
    monkeypatch.setattr(
        sys, "argv", ["pipeline.cli", "review", "--run-id", run_id, "--host", "0.0.0.0", "--port", "8123"]
    )

    assert main() == 0
    assert capsys.readouterr().out.strip() == str(review_dir)
    assert captured == {"run_id": run_id, "root": root, "host": "0.0.0.0", "port": 8123}


def test_run_pipeline_processes_ideas_in_parallel_with_stable_output_order(tmp_path: Path):
    root = tmp_path
    idea_path = root / "pipeline" / "game_ideas" / "ideas.json"
    idea_path.parent.mkdir(parents=True, exist_ok=True)
    idea_path.write_text(
        "\n".join(
            [
                "[",
                '  {"title": "Slow Game", "description": "Takes longer.", "num_levels": 4},',
                '  {"title": "Fast Game", "description": "Finishes quickly.", "num_levels": 3}',
                "]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec_instructions_path = root / "pipeline" / "prompts_and_schemas" / "spec_generation_instructions.md"
    spec_instructions_path.parent.mkdir(parents=True, exist_ok=True)
    spec_instructions_path.write_text("Write a structured game spec.\n", encoding="utf-8")
    implementation_prompt_path = root / "pipeline" / "prompts_and_schemas" / "implementation_codex_prompt.md"
    implementation_prompt_path.write_text("Implement one game from the provided spec.\n", encoding="utf-8")
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        (root / filename).write_text(f"{filename}\n", encoding="utf-8")
    (root / "re_arc" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "re_arc" / "__init__.py").write_text("", encoding="utf-8")
    (root / "re_arc" / "dsl" / "agents" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "re_arc" / "dsl" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "test_sample.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    for slug in ("ft09", "ls20", "vc33"):
        source_env_dir = root / "re_arc" / "environment_files" / slug / "0001"
        source_env_dir.mkdir(parents=True, exist_ok=True)
        (source_env_dir / f"{slug}.py").write_text(f"# env {slug}\n", encoding="utf-8")
        (source_env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        source_agent_path = root / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        source_agent_path.write_text(f"# agent {slug}\n", encoding="utf-8")
    for slug in ("ft09_close", "ls20_close"):
        reference_env_dir = root / "pipeline" / "reference_games" / "re_arc" / "environment_files" / slug / "0001"
        reference_env_dir.mkdir(parents=True, exist_ok=True)
        (reference_env_dir / f"{slug}.py").write_text(f"# env {slug}\n", encoding="utf-8")
        (reference_env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        reference_agent_path = root / "pipeline" / "reference_games" / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        reference_agent_path.parent.mkdir(parents=True, exist_ok=True)
        reference_agent_path.write_text(f"# agent {slug}\n", encoding="utf-8")

    completion_order: list[str] = []

    def fake_spec_provider(*, api_key: str, model: str, prompt: str, reasoning_effort: str) -> str:
        _ = api_key, model, reasoning_effort
        if "Slow Game" in prompt:
            time.sleep(0.05)
            completion_order.append("slow-game")
            return "# Slow Game\n"
        time.sleep(0.01)
        completion_order.append("fast-game")
        return "# Fast Game\n"

    def fake_codex_runner(cmd, cwd, log_path):
        output_path = Path(str(cmd[cmd.index("-o") + 1]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("implementation complete\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("codex log\n", encoding="utf-8")
        slug = cwd.parent.name.replace("-", "_")
        env_dir = cwd / "re_arc" / "environment_files" / slug / "0001"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / f"{slug}.py").write_text("# env\n", encoding="utf-8")
        (env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        agent_path = cwd / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        agent_path.write_text("# agent\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    def fake_artifact_runner(cmd, cwd, log_path):
        _ = cwd
        gif_path = Path(str(cmd[cmd.index("--gif") + 1]))
        replay_path = Path(str(cmd[cmd.index("--replay") + 1]))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.write_text("gif\n", encoding="utf-8")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text('{"version":2}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("artifact generation log\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    result = run_pipeline(
        api_key="test-key",
        config=PipelineRunConfig(
            idea_json="pipeline/game_ideas/ideas.json",
            implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
            spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
        ),
        run_id="20260327-120000",
        root=root,
        spec_provider=fake_spec_provider,
        codex_runner=fake_codex_runner,
        artifact_runner=fake_artifact_runner,
    )

    assert completion_order == ["fast-game", "slow-game"]
    assert [entry["idea_id"] for entry in result["outputs"]] == ["slow-game", "fast-game"]
    assert [entry["idea_id"] for entry in result["worker_packages"]] == ["slow-game", "fast-game"]


def test_run_pipeline_records_worker_failure_and_continues(tmp_path: Path):
    root = tmp_path
    idea_path = root / "pipeline" / "game_ideas" / "ideas.json"
    idea_path.parent.mkdir(parents=True, exist_ok=True)
    idea_path.write_text(
        "\n".join(
            [
                "[",
                '  {"title": "Good Game", "description": "Completes.", "num_levels": 4},',
                '  {"title": "Bad Game", "description": "Fails.", "num_levels": 3}',
                "]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec_instructions_path = root / "pipeline" / "prompts_and_schemas" / "spec_generation_instructions.md"
    spec_instructions_path.parent.mkdir(parents=True, exist_ok=True)
    spec_instructions_path.write_text("Write a structured game spec.\n", encoding="utf-8")
    implementation_prompt_path = root / "pipeline" / "prompts_and_schemas" / "implementation_codex_prompt.md"
    implementation_prompt_path.write_text("Implement one game from the provided spec.\n", encoding="utf-8")
    for filename in ("AGENTS.md", "README.md", "config.env", "pyproject.toml", "uv.lock"):
        (root / filename).write_text(f"{filename}\n", encoding="utf-8")
    (root / "re_arc" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "re_arc" / "__init__.py").write_text("", encoding="utf-8")
    agents_root = root / "re_arc" / "dsl" / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)
    (agents_root / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "test_sample.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    for slug in ("ft09", "ls20", "vc33"):
        source_env_dir = root / "re_arc" / "environment_files" / slug / "0001"
        source_env_dir.mkdir(parents=True, exist_ok=True)
        (source_env_dir / f"{slug}.py").write_text(f"# env {slug}\n", encoding="utf-8")
        (source_env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        (agents_root / f"{slug}.py").write_text(f"# agent {slug}\n", encoding="utf-8")
    for slug in ("ft09_close", "ls20_close"):
        reference_env_dir = root / "pipeline" / "reference_games" / "re_arc" / "environment_files" / slug / "0001"
        reference_env_dir.mkdir(parents=True, exist_ok=True)
        (reference_env_dir / f"{slug}.py").write_text(f"# env {slug}\n", encoding="utf-8")
        (reference_env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        reference_agent_path = root / "pipeline" / "reference_games" / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        reference_agent_path.parent.mkdir(parents=True, exist_ok=True)
        reference_agent_path.write_text(f"# agent {slug}\n", encoding="utf-8")

    def fake_spec_provider(*, api_key: str, model: str, prompt: str, reasoning_effort: str) -> str:
        _ = api_key, model, reasoning_effort
        if "Bad Game" in prompt:
            raise RuntimeError("transient model failure")
        return "# Good Game\n"

    def fake_codex_runner(cmd, cwd, log_path):
        output_path = Path(str(cmd[cmd.index("-o") + 1]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("implementation complete\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("codex log\n", encoding="utf-8")
        slug = cwd.parent.name.replace("-", "_")
        env_dir = cwd / "re_arc" / "environment_files" / slug / "0001"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / f"{slug}.py").write_text("# env\n", encoding="utf-8")
        (env_dir / "metadata.json").write_text(f'{{"game_id": "{slug}-0001"}}\n', encoding="utf-8")
        agent_path = cwd / "re_arc" / "dsl" / "agents" / f"{slug}.py"
        agent_path.write_text("# agent\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    def fake_artifact_runner(cmd, cwd, log_path):
        _ = cwd
        gif_path = Path(str(cmd[cmd.index("--gif") + 1]))
        replay_path = Path(str(cmd[cmd.index("--replay") + 1]))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.write_text("gif\n", encoding="utf-8")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text('{"version":2}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("artifact generation log\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    result = run_pipeline(
        api_key="test-key",
        config=PipelineRunConfig(
            idea_json="pipeline/game_ideas/ideas.json",
            implementation_prompt_path="pipeline/prompts_and_schemas/implementation_codex_prompt.md",
            spec_instructions_path="pipeline/prompts_and_schemas/spec_generation_instructions.md",
        ),
        run_id="20260327-130000",
        root=root,
        spec_provider=fake_spec_provider,
        codex_runner=fake_codex_runner,
        artifact_runner=fake_artifact_runner,
    )

    assert result["run_status"] == "completed_with_failures"
    assert result["succeeded_count"] == 1
    assert result["failed_count"] == 1
    assert [entry["idea_id"] for entry in result["outputs"]] == ["good-game", "bad-game"]
    assert result["outputs"][1]["status"] == "failed"
    assert result["outputs"][1]["error"] == "transient model failure"
    assert result["worker_packages"][0]["idea_id"] == "good-game"
    assert result["worker_packages"][1]["status"] == "failed"

    run_json = json.loads((root / "pipeline" / "runs" / "20260327-130000" / "run.json").read_text(encoding="utf-8"))
    assert run_json == result
