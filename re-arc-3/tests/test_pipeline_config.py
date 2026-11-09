from __future__ import annotations

from pathlib import Path

from pipeline.config import load_run_config


def test_load_run_config_from_yaml(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/debugging_ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "model: gpt-5.4",
                "reasoning_effort: high",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.idea_json == "pipeline/game_ideas/debugging_ideas.json"
    assert config.model == "gpt-5.4"
    assert config.reasoning_effort == "high"
    assert config.executor == "local"
    assert config.slurm_partition == "llm"
    assert config.slurm_time == "01:30:00"
    assert config.slurm_mem == "8G"
    assert config.slurm_ntasks == 1
    assert config.slurm_cpus_per_task == 2
    assert config.implementation_prompt_path == "pipeline/prompts_and_schemas/implementation_codex_prompt.md"
    assert config.implementation_backend == "codex"


def test_load_run_config_accepts_slurm_executor(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/debugging_ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
                "executor: slurm",
                "slurm_partition: llm",
                "slurm_time: 01:00:00",
                "slurm_mem: 16G",
                "slurm_ntasks: 1",
                "slurm_cpus_per_task: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.executor == "slurm"
    assert config.slurm_partition == "llm"
    assert config.slurm_time == "01:00:00"
    assert config.slurm_mem == "16G"
    assert config.slurm_ntasks == 1
    assert config.slurm_cpus_per_task == 4


def test_load_run_config_accepts_official_description_source(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "official_description_paths:",
                "  - arc_official_description/ft09_flip_tiles.md",
                "  - arc_official_description/ls20_locksmith.md",
                "official_description_num_levels: 5",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.idea_json is None
    assert config.official_description_paths == (
        "arc_official_description/ft09_flip_tiles.md",
        "arc_official_description/ls20_locksmith.md",
    )
    assert config.official_description_num_levels == 5


def test_load_run_config_accepts_claude_backend(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/debugging_ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
                "implementation_backend: claude",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.implementation_backend == "claude"


def test_load_run_config_accepts_xhigh_reasoning_effort(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/debugging_ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
                "reasoning_effort: xhigh",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.reasoning_effort == "xhigh"


def test_load_run_config_accepts_background_spec_option(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/debugging_ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
                "spec_background: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.spec_background is True


def test_load_run_config_accepts_codex_reasoning_effort(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "idea_json: pipeline/game_ideas/debugging_ideas.json",
                "implementation_prompt_path: pipeline/prompts_and_schemas/implementation_codex_prompt.md",
                "spec_instructions_path: pipeline/prompts_and_schemas/spec_generation_instructions.md",
                "codex_reasoning_effort: high",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.codex_reasoning_effort == "high"
