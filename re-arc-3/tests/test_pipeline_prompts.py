from __future__ import annotations

from pathlib import Path

from pipeline.config import load_run_config
from pipeline.idea_loading import load_idea_specs

ROOT = Path(__file__).resolve().parents[1]


def test_compact_axis_generation_inputs_are_complete():
    config_path = ROOT / "pipeline" / "configs" / "compact_axis_seed_gpt_pro_human.yaml"
    config = load_run_config(config_path)

    assert config.idea_json is not None
    idea_path = ROOT / config.idea_json
    implementation_prompt = ROOT / config.implementation_prompt_path
    spec_prompt = ROOT / config.spec_instructions_path

    assert len(load_idea_specs(idea_path)) == 30
    assert implementation_prompt.is_file()
    assert spec_prompt.is_file()


def test_pre_pipeline_prompts_preserve_the_recent_generation_contract():
    prompt_root = ROOT / "pipeline" / "prompts"
    axis_prompt = (prompt_root / "axis_generation.md").read_text(encoding="utf-8")
    sample_prompt = (prompt_root / "sample_from_axis.md").read_text(encoding="utf-8")
    idea_prompt = (prompt_root / "idea_prompt.md").read_text(encoding="utf-8")

    assert "exactly 10" in axis_prompt
    assert "## sampling" not in axis_prompt
    assert "Generate exactly 30 game seeds" in sample_prompt
    assert "exactly 3 or 4 axes" in sample_prompt
    assert "The supplied seed is authoritative" in idea_prompt
    assert "Expand that seed into exactly one complete game idea" in idea_prompt
    assert "Default to a click-free design" in idea_prompt


def test_axis_prompt_replaces_only_the_category_section():
    axis_prompt = (ROOT / "pipeline" / "prompts" / "axis_generation.md").read_text(encoding="utf-8")
    category_slot = (
        "**Replace only this section when exploring a different game category.**\n\n"
        "[GAME CATEGORY SPECIFICATION GOES HERE]"
    )
    click_category = (
        "The player interacts only by clicking visibly interactive cells, objects, or regions; "
        "there is no persistent navigable agent.\n\n"
        "Solving the game must require reasoning about how each click transforms the visible state "
        "and changes which future interactions or configurations remain possible."
    )

    assert axis_prompt.count(category_slot) == 1
    rendered_prompt = axis_prompt.replace(category_slot, click_category)
    assert rendered_prompt.replace(click_category, category_slot) == axis_prompt
    assert "GAME CATEGORY SPECIFICATION GOES HERE" not in rendered_prompt
    assert "Replace only this section" not in rendered_prompt


def test_pipeline_prompts_require_learnability_and_game_native_undo():
    prompt_root = ROOT / "pipeline" / "prompts"
    spec_prompt = (prompt_root / "spec_prompt.md").read_text(encoding="utf-8")
    implementation_prompt = (prompt_root / "codex_check_visual_prompt.md").read_text(encoding="utf-8")

    assert "The most important design goal is epistemic solvability" in spec_prompt
    assert "Introduce at most one new causal unknown per level" in spec_prompt
    assert "Every generated game must provide game-native single-step undo" in implementation_prompt
    assert "GameAction.ACTION7" in implementation_prompt
