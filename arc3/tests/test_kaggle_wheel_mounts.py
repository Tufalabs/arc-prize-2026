"""Kaggle presets link one engine descriptor (§1): the solver derives the
kernel sources, setup commands, exclusions and payload from it, and the duck
alone overlays the vLLM fork on top of what the wheels install."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest
import ullm.kaggle as ullm_kaggle
import ullm.vllm_overlay as vllm_overlay

import platypus
from platypus import RunConfig
from reference_agents import PromptedRandomAgent

duck_presets = pytest.importorskip("duck.presets", reason="needs the vllm extra (linux)")
duck_agent = pytest.importorskip("duck.agent", reason="needs the vllm extra (linux)")


def _stock_agent() -> PromptedRandomAgent:
    """An agent linking the STOCK engine, which is the contrast every test
    below is about. Its run config is the whole input: the solver derives the
    sources, commands and payload from ``kaggle_engine`` alone."""
    return PromptedRandomAgent(run_config=RunConfig(kaggle_engine=ullm_kaggle.VLLM_STOCK))


def test_serving_columns_link_the_shared_engine() -> None:
    """Both install from the one wheelhouse with the one requirement set;
    what differs is whether the fork rides along, and which checkpoints the
    cell has read ahead of its loader."""
    duck_config = duck_agent.DuckAgent.build(environment="kaggle-rtx").run_config
    stock_config = _stock_agent().run_config

    assert duck_config.kaggle_engine is not None
    assert dataclasses.replace(duck_config.kaggle_engine, prefetch=()) == ullm_kaggle.VLLM_FORK
    assert duck_config.kaggle_engine.prefetch  # the Flash cell names its checkpoints
    assert stock_config.kaggle_engine is ullm_kaggle.VLLM_STOCK
    assert duck_config.kaggle_kernel_sources == stock_config.kaggle_kernel_sources == ()
    assert duck_config.kaggle_setup_commands == stock_config.kaggle_setup_commands == ()


def test_the_solver_puts_the_engine_ahead_of_the_agent_s_own() -> None:
    """TAAF reads the solver's properties: the engine's wheelhouse, install
    and overlay come first, the agent's own sources and commands after, and
    the fork's repo is excluded once however many ways it is named."""
    agent = duck_agent.DuckAgent.build(environment="kaggle-rtx")
    solver = platypus.LLMAgentSolver(agent=agent)
    expected = ullm_kaggle.kaggle_site_install_command(*ullm_kaggle.KAGGLE_VLLM_REQUIREMENTS)

    assert solver.kaggle_kernel_sources == [ullm_kaggle.KAGGLE_VLLM_WHEELS]
    assert agent.run_config.kaggle_engine is not None
    assert solver.kaggle_setup_commands == [
        ullm_kaggle.kaggle_prefetch_command(*agent.run_config.kaggle_engine.prefetch),
        expected,
        ullm_kaggle.kaggle_cutlass_dsl_path_command(),
        ullm_kaggle.kaggle_vllm_overlay_command(),
    ]
    assert solver.kaggle_exclude_source_repos == [ullm_kaggle.VLLM_FORK_REPO]

    stock = platypus.LLMAgentSolver(agent=_stock_agent())
    assert stock.kaggle_setup_commands == [expected, ullm_kaggle.kaggle_cutlass_dsl_path_command()]
    assert stock.kaggle_exclude_source_repos == [ullm_kaggle.VLLM_FORK_REPO]  # the workspace's own exclusion


def test_solver_builds_the_overlay_payload_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """TAAF reads ``kaggle_bundle_files`` once on the launcher while it
    assembles a deployment, which is where the overlay gets built against
    the workspace the deploy runs from — so the agent that does not want one
    must not pay for it. The build is the LLM solver's; the LLM-free base
    ships only what the run config names, even for a run that links the
    fork engine."""
    built = pathlib.Path("/tmp/built") / vllm_overlay.OVERLAY_PAYLOAD_NAME
    seen: list[pathlib.Path | None] = []

    def _fake_build(dest_dir: pathlib.Path | None = None, *, workspace: pathlib.Path | None = None) -> pathlib.Path:
        seen.append(workspace)
        return built

    monkeypatch.setattr(vllm_overlay, "build_for_deploy", _fake_build)

    stock = platypus.LLMAgentSolver(agent=_stock_agent())
    assert stock.kaggle_bundle_files == []
    assert not seen

    forked = platypus.LLMAgentSolver(agent=duck_agent.DuckAgent.build(environment="kaggle-rtx"))
    assert forked.kaggle_bundle_files == [str(built)]
    assert seen == [pathlib.Path.cwd()]

    assert platypus.AgentSolver(agent=duck_agent.DuckAgent.build(environment="kaggle-rtx")).kaggle_bundle_files == []
    assert len(seen) == 1
