"""The Kaggle column's offline deployment contract.

Everything a column declares has to exist on Kaggle before the engine is
asked for, and a mismatch surfaces only in a run that has already paid for
the hardware — so the contract is asserted here rather than discovered
there. An agent links one engine descriptor: the solver derives the kernel
sources, setup commands, exclusions and payload from it, and the duck alone
overlays the vLLM fork on top of what the wheels install.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest
import ullm.kaggle as ullm_kaggle
from duck import DuckAgent, presets
from ullm import vllm_overlay

import platypus
from platypus import RunConfig
from platypus.reference_agents import PromptedFlipAgent


def _stock_agent() -> PromptedFlipAgent:
    """An agent linking the STOCK engine, which is the contrast every test
    below is about. Its run config is the whole input: the solver derives the
    sources, commands and payload from ``kaggle_engine`` alone."""
    return PromptedFlipAgent(run_config=RunConfig(kaggle_engine=ullm_kaggle.VLLM_STOCK))


def _kaggle_duck() -> DuckAgent:
    return DuckAgent.build(environment="kaggle-4xl4")


def test_serving_columns_link_the_shared_engine() -> None:
    """Both install the one requirement set; what differs is whether the fork
    rides along, and the duck's wheelhouse, ARC2's: the same wheels, with the
    record an ARC2 kernel makes in place of ullm's, which starts after the
    ARC3 competition's install. The four-L4 Flash cell reads nothing ahead of
    its loader: the engine prefetches, since the kernel's bounded reader
    drops shards its four loaders have yet to read, and it puts no checkpoint
    together: ARC2's REAP-307 is one Kaggle model."""
    duck_config = _kaggle_duck().run_config
    stock_config = _stock_agent().run_config

    assert duck_config.kaggle_engine == dataclasses.replace(ullm_kaggle.VLLM_FORK, wheels=presets.KAGGLE_WHEELS)
    assert stock_config.kaggle_engine is ullm_kaggle.VLLM_STOCK
    assert duck_config.kaggle_kernel_sources == stock_config.kaggle_kernel_sources == ()
    assert duck_config.kaggle_setup_commands == stock_config.kaggle_setup_commands == ()


def test_the_solver_puts_the_engine_ahead_of_the_agent_s_own() -> None:
    """TAAF reads the solver's properties: the engine's wheelhouse, the
    install and the overlay come first, the agent's own sources and commands
    after, and the fork's repo is excluded once however many ways it is
    named."""
    agent = _kaggle_duck()
    solver = platypus.LLMAgentSolver(agent=agent)
    expected = ullm_kaggle.kaggle_site_install_command(
        *ullm_kaggle.KAGGLE_VLLM_REQUIREMENTS, wheels=presets.KAGGLE_WHEELS
    )

    assert solver.kaggle_kernel_sources == [presets.KAGGLE_WHEELS]
    assert agent.run_config.kaggle_engine is not None
    assert solver.kaggle_setup_commands == [
        expected,
        ullm_kaggle.kaggle_cutlass_dsl_path_command(),
        ullm_kaggle.kaggle_vllm_overlay_command(),
    ]
    assert solver.kaggle_exclude_source_repos == [ullm_kaggle.VLLM_FORK_REPO]

    stock = platypus.LLMAgentSolver(agent=_stock_agent())
    assert stock.kaggle_setup_commands == [
        ullm_kaggle.kaggle_site_install_command(*ullm_kaggle.KAGGLE_VLLM_REQUIREMENTS),
        ullm_kaggle.kaggle_cutlass_dsl_path_command(),
    ]
    assert stock.kaggle_exclude_source_repos == [ullm_kaggle.VLLM_FORK_REPO]  # the workspace's own exclusion


def test_the_kernels_run_on_the_image_the_record_was_built_on() -> None:
    """The record replays only onto the installed state it started from, so
    the duck's Kaggle kernels are pinned to the image its notebook is pinned
    to, and install from the wheelhouse that notebook writes."""
    builder = pathlib.Path(__file__).resolve().parents[1] / "tools/kaggle/arc2-site-0290/kernel-metadata.json"
    metadata = json.loads(builder.read_text())
    engine = _kaggle_duck().run_config.kaggle_engine
    assert engine is not None
    assert metadata["docker_image"] == engine.docker_image == ullm_kaggle.KAGGLE_DOCKER_IMAGE
    assert metadata["id"] == engine.wheels == presets.KAGGLE_WHEELS


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

    forked = platypus.LLMAgentSolver(agent=_kaggle_duck())
    assert forked.kaggle_bundle_files == [str(built)]
    assert seen == [pathlib.Path.cwd()]

    assert platypus.AgentSolver(agent=_kaggle_duck()).kaggle_bundle_files == []
    assert len(seen) == 1
