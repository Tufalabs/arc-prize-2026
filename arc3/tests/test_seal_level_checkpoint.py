"""The seal under level checkpoints (I2 ``"level"``, A4) — no API.

On a scripted game: every ``persistence`` that declares ``"level"`` must leave
no kernel at a level start, since the checkpoint is a pickle and a live
kernel refuses one (A4). The duck's second half, a branch from its pathfinder
reference's level checkpoint replayed against the recording's remaining calls,
waits for a seal recording of that game.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Literal

import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from platypus.checkpoints import AGENT_FILE, CHECKPOINTS_DIR, MANIFEST_FILE
from reference_support import tool_call
from seal import SealAgent
from test_seal_scripted_reference import CornerGame


@pytest.mark.parametrize("persistence", ["round", "move", "level"])
async def test_every_persistence_declaring_level_leaves_no_kernel_at_a_level_start(
    tmp_path: Path, persistence: Literal["round", "move", "level"]
) -> None:
    # A mode that declares "level" yet keeps its kernel across the move
    # boundary crashes the game at its first level win, where the pickle
    # refuses the live kernel.
    script = [
        tool_call("python", {"code": "submit_action([Action('UP'), Action('UP')])\n"}),  # wins level 0 in a snippet
        ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="Level won.")),  # ends the move there
        tool_call("python", {"code": "submit_action([Action('DOWN'), Action('DOWN')])\n"}),  # wins the game
    ]
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        multimodal_context="",
        persistence=persistence,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=ullm.backends.stub.StubBackend(responses=script, on_exhausted="repeat_last")
            ),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess", checkpointing="level"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[CornerGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=180)
    (run,) = benchmark.game_runs
    assert run.state == "won", run.solver_note

    (checkpoint,) = sorted((tmp_path / CHECKPOINTS_DIR).glob("*/p0/L*"))
    assert json.loads((checkpoint / MANIFEST_FILE).read_text())["level"] == 1
    seal = pickle.loads((checkpoint / AGENT_FILE).read_bytes())
    assert seal.repl is None or seal.repl.is_fresh
