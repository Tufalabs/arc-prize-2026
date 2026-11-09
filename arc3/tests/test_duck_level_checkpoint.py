"""The duck under level checkpoints (I2 ``"level"``, I4) — no API.

On a scripted game: every ``persistence`` that declares ``"level"`` must leave
no kernel at a level start, since the checkpoint is a pickle and a live
kernel refuses one (A4).

On the pathfinder reference's recorded calls alone: one replay of the
reference with checkpoints on must still reproduce it bit-exact: taking a
checkpoint may not change play. The checkpoint it leaves
where the game crosses from level index 1 into 2 must hold a duck ready to branch
from — kernel gone (A4), conversation carried, no engine thread name of the
source's. And a game started from that checkpoint (its engine and history
rebuilt by replaying the source's actions), replayed against only the calls
the recording made after it (``branch_rng="resume"``, so its seeds are the
recording's), must make exactly those calls and play exactly those actions:
the branch IS the recorded game's continuation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Literal

import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.capture
import ullm.conv
import ullm.factory

import platypus.checkpoints
import platypus.llm
from duck import DuckAgent
from platypus import LLMAgentSolver, LLMConfig, RunConfig
from platypus.checkpoint_games import CheckpointGameAPI
from platypus.checkpoints import AGENT_FILE, CHECKPOINTS_DIR, MANIFEST_FILE
from platypus.reference import (
    game_run_dir,
    load_reference_seeds,
    load_references,
    replay_references,
)
from reference_support import ReferenceRun, replay_job_dir, tool_call
from test_duck_pathfinder_reference import CASE, START_AT_LEVEL, STOP_AT_LEVEL
from test_duck_scripted_reference import CornerGame

GAME_ID = "pathfinder-0001"
BRANCH_ID = f"{GAME_ID}@L{START_AT_LEVEL + 1}"


@pytest.mark.parametrize("persistence", ["round", "move", "level"])
async def test_every_persistence_declaring_level_leaves_no_kernel_at_a_level_start(
    tmp_path: Path, persistence: Literal["round", "move", "level"]
) -> None:
    # A mode that declares "level" yet keeps its kernel across the move
    # boundary crashes the game at its first level win, where the pickle
    # refuses the live kernel.
    script = [
        tool_call("python", {"code": "submit_action(['UP','UP'])\n"}),  # wins level 0 in a snippet
        ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="Level won.")),  # ends the move there
        tool_call("python", {"code": "submit_action(['DOWN','DOWN'])\n"}),  # wins the game
    ]
    agent = DuckAgent(
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
    duck = pickle.loads((checkpoint / AGENT_FILE).read_bytes())
    assert duck.repl is None or duck.repl.is_fresh


def _build(run: ReferenceRun, **run_config: Any) -> taaf.benchmark.Benchmark:
    benchmark = CASE.build(run)
    solver = benchmark.solver
    assert isinstance(solver, LLMAgentSolver)
    agent = solver.agent
    assert isinstance(agent, DuckAgent)
    agent.run_config = dataclasses.replace(agent.run_config, **run_config)
    return benchmark


def _actions(benchmark: taaf.benchmark.Benchmark) -> list[str]:
    (run,) = benchmark.game_runs
    return [record.action.id.name for record in run.history]


def _unconsumed(benchmark: taaf.benchmark.Benchmark) -> int:
    (run,) = benchmark.game_runs
    assert benchmark.job_dir is not None
    sidecar = game_run_dir(benchmark.job_dir, run.run_id) / platypus.llm.REPLAY_CONSUMPTION_FILE
    return int(json.loads(sidecar.read_text())["unconsumed"])


def _stream_from(source: Path, dest: Path, first: int) -> None:
    """The recorded stream from call ``first`` on, renumbered from 0: what a
    game resuming after ``first`` calls asks for next."""
    dest.mkdir(parents=True)
    shutil.copy(source / "blocks.jsonl", dest / "blocks.jsonl")
    records = [json.loads(line) for line in (source / "records.jsonl").read_text().splitlines()]
    with (dest / "records.jsonl").open("w") as out:
        for index, record in enumerate(records[first:]):
            out.write(json.dumps({**record, "index": index}) + "\n")


async def test_duck_branches_from_a_level_checkpoint_as_the_recording_continued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The source replay, in process so the calls it serves and the moment it
    # checkpoints can be put in one order: the branch's stream starts at the
    # first call the recording made after the checkpoint.
    order: list[str] = []
    next_record = ullm.capture.ReplayBackend._next_record
    text_token_count = ullm.capture.ReplayBackend.text_token_count
    write = platypus.checkpoints.CheckpointWriter.write

    def logged_next_record(self: ullm.capture.ReplayBackend, *args: Any) -> Any:
        order.append("call")
        return next_record(self, *args)

    async def logged_text_token_count(self: ullm.capture.ReplayBackend, text: str) -> Any:
        order.append("call")
        return await text_token_count(self, text)

    def logged_write(self: platypus.checkpoints.CheckpointWriter, *args: Any) -> Path:
        order.append("checkpoint")
        return write(self, *args)

    monkeypatch.setattr(ullm.capture.ReplayBackend, "_next_record", logged_next_record)
    monkeypatch.setattr(ullm.capture.ReplayBackend, "text_token_count", logged_text_token_count)
    monkeypatch.setattr(platypus.checkpoints.CheckpointWriter, "write", logged_write)

    paths = CASE.paths()
    unpacked = tmp_path / "unpacked"
    expected = load_references(paths, unpacked)
    source_dir = replay_job_dir(tmp_path)
    source = _build(
        ReferenceRun(
            llm_config=LLMConfig(sampling_profiles=CASE.profiles, replay_dir=str(unpacked)),
            job_dir=source_dir,
            seeds=load_reference_seeds(paths),
        ),
        checkpointing="level",
        process_mode="inprocess",
    )
    await replay_references(source, expected)
    monkeypatch.undo()

    # The checkpoint: one, at the level the game crossed into, holding a duck
    # with no kernel and no thread of its own yet.
    (checkpoint,) = sorted((source_dir / CHECKPOINTS_DIR).glob("*/p0/L*"))
    manifest = json.loads((checkpoint / MANIFEST_FILE).read_text())
    assert manifest["level"] == START_AT_LEVEL + 1
    assert manifest["start_level"] == START_AT_LEVEL, "the source played a level window: its replay starts there"
    duck = pickle.loads((checkpoint / AGENT_FILE).read_bytes())
    assert isinstance(duck, DuckAgent)
    assert duck.messages, "the conversation did not travel with the checkpoint"
    assert duck.repl is None or duck.repl.is_fresh
    assert not duck.pending_level_reset
    assert "_thread_conversation_id" not in vars(duck)
    assert order.count("checkpoint") == 1
    calls_before = order.index("checkpoint")
    calls_after = len(order) - calls_before - 1
    assert calls_after, "the recording made no call after the checkpoint: nothing to branch into"

    # The branch: that level on its own, from the checkpoint, against only
    # the calls the recording made after it.
    branch_streams = tmp_path / "branch_streams"
    _stream_from(unpacked / GAME_ID, branch_streams / BRANCH_ID, calls_before)
    branch = _build(
        ReferenceRun(
            llm_config=LLMConfig(sampling_profiles=CASE.profiles, replay_dir=str(branch_streams)),
            job_dir=tmp_path / "branch",
            seeds={},
        ),
        branch_rng="resume",
    )
    branch.games = [
        CheckpointGameAPI(
            env_name=GAME_ID, external_game_id=BRANCH_ID, checkpoint=str(checkpoint), stop_at_level=STOP_AT_LEVEL
        )
    ]
    await branch.run()

    (source_run,) = source.game_runs
    (branch_run,) = branch.game_runs
    assert branch_run.state == source_run.state, branch_run.solver_note
    assert _actions(branch) == _actions(source)[len(manifest["actions"]) :]
    assert _unconsumed(branch) == 0, "the branch made fewer calls than the recording did after the checkpoint"
