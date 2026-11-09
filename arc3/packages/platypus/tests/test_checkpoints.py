"""Level checkpoints (I2 ``"level"``) and games branched from them (I4).

The shape: a source run over three games stores a checkpoint at
every level start past the first — five levels across the games, five passes
each — and a branch run then plays each of those 25 checkpoints as a game of
its own, from its level to the next, three passes apiece. Each branch game
replays its source's actions before play, which rebuilds the engine and the
history the agent reads (``LevelLearnerAgent`` learns from ``ctx.history``).
Apart from that shape: a move that plays on past a level's first observation
leaves that level no checkpoint, and the refusals.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import arcengine
import pytest
from infra_fixtures import LevelLearnerAgent
from taaf.benchmark import Benchmark
from taaf.game import Game
from taaf.game_examples import ExampleGame

from platypus import Action, Agent, AgentSolver, GiveUp, RunConfig
from platypus.checkpoints import AGENT_FILE, CHECKPOINTS_DIR, MANIFEST_FILE, CheckpointStart, checkpoint_dir
from platypus.context import MoveContext
from platypus.contract import Observation
from platypus.diagnostics import build_span_tree, read_game_events

A1, A2, A3 = arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2, arcengine.GameAction.ACTION3

# Three games, five level starts past the first: a1, a2, b1, b2, c1.
SEQUENCES = {
    "a": [[A1, A2, A3], [A3, A1, A1], [A2, A2, A3]],
    "b": [[A2, A1, A3], [A1, A3, A2], [A3, A3, A1]],
    "c": [[A3, A2, A1], [A1, A1, A2]],
}
SOURCE_PASSES = 5
BRANCH_PASSES = 3


@dataclass
class CheckpointExampleGame(CheckpointStart, ExampleGame):
    """An ``ExampleGame`` started from a level checkpoint."""


def _config(process_mode: str, **overrides: object) -> RunConfig:
    return RunConfig(process_mode=process_mode, rng_seed=1000, **overrides)  # type: ignore[arg-type]


def _source_games() -> list[Game]:
    return [ExampleGame(label=name, target_sequence_per_level=seq) for name, seq in SEQUENCES.items()]


def _branch_points() -> list[tuple[str, int]]:
    return [(name, level) for name, seq in SEQUENCES.items() for level in range(1, len(seq))]


def _branch_id(name: str, level: int, source_pass: int) -> str:
    return f"{name}@L{level}#p{source_pass}"


def _branch_game(root: Path, name: str, level: int, source_pass: int, **fields: object) -> Game:
    """One checkpoint as a game: from its level to the next."""
    return CheckpointExampleGame(
        label=_branch_id(name, level, source_pass),
        target_sequence_per_level=SEQUENCES[name],
        checkpoint=str(checkpoint_dir(root, name, source_pass, level)),
        stop_at_level=level + 1,
        **fields,  # type: ignore[arg-type]
    )


async def _source_run(job_dir: Path, *, process_mode: str, own_rng: bool) -> Benchmark:
    agent = LevelLearnerAgent(own_rng=own_rng, run_config=_config(process_mode, checkpointing="level"))
    benchmark = Benchmark(
        games=_source_games(), solver=AgentSolver(agent=agent), n_passes=SOURCE_PASSES, job_dir=job_dir
    )
    await benchmark.run()
    assert all(run.state == "won" for run in benchmark.game_runs)
    return benchmark


async def _branch_run(
    job_dir: Path, source_root: Path, *, process_mode: str, own_rng: bool, branch_rng: str = "fresh"
) -> Benchmark:
    games = [
        _branch_game(source_root, name, level, source_pass)
        for name, level in _branch_points()
        for source_pass in range(SOURCE_PASSES)
    ]
    # The template only hosts: every game's agent comes from its checkpoint.
    agent = LevelLearnerAgent(own_rng=own_rng, run_config=_config(process_mode, branch_rng=branch_rng))
    benchmark = Benchmark(games=games, solver=AgentSolver(agent=agent), n_passes=BRANCH_PASSES, job_dir=job_dir)
    await benchmark.run()
    return benchmark


def _actions(benchmark: Benchmark, game_id: str, pass_index: int) -> list[str]:
    (run,) = [r for r in benchmark.game_runs if r.game_id == game_id and r.pass_index == pass_index]
    return [record.action.id.name for record in run.history]


def _manifest(root: Path, name: str, source_pass: int, level: int) -> dict[str, object]:
    return json.loads((checkpoint_dir(root, name, source_pass, level) / MANIFEST_FILE).read_text())


def _level_actions(source: Benchmark, root: Path, name: str, source_pass: int, level: int) -> list[str]:
    """What the source game played on ``level``: its actions from that
    level's checkpoint up to the next one's (or to the end)."""
    played = _actions(source, name, source_pass)
    start = len(_manifest(root, name, source_pass, level)["actions"])  # type: ignore[arg-type]
    end = len(played)
    if level + 1 < len(SEQUENCES[name]):
        end = len(_manifest(root, name, source_pass, level + 1)["actions"])  # type: ignore[arg-type]
    return played[start:end]


@pytest.mark.parametrize("process_mode", ["inprocess", "subprocess"])
async def test_branches_replay_their_source_exactly(tmp_path: Path, process_mode: str) -> None:
    """With the agent's random stream in its own state, a branch IS the source
    game's continuation: every pass of every checkpoint plays exactly what the
    source played on that level. Anything the checkpoint dropped or the
    restore got wrong — learnt steps, the stream, the observation numbering
    — would show up as a different action somewhere."""
    source = await _source_run(tmp_path / "source", process_mode=process_mode, own_rng=True)
    root = tmp_path / "source" / CHECKPOINTS_DIR

    stored = sorted(p.relative_to(root).as_posix() for p in root.glob("*/p*/L*"))
    assert stored == sorted(f"{name}/p{p}/L{level}" for name, level in _branch_points() for p in range(SOURCE_PASSES))
    for name, level in _branch_points():
        histories = {json.dumps(_manifest(root, name, p, level)["actions"]) for p in range(SOURCE_PASSES)}
        assert len(histories) > 1, f"{name} L{level}: every pass reached it the same way — no history to vary"

    branch = await _branch_run(tmp_path / "branch", root, process_mode=process_mode, own_rng=True)
    assert len(branch.game_runs) == len(_branch_points()) * SOURCE_PASSES * BRANCH_PASSES
    for name, level in _branch_points():
        for source_pass in range(SOURCE_PASSES):
            expected = _level_actions(source, root, name, source_pass, level)
            for branch_pass in range(BRANCH_PASSES):
                game_id = _branch_id(name, level, source_pass)
                (run,) = [r for r in branch.game_runs if r.game_id == game_id and r.pass_index == branch_pass]
                assert run.state == "won"
                assert _actions(branch, game_id, branch_pass) == expected, (game_id, branch_pass)


async def test_resumed_rng_replays_the_source_exactly(tmp_path: Path) -> None:
    """``branch_rng="resume"``: an agent drawing from ``ctx.rng`` itself —
    nothing random in its own state — still replays its source exactly,
    because the game's stream continues from where the source's stood."""
    source = await _source_run(tmp_path / "source", process_mode="subprocess", own_rng=False)
    root = tmp_path / "source" / CHECKPOINTS_DIR
    branch = await _branch_run(tmp_path / "branch", root, process_mode="subprocess", own_rng=False, branch_rng="resume")
    for name, level in _branch_points():
        for source_pass in range(SOURCE_PASSES):
            expected = _level_actions(source, root, name, source_pass, level)
            for branch_pass in range(BRANCH_PASSES):
                assert _actions(branch, _branch_id(name, level, source_pass), branch_pass) == expected


async def test_branches_resume_the_stored_agent(tmp_path: Path) -> None:
    """With randomness from the run, branches of one checkpoint diverge — yet
    each resumes the stored agent: its log continues the checkpoint's, and the
    first observation it is handed carries the index the source's had."""
    await _source_run(tmp_path / "source", process_mode="inprocess", own_rng=False)
    root = tmp_path / "source" / CHECKPOINTS_DIR
    LevelLearnerAgent.finishes.clear()
    branch = await _branch_run(tmp_path / "branch", root, process_mode="inprocess", own_rng=False)
    assert all(run.state == "won" for run in branch.game_runs)

    logs = {(game_id, pass_index): log for game_id, pass_index, log in LevelLearnerAgent.finishes}
    diverged = 0
    for name, level in _branch_points():
        for source_pass in range(SOURCE_PASSES):
            path = checkpoint_dir(root, name, source_pass, level)
            stored = pickle.loads((path / AGENT_FILE).read_bytes())
            start_index = _manifest(root, name, source_pass, level)["observation_index"]
            game_id = _branch_id(name, level, source_pass)
            continuations: set[tuple[tuple[int, int, str], ...]] = set()
            for branch_pass in range(BRANCH_PASSES):
                log = logs[(game_id, branch_pass)]
                assert log[: len(stored.log)] == stored.log
                assert log[len(stored.log)][:2] == (start_index, level)
                continuations.add(tuple(log[len(stored.log) :]))
            diverged += len(continuations) > 1
    assert diverged, "no checkpoint's branches differed: the passes are not drawing their own randomness"


@dataclass(kw_only=True)
class MovePlanAgent(Agent):
    """Plays a fixed plan through ``ctx.execute``: one list of actions per move."""

    plan: list[list[str]] = field(default_factory=lambda: list[list[str]]())
    move: int = 0

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp | None:
        assert ctx.execute is not None
        for name in self.plan[self.move]:
            await ctx.execute(Action(name))
        self.move += 1
        return None


async def test_a_move_that_plays_past_a_level_start_leaves_that_level_no_checkpoint(tmp_path: Path) -> None:
    """A checkpoint is the agent between moves, standing on a level's first
    observation. A move that wins a level and acts again on the next leaves
    no such moment: that level is recorded as skipped and stored nowhere —
    a snapshot taken there would hold an agent already past the level's
    start. A move that ends on the level start stores it as usual."""
    game = ExampleGame(label="past", target_sequence_per_level=[[A1], [A2, A3], [A1]])
    agent = MovePlanAgent(
        plan=[["ACTION1", "ACTION2"], ["ACTION3"], ["ACTION1"]],  # wins L0 and acts on L1; wins L1; wins
        run_config=_config("inprocess", checkpointing="level"),
    )
    job_dir = tmp_path / "job"
    benchmark = Benchmark(games=[game], solver=AgentSolver(agent=agent), n_passes=1, job_dir=job_dir)
    await benchmark.run()
    (run,) = benchmark.game_runs
    assert run.state == "won", run.solver_note

    root = job_dir / CHECKPOINTS_DIR
    assert [p.relative_to(root).as_posix() for p in root.glob("*/p*/L*")] == ["past/p0/L2"]
    manifest = _manifest(root, "past", 0, 2)
    assert [action[0] for action in manifest["actions"]] == ["ACTION1", "ACTION2", "ACTION3"]  # type: ignore[union-attr]
    (tree,) = build_span_tree(read_game_events(job_dir, run.run_id))
    events = [event["values"] for event in tree.events if event["kind"] == "infra/checkpoint"]
    assert events == [
        {"level": 1, "skipped": "the move acted past the level start"},
        {
            "level": 2,
            "observation_index": manifest["observation_index"],
            "path": str(checkpoint_dir(root, "past", 0, 2)),
        },
    ]


async def test_a_replay_that_does_not_arrive_where_its_source_stood_is_refused(tmp_path: Path) -> None:
    """Every game instance checks its own replay as it starts, before any
    game plays: the level, the board and the observation numbering must be
    the source's, or the agent would resume in a game its history does not
    describe."""
    await _source_run(tmp_path / "source", process_mode="inprocess", own_rng=False)
    root = tmp_path / "source" / CHECKPOINTS_DIR
    path = checkpoint_dir(root, "a", 0, 2)
    manifest = json.loads((path / MANIFEST_FILE).read_text())

    def refused(match: str, **changes: object) -> None:
        (path / MANIFEST_FILE).write_text(json.dumps({**manifest, **changes}))
        with pytest.raises(ValueError, match=match):
            _branch_game(root, "a", 2, 0).start_game()

    refused("reached level index 1", actions=manifest["actions"][:-1])  # one short of the level win
    frame = json.loads(json.dumps(manifest["frame"]))
    frame[0][0] = (frame[0][0] + 1) % 16
    refused("different board", frame=frame)
    refused("numbered the level's first observation", observation_index=manifest["observation_index"] + 1)

    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not start_at_level=1"):
        _branch_game(root, "a", 2, 0, start_at_level=1).start_game()


def test_level_checkpointing_needs_a_job_dir() -> None:
    solver = AgentSolver(agent=LevelLearnerAgent(run_config=_config("inprocess", checkpointing="level")))
    with pytest.raises(ValueError, match="job dir"):
        solver.setup()
