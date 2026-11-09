"""Two of the kernel package's options over one counter game, each its own
scripted reference, read through the viewer as their review.

- ``seal_scripted_undo``: ``expose_undo`` on a seal without a world model, as
  the duck has it. UNDO named among the game's actions and played like any
  move: two in a row stepping back twice, one after a RESET doing nothing,
  each spending energy.
- ``seal_scripted_plan``: ``plan_argument="first"``. The python tool's
  ``plan`` asked for ahead of ``code``, and kept on every turn its call
  played, as ``history[i].plan`` and in ``help(Turn)``; a call giving only a
  plan refused for its missing code rather than run as code; with
  ``expose_undo`` off, UNDO not named and refused.

Every python cell opens with comments saying what it exercises.

The plan case's calls write their plan after the code, though the schema asks
for it first. The capture stores a call's arguments with sorted keys, so a
replay hands them to the tool code first whatever order was recorded, and the
``seal/moves`` span's ``plan_first`` must come out the same both times. That
the order is read is pinned in ``test_seal_plan_argument``, on a live game.

Regenerate (rewrites the committed files; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_counter'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import arcengine
import numpy as np
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
from taaf.game import Game, GameState, RunSession

from platypus import LLMAgentSolver, RunConfig
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    regenerate_case,
    replay_case,
    require_regeneration,
    tool_call,
)
from seal import SealAgent
from seal.prompts import SealPrompts

RNG_SEED = 20261007


@dataclass
class UndoCounterGame(Game):
    """A counter, with UNDO.

    UP adds one and RIGHT adds two; reaching three wins the one level. UNDO
    puts the counter back where the last move not yet undone found it, with an
    empty stack after a RESET. The board's fill is the counter; row 0 is an
    energy meter of ten cells, one lit per unit left. Every move spends one,
    UNDO included and never refunded; RESET refills it, and running out loses.
    """

    label: str = field(default="undo_counter_game", init=True)
    _count: int = field(default=0, init=False, repr=False)
    _energy: int = field(default=0, init=False, repr=False)
    _undone: list[int] = field(default_factory=lambda: list[int](), init=False, repr=False)

    _START_ENERGY = 10
    _TARGET = 3
    #: UP, RIGHT and UNDO.
    _ACTION_IDS: tuple[int, ...] = (1, 4, 7)

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = 1
        self.base_actions_per_level = [2]
        self.hint = "scripted counter exercise"
        self.grid_size = (64, 64)
        return self._reset()

    def _reset(self) -> GameState:
        self._count, self._energy, self._undone = 0, self._START_ENERGY, []
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            return self._reset()
        self._energy -= 1
        if action.id == arcengine.GameAction.ACTION7:
            if self._undone:
                self._count = self._undone.pop()
        else:
            self._undone.append(self._count)
            self._count += 1 if action.id == arcengine.GameAction.ACTION1 else 2
        if self._count == self._TARGET:
            return self._build(arcengine.GameState.WIN)
        if self._energy <= 0:
            return self._build(arcengine.GameState.GAME_OVER)
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _build(self, state: arcengine.GameState) -> GameState:
        grid = np.full((64, 64), self._count % 16, dtype=np.int8)
        grid[0, : self._START_ENERGY] = 8
        grid[0, : self._energy] = 11
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=1 if state == arcengine.GameState.WIN else 0,
            win_levels=1,
            available_actions=list(self._ACTION_IDS),
        )
        raw.frame = [grid]
        return GameState(raw=raw)


def _python(code: str, plan: str | None = None) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet, with a plan when given one."""
    return tool_call("python", {"code": code, **({"plan": plan} if plan is not None else {})})


#: What each cell of the UNDO case prints of the level's turns: the action, the
#: counter the board shows, and the energy lit on its meter.
_TURNS = "print([(t.action, int(t.frame.grid[5, 5]), int((t.frame.grid[0] == 11).sum())) for t in history[1:]])\n"


def _undo_script() -> list[ullm.conv.LLMResponse]:
    """The UNDO case's replies, in order."""
    return [
        _python(
            "# Two RIGHTs overshoot the three this level wants, landing on four.\n"
            "submit_action([Action('RIGHT'), Action('RIGHT')])\n"
        ),
        _python(
            "# UNDO is a move like any other: two in a row step back twice (four,\n"
            "# two, zero), and after a RESET there is nothing to undo, though the\n"
            "# UNDO still spends energy.\n"
            "submit_action([Action('UNDO'), Action('UNDO'), Action('RESET'), Action('UNDO')])\n" + _TURNS
        ),
        _python("# Back at zero: RIGHT and UP win.\nsubmit_action([Action('RIGHT'), Action('UP')])\n"),
    ]


def _plan_script() -> list[ullm.conv.LLMResponse]:
    """The plan case's replies: a planned call, one with nothing but a plan,
    one without a plan, and a planned win."""
    return [
        _python(
            "# A call with a plan: kept on the turn it plays.\nsubmit_action([Action('RIGHT')])\n",
            plan="RIGHT once, to see what it adds",
        ),
        # Nothing but a plan: refused for its missing code, not run as code. It
        # does not act, so the next call is the same turn's.
        tool_call("python", {"plan": "a plan and no code"}),
        _python(
            "# No plan: the turns keep the plan of the call that played them, and\n"
            "# help(Turn) names the attribute. UNDO is not offered, so it is refused\n"
            "# with nothing executed; then a RESET, back to zero.\n"
            "print([t.plan for t in history])\n"
            "help(Turn)\n"
            "try:\n"
            "    submit_action([Action('UNDO')])\n"
            "except ValueError as error:\n"
            "    print('refused:', error)\n"
            "submit_action([Action('RESET')])\n"
        ),
        _python(
            "# The win, planned: both moves keep the plan.\nsubmit_action([Action('RIGHT'), Action('UP')])\n",
            plan="RIGHT then UP reach three",
        ),
    ]


def _agent(run: ReferenceRun, **fields: object) -> SealAgent:
    """The seal without a world model, over the reference run's backend."""
    return SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=2000,
        # No side turns: a reflection or an audit would take the script's
        # next replies, and none of the corners here is theirs.
        level_reflection="none",
        strategy_audit="none",
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
        use_world_model="none",
        **fields,  # pyright: ignore[reportArgumentType]
    )


def _build_undo(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The UNDO case: offered."""
    return taaf.benchmark.Benchmark(
        games=[UndoCounterGame()],
        solver=LLMAgentSolver(agent=_agent(run, expose_undo=True)),
        n_passes=1,
        job_dir=run.job_dir,
    )


def _build_plan(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The plan case: the plan argument asked for first, UNDO not offered."""
    return taaf.benchmark.Benchmark(
        games=[UndoCounterGame()],
        solver=LLMAgentSolver(agent=_agent(run, plan_argument="first")),
        n_passes=1,
        job_dir=run.job_dir,
    )


_PROFILES = {"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)}
UNDO = ReferenceCase(name="seal_scripted_undo", build=_build_undo, profiles=_PROFILES)
PLAN = ReferenceCase(name="seal_scripted_plan", build=_build_plan, profiles=_PROFILES)


async def test_undo_reference_replays_and_undo_plays_as_a_move(tmp_path: Path) -> None:
    (ref,) = await replay_case(UNDO, tmp_path)
    text = ref.all_request_text()

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == ["ACTION4", "ACTION4", "ACTION7", "ACTION7", "RESET", "ACTION7", "ACTION4", "ACTION1"]
    assert "The available actions for this game are: RESET, UP, RIGHT, UNDO." in text
    assert (
        "Executed actions:\n- UNDO (4087 pix changed, colors 4,11->2,8)\n- UNDO (4087 pix changed, colors 2,11->0,8)\n"
        "- RESET (4 pix changed, colors 8->11)\n- UNDO (1 pix changed, colors 11->8)"
    ) in text
    # Four, two, zero; the RESET's zero with a full meter; the UNDO after it
    # going nowhere and spending one.
    assert (
        "[(Action('RIGHT'), 2, 9), (Action('RIGHT'), 4, 8), (Action('UNDO'), 2, 7), (Action('UNDO'), 0, 6), "
        "(Action('RESET'), 0, 10), (Action('UNDO'), 0, 9)]"
    ) in text


async def test_plan_reference_replays_and_keeps_the_plan(tmp_path: Path) -> None:
    (ref,) = await replay_case(PLAN, tmp_path)
    text = ref.all_request_text()
    surface = SealPrompts()

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == ["ACTION4", "RESET", "ACTION4", "ACTION1"]
    # Not offered: UNDO not named, and refused with nothing executed.
    assert "The available actions for this game are: RESET, UP, RIGHT." in text
    assert "refused: " + surface.stage_error_undo_disabled.format(index=1) in text
    # The parameter offered, and the plan kept on the turn its call played.
    assert surface.turn_plan_line.strip() in text
    assert "[None, 'RIGHT once, to see what it adds']" in text
    assert "plan: str | None — the plan you gave the python call that played this" in text
    # A plan with no code: refused by the schema, not remapped onto `code`.
    assert "Invalid arguments for tool 'python': 'code' is a required property" in text


async def test_regenerate_seal_scripted_counter(tmp_path: Path) -> None:
    """Regenerates the committed references (stub-driven, no API spend); gated
    because it rewrites committed files. Validated by immediate replay (I13)."""
    require_regeneration()
    for case, script in ((UNDO, _undo_script()), (PLAN, _plan_script())):
        backend = ullm.backends.stub.StubBackend(responses=script, on_exhausted="raise")
        await regenerate_case(case, backend=backend, tmp_path=tmp_path / case.name)
