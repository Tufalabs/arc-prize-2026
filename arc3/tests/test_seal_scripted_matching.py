"""The matching settings: how a world model's renders are held to the frames,
the same in ``test_world_model()``, the check on every move and the host's test.

Two scripted references over one meter game, read through the viewer as the
review of the harness. Every python cell opens with comments saying what it
exercises. The seal runs its default world model (optional) with the search on.

``seal_scripted_matching`` offers the pixel tolerance (``expose_pixel_tolerance``):

- Level 1, a model that draws the energy meter full, missing one more cell
  with every move: its test fails on one cell and suggests the tolerance;
  setters refuse what they cannot take; at 2 the test passes loosely and says
  how loosely, and the search after it (the host's test, under the same
  settings) runs; a move missing three cells is stopped with the suggestion; at
  4, with strict energy, a move missing four passes, its frame described from
  the render; the level won.
- Level 2: the settings kept from level 1, the carried model passing loosely
  on a marker cell it does not draw. A model that draws the meter but drains
  it twice as fast: at a tolerance of 1 a move passes on an energy match and
  the marker together; the test says both; strict energy fails it, with the
  energy fix; a tolerance of 0 fails it however the energy goes; the marker
  rendered -1 passes it on energy and the wildcard alone; the game won.

``seal_scripted_matching_not_offered`` does not: ``set_pixel_tolerance`` is not
a name, a failure on one cell suggests nothing, and ``set_strict_energy``
reports the energy alone.

Regenerate (rewrites the committed files; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_matching'
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

RNG_SEED = 20261008


@dataclass
class MeterGame(Game):
    """A counter with an energy meter, over ``levels`` levels.

    UP adds one to the count and RIGHT two; level 1 is won at five, level 2 at
    three. The board's fill is the count, plus 8 on level 2; row 0 is an energy
    meter of ten cells, lit (11) one per unit left and spent (8) after. Every
    move spends one unit, RESET and a new level refill it, and running out
    loses. Level 2 also shows a marker cell (14) in its bottom-left corner.
    """

    label: str = field(default="meter_game", init=True)
    levels: int = 2
    _level: int = field(default=0, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)
    _energy: int = field(default=0, init=False, repr=False)

    _TARGETS = (5, 3)
    _START_ENERGY = 10
    #: UP and RIGHT.
    _ACTION_IDS: tuple[int, ...] = (1, 4)

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = self.levels
        self.base_actions_per_level = list(self._TARGETS[: self.levels])
        self.hint = "scripted matching exercise"
        self.grid_size = (64, 64)
        self._level = 0
        return self._restart()

    def _restart(self) -> GameState:
        self._count, self._energy = 0, self._START_ENERGY
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            return self._restart()
        self._energy -= 1
        self._count += 1 if action.id == arcengine.GameAction.ACTION1 else 2
        if self._count == self._TARGETS[self._level]:
            self._level += 1
            if self._level == self.levels:
                return self._build(arcengine.GameState.WIN)
            return self._restart()
        if self._energy <= 0:
            return self._build(arcengine.GameState.GAME_OVER)
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _build(self, state: arcengine.GameState) -> GameState:
        grid = np.full((64, 64), (8 * self._level + self._count) % 16, dtype=np.int8)
        grid[0, : self._START_ENERGY] = 8
        grid[0, : self._energy] = 11
        if self._level == 1:
            grid[63, 0] = 14
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=self._level,
            win_levels=self.levels,
            available_actions=list(self._ACTION_IDS),
        )
        raw.frame = [grid]
        return GameState(raw=raw)


#: The model level 1 writes: right but for the meter, which it draws full, so
#: it misses one more cell with every move.
METERLESS_MODEL = """\
import numpy as np

BASE = int(history[0].frame.grid[5, 5])
TARGET = 5 if BASE == 0 else 3
START_ENERGY = 10


def initial_state():
    return {'outcome': 'playing', 'energy': START_ENERGY, 'count': 0}


def do_move(state, move):
    state['energy'] -= 1
    state['count'] += 1 if move.name == 'UP' else 2
    if state['count'] == TARGET:
        state['outcome'] = 'won'


def render(state):
    # The meter is drawn full: this model leaves the energy out of the board.
    grid = np.full((64, 64), BASE + state['count'], dtype=np.int8)
    grid[0, :START_ENERGY] = 11
    return grid
"""

#: The model level 2 writes: the meter drawn from the energy, which it drains
#: two a move where the game drains one, and no marker.
DRAINING_MODEL = """\
import numpy as np

BASE = int(history[0].frame.grid[5, 5])
TARGET = 5 if BASE == 0 else 3
START_ENERGY = 10


def initial_state():
    return {'outcome': 'playing', 'energy': START_ENERGY, 'count': 0}


def do_move(state, move):
    state['energy'] -= 2
    state['count'] += 1 if move.name == 'UP' else 2
    if state['count'] == TARGET:
        state['outcome'] = 'won'


def render(state):
    grid = np.full((64, 64), BASE + state['count'], dtype=np.int8)
    grid[0, :START_ENERGY] = 8
    grid[0, :max(0, int(state['energy']))] = 11
    return grid
"""


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _create(path: str, content: str) -> ullm.conv.LLMResponse:
    """The scripted model writing a whole file."""
    return tool_call("create_file", {"path": path, "content": content})


def _replace(path: str, old: str, new: str) -> ullm.conv.LLMResponse:
    """The scripted model editing a file in place."""
    return tool_call("str_replace", {"path": path, "old_string": old, "new_string": new})


def _offered_script() -> list[ullm.conv.LLMResponse]:
    """The offered case's replies, in order."""
    return [
        # === level 1, decision 1 ===
        _python("# Two UPs, unchecked: no model of this game yet.\nsubmit_action([Action('UP'), Action('UP')])\n"),
        # === decision 2 ===
        _create("world_model.py", METERLESS_MODEL),
        _python(
            "# The meter drawn full misses one cell after the first move: the test\n"
            "# fails, and a miss this small suggests the pixel tolerance.\n"
            "print(test_world_model())\n"
        ),
        _python(
            "# The setters refuse what they cannot take, and change nothing.\n"
            "for call in ('set_pixel_tolerance(-1)', 'set_pixel_tolerance(1.5)', 'set_strict_energy(\"yes\")'):\n"
            "    try:\n"
            "        eval(call)\n"
            "    except TypeError as error:\n"
            "        print(call, '->', error)\n"
            "# A tolerance of 2: the test passes loosely and says how loosely, and the\n"
            "# search after it reads the same settings.\n"
            "set_pixel_tolerance(2)\n"
            "print(test_world_model())\n"
        ),
        _python(
            "# The third move misses three cells, past the tolerance: the check stops\n"
            "# the batch with the same suggestion.\n"
            "submit_action([Action('UP')])\n"
        ),
        # === decision 3 ===
        _python(
            "# A tolerance of 4, and strict energy, which this model's frames do not\n"
            "# lean on: the fourth move misses four cells and passes.\n"
            "set_pixel_tolerance(4)\n"
            "set_strict_energy(True)\n"
            "print(test_world_model())\n"
            "submit_action([Action('UP')])\n"
        ),
        # === decision 4 ===
        _python("# The fifth UP wins the level.\nsubmit_action([Action('UP')])\n"),
        # === level 2, decision 5 ===
        _python(
            "# A new level and a new kernel: the settings are the game's, so the\n"
            "# carried model passes loosely on the marker it does not draw.\n"
            "print(test_world_model())\n"
        ),
        _create("world_model.py", DRAINING_MODEL),
        _python(
            "# The meter drawn, but drained two a move: at a tolerance of 1 the move\n"
            "# passes on an energy match and the marker together.\n"
            "set_pixel_tolerance(1)\n"
            "set_strict_energy(False)\n"
            "print(test_world_model())\n"
            "submit_action([Action('UP')])\n"
        ),
        # === decision 6 ===
        _python(
            "# The test says both. Strict energy fails it, with the energy fix; a\n"
            "# tolerance of 0 fails it however the energy goes.\n"
            "print(test_world_model())\n"
            "set_strict_energy(True)\n"
            "print(test_world_model())\n"
            "set_strict_energy(False)\n"
            "set_pixel_tolerance(0)\n"
            "print(test_world_model())\n"
        ),
        _replace("world_model.py", "    return grid\n", "    grid[63, 0] = -1\n    return grid\n"),
        _python(
            "# The marker rendered -1: the test passes on energy and the wildcard\n"
            "# alone, and so does the next move.\n"
            "print(test_world_model())\n"
            "submit_action([Action('UP')])\n"
        ),
        # === decision 7 ===
        _python("# The third UP wins the game.\nsubmit_action([Action('UP')])\n"),
    ]


def _not_offered_script() -> list[ullm.conv.LLMResponse]:
    """The not-offered case's replies, in order."""
    return [
        _python("# Two UPs, unchecked: no model of this game yet.\nsubmit_action([Action('UP'), Action('UP')])\n"),
        _create("world_model.py", METERLESS_MODEL),
        _python(
            "# The same one-cell miss suggests nothing: the tolerance is not offered,\n"
            "# and set_pixel_tolerance is not a name. set_strict_energy reports the\n"
            "# energy alone. Then a move, unchecked under the failing model.\n"
            "print(test_world_model())\n"
            "try:\n"
            "    set_pixel_tolerance(2)\n"
            "except NameError as error:\n"
            "    print(error)\n"
            "set_strict_energy(True)\n"
            "submit_action([Action('UP')])\n"
        ),
        _python("# Two more UPs win the level, and with it this one-level game.\nsubmit_action([Action('UP')] * 2)\n"),
    ]


def _agent(run: ReferenceRun, *, offered: bool) -> SealAgent:
    """The seal with its default world model, the tolerance offered or not."""
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
        expose_pixel_tolerance=offered,
        max_llm_calls_per_move=20,
    )


def _build_offered(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The offered case: two levels."""
    return taaf.benchmark.Benchmark(
        games=[MeterGame()], solver=LLMAgentSolver(agent=_agent(run, offered=True)), n_passes=1, job_dir=run.job_dir
    )


def _build_not_offered(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The not-offered case: level 1 alone."""
    return taaf.benchmark.Benchmark(
        games=[MeterGame(levels=1)],
        solver=LLMAgentSolver(agent=_agent(run, offered=False)),
        n_passes=1,
        job_dir=run.job_dir,
    )


_PROFILES = {"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)}
OFFERED = ReferenceCase(name="seal_scripted_matching", build=_build_offered, profiles=_PROFILES)
NOT_OFFERED = ReferenceCase(name="seal_scripted_matching_not_offered", build=_build_not_offered, profiles=_PROFILES)


async def test_matching_reference_replays_and_every_corner_fired(tmp_path: Path) -> None:
    (ref,) = await replay_case(OFFERED, tmp_path)
    text = ref.all_request_text()
    surface = SealPrompts()

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == ["ACTION1"] * 8
    # The system prompt: energy by the setter, the tolerance offered.
    assert "`set_strict_energy(True)` holds every frame to your own value" in text
    assert surface.world_model_pixel_tolerance_line.format(start=0).strip() in text
    # Level 1: the suggestion, the refused setters, the loose pass...
    assert (
        "render does not match actual frame (1 pixel differs).\n"
        "  If that pixel is a detail your model need not predict, set_pixel_tolerance(1) lets such a frame pass"
    ) in text
    assert "set_pixel_tolerance(-1) -> set_pixel_tolerance takes a whole number, 0 or more." in text
    assert "set_pixel_tolerance(1.5) -> set_pixel_tolerance takes a whole number, 0 or more." in text
    assert 'set_strict_energy("yes") -> set_strict_energy takes True or False.' in text
    assert "Now in force: loose energy, pixel tolerance 2." in text
    assert (
        "pixels: render missed up to 2 pixels on 2 turns (first history[1]), within your tolerance of 2; "
        "set_pixel_tolerance(0) requires every pixel."
    ) in text
    assert "[search] Your world model passed" in text
    # ...the stop past the tolerance, and the pass within it.
    assert "render does not match the new frame after Action('UP') (3 pixels differ)." in text
    assert "If those 3 pixels are a detail your model need not predict, set_pixel_tolerance(3)" in text
    assert "Now in force: strict energy, pixel tolerance 4." in text
    assert "The current frame passed your world model's check within your pixel tolerance; your render missed 4" in text
    assert "delta from your world model's render" in text
    # Level 2: the settings kept, energy and pixels together, strict energy and
    # a zero tolerance failing, the wildcard passing.
    assert "pixels: render missed up to 1 pixel on 1 turn (first history[0]), within your tolerance of 4;" in text
    assert (
        "energy: render matched only under another state['energy'] on 1 turn (first history[1]: 9, yours was 8)" in text
    )
    assert "pixels: render missed up to 1 pixel on 2 turns (first history[0]), within your tolerance of 1;" in text
    assert (
        "The current frame passed your world model's check within your pixel tolerance, once its energy is "
        "reconciled with the frame's."
    ) in text
    # Strict energy: the energy fix, and no tolerance offered for what it explains.
    assert "Energy fix: render would match if state['energy'] were 9 instead of 8." in text
    assert "set_pixel_tolerance(2) lets such a frame pass" not in text
    assert "render does not match initial frame (1 pixel differs)." in text
    assert "wildcards: render declined cells with -1 on" in text
    # The sweep's counts.
    spans = "".join(path.read_text() for path in tmp_path.rglob("spans.jsonl"))
    assert '"reason":"tolerated"' in spans and '"pixels":4' in spans
    assert '"pixels_tolerated":2' in spans


async def test_not_offered_reference_replays_and_suggests_nothing(tmp_path: Path) -> None:
    (ref,) = await replay_case(NOT_OFFERED, tmp_path)
    text = ref.all_request_text()
    surface = SealPrompts()

    assert ref.expected["state"] == "won"
    assert surface.world_model_pixel_tolerance_line.format(start=0).strip() not in text
    assert "render does not match actual frame (1 pixel differs)." in text
    assert "set_pixel_tolerance(" not in text.replace("set_pixel_tolerance(2)\n", "")
    assert "name 'set_pixel_tolerance' is not defined" in text
    assert "Now in force: strict energy." in text


async def test_regenerate_seal_scripted_matching(tmp_path: Path) -> None:
    """Regenerates the committed references (stub-driven, no API spend); gated
    because it rewrites committed files. Validated by immediate replay (I13)."""
    require_regeneration()
    for case, script in ((OFFERED, _offered_script()), (NOT_OFFERED, _not_offered_script())):
        backend = ullm.backends.stub.StubBackend(responses=script, on_exhausted="raise")
        await regenerate_case(case, backend=backend, tmp_path=tmp_path / case.name)
