"""The world model, phase 1: exploring, writing one, testing it, and acting under it.

One scripted reference over a meter game — the corner game with an energy bar
across the top row — read through the viewer as the review of the harness.
Every python cell opens with comments saying what it exercises. The seal runs
the mandatory gate (not the default), with small exploration budgets
so a short script reaches their edges.

- Exploration: with no passing model a move runs only as exploration — a
  single action, one per python call, from the level's budget; a batch or a
  second call is refused with nothing executed, and a failing model explores
  the same way.
- ``test_world_model()`` failing once per check class — a module that does
  not load, a missing ``render``, a broken state contract, a wrong starting
  board — then passing loosely on energy (and failing the same model under
  strict energy), then passing, with ``history[i].state`` filled.
- The first passing model ends exploration for the level. Its batch runs
  stepped and silent while the claims hold, and the first contradicted frame
  stops it and interrupts the cell with ``WorldModelStop``; with the model
  now failing, exploration stays over and acting is refused until it passes.
- ``Unknown`` live: the declined move is played and nothing after it runs in
  the cell — a later ``submit_action`` refused, a batch stopped at the next
  move, nothing to stop after a game-ending move — and refused by the replay
  once the move is in history.
- The optional ``distance_to_win`` checked; module-level state mutated during
  the replay caught, with world_model.py rewritten and tested in one cell.
- Moves as Actions: a plan stepped through ``world_model.do_move`` and
  submitted as the same list, and a click in ``history`` with its
  coordinates.
- RESET stepped through ``initial_state()``, and the winning move read in the
  rest of its call: the last turn of the level's ``history``, on the win
  screen, with the ``'won'`` state the check stepped the model to, and a
  ``test_world_model()`` that replays the win.
- The next level: a fresh kernel, the files still there, constants
  recalibrated from the new history; the level's own exploration budget,
  spent and then refused; the game won under the model.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_world_model'
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
from seal import SealAgent, seeds
from seal.prompts import SealPrompts

RNG_SEED = 20260915


@dataclass
class MeterGame(Game):
    """The corner game with an energy meter.

    Two levels ([UP, UP], [DOWN, DOWN]); any other action — MOUSE included,
    offered so the prompt's MOUSE sentences render — resets the level's
    progress, and LEFT is an instant GAME_OVER. The board's fill encodes
    (level, progress); row 0 is a meter track of ten cells with one lit per
    unit of energy. Energy starts at ten, every non-RESET action spends one,
    and RESET or a new level refills it.
    """

    label: str = field(default="meter_game", init=True)
    _level: int = field(default=0, init=False, repr=False)
    _progress: int = field(default=0, init=False, repr=False)
    _energy: int = field(default=0, init=False, repr=False)

    _TARGETS = ([arcengine.GameAction.ACTION1] * 2, [arcengine.GameAction.ACTION2] * 2)
    _START_ENERGY = 10
    #: UP, DOWN, LEFT, and MOUSE, which renders the prompt's MOUSE sentences.
    _ACTION_IDS: tuple[int, ...] = (1, 2, 3, 6)

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = len(self._TARGETS)
        self.base_actions_per_level = [len(seq) for seq in self._TARGETS]
        self.hint = "scripted world-model exercise"
        self.grid_size = (64, 64)
        self._level = 0
        self._progress = 0
        self._energy = self._START_ENERGY
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            self._progress = 0
            self._energy = self._START_ENERGY
            return self._build(arcengine.GameState.NOT_FINISHED)
        self._energy = max(0, self._energy - 1)
        if action.id == arcengine.GameAction.ACTION3:
            return self._build(arcengine.GameState.GAME_OVER)
        target = self._TARGETS[self._level]
        if action.id == target[self._progress]:
            self._progress += 1
            if self._progress >= len(target):
                self._level += 1
                self._progress = 0
                self._energy = self._START_ENERGY
                if self._level >= len(self._TARGETS):
                    return self._build(arcengine.GameState.WIN)
        else:
            self._progress = 0
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _build(self, state: arcengine.GameState) -> GameState:
        if state == arcengine.GameState.WIN:
            fill, levels_completed = 14, len(self._TARGETS)
        elif state == arcengine.GameState.GAME_OVER:
            fill, levels_completed = 9, self._level
        else:
            fill, levels_completed = (self._level * 4 + self._progress) % 16, self._level
        grid = np.full((64, 64), fill, dtype=np.int8)
        if state == arcengine.GameState.NOT_FINISHED:
            grid[0, : self._START_ENERGY] = 8
            grid[0, : self._energy] = 11
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=levels_completed,
            win_levels=len(self._TARGETS),
            available_actions=list(self._ACTION_IDS),
        )
        raw.frame = [grid]
        return GameState(raw=raw)


# ---------------------------------------------------------------------------
# The files the scripted model writes.
# ---------------------------------------------------------------------------

RULES = "# How much one step of progress adds to the board's fill.\nSTEP = 1\n"

#: Everything the first two moves showed, and one belief they cannot yet
#: contradict: a non-target action leaves progress alone (the only one seen,
#: DOWN, was played at progress 0, where a reset looks the same).
NAIVE_MODEL = """\
import numpy as np
from rules import STEP

# Calibrated from this level's first frame: the fill encodes the level, and
# the level decides which action advances it.
BASE = int(history[0].frame.grid[5, 5])
TARGET = 'UP' if BASE == 0 else 'DOWN'
START_ENERGY = 10


def initial_state():
    return {'outcome': 'playing', 'energy': START_ENERGY, 'progress': 0}


def do_move(state, action):
    state['energy'] = max(0, state['energy'] - 1)
    if action.name == TARGET:
        state['progress'] += 1
        if state['progress'] == 2:
            state['outcome'] = 'won'


def render(state):
    grid = np.full((64, 64), BASE + STEP * state['progress'], dtype=np.int8)
    grid[0, :START_ENERGY] = 8
    grid[0, :int(state['energy'])] = 11
    return grid
"""

#: The corrected rule, and an honest refusal for the move never played.
FIXED_MODEL = NAIVE_MODEL.replace(
    """            state['outcome'] = 'won'
""",
    """            state['outcome'] = 'won'
    elif action.name == 'LEFT':
        raise Unknown('LEFT has never been played')
    else:
        state['progress'] = 0
""",
)


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


@dataclass
class KeyboardMeterGame(MeterGame):
    """The meter game without MOUSE: its seal starts from the keyboard example."""

    label: str = field(default="keyboard_meter_game", init=True)
    _ACTION_IDS: tuple[int, ...] = (1, 2, 3)


def _view(path: str) -> ullm.conv.LLMResponse:
    """The scripted model reading a file."""
    return tool_call("view_file", {"path": path})


def _create(path: str, content: str) -> ullm.conv.LLMResponse:
    """The scripted model writing a whole file."""
    return tool_call("create_file", {"path": path, "content": content})


def _replace(path: str, old: str, new: str) -> ullm.conv.LLMResponse:
    """The scripted model editing a file in place."""
    return tool_call("str_replace", {"path": path, "old_string": old, "new_string": new})


def _script() -> list[ullm.conv.LLMResponse]:
    """One entry per backend call. A decision ends at the cell that acts."""
    return [
        # === decision 1 (level 1, 4 exploratory moves; history: the seed turn)
        # The game starts with the example world model for a game with MOUSE.
        _view("world_model.py"),
        _python(
            "# No world_model.py yet, so a move can only be exploration: one action\n"
            "# per python call, from this level's budget. Checking the model here is\n"
            "# silent — exploring does not need one. A batch of two is refused on\n"
            "# exploration's own rule, and nothing executes.\n"
            "submit_action([Action('DOWN')] * 2)\n"
        ),
        _python(
            "# A single exploratory move runs (history grows from 1 turn to 2). A\n"
            "# second submit_action in the same python call is refused.\n"
            "r = submit_action([Action('DOWN')])\n"
            "print(r['executed_actions'], len(history))\n"
            "submit_action([Action('UP')])\n"
        ),
        # === decision 2 (3 exploratory moves left; history: seed, DOWN) =======
        _create("world_model.py", "def initial_state(:\n    return {}\n"),
        _python(
            "# Check class 1: a module that does not load. test_world_model()\n"
            "# reports the SyntaxError from world_model.py itself.\n"
            "print(test_world_model())\n"
        ),
        _create(
            "world_model.py",
            "def initial_state():\n    return {'outcome': 'playing', 'energy': 10}\n\n\n"
            "def do_move(state, action):\n    pass\n",
        ),
        _python("# Check class 2: a required function missing (no render).\nprint(test_world_model())\n"),
        _create("rules.py", RULES),
        _create("world_model.py", NAIVE_MODEL.replace("'energy': START_ENERGY, ", "")),
        _python(
            "# Check class 3: the state contract. initial_state() no longer sets\n"
            "# 'energy', which every state must carry as a number.\n"
            "print(test_world_model())\n"
        ),
        _create("world_model.py", NAIVE_MODEL.replace("BASE + STEP", "BASE + 3 + STEP")),
        _python(
            "# Check class 4: a wrong starting board. render(initial_state()) is\n"
            "# compared with history[0] before any move is replayed.\n"
            "print(test_world_model())\n"
            "# A failing model is no model: this UP runs as exploration.\n"
            "r = submit_action([Action('UP')])\n"
            "print(r['executed_actions'])\n"
        ),
        # === decision 3 (2 exploratory moves left; history: seed, DOWN, UP) ===
        _create("world_model.py", NAIVE_MODEL.replace("state['energy'] - 1", "state['energy'] - 2")),
        _python(
            "# Energy is checked flexibly: do_move spends 2 per move where the game\n"
            "# spends 1, so the meter is off. Each frame WOULD match with another\n"
            "# energy value, so the test passes and says so...\n"
            "print(test_world_model())\n"
            "# ...and strict energy holds the frame to the model's own value.\n"
            "set_strict_energy(True)\n"
            "print(test_world_model())\n"
            "set_strict_energy(False)\n"
        ),
        _create("world_model.py", NAIVE_MODEL),
        _python(
            "# A passing model. test_world_model() also fills history[i].state\n"
            "# with the predicted state after each turn.\n"
            "print(test_world_model())\n"
            "print(history[1].state, history[2].state)\n"
            "history[2]\n"
        ),
        _python(
            "# Acting under the passing model ends exploration on this level. The\n"
            "# batch runs stepped. The model still believes a non-target move\n"
            "# leaves progress alone — nothing in history contradicts it — but DOWN\n"
            "# at progress 1 resets the board to fill 0 where the model predicts 1.\n"
            "# The batch stops at that move, the two UPs never run, and the cell is\n"
            "# interrupted before its print.\n"
            "submit_action([Action('DOWN'), Action('UP'), Action('UP')])\n"
            "print('not reached')\n"
        ),
        # === decision 4 (exploration over; history: seed, DOWN, UP, DOWN) ====
        _python(
            "# The mispredicted turn is in history now, so the replay fails on it.\n"
            "# The stop leaves the move it played in history, and the two UPs out.\n"
            "print(test_world_model())\n"
            "print('after the stop:', len(history), history[-1].action)\n"
            "# Exploration stays over even with 2 moves left in the budget, so the\n"
            "# model is required now: acting fails loudly, saying why a model was\n"
            "# required and how it fails.\n"
            "submit_action([Action('UP')])\n"
        ),
        _create("world_model.py", FIXED_MODEL),
        _python(
            "# Fixed: a non-target move resets progress, and LEFT — never played —\n"
            "# raises Unknown instead of guessing.\n"
            "print(test_world_model())\n"
            "print(history[3].state)\n"
            "# With the states filled, world_model — bound by the REPL, no import\n"
            "# needed — plays a plan forward without spending an action. The plan\n"
            "# is a list of Actions, the same list submit_action would take.\n"
            "ahead = copy.deepcopy(history[-1].state)\n"
            "for move in [Action('UP')] * 2:\n"
            "    world_model.do_move(ahead, move)\n"
            "print(ahead['progress'], ahead['outcome'])\n"
        ),
        _replace(
            "world_model.py",
            "def do_move(state, action):\n",
            "def do_move(state, action):\n"
            "    if state['energy'] <= 6:\n"
            "        raise Unknown('the meter has never been this low')\n",
        ),
        _python(
            "# The model now declines every move once the meter is at 6 or below —\n"
            "# history never went that low, so the test still passes.\n"
            "print(test_world_model())\n"
            "# Live, an Unknown move is played (the model made no claim to hold\n"
            "# it to), and then no further move runs in this cell. The first DOWN\n"
            "# is at energy 7 and checked; the second is at 6 and declined, but it\n"
            "# is the batch's last move, so the call returns normally...\n"
            "submit_action([Action('DOWN')])\n"
            "r = submit_action([Action('DOWN')])\n"
            "print(r['executed_actions'], len(history))\n"
            "# Each move the model was checked on carries its predicted state with no\n"
            "# new test_world_model() call; the declined one has none.\n"
            "print(history[-2].state, history[-1].state)\n"
            "# ...and the next submit_action in the same cell is refused unexecuted.\n"
            "submit_action([Action('UP')])\n"
        ),
        # === decision 5 (history: seed, DOWN, UP, DOWN, DOWN, DOWN) ===========
        _python(
            "# The two declined DOWNs are in history now, so Unknown for them fails\n"
            "# the test (at the one at energy 6).\n"
            "print(test_world_model())\n"
        ),
        _replace("world_model.py", "<= 6", "<= 5"),
        _python(
            "# Declining only at 5 and below, the replay passes again. A click (at\n"
            "# 5, declined) is played; the UP after it is not, and the cell stops.\n"
            "print(test_world_model())\n"
            "submit_action([Action('MOUSE', row=4, col=7), Action('UP')])\n"
        ),
        # === decision 6 (history: ..., a click at energy 5) ==================
        _replace(
            "world_model.py",
            "    if state['energy'] <= 5:\n        raise Unknown('the meter has never been this low')\n",
            "",
        ),
        _python(
            "# The click is in history as the Action that was played, coordinates\n"
            "# and all; the model's catch-all branch resets progress for it, as the\n"
            "# game did.\n"
            "print(history[-1].action, history[-1].action.row, history[-1].action.col)\n"
            "# The meter rule dropped again: LEFT is the one move still declined.\n"
            "# Unknown on the batch's only move: played, and it loses the game, so\n"
            "# there is nothing after it to stop.\n"
            "print(test_world_model())\n"
            "r = submit_action([Action('LEFT')])\n"
            "print(r['executed_actions'], r.get('state'))\n"
        ),
        # === decision 7 (history: ..., LEFT lost) =============================
        _python(
            "# Once LEFT is in history, Unknown for it is a failure: its effect has\n"
            "# been observed and must be modelled.\n"
            "print(test_world_model())\n"
        ),
        _replace(
            "world_model.py",
            "        raise Unknown('LEFT has never been played')\n",
            "        state['outcome'] = 'lost'\n",
        ),
        _python("# LEFT modelled as a loss.\nprint(test_world_model())\n"),
        _replace(
            "world_model.py",
            "def render(state):",
            "def distance_to_win(state):\n    return 'far'\n\n\ndef render(state):",
        ),
        _python(
            "# distance_to_win is optional, but if defined it must return a number:\n"
            "# a search will order states by it.\n"
            "print(test_world_model())\n"
        ),
        _replace("world_model.py", "    return 'far'\n", "    return 2 - state['progress']\n"),
        _python(
            "# Module-level data must not change during the replay. Rewrite the file\n"
            "# with a call log appended from do_move, test it in the same cell —\n"
            "# test_world_model() loads from disk, not from an imported module —\n"
            "# then put the clean version back.\n"
            "clean = open('world_model.py').read()\n"
            "logged = clean.replace('START_ENERGY = 10\\n', 'START_ENERGY = 10\\nCALLS = []\\n')\n"
            "logged = logged.replace(\"    state['energy'] = max\", \"    CALLS.append(action.name)\\n    state['energy'] = max\")\n"
            "with open('world_model.py', 'w') as f:\n"
            "    f.write(logged)\n"
            "print(test_world_model())\n"
            "with open('world_model.py', 'w') as f:\n"
            "    f.write(clean)\n"
            "print(test_world_model())\n"
        ),
        _python(
            "# RESET is stepped through initial_state(); the second UP wins the\n"
            "# level. Until the move ends, history is still level 1's, ending on\n"
            "# the winning turn: its frame is the win screen, the check stepped\n"
            "# the model over it, and the test replays the win.\n"
            "r = submit_action([Action('RESET'), Action('UP'), Action('UP')])\n"
            "print(r['executed_actions'], r.get('stop_reason'))\n"
            "won = history[-1]\n"
            "print(won.action, won.outcome, int(won.frame.grid[5, 5]), won.state['outcome'])\n"
            "print(test_world_model())\n"
        ),
        # === decision 8 (level 2, a fresh kernel, 2 exploratory moves) ========
        _python(
            "# A new level: the kernel was cleared, but world_model.py and rules.py\n"
            "# survived. The REPL re-imported world_model from disk for this cell,\n"
            "# so BASE and TARGET recalibrated from this level's history and the\n"
            "# model already passes — acting now would end exploration at once.\n"
            "print(len(history), int(history[-1].frame.grid[5, 5]))\n"
            "print(world_model.BASE, world_model.TARGET)\n"
            "print(test_world_model())\n"
        ),
        _replace(
            "world_model.py",
            "BASE = int(history[0].frame.grid[5, 5])",
            "BASE = 0",
        ),
        _python(
            "# To show this level's own exploration budget, hard-code level 1's\n"
            "# base colour: the model fails on level 2's first frame, so the UP\n"
            "# runs as exploration (1 left).\n"
            "print(test_world_model())\n"
            "print(submit_action([Action('UP')])['executed_actions'])\n"
        ),
        # === decision 9 (level 2; 1 exploratory move left) ====================
        _python(
            "# The last exploratory move of the level.\nprint(submit_action([Action('UP')])['executed_actions'])\n"
        ),
        # === decision 10 (level 2; the budget is spent) =======================
        _python(
            "# No exploratory moves are left, so the model is required, and it fails.\nsubmit_action([Action('DOWN')])\n"
        ),
        _replace(
            "world_model.py",
            "BASE = 0",
            "BASE = int(history[0].frame.grid[5, 5])",
        ),
        _python(
            "# Recalibrated, the model passes again. The plan is played through the\n"
            "# model first; it predicts a win, so the same list runs stepped and\n"
            "# wins the game.\n"
            "print(test_world_model())\n"
            "plan = [Action('DOWN')] * 2\n"
            "ahead = copy.deepcopy(history[-1].state)\n"
            "for move in plan:\n"
            "    world_model.do_move(ahead, move)\n"
            "print(ahead['outcome'])\n"
            "r = submit_action(plan)\n"
            "print(r['executed_actions'])\n"
        ),
    ]


def _keyboard_script() -> list[ullm.conv.LLMResponse]:
    """The keyboard game: read the example, see it fail here, explore to a win."""
    return [
        # === decision 1 (level 1; history: the seed turn) ===================
        # A game without MOUSE starts with the keyboard example.
        _view("world_model.py"),
        _python(
            "# The example models a different game, so it fails on this one: its\n"
            "# module-level code finds no square to calibrate from.\n"
            "print(test_world_model())\n"
        ),
        _python("# Exploring, one action per call.\nprint(submit_action([Action('UP')])['executed_actions'])\n"),
        # === decision 2 ======================================================
        _python("print(submit_action([Action('UP')])['executed_actions'])\n"),
        # === decision 3 (level 2) ============================================
        _python("print(submit_action([Action('DOWN')])['executed_actions'])\n"),
        # === decision 4 ======================================================
        _python("print(submit_action([Action('DOWN')])['executed_actions'])\n"),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The seal over the meter game, with the world model mandatory."""
    agent = SealAgent(
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
        # The corners below are the mandatory gate's: exploration budgets,
        # and acting refused until a model passes.
        use_world_model="mandatory",
        # Small budgets, so the script reaches a spent one in a few moves.
        exploration_moves_new_game=4,
        exploration_moves_new_level=2,
        # Decision 2 writes and tests the model through every check class
        # before it acts, the longest run of calls in one decision.
        max_llm_calls_per_move=20,
    )
    return taaf.benchmark.Benchmark(
        games=[MeterGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


CASE = ReferenceCase(
    name="seal_scripted_world_model",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


def _build_keyboard(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The seal over the keyboard meter game, with the default budgets."""
    agent = SealAgent(
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
    )
    return taaf.benchmark.Benchmark(
        games=[KeyboardMeterGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


KEYBOARD = ReferenceCase(
    name="seal_scripted_world_model_keyboard",
    build=_build_keyboard,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_world_model_reference_replays_and_every_corner_fired(tmp_path: Path) -> None:
    (ref,) = await replay_case(CASE, tmp_path)
    text = ref.all_request_text()
    raw = ref.path.read_text(encoding="utf-8")

    assert ref.expected["state"] == "won"
    # Stepped and exploratory moves execute exactly what the cells asked for;
    # stopped batches contribute only the moves before their stop.
    assert ref.action_ids() == [
        "ACTION2",  # exploration
        "ACTION1",  # exploration, under a failing model
        "ACTION2",  # the mispredicted move
        "ACTION2",
        "ACTION2",  # declined as the batch's last move
        "ACTION6",  # a click, declined mid-batch
        "ACTION3",
        "RESET",
        "ACTION1",
        "ACTION1",  # level 1 won
        "ACTION1",  # level 2 exploration
        "ACTION1",
        "ACTION2",
        "ACTION2",  # the game won
    ]
    # Exploration: a batch refused, one move per call, a failing model explores.
    assert (
        "ValueError: While exploring, an action batch holds one action; this one has 2. Nothing was executed." in text
    )
    assert "[Action('DOWN')] 2" in text
    assert "ValueError: While exploring, one action runs per python call. Nothing was executed." in text
    # The wrong starting board, explained from the render: the whole board's 3 recoloured to 0, the bar kept.
    assert (
        "State at failure: history[0].state (your initial_state() output)\n"
        "The actual frame at history[0] can be most easily understood as a delta from your mispredicted frame as "
        "follows (this is based on visual decomposition, without understanding the game). "
        "The code is exact: it was run against these frames and its assert holds, so there is no need to check it "
        "against the pixels.\n\n"
        "```python\ng = history[0].predicted_frame.copy()\ng[0:64, 0:64][g[0:64, 0:64] == 3] = 0\n"
        "assert (g == history[0].frame.grid).all()\n```\n[Action('UP')]"
    ) in text
    # Exploring is silent about the model.
    assert "There is no world_model.py" not in text
    # One failure per check class, in the harness's own words.
    assert 'File "<harness>", in test_world_model\n  File "world_model.py", line 1' in text
    assert 'File "test.py"' not in text
    assert "world model has no render() function" in text
    assert "initial_state() must set state['energy'] to a numeric value." in text
    assert "render does not match initial frame (4086 pixels differ)." in text
    # Energy: loose by default, and saying so; strict on request.
    assert (
        "energy: render matched only under another state['energy'] on 2 turns (first history[1]: 9, yours was 8)"
    ) in text
    assert "Energy fix: render would match if state['energy'] were 9 instead of 8." in text
    # A pass fills the state chain.
    assert (
        "{'outcome': 'playing', 'energy': 9, 'progress': 0} {'outcome': 'playing', 'energy': 8, 'progress': 1}" in text
    )
    # The first passing model's batch interrupts at the move it mispredicts...
    assert (
        "runtime.game.WorldModelStop: Your world model mispredicted history[-1]: render does not match the "
        "new frame after Action('DOWN') (4086 pixels differ). The batch stopped after 1 of 3 actions"
    ) in text
    assert "after the stop: 4 Action('DOWN')" in text
    # ...and exploration stays over with budget left.
    assert (
        "runtime.game.WorldModelRequired: A passing test_world_model() is required to make moves.\n"
        "Nothing was executed. test_world_model() gives:\nTestResult: FAIL at history[3]"
    ) in text
    # Unknown live: played; a later call refused; a batch stopped at the next move.
    assert "[Action('DOWN')] 6" in text
    # The checked move's state is on its turn without a new test; the declined one's is None.
    assert "[Action('DOWN')] 6\n{'outcome': 'playing', 'energy': 6, 'progress': 0} None" in text
    assert (
        "Your world model declined to predict an earlier move in this cell (do_move raised "
        "Unknown('the meter has never been this low') for Action('DOWN')), so no further action runs here."
    ) in text
    assert (
        "Your world model declined to predict history[-1]: do_move raised Unknown('the meter has never been "
        "this low') for Action('MOUSE', row=4, col=7). The move was executed, but nothing further runs in this "
        "cell. The batch stopped "
        "after 1 of 2 actions"
    ) in text
    assert "[Action('LEFT')] GAME_OVER" in text
    # The click is in history with its coordinates.
    assert "Action('MOUSE', row=4, col=7) 4 7" in text
    assert "do_move raised Unknown('LEFT has never been played') on a move that IS in history" in text
    # The state test_world_model() filled, and a plan of Actions stepped
    # through the world_model the REPL bound: two UPs from there win.
    assert "{'outcome': 'playing', 'energy': 7, 'progress': 0}\n2 won" in text
    # The optional heuristic and the module-state guard.
    assert "distance_to_win(state) returned 'far' on the initial state, which is not numeric" in text
    assert "CALLS: mutated during replay" in text
    # RESET stepped; level 2 recalibrated.
    # The winning turn, read in the rest of its call: the win screen (TAAF's
    # green), the model's 'won' state from the check, and a test over the win.
    assert (
        "[Action('RESET'), Action('UP'), Action('UP')] None\nAction('UP') won 14 won\nTestResult: ALL PASSED."
    ) in text
    assert "1 4\n4 DOWN\nTestResult: ALL PASSED." in text
    # Level 2's own budget, spent and then refused. The refusal's cell is read
    # from the file: the call that follows wins the game.
    assert raw.count("A passing test_world_model() is required to make moves.") >= 2

    # The MOUSE example, read in full.
    for line in seeds.seed_source(mouse=True).splitlines():
        assert line in text, line
    # Every world-model sentence the prompts can render reached the model.
    surface = SealPrompts()
    for fragment in (
        surface.world_model_intro_line,
        surface.world_model_tool_lines,
        surface.world_model_seed_line,
        surface.world_model_check_lines,
        surface.world_model_search_line,
        surface.world_model_contract_lines.format(
            mouse_note=surface.world_model_mouse_note, search_note=surface.world_model_search_unknown_line
        ),
        surface.world_model_energy_lines,
        surface.world_model_workflow_line,
        surface.turn_state_line,
        surface.world_model_batch_stop_line,
        surface.tool_world_model_sentence,
        surface.exploration_status_line.format(n=4, moves="moves"),
        surface.exploration_status_line.format(n=1, moves="move"),
        surface.world_model_required_line,
        surface.summary_world_model_mismatch,
        surface.summary_world_model_unknown,
        surface.new_level_history_line,
        surface.new_level_world_model_lines,
        surface.checklist_decide_world_model,
        surface.checklist_decide_world_model_game_start,
        surface.checklist_act_world_model_without_journal,
        surface.exploration_single_action_refusal.format(count=2),
        surface.exploration_one_per_call_refusal,
        surface.world_model_required,
    ):
        assert fragment.strip() in text, fragment[:80]

    assert "/scratch" not in text and "/tmp/ullm" not in text and " at 0x" not in text
    assert "seal.harness" not in text and "_repl_state" not in text


async def test_keyboard_reference_replays_and_read_its_example(tmp_path: Path) -> None:
    (ref,) = await replay_case(KEYBOARD, tmp_path)
    text = ref.all_request_text()
    assert ref.expected["state"] == "won"
    assert ref.action_ids() == ["ACTION1", "ACTION1", "ACTION2", "ACTION2"]
    # The keyboard example, read in full, and its failure on this game.
    for line in seeds.seed_source(mouse=False).splitlines():
        assert line in text, line
    assert 'File "world_model.py", line' in text and "IndexError" in text
    # No MOUSE on this game, so no MOUSE sentence.
    assert SealPrompts().world_model_mouse_note not in text


async def test_regenerate_seal_scripted_world_model(tmp_path: Path) -> None:
    """Regenerates the committed references (stub-driven, no API spend); gated
    because it rewrites committed files. Validated by immediate replay (I13)."""
    require_regeneration()
    for case, script in ((CASE, _script()), (KEYBOARD, _keyboard_script())):
        backend = ullm.backends.stub.StubBackend(responses=script, on_exhausted="raise")
        await regenerate_case(case, backend=backend, tmp_path=tmp_path / case.name)
