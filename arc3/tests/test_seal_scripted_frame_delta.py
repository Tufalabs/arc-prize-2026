"""The frame delta: a new frame explained to the model as a block of code from a frame it knew.

One scripted reference over a delta game, read through the viewer as the
review of the surface. Every python cell opens with comments saying what it
exercises. The game: a 3x3 sprite on a black board with an energy bar along
the bottom row, moved four cells by UP/DOWN/LEFT/RIGHT (eight on level 2);
UP at the top edge does nothing and spends nothing; DOWN leaves a trail
cell behind; LEFT past the edge loses, onto a red screen with a patterned
mark; SPACE scrambles the top rows into a checkerboard; the level is won at
the right side.

- Without a world model, every new frame is explained from the previous
  frame: a move that changed nothing is the comparison alone, a move is a
  move, and the scramble is beyond the limit (150 bits), so the prompt says
  there is no concise delta.
- Once ``test_world_model()`` passes the level is understood and the limit
  drops to 50 bits. The model renders the bar as wildcards and does not
  know the trail: the batch stops at the mispredicted DOWN, and the next
  prompt explains the frame from the mispredicted render, one cell, the
  wildcards resolved. ``test_world_model()`` then fails on that turn and its
  result carries the same delta; ``history[i].predicted_frame`` holds the
  render on every turn the model was checked on.
- The game-over screen costs between the two limits: under the understood
  level it is not shown, and the prompt says so. A RESET the model
  predicts right gets the line saying the frame is as predicted and no
  block; a level win's new frame gets nothing.
- Level 2 moves eight cells: the recalibrated model mispredicts its first
  move, explained from the render as one move.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_frame_delta'
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

from framediff import Decomposer, Source
from framediff import grid as framediff_grid
from platypus import LLMAgentSolver, RunConfig
from platypus.diagnostics import job_block_store, resolve_blocks
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    descendants,
    regenerate_case,
    replay_case,
    replay_job_dir,
    replay_spans,
    require_regeneration,
    tool_call,
)
from seal import SealAgent
from seal import frame_delta as fd
from seal.palette import ASCII_SYMBOLS

RNG_SEED = 20260917


@dataclass
class DeltaGame(Game):
    """A sprite on a board, with the edges and effects the delta must describe.

    Two levels, black then grey, the sprite blue then orange, a step of four
    then eight. The energy bar is the bottom row: twenty cells, one lit per
    unit, a unit per effective move.
    """

    label: str = field(default="delta_game", init=True)
    _level: int = field(default=0, init=False, repr=False)
    _row: int = field(default=0, init=False, repr=False)
    _col: int = field(default=0, init=False, repr=False)
    _energy: int = field(default=0, init=False, repr=False)
    _scrambled: bool = field(default=False, init=False, repr=False)
    _trail: list[tuple[int, int]] = field(default_factory=lambda: list[tuple[int, int]](), init=False, repr=False)

    _ACTION_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
    _START = (2, 10)
    _STEPS = (4, 8)
    _BACKGROUNDS = (5, 1)
    _SPRITES = (9, 12)
    _TRAIL = 13
    _WIN_COL = 30
    _START_ENERGY = 20
    _LEVELS = 2

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = self._LEVELS
        self.base_actions_per_level = [5, 3]
        self.hint = "scripted frame-delta exercise"
        self.grid_size = (64, 64)
        self._level = 0
        self._restart_level()
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _restart_level(self) -> None:
        """The level as it opens."""
        self._row, self._col = self._START
        self._energy = self._START_ENERGY
        self._scrambled = False
        self._trail = []

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        step = self._STEPS[self._level]
        if action.id == arcengine.GameAction.RESET:
            self._restart_level()
        elif action.id == arcengine.GameAction.ACTION1:
            if self._row > 2:
                self._row -= step
                self._energy -= 1
        elif action.id == arcengine.GameAction.ACTION2:
            self._trail.append((self._row, self._col))
            self._row += step
            self._energy -= 1
        elif action.id == arcengine.GameAction.ACTION3:
            self._energy -= 1
            if self._col - step < 4:
                return self._build(arcengine.GameState.GAME_OVER)
            self._col -= step
        elif action.id == arcengine.GameAction.ACTION4:
            self._col += step
            self._energy -= 1
            if self._col >= self._WIN_COL:
                self._level += 1
                self._restart_level()
                if self._level >= self._LEVELS:
                    return self._build(arcengine.GameState.WIN)
        elif action.id == arcengine.GameAction.ACTION5:
            self._scrambled = not self._scrambled
            self._energy -= 1
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _build(self, state: arcengine.GameState) -> GameState:
        if state == arcengine.GameState.WIN:
            grid = np.full((64, 64), 14, dtype=np.int8)
            levels_completed = self._LEVELS
        elif state == arcengine.GameState.GAME_OVER:
            grid = np.full((64, 64), 8, dtype=np.int8)
            grid[27:36, 27:36] = np.random.default_rng(7).integers(0, 16, (9, 9))
            levels_completed = self._level
        else:
            grid = np.full((64, 64), self._BACKGROUNDS[self._level], dtype=np.int8)
            if self._scrambled:
                rows, cols = np.indices((10, 64))
                grid[8:18, :] = np.where((rows + cols) % 2 == 0, 3, 4)
            for row, col in self._trail:
                grid[row, col] = self._TRAIL
            grid[self._row : self._row + 3, self._col : self._col + 3] = self._SPRITES[self._level]
            grid[63, : self._START_ENERGY] = 8
            grid[63, : self._energy] = 11
            levels_completed = self._level
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=levels_completed,
            win_levels=self._LEVELS,
            available_actions=list(self._ACTION_IDS),
        )
        raw.frame = [grid]
        return GameState(raw=raw)


MODEL = """\
import numpy as np

_first = history[0].frame.grid
BG = int(_first[0, 0])
SPRITE = int(_first[2, 10])
STEP = 4
START_ENERGY = 20


def initial_state():
    return {'outcome': 'playing', 'energy': START_ENERGY, 'row': 2, 'col': 10, 'scrambled': False, 'trail': []}


def do_move(state, action):
    if action.name == 'UP':
        if state['row'] > 2:
            state['row'] -= STEP
            state['energy'] -= 1
    elif action.name == 'DOWN':
        state['row'] += STEP
        state['energy'] -= 1
    elif action.name == 'LEFT':
        state['energy'] -= 1
        if state['col'] - STEP < 4:
            state['outcome'] = 'lost'
        else:
            state['col'] -= STEP
    elif action.name == 'RIGHT':
        state['col'] += STEP
        state['energy'] -= 1
        if state['col'] >= 30:
            state['outcome'] = 'won'
    elif action.name == 'SPACE':
        state['scrambled'] = not state['scrambled']
        state['energy'] -= 1


def render(state):
    grid = np.full((64, 64), BG, dtype=np.int8)
    if state['scrambled']:
        rows, cols = np.indices((10, 64))
        grid[8:18, :] = np.where((rows + cols) % 2 == 0, 3, 4)
    for row, col in state['trail']:
        grid[row, col] = 13
    r, c = state['row'], state['col']
    grid[r:r + 3, c:c + 3] = SPRITE
    grid[63, :START_ENERGY] = -1
    return grid
"""

MODEL_WITH_TRAIL = MODEL.replace(
    "    elif action.name == 'DOWN':\n        state['row'] += STEP\n",
    "    elif action.name == 'DOWN':\n        state['trail'].append((state['row'], state['col']))\n        state['row'] += STEP\n",
)


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _create(path: str, content: str) -> ullm.conv.LLMResponse:
    """The scripted model writing a whole file."""
    return tool_call("create_file", {"path": path, "content": content})


def _replace(path: str, old: str, new: str) -> ullm.conv.LLMResponse:
    """The scripted model editing a file in place."""
    return tool_call("str_replace", {"path": path, "old_string": old, "new_string": new})


def _script() -> list[ullm.conv.LLMResponse]:
    """One entry per backend call. A decision ends at the cell that acts."""
    return [
        # === decision 1 (level 1; history: the seed turn) ===================
        _python(
            "# Exploring, one action per call. UP at the top edge changes nothing:\n"
            "# the new frame equals the previous one, and the next prompt says so\n"
            "# as the comparison alone.\n"
            "print(submit_action([Action('UP')])['executed_actions'])\n"
        ),
        # === decision 2 =====================================================
        _python(
            "# RIGHT: the sprite moves four columns and the bar loses a cell. The\n"
            "# next prompt explains the frame from the previous one as a move that\n"
            "# leaves the background behind, and one cell.\n"
            "print(submit_action([Action('RIGHT')])['executed_actions'])\n"
        ),
        # === decision 3 =====================================================
        _python(
            "# SPACE scrambles the top rows into a checkerboard: nothing short\n"
            "# describes that (the limit is 150 bits), and the prompt says so.\n"
            "print(submit_action([Action('SPACE')])['executed_actions'])\n"
        ),
        # === decision 4 =====================================================
        _create("world_model.py", MODEL),
        _python(
            "# A passing model: the level is understood, and from here a delta is\n"
            "# shown only under 50 bits. The model renders the bar as wildcards.\n"
            "# test_world_model() also sets history[i].predicted_frame.\n"
            "print(test_world_model())\n"
            "print(history[2].predicted_frame[63, 0], history[2].frame.grid[63, 0])\n"
            "# DOWN leaves a trail cell the model does not know. The batch stops\n"
            "# there; the next prompt explains the frame from the mispredicted\n"
            "# render: one cell, the wildcards resolved.\n"
            "submit_action([Action('RIGHT'), Action('DOWN'), Action('RIGHT')])\n"
            "print('not reached')\n"
        ),
        # === decision 5 (history: seed, UP, RIGHT, SPACE, RIGHT, DOWN) ======
        _python(
            "# The mispredicted turn is in history: test_world_model() fails on its\n"
            "# render, and the result carries the same delta. The move's render is\n"
            "# on the turn either way.\n"
            "print(test_world_model())\n"
            "print(history[-1].predicted_frame[2, 18], history[-1].frame.grid[2, 18])\n"
        ),
        _create("world_model.py", MODEL_WITH_TRAIL),
        _python(
            "# Fixed. LEFT past the edge loses, and the model predicts the loss, so\n"
            "# the batch runs to its end. The game-over screen is a red board with a\n"
            "# patterned mark: under 150 bits but over 50, so on this understood\n"
            "# level the prompt says there is no concise delta.\n"
            "print(test_world_model())\n"
            "r = submit_action([Action('LEFT')] * 4)\n"
            "print(r['executed_actions'][-1], r.get('state'))\n"
        ),
        # === decision 6 (lost) ==============================================
        _python(
            "# RESET restores the level start. The model predicts it exactly (on\n"
            "# the cells it predicts: the bar is wildcards), so the next prompt\n"
            "# says the frame is as predicted, and carries no delta.\n"
            "print(submit_action([Action('RESET')])['executed_actions'])\n"
        ),
        # === decision 7 =====================================================
        _python(
            "# Five RIGHTs win the level; a new level's first frame gets no delta.\n"
            "r = submit_action([Action('RIGHT')] * 5)\n"
            "print(r['executed_actions'], len(history))\n"
        ),
        # === decision 8 (level 2, a fresh kernel) ===========================
        _python(
            "# Level 2 moves eight cells. The model recalibrates its colours from\n"
            "# history[0] and passes silently at the move, then mispredicts it: the\n"
            "# prompt explains the frame from the render as one move of the sprite.\n"
            "print(submit_action([Action('RIGHT')])['executed_actions'])\n"
        ),
        # === decision 9 =====================================================
        _replace("world_model.py", "STEP = 4", "STEP = 8"),
        _python(
            "# With the step fixed the model passes again and two RIGHTs win the game.\n"
            "print(test_world_model())\n"
            "print(submit_action([Action('RIGHT')] * 2)['executed_actions'])\n"
        ),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The seal over the delta game, with the world model mandatory and the default explainer."""
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
        # Three exploratory moves before the model is written, one on level 2.
        exploration_moves_new_game=3,
        exploration_moves_new_level=1,
        # Decision 5 tests, rewrites and tests again before it acts.
        max_llm_calls_per_move=8,
    )
    return taaf.benchmark.Benchmark(
        games=[DeltaGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


CASE = ReferenceCase(
    name="seal_scripted_frame_delta",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_frame_delta_reference_replays_and_every_corner_fired(tmp_path: Path) -> None:
    (ref,) = await replay_case(CASE, tmp_path)
    text = ref.all_request_text()

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == [
        "ACTION1",  # UP at the edge: nothing changes
        "ACTION4",
        "ACTION5",  # the scramble
        "ACTION4",
        "ACTION2",  # the mispredicted DOWN; the batch stops
        "ACTION3",
        "ACTION3",
        "ACTION3",
        "ACTION3",  # lost
        "RESET",
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION4",  # level 1 won
        "ACTION4",  # level 2, mispredicted
        "ACTION4",
        "ACTION4",  # the game won
    ]
    # Without a model: the comparison alone, a move, and nothing concise.
    assert (
        "The current frame can be most easily understood as a delta from the previous frame as follows "
        "(this is based on visual decomposition, without understanding the game). "
        "The code is exact: it was run against these frames and its assert holds, so there is no need to check it "
        "against the pixels.\n\n"
        "```python\nassert (history[-2].frame.grid == history[-1].frame.grid).all()\n```"
    ) in text
    assert (
        "```python\ng = history[-2].frame.grid.copy()\n"
        "g[2:5, 10:13] = 5  # cleared behind the move below\n"
        "g[2:5, 14:17] = history[-2].frame.grid[2:5, 10:13]  # moved right 4 (colors: 9)\n"
        "g[63, 19] = 8\n"
        "assert (g == history[-1].frame.grid).all()\n```"
    ) in text
    assert (
        "The current frame has no concise description as a delta from the previous frame or the level start frame "
        "(this is based on visual decomposition, without understanding the game)."
    ) in text
    # The render on a checked turn, and the mispredicted frame explained from it.
    assert "-1 11" in text
    assert (
        "The current frame can be most easily understood as a delta from your mispredicted frame as follows "
        "(this is based on visual decomposition, without understanding the game). "
        "The code is exact: it was run against these frames and its assert holds, so there is no need to check it "
        "against the pixels.\n\n"
        "```python\ng = history[-1].predicted_frame.copy()\ng[2, 18] = 13\n"
        "g[history[-1].predicted_frame == -1] = history[-1].frame.grid[history[-1].predicted_frame == -1]"
        "  # resolve wildcards\nassert (g == history[-1].frame.grid).all()\n```"
    ) in text
    # The test that fails on that turn carries the same delta, by index.
    assert (
        "TestResult: FAIL at history[5]\n  render does not match actual frame (1 pixel differs).\n"
        "  If that pixel is a detail your model need not predict, set_pixel_tolerance(1) lets such a frame pass; "
        "first check that it is not the only sign of a change.\n"
        "  State at failure, move was Action('DOWN'): history[4].state (input) → history[5].state (output)\n"
        "The actual frame at history[5] can be most easily understood as a delta from your mispredicted frame as "
        "follows (this is based on visual decomposition, without understanding the game). "
        "The code is exact: it was run against these frames and its assert holds, so there is no need to check it "
        "against the pixels.\n\n"
        "```python\ng = history[5].predicted_frame.copy()\ng[2, 18] = 13\n"
        "g[history[5].predicted_frame == -1] = history[5].frame.grid[history[5].predicted_frame == -1]"
        "  # resolve wildcards\nassert (g == history[5].frame.grid).all()\n```\n5 13"
    ) in text
    # The game-over screen, between the limits, on an understood level: skipped (the span below says which).
    assert "Action('LEFT') GAME_OVER\n" in text
    # The RESET the model predicted right: the line, no block.
    assert (
        "Executed actions:\n- RESET (4093 pix changed, colors 0,1,2,3,4,6,7,8,9,10,11,12,13,14,15->5,9,11)\n"
        "You are still on the same level.\nCurrent state: step 11, level 1.\n"
        "A passing `test_world_model()` is required to make moves.\n"
        "The current frame is as predicted by your world model (on the cells it predicted)."
    ) in text
    # Level 2's first move, explained from the render as a move.
    assert (
        "```python\ng = history[-1].predicted_frame.copy()\n"
        "g[2:5, 14:17] = 1  # cleared behind the move below\n"
        "g[2:5, 18:21] = history[-1].predicted_frame[2:5, 14:17]  # moved right 4 (colors: 12)\n"
    ) in text
    assert "/scratch" not in text and "/tmp/ullm" not in text and " at 0x" not in text
    assert "seal.harness" not in text and "_repl_state" not in text and "frame_delta" not in text

    # Every delta, shown or skipped, is on a span with its code.
    spans = descendants(replay_spans(tmp_path), "seal/frame_delta")
    records = [span.kv for span in spans]
    assert [
        (r["event"], r["reason"], r["turn_index"], r.get("chosen"), r["shown"], r["understood"]) for r in records
    ] == [
        ("frame", "unpredicted", 1, "previous", True, False),
        ("frame", "unpredicted", 2, "previous", True, False),
        ("frame", "unpredicted", 3, "previous", False, False),
        ("frame", "mispredicted", 5, "prediction", True, True),
        ("test", "test_failure", 5, "prediction", True, True),
        ("frame", "lost", 9, "previous", False, True),
        ("frame", "predicted", 10, None, False, True),
        ("frame", "mispredicted", 1, "prediction", True, True),
    ]
    # The block and the text are on every decomposition's record (auto-blocked when long, so not compared here).
    assert all(r["text"] and (r["reason"] == "predicted" or r["code"]) for r in records)
    assert [r["limit"] for r in records if r["reason"] != "predicted"] == [
        Decomposer().max_bits,
        Decomposer().max_bits,
        Decomposer().max_bits,
        fd.MAX_BITS_UNDERSTOOD,
        fd.MAX_BITS_UNDERSTOOD,
        fd.MAX_BITS_UNDERSTOOD,
        fd.MAX_BITS_UNDERSTOOD,
    ]

    # Each decomposition keeps the frames it was given, and they are the frames it ran on: searched again
    # from them, every block comes out as recorded. The render keeps its wildcards.
    store = job_block_store(replay_job_dir(tmp_path))
    searched = [span for span in spans if "code" in span.kv]
    renders: list[np.ndarray] = []
    # Searched in the alphabet the scenario's kernel wrote its blocks in.
    process_alphabet = framediff_grid.alphabet()
    framediff_grid.set_alphabet(ASCII_SYMBOLS[SealAgent().ascii_symbols])
    try:
        for span in searched:
            stored = {a["name"]: store.load(a["block"], "json") for a in span.artifacts}
            grids = {name: fd.unpack_grid(packed) for name, packed in stored["grids"].items()}
            assert set(grids) == {"target", *(b["kind"] for b in stored["baselines"])}
            sources = [Source(b["expression"], grids[b["kind"]], b["handicap"]) for b in stored["baselines"]]
            found = Decomposer(max_bits=span.kv["limit"]).explain(sources, grids["target"], span.kv["target"])
            assert found.code == resolve_blocks(store, span.kv["code"]), span.kv
            renders += [grids["prediction"]] if span.kv["reason"] == "mispredicted" else []
    finally:
        framediff_grid.set_alphabet(process_alphabet)
    assert any((render == -1).any() for render in renders)


async def test_regenerate_seal_scripted_frame_delta(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend); gated
    because it rewrites a committed file. Validated by immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_script(), on_exhausted="raise")
    await regenerate_case(CASE, backend=backend, tmp_path=tmp_path / CASE.name)
