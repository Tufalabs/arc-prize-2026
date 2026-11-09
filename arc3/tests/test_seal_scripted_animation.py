"""Animation frames: which moves had them, and their description as a block of code.

Six scripted references over one animation game, read through the viewer
as the review of the surface; every python cell opens with comments saying
what it exercises. The game: a 3x3 sprite on a board, two levels, won at
the right side. UP moves it with no animation; DOWN slides it in three
frames; RIGHT's animation is padded with a copy of the board before it, a
repeated frame and a hold on the settled board, which the kernel's history
drops, keeping two; SPACE flashes a noise patch for four frames that no
short script describes; LEFT bounces the sprite out and back in eleven
frames, flickering between two colours so that no loop draws it, the second
half repeating the first.

Each case sets the three switches: the settled delta, the animation
description (none, available, auto) and the marks.

- Settled delta, description auto, marks (the character limits reduced so
  the bounce needs a stride): the first animation on a level is described below the prompt
  and its move says so; an animation no longer than one described already
  is not, and its move says how to ask; the noise has no concise
  description; the bounce is built every n-th frame and the text says how
  to get every one. ``describe_animation`` refuses a turn, a frame, an
  array, a bool, an index out of range and a zero stride with a message,
  says when a turn has no animation, and at a stride the model gives cuts a
  long block off. A long batch keeps its fully marked move between the
  elided moves and counts the animated ones it hides; the moves of a level
  won within a batch keep only their counts; a new level describes again.
- Settled delta, description available, marks: nothing is described
  unasked; the first animated move of each prompt says how to ask, and
  asking works.
- Description none, marks, with the settled delta and without: every
  animated move is still marked, the first with where its frames are;
  ``describe_animation`` is not a name.
- Settled delta, description auto, no marks: a described animation stands
  on its own, and one with no concise description says so in a line of its
  own, since no mark does.
- Settled delta, description auto, marks, with a world model: a move
  predicted right has no delta, only the line saying so, and its animation
  is described after it.

Regenerate (rewrites the committed files; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_animation'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import arcengine
import numpy as np
import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
from taaf.game import Game, GameState, RunSession

from platypus import LLMAgentSolver, RunConfig
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    descendants,
    regenerate_case,
    replay_case,
    replay_spans,
    require_regeneration,
    tool_call,
)
from seal import SealAgent
from seal.frame_delta import AnimationDescription
from seal.prompts import SealPrompts

RNG_SEED = 20260925

#: The reduced character limits of the main case: the bounce passes the first at every frame, and a block at
#: stride 1 passes the second.
MAX_CHARS = 680
CUT_CHARS = 690


@dataclass
class AnimationGame(Game):
    """A sprite whose moves animate in each of the ways the seal has to say."""

    label: str = field(default="animation_game", init=True)
    _level: int = field(default=0, init=False, repr=False)
    _row: int = field(default=0, init=False, repr=False)
    _col: int = field(default=0, init=False, repr=False)

    _ACTION_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
    _START = (30, 10)
    _BACKGROUNDS = (0, 5)
    _SPRITE = 9
    _FLICKER = 8
    _WIN_COL = 34
    _LEVELS = 2

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = self._LEVELS
        self.base_actions_per_level = [6, 6]
        self.hint = "scripted animation exercise"
        self.grid_size = (64, 64)
        self._level = 0
        self._row, self._col = self._START
        return self._build(arcengine.GameState.NOT_FINISHED, [])

    def _board(self, row: int | None = None, col: int | None = None, sprite: int | None = None) -> np.ndarray:
        """The board with the sprite at ``row``, ``col`` (its place by default), in colour ``sprite``."""
        grid = np.full((64, 64), self._BACKGROUNDS[self._level], dtype=np.int8)
        row = self._row if row is None else row
        col = self._col if col is None else col
        grid[row : row + 3, col : col + 3] = self._SPRITE if sprite is None else sprite
        return grid

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        row, col = self._row, self._col
        animation: list[np.ndarray] = []
        if action.id == arcengine.GameAction.ACTION1:
            if self._row > 2:
                self._row -= 4
        elif action.id == arcengine.GameAction.ACTION2:
            animation = [self._board(row + step, col) for step in (1, 2, 3)]
            self._row += 4
        elif action.id == arcengine.GameAction.ACTION3:
            # The sprite flickers as it goes, so no loop draws the bounce and its frames each need a piece.
            offsets = [*range(1, 7), *range(5, 0, -1)]
            animation = [
                self._board(row, col - offset, self._FLICKER if k % 2 else None) for k, offset in enumerate(offsets)
            ]
        elif action.id == arcengine.GameAction.ACTION4:
            first, second = self._board(row, col + 1), self._board(row, col + 2)
            self._col += 4
            animation = [self._board(row, col), first, first.copy(), second, self._board()]
            if self._col >= self._WIN_COL:
                self._level += 1
                self._row, self._col = self._START
                if self._level >= self._LEVELS:
                    return self._build(arcengine.GameState.WIN, animation)
        elif action.id == arcengine.GameAction.ACTION5:
            rng = np.random.default_rng(len(self.label) + row + col)
            for _ in range(4):
                frame = self._board()
                frame[50:60, 50:60] = rng.integers(1, 16, (10, 10))
                animation.append(frame)
        return self._build(arcengine.GameState.NOT_FINISHED, animation)

    def _build(self, state: arcengine.GameState, animation: list[np.ndarray]) -> GameState:
        if state == arcengine.GameState.WIN:
            settled = np.full((64, 64), 14, dtype=np.int8)
            levels_completed = self._LEVELS
        else:
            settled = self._board()
            levels_completed = self._level
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=levels_completed,
            win_levels=self._LEVELS,
            available_actions=list(self._ACTION_IDS),
        )
        raw.frame = [*animation, settled]
        return GameState(raw=raw)


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _animation_script() -> list[ullm.conv.LLMResponse]:
    """The main case, animations described. One entry per backend call; a decision ends at the cell that acts."""
    return [
        # === decision 1 (level 1; history: the seed turn) ===================
        _python(
            "# UP has no animation: its move is named plainly.\n"
            "print(submit_action([Action('UP')])['executed_actions'])\n"
        ),
        # === decision 2 =====================================================
        _python(
            "# DOWN slides in three frames, the first animation on the level: the\n"
            "# next prompt describes it below and its move says so.\n"
            "print(submit_action([Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 3 =====================================================
        _python(
            "# DOWN again: no longer than the animation already described, so it is\n"
            "# not described, and its move says how to ask.\n"
            "print(submit_action([Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 4 =====================================================
        _python(
            "# describe_animation refuses what is not a turn's index, and says so.\n"
            "for call in (\n"
            "    lambda: describe_animation(history[-1]),\n"
            "    lambda: describe_animation(history[-1].frame),\n"
            "    lambda: describe_animation(history[-1].frame.grid),\n"
            "    lambda: describe_animation(True),\n"
            "    lambda: describe_animation(99),\n"
            "    lambda: describe_animation(-1, stride=0),\n"
            "):\n"
            "    try:\n"
            "        call()\n"
            "    except Exception as error:\n"
            "        print(type(error).__name__, error)\n"
            "# A turn without animation frames: UP's, and the level's opening turn.\n"
            "describe_animation(1)\n"
            "describe_animation(0)\n"
            "help(describe_animation)\n"
        ),
        _python(
            "# Asked for: the DOWN the last prompt did not describe.\n"
            "describe_animation(-1)\n"
            "# RIGHT's animation is sent padded; history keeps two frames, and the\n"
            "# move is marked with those two.\n"
            "print(submit_action([Action('RIGHT')])['executed_actions'])\n"
        ),
        # === decision 5 =====================================================
        _python(
            "# RIGHT's padded animation, as history keeps it.\n"
            "print(len(history[-1].animation_frames))\n"
            "# SPACE flashes noise for four frames, longer than any described: it is\n"
            "# described, as having no concise description.\n"
            "print(submit_action([Action('SPACE')])['executed_actions'])\n"
        ),
        # === decision 6 =====================================================
        _python(
            "# LEFT bounces out and back in eleven frames. Every frame is over the\n"
            "# (reduced) limit, so every n-th frame is built, and the text says how\n"
            "# to get every one.\n"
            "print(submit_action([Action('LEFT')])['executed_actions'])\n"
        ),
        # === decision 7 =====================================================
        _python(
            "# Every frame, at the stride asked for: past the (reduced) cut, the\n"
            "# block is cut off. A coarser stride fits whole.\n"
            "describe_animation(-1, stride=1)\n"
            "describe_animation(-1, stride=4)\n"
        ),
        _python(
            "# A long batch: its line keeps the first and last five moves. The first\n"
            "# animated move (RIGHT) falls between them and is kept, marked in full;\n"
            "# the gap after it says it hid an animated move (DOWN).\n"
            "r = submit_action([Action('UP')] * 5 + [Action('RIGHT'), Action('DOWN')] + [Action('UP')] * 5)\n"
            "print(len(r['executed_actions']))\n"
        ),
        # === decision 8 =====================================================
        _python(
            "# A batch that wins the level: its moves' turns leave with the level, so\n"
            "# they keep only their counts.\n"
            "print(submit_action([Action('DOWN')] + [Action('RIGHT')] * 4)['executed_actions'])\n"
        ),
        # === decision 9 (level 2, a fresh kernel) ===========================
        _python(
            "# A new level: its first animation is described again.\n"
            "print(submit_action([Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 10 ====================================================
        _python("print(submit_action([Action('RIGHT')] * 6)['executed_actions'])\n"),
    ]


def _plain_script() -> list[ullm.conv.LLMResponse]:
    """The ``settled`` and ``none`` cases: the marks alone."""
    return [
        # === decision 1 (level 1) ===========================================
        _python(
            "# DOWN slides in three frames: its move says where they are.\n"
            "print(submit_action([Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 2 =====================================================
        _python(
            "# Animations are not described in this mode: no such function.\n"
            "try:\n"
            "    describe_animation(-1)\n"
            "except NameError as error:\n"
            "    print(error)\n"
            "# Two animated moves: the first marked in full, the second with its count.\n"
            "print(submit_action([Action('DOWN'), Action('RIGHT')])['executed_actions'])\n"
        ),
        # === decision 3 =====================================================
        _python("print(submit_action([Action('RIGHT')] * 5)['executed_actions'])\n"),
        # === decision 4 (level 2) ===========================================
        _python("print(submit_action([Action('RIGHT')] * 6)['executed_actions'])\n"),
    ]


ANIMATION_MODEL = """\
import numpy as np

BG = int(history[0].frame.grid[0, 0])


def initial_state():
    return {'outcome': 'playing', 'energy': 2000, 'row': 30, 'col': 10}


def do_move(state, action):
    state['energy'] -= 1
    if action.name == 'UP':
        if state['row'] > 2:
            state['row'] -= 4
    elif action.name == 'DOWN':
        state['row'] += 4
    elif action.name == 'RIGHT':
        state['col'] += 4
        if state['col'] >= 34:
            state['outcome'] = 'won'


def render(state):
    grid = np.full((64, 64), BG, dtype=np.int8)
    r, c = state['row'], state['col']
    grid[r:r + 3, c:c + 3] = 9
    return grid
"""


def _on_request_script() -> list[ullm.conv.LLMResponse]:
    """The on-request case: nothing described unasked, the marks saying how to ask."""
    return [
        # === decision 1 (level 1) ===========================================
        _python(
            "# DOWN slides in three frames. Nothing is described unasked: its move\n"
            "# says where the frames are and how to have them described.\n"
            "print(submit_action([Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 2 =====================================================
        _python(
            "# Asked for, the way the mark said.\n"
            "describe_animation(-1)\n"
            "# Two animated moves: the first says how to ask, the second has its count.\n"
            "print(submit_action([Action('DOWN'), Action('RIGHT')])['executed_actions'])\n"
        ),
        # === decision 3 =====================================================
        _python("print(submit_action([Action('RIGHT')] * 5)['executed_actions'])\n"),
        # === decision 4 (level 2) ===========================================
        _python("print(submit_action([Action('RIGHT')] * 6)['executed_actions'])\n"),
    ]


def _unmarked_script() -> list[ullm.conv.LLMResponse]:
    """The unmarked case: animations described on their own, no move marked."""
    return [
        # === decision 1 (level 1) ===========================================
        _python(
            "# DOWN's three frames, the level's first animation: described below the\n"
            "# prompt, with no mark on the move to point at it.\n"
            "print(submit_action([Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 2 =====================================================
        _python(
            "# SPACE's noise, longer: described, as having no concise description, in a\n"
            "# line of its own, since no mark says so.\n"
            "print(submit_action([Action('SPACE')])['executed_actions'])\n"
        ),
        # === decision 3 =====================================================
        _python("print(submit_action([Action('RIGHT')] * 6)['executed_actions'])\n"),
        # === decision 4 (level 2) ===========================================
        _python("print(submit_action([Action('RIGHT')] * 6)['executed_actions'])\n"),
    ]


def _world_model_script() -> list[ullm.conv.LLMResponse]:
    """The world-model case: a move predicted right, whose animation is still described."""
    return [
        # === decision 1 (level 1) ===========================================
        tool_call("create_file", {"path": "world_model.py", "content": ANIMATION_MODEL}),
        _python(
            "# The model predicts DOWN's settled board: the prompt says the frame is as\n"
            "# predicted, with no delta, and still describes the animation after it.\n"
            "print(test_world_model())\n"
            "print(submit_action([Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 2 =====================================================
        _python("print(submit_action([Action('RIGHT')] * 6)['executed_actions'])\n"),
        # === decision 3 (level 2) ===========================================
        _python("print(submit_action([Action('RIGHT')] * 6)['executed_actions'])\n"),
    ]


def _optional_script() -> list[ullm.conv.LLMResponse]:
    """The optional-world-model case: moves unchecked, then checked, then stopped, each still marked."""
    return [
        # === decision 1 (level 1) ===========================================
        _python(
            "# world_model.py is still the example, which fails here: the batch runs\n"
            "# unchecked, and its moves are marked as without a world model.\n"
            "print(submit_action([Action('DOWN'), Action('DOWN')])['executed_actions'])\n"
        ),
        # === decision 2 =====================================================
        tool_call("create_file", {"path": "world_model.py", "content": _WRONG_RIGHT_MODEL}),
        _python(
            "# The model passes on the two DOWNs and checks the batch: DOWN as predicted,\n"
            "# RIGHT not, which stops it.\n"
            "print(test_world_model())\n"
            "print(submit_action([Action('DOWN'), Action('RIGHT'), Action('RIGHT')])['executed_actions'])\n"
        ),
        # === decision 3 =====================================================
        tool_call("create_file", {"path": "world_model.py", "content": ANIMATION_MODEL}),
        _python(
            "# Fixed: the batch runs checked, and the prompt says so below its marks.\n"
            "print(test_world_model())\n"
            "print(submit_action([Action('RIGHT')] * 2)['executed_actions'])\n"
        ),
        # === decision 4 =====================================================
        _python("print(submit_action([Action('RIGHT')] * 3)['executed_actions'])\n"),
        # === decision 5 (level 2) ===========================================
        _python("print(submit_action([Action('RIGHT')] * 6)['executed_actions'])\n"),
    ]


#: A model that moves RIGHT one cell short: it passes until a RIGHT is played.
_WRONG_RIGHT_MODEL = ANIMATION_MODEL.replace("state['col'] += 4", "state['col'] += 3")


def _builder(
    *,
    settled_delta: bool,
    animation_description: AnimationDescription,
    animation_marks: bool,
    max_chars: int | None = None,
    cut_chars: int | None = None,
    use_world_model: Literal["none", "optional", "mandatory"] = "none",
) -> Callable[[ReferenceRun], taaf.benchmark.Benchmark]:
    """A benchmark builder for one setting of the three switches, with or without a world model."""

    def build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
        """The seal over the animation game; with a world model, no exploratory moves."""
        agent = SealAgent(
            derive_max_model_len=False,
            derive_max_num_seqs=False,
            derive_max_pixels=False,
            reply_cap_tokens=2000,
            # No side turns: a reflection or an audit would take the script's
            # next replies, and none of the corners here is theirs.
            level_reflection="none",
            strategy_audit="none",
            # The marks as the batch line carries them; each move's own line
            # carries them too (``test_seal_move_changes``).
            move_changes=False,
            llm_config=run.llm_config,
            run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
            use_world_model=use_world_model,
            settled_delta=settled_delta,
            animation_description=animation_description,
            animation_marks=animation_marks,
            max_llm_calls_per_move=8,
            exploration_moves_new_game=0,
            exploration_moves_new_level=0,
        )
        if max_chars is not None:
            agent.animation_max_chars = max_chars
        if cut_chars is not None:
            agent.animation_cut_chars = cut_chars
        return taaf.benchmark.Benchmark(
            games=[AnimationGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
        )

    return build


_PROFILES = {"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)}

CASE = ReferenceCase(
    name="seal_scripted_animation",
    build=_builder(
        settled_delta=True,
        animation_description="auto",
        animation_marks=True,
        max_chars=MAX_CHARS,
        cut_chars=CUT_CHARS,
    ),
    profiles=_PROFILES,
)
SETTLED_CASE = ReferenceCase(
    name="seal_scripted_animation_settled",
    build=_builder(settled_delta=True, animation_description="none", animation_marks=True),
    profiles=_PROFILES,
)
NONE_CASE = ReferenceCase(
    name="seal_scripted_animation_none",
    build=_builder(settled_delta=False, animation_description="none", animation_marks=True),
    profiles=_PROFILES,
)
ON_REQUEST_CASE = ReferenceCase(
    name="seal_scripted_animation_on_request",
    build=_builder(settled_delta=True, animation_description="available", animation_marks=True),
    profiles=_PROFILES,
)
UNMARKED_CASE = ReferenceCase(
    name="seal_scripted_animation_unmarked",
    build=_builder(settled_delta=True, animation_description="auto", animation_marks=False),
    profiles=_PROFILES,
)
WORLD_MODEL_CASE = ReferenceCase(
    name="seal_scripted_animation_world_model",
    build=_builder(settled_delta=True, animation_description="auto", animation_marks=True, use_world_model="mandatory"),
    profiles=_PROFILES,
)
OPTIONAL_CASE = ReferenceCase(
    name="seal_scripted_animation_optional",
    build=_builder(settled_delta=True, animation_description="auto", animation_marks=True, use_world_model="optional"),
    profiles=_PROFILES,
)


async def test_animation_reference_replays_and_every_corner_fired(tmp_path: Path) -> None:
    (ref,) = await replay_case(CASE, tmp_path)
    text = ref.all_request_text()
    exact = (
        "(this is based on visual decomposition, without understanding the game). The code is exact: it was run "
        "against these frames and its assert holds, so there is no need to check it against the pixels."
    )

    assert ref.expected["state"] == "won"
    assert SealPrompts().runtime_animation_line in text
    # No animation, no mark.
    assert "Executed actions: UP.\n" in text
    # The level's first animation: described below, and its move says so.
    assert (
        "Executed actions: DOWN (3 animation frames, history[2].animation_frames, concise code description below).\n"
    ) in text
    assert (
        f"The animation of history[2] (3 frames) can be rebuilt as follows {exact}\n\n```python\nanim = {{}}\n\n"
        "# Frames 0-2: the shape below moving down 1 a frame over a background that stays put\n"
        "shape = history[1].frame.grid[26:29, 10:13].copy()\n"
    ) in text
    assert (
        "    anim[k] = g.copy()\n\n"
        "assert all((frame == history[2].animation_frames[k].grid).all() for k, frame in anim.items())\n```"
    ) in text
    # No longer than one described: how to ask.
    assert (
        "Executed actions: DOWN (3 animation frames, history[3].animation_frames, use `describe_animation(3)` for a "
        "concise code description).\n"
    ) in text
    # What is not a turn's index, and a turn without animation frames.
    assert (
        "TypeError describe_animation takes the index of a turn in history, such as describe_animation(-1) or "
        "describe_animation(5); got a Turn.\n"
    ) in text
    for what in ("a Frame", "an array", "a bool"):
        assert f"describe_animation(5); got {what}.\n" in text
    assert "IndexError history has 4 turns, history[0] to history[3]; got 99.\n" in text
    assert "ValueError stride is a whole number of frames, 1 or more, or None to have one chosen; got 0.\n" in text
    assert (
        "history[1] has no animation frames; history[2] has 3.\n"
        "history[0] has no animation frames; history[2] has 3.\n"
        "Help on function describe_animation in module runtime.game:\n\ndescribe_animation(i, stride=None)\n"
    ) in text
    # Asked for, then counted: RIGHT's two kept frames are no longer than it.
    assert f"The animation of history[3] (3 frames) can be rebuilt as follows {exact}" in text
    assert (
        "Executed actions: RIGHT (2 animation frames, history[4].animation_frames, use `describe_animation(4)` for a "
        "concise code description).\n"
    ) in text
    assert "2\n[Action('SPACE')]" in text
    # Nothing concise: the move says so, and nothing follows.
    assert (
        "Executed actions: SPACE (4 animation frames, history[5].animation_frames, no concise code description "
        "because the visuals are too complex).\n"
    ) in text
    assert "The animation of history[5]" not in text
    # The bounce, every 4th frame, and how to get every one.
    assert (
        "The animation of history[6] (11 frames) can be rebuilt as follows (this is based on visual decomposition, "
        f"without understanding the game). Only every 4th frame is built, to keep this under {MAX_CHARS} "
        "characters; `describe_animation(6, stride=1)` builds every frame. The code is exact"
    ) in text
    assert "# Frame 4: from frame 0\n" in text and "# Frame 8: from frame 4\n" in text
    # A stride asked for: the block cut off, or whole at a coarser one.
    assert (
        "The animation of history[6] (11 frames) can be rebuilt as follows (this is based on visual decomposition, "
        "without understanding the game). The whole block was run against these frames and its assert holds, but "
        f"it is cut off here at {CUT_CHARS} characters; a larger `stride` shows all of it."
    ) in text
    assert "anim[3] = g.copy()\n\n# ... cut off\n```" in text
    # A long batch: the fully marked move kept between the elided ones, the animated move a gap hid counted.
    assert (
        "Executed actions (first 5, the move with its animation named, and last 5): UP, UP, UP, UP, UP, "
        "RIGHT (2 animation frames, history[12].animation_frames, use `describe_animation(12)` for a concise code "
        "description), ... (1 more, 1 with animation frames) ..., UP, UP, UP, UP, UP.\n"
    ) in text
    # A batch that wins the level: counts only.
    assert (
        "Executed actions: DOWN (3 anim frames), RIGHT (2 anim frames), RIGHT (2 anim frames), RIGHT (2 anim frames), "
        "RIGHT (2 anim frames).\nYou have progressed to a new level!"
    ) in text
    # A new level describes again.
    assert (
        "Executed actions: DOWN (3 animation frames, history[1].animation_frames, concise code description below).\n"
    ) in text
    assert "/scratch" not in text and "/tmp/ullm" not in text and " at 0x" not in text
    assert "seal.harness" not in text and "_repl_state" not in text and "frame_delta" not in text

    # Every description, automatic or asked for, is on a span.
    records = [span.kv for span in descendants(replay_spans(tmp_path), "seal/frame_delta")]
    animations = [record for record in records if record["event"] == "animation"]
    assert [(r["reason"], r["turn_index"], r["frames"], r["stride"], r["shown"], r["cut"]) for r in animations] == [
        ("automatic", 2, 3, 1, True, False),
        ("requested", 3, 3, 1, True, False),
        ("automatic", 5, 4, 1, False, False),
        ("automatic", 6, 11, 4, True, False),
        ("requested", 6, 11, 1, True, True),
        ("requested", 6, 11, 4, True, False),
        ("automatic", 1, 3, 1, True, False),
    ]


@pytest.mark.parametrize("case", [SETTLED_CASE, NONE_CASE], ids=lambda case: case.name)
async def test_plain_references_replay_and_mark_the_moves(case: ReferenceCase, tmp_path: Path) -> None:
    (ref,) = await replay_case(case, tmp_path)
    text = ref.all_request_text()
    assert ref.expected["state"] == "won"
    assert SealPrompts().runtime_animation_line not in text
    assert "Executed actions: DOWN (3 animation frames, history[1].animation_frames).\n" in text
    assert "name 'describe_animation' is not defined\n" in text
    assert "Executed actions: DOWN (3 animation frames, history[2].animation_frames), RIGHT (2 anim frames).\n" in text
    settled = "The current frame can be most easily understood as a delta from the previous frame" in text
    assert settled == (case is SETTLED_CASE)


async def test_on_request_nothing_is_described_unasked(tmp_path: Path) -> None:
    """``settled+anim``: the function is offered and works, each prompt's first animated move says how to ask,
    and no description comes without asking."""
    (ref,) = await replay_case(ON_REQUEST_CASE, tmp_path)
    text = ref.all_request_text()
    assert ref.expected["state"] == "won"
    assert SealPrompts().runtime_animation_line in text
    assert (
        "Executed actions: DOWN (3 animation frames, history[1].animation_frames, use `describe_animation(1)` for a "
        "concise code description).\n"
    ) in text
    assert (
        "Executed actions: DOWN (3 animation frames, history[2].animation_frames, use `describe_animation(2)` for a "
        "concise code description), RIGHT (2 anim frames).\n"
    ) in text
    assert "concise code description below" not in text
    # The whole conversation, as the last request carries it: the one description is the one asked for.
    last = ref.request_text(ref.records(kind=None)[-1])
    assert last.count("The animation of history[") == 1, "only the one asked for"


async def test_without_marks_the_descriptions_stand_on_their_own(tmp_path: Path) -> None:
    """Description auto, marks off: moves are named plainly; a described animation carries its own turn, and
    one with no concise description says so in a line of its own."""
    (ref,) = await replay_case(UNMARKED_CASE, tmp_path)
    text = ref.all_request_text()
    assert ref.expected["state"] == "won"
    assert SealPrompts().runtime_animation_line in text
    assert "Executed actions: DOWN.\n" in text and "Executed actions: SPACE.\n" in text
    assert "animation frames, history[" not in text and "anim frames" not in text
    assert "The animation of history[1] (3 frames) can be rebuilt as follows" in text
    assert (
        "The animation of history[2] (4 frames) has no concise description "
        "(this is based on visual decomposition, without understanding the game)."
    ) in text


async def test_a_predicted_frame_still_has_its_animation_described(tmp_path: Path) -> None:
    """With the settled board predicted right there is no delta, only the line saying so, and the animation
    block follows it."""
    (ref,) = await replay_case(WORLD_MODEL_CASE, tmp_path)
    text = ref.all_request_text()
    assert ref.expected["state"] == "won"
    assert (
        "Executed actions: DOWN (3 animation frames, history[1].animation_frames, concise code description below).\n"
    ) in text
    assert (
        "The current frame is as predicted by your world model.\n"
        "The animation of history[1] (3 frames) can be rebuilt as follows"
    ) in text


async def test_an_optional_world_model_leaves_the_marks_and_blocks_in_place(tmp_path: Path) -> None:
    """Under an optional world model the moves are marked and described as without one, whether the batch
    ran unchecked, was stopped by the model, or ran checked; the line on how it ran follows the marks, and
    the blocks follow the state line."""
    (ref,) = await replay_case(OPTIONAL_CASE, tmp_path)
    text = ref.all_request_text()
    prompts = SealPrompts()
    assert ref.expected["state"] == "won"
    assert (
        "Executed actions: DOWN (3 anim frames), DOWN (3 animation frames, history[2].animation_frames, concise "
        f"code description below).\nYou are still on the same level.\n{prompts.summary_unchecked_line}\n"
        "Current state: step 3, level 1.\nThe current frame can be most easily understood as a delta"
    ) in text
    assert "```\nThe animation of history[2] (3 frames) can be rebuilt as follows" in text
    assert (
        "Executed actions: DOWN (3 animation frames, history[3].animation_frames, use `describe_animation(3)` for a "
        "concise code description), RIGHT (2 anim frames).\nYou are still on the same level.\n"
        f"{prompts.summary_world_model_mismatch}\n{prompts.summary_world_model_stop_choice}\n"
        "Current state: step 5, level 1.\n"
        "The current frame can be most easily understood as a delta from your mispredicted frame"
    ) in text
    assert (
        "Executed actions: RIGHT (2 animation frames, history[5].animation_frames, use `describe_animation(5)` for a "
        f"concise code description), RIGHT (2 anim frames).\nYou are still on the same level.\n"
        f"{prompts.summary_checked_line}\nCurrent state: step 7, level 1.\n"
        "The current frame is as predicted by your world model.\n"
    ) in text


@pytest.mark.parametrize(
    ("case", "script"),
    [
        (CASE, _animation_script),
        (SETTLED_CASE, _plain_script),
        (NONE_CASE, _plain_script),
        (ON_REQUEST_CASE, _on_request_script),
        (UNMARKED_CASE, _unmarked_script),
        (WORLD_MODEL_CASE, _world_model_script),
        (OPTIONAL_CASE, _optional_script),
    ],
    ids=lambda value: value.name if isinstance(value, ReferenceCase) else "",
)
async def test_regenerate_seal_scripted_animation(
    case: ReferenceCase, script: Callable[[], list[ullm.conv.LLMResponse]], tmp_path: Path
) -> None:
    """Regenerates the committed references (stub-driven, no API spend); gated
    because it rewrites committed files. Validated by immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=script(), on_exhausted="raise")
    await regenerate_case(case, backend=backend, tmp_path=tmp_path / case.name)
