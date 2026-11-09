"""The move that wins a level, in the finished level's ``history``.

For the rest of the move that won it, the kernel's ``history`` is the level
just won, ending on the winning turn: the win screen as its frame (the last
frame of the level, as platypus partitions a level change) and the level's
own animation frames. The per-move check holds a winning move to nothing:
a world model that does not predict the win is silent there, and only
``test_world_model()`` fails it, on the outcome alone. A later
``submit_action`` in the move is refused before it can touch the next
level's exploration, and the next move opens on the new level. Played in a
real game, on a real kernel, through scripted replies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import arcengine
import numpy as np
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game import Game, GameState, RunSession

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from seal import SealAgent
from seal.prompts import SealPrompts

SIDE = 64
WIN_SCREEN = 13
ANIMATION = 7


@dataclass
class WinGame(Game):
    """Two levels, each won by two UPs; any other move resets the level's
    progress. The board's fill is ``4 * level + progress``. Winning a level
    animates: one frame of its own, the win screen, then the next level's
    board."""

    label: str = field(default="win_game", init=True)
    _level: int = field(default=0, init=False, repr=False)
    _progress: int = field(default=0, init=False, repr=False)

    _LEVELS = 2

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = self._LEVELS
        self.base_actions_per_level = [2] * self._LEVELS
        self.hint = "the winning turn"
        self.grid_size = (SIDE, SIDE)
        self._level = 0
        self._progress = 0
        return self._build(arcengine.GameState.NOT_FINISHED, [self._board()])

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id != arcengine.GameAction.ACTION1:
            self._progress = 0
            return self._build(arcengine.GameState.NOT_FINISHED, [self._board()])
        self._progress += 1
        if self._progress < 2:
            return self._build(arcengine.GameState.NOT_FINISHED, [self._board()])
        self._level += 1
        self._progress = 0
        if self._level >= self._LEVELS:
            return self._build(arcengine.GameState.WIN, [self._fill(14)])
        return self._build(
            arcengine.GameState.NOT_FINISHED, [self._fill(ANIMATION), self._fill(WIN_SCREEN), self._board()]
        )

    def _board(self) -> np.ndarray:
        return self._fill(4 * self._level + self._progress)

    @staticmethod
    def _fill(value: int) -> np.ndarray:
        return np.full((SIDE, SIDE), value, dtype=np.int8)

    def _build(self, state: arcengine.GameState, frames: list[np.ndarray]) -> GameState:
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=self._LEVELS if state == arcengine.GameState.WIN else self._level,
            win_levels=self._LEVELS,
            available_actions=[1, 2],
        )
        raw.frame = frames
        return GameState(raw=raw)


#: Predicts the game, the win included; the base fill is read off the level's
#: first board, so the same file passes on level 2.
MODEL = """\
import numpy as np

BASE = int(history[0].frame.grid[0, 0])


def initial_state():
    return {'outcome': 'playing', 'energy': 2000, 'progress': 0}


def do_move(state, action):
    if action.name == 'UP':
        state['progress'] += 1
        if state['progress'] == 2:
            state['outcome'] = 'won'
    else:
        state['progress'] = 0


def render(state):
    return np.full((64, 64), BASE + state['progress'], dtype=np.int8)
"""

#: The same model, blind to the win: every frame right, every outcome 'playing'.
BLIND_MODEL = MODEL.replace(
    "        if state['progress'] == 2:\n            state['outcome'] = 'won'\n",
    "",
)

#: Wins level 1, then reads the win in the rest of the call and tries to act on.
WINNING_CELL = """\
open('world_model.py', 'w').write({model!r})
submit_action([Action('UP'), Action('UP')])
last = history[-1]
print('WON', len(history), last.action, last.outcome, int(last.frame.grid[0, 0]), last.frame.level,
      len(last.animation_frames), last.frame.diff_sparse is not None)
print('STATE', last.state and last.state['outcome'])
print(test_world_model())
describe_animation(len(history) - 1)
later = submit_action(Action('UP'))
print('LATER', later['executed'], later['stop_reason'])
"""

LOOK_CELL = "print('NEXT', len(history), history[0].frame.level, int(history[0].frame.grid[0, 0]))"


def _python(code: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code})]
        ),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=5),
    )


class _Script(ullm.backends.stub.StubBackend):
    """Scripted replies. Records, per request, the last user message and the
    conversation's text; class-level, since the game plays on a copy of the
    backend the test built."""

    last_user: ClassVar[list[str]] = []
    texts: ClassVar[list[str]] = []

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        """Record what the request carries, then pick the scripted reply."""
        users = [e.content.to_string() for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)]
        type(self).last_user.append(users[-1] if users else "")
        type(self).texts.append(conversation.to_string())
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)


async def _play(tmp_path: Path, model: str, mode: Literal["optional", "mandatory"]) -> Any:
    """Level 1 won by the winning cell, a look at level 2, and the game won."""
    _Script.last_user = []
    _Script.texts = []
    replies = [
        _python(WINNING_CELL.format(model=model)),
        _python(LOOK_CELL),
        _python("submit_action([Action('UP'), Action('UP')])"),
    ]
    backend = _Script(responses=replies, on_exhausted="repeat_last")
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        use_world_model=mode,
        # The winning call's own turn is under test, not the reflection after it.
        level_reflection="none",
        use_search=False,
        animation_description="auto",
        exploration_moves_new_level=2,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[WinGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=120)
    (run,) = benchmark.game_runs
    return run


async def test_the_call_that_won_reads_the_win(tmp_path: Path) -> None:
    run = await _play(tmp_path, MODEL, "optional")
    assert run.state == "won"
    # Request 1 is the move after the win: its conversation holds the
    # winning cell's output, its user prompt opens level 2.
    won_cell, next_prompt = _Script.texts[1], _Script.last_user[1]
    # The winning turn ends the level's history, on the win screen, with the
    # level's own animation frame and a diff from the board before it.
    assert f"WON 3 Action('UP') won {WIN_SCREEN} 1 1 True" in won_cell
    # The check stepped the model over the win and kept its state.
    assert "STATE won" in won_cell
    assert "TestResult: ALL PASSED." in won_cell
    # The winning animation is there to ask about, and only asked about: no
    # automatic block lands on the next level's prompt.
    assert "The animation of history[2] (1 frames)" in won_cell
    assert "The animation of history[" not in next_prompt
    # Nothing more runs on the finished level.
    assert "LATER False previous_level_completed" in won_cell
    assert "runtime.game.WorldModelStop:" not in won_cell
    # The next move opens on the new level.
    assert "NEXT 1 2 4" in _Script.texts[2]


async def test_a_model_blind_to_the_win_is_not_stopped(tmp_path: Path) -> None:
    run = await _play(tmp_path, BLIND_MODEL, "optional")
    assert run.state == "won"
    won_cell = _Script.texts[1]
    assert f"WON 3 Action('UP') won {WIN_SCREEN} 1 1 True" in won_cell
    # Silent at the move: no stop, no state on the turn.
    assert "runtime.game.WorldModelStop:" not in won_cell
    assert "STATE None" in won_cell
    # test_world_model() fails it on the outcome alone.
    assert "do_move predicted outcome='playing', expected 'won'." in won_cell
    assert "LATER False previous_level_completed" in won_cell


async def test_acting_after_the_win_leaves_the_next_levels_exploration_whole(tmp_path: Path) -> None:
    """Under the mandatory gate the refused move after the win went the
    checked route, which ends a level's exploration; the refusal comes first,
    so level 2 opens with its exploration open."""
    run = await _play(tmp_path, MODEL, "mandatory")
    assert run.state == "won"
    assert "LATER False previous_level_completed" in _Script.texts[1]
    surface = SealPrompts()
    next_prompt = _Script.last_user[1]
    assert surface.exploration_status_line.format(n=2, moves="moves") in next_prompt
    assert surface.world_model_required_line not in next_prompt
