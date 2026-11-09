"""A RESET that does not restore its level: one scripted reference over a game
whose RESET leaves part of the board as it was, read through the viewer as the
review of the flow.

- The model writes a world model that passes, and plays under it: a move, and
  a click that lights a panel.
- A RESET puts the counter back but leaves the panel lit, so the board differs
  from the level's first; the replay rebuilds every RESET from
  ``initial_state()``, so the world model now fails on that turn.
- The next prompt says the RESET led to a different frame, that this is
  cosmetic, and that the cells are for wildcards; the model renders the panel
  as -1, its world model passes again, and the game is won under it.

Every python cell opens with comments saying what it exercises.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_irregular_reset'
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
    replay_spans,
    require_regeneration,
    span_events,
    tool_call,
)
from seal import SealAgent
from seal.prompts import SealPrompts

RNG_SEED = 20261010


@dataclass
class PanelGame(Game):
    """A counter with a panel that RESET leaves alone.

    UP adds one, and three wins the one level. A click anywhere lights the
    panel, four cells of row 60; RESET puts the counter back to zero and leaves
    the panel as it is. The board's fill is the counter.
    """

    label: str = field(default="panel_game", init=True)
    _count: int = field(default=0, init=False, repr=False)
    _panel: bool = field(default=False, init=False, repr=False)

    _TARGET = 3
    #: UP and MOUSE.
    _ACTION_IDS: tuple[int, ...] = (1, 6)

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = 1
        self.base_actions_per_level = [self._TARGET]
        self.hint = "scripted irregular RESET exercise"
        self.grid_size = (64, 64)
        self._count, self._panel = 0, False
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            self._count = 0
        elif action.id == arcengine.GameAction.ACTION6:
            self._panel = True
        else:
            self._count += 1
            if self._count == self._TARGET:
                return self._build(arcengine.GameState.WIN)
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _build(self, state: arcengine.GameState) -> GameState:
        grid = np.full((64, 64), self._count % 16, dtype=np.int8)
        if self._panel:
            grid[60, 0:4] = 9
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=1 if state == arcengine.GameState.WIN else 0,
            win_levels=1,
            available_actions=list(self._ACTION_IDS),
        )
        raw.frame = [grid]
        return GameState(raw=raw)


#: A world model right about everything but RESET, which it cannot get right:
#: the replay rebuilds RESET from ``initial_state()``.
MODEL = """\
import numpy as np


def initial_state():
    return {'outcome': 'playing', 'energy': 0, 'count': 0, 'panel': False}


def do_move(state, action):
    if action.name == 'UP':
        state['count'] += 1
        if state['count'] == 3:
            state['outcome'] = 'won'
    else:
        state['panel'] = True


def render(state):
    grid = np.full((64, 64), state['count'], dtype=np.int8)
    if state['panel']:
        grid[60, 0:4] = 9
    return grid
"""

#: The same model with the panel left to wildcards.
WILDCARD_MODEL = MODEL.replace(
    """    if state['panel']:
        grid[60, 0:4] = 9
""",
    """    grid[60, 0:4] = -1
""",
)


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _script() -> list[ullm.conv.LLMResponse]:
    """The replies, in order."""
    return [
        _python(
            "# A world model that passes on the opening board, and a move under it.\n"
            f"open('world_model.py', 'w').write({MODEL!r})\n"
            "print(test_world_model())\n"
            "submit_action([Action('UP')])\n"
        ),
        _python(
            "# A click lights the panel; the world model still passes.\n"
            "submit_action([Action('MOUSE', row=10, col=10)])\n"
            "print(test_world_model())\n"
        ),
        _python(
            "# A RESET: the counter goes back to zero and the panel stays lit, which\n"
            "# the replay's initial_state() cannot match.\n"
            "submit_action([Action('RESET')])\n"
            "print(test_world_model())\n"
        ),
        _python(
            "# The panel left to wildcards: the world model passes again, and a move\n"
            "# under it.\n"
            f"open('world_model.py', 'w').write({WILDCARD_MODEL!r})\n"
            "print(test_world_model())\n"
            "submit_action([Action('UP')])\n"
        ),
        _python("# The win.\nsubmit_action([Action('UP'), Action('UP')])\n"),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The seal with the world model optional and no search, over the panel game."""
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
        use_world_model="optional",
        use_search=False,
    )
    return taaf.benchmark.Benchmark(
        games=[PanelGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


_PROFILES = {"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)}
IRREGULAR = ReferenceCase(name="seal_scripted_irregular_reset", build=_build, profiles=_PROFILES)


async def test_a_reset_that_does_not_restore_the_level_is_pointed_at_wildcards(tmp_path: Path) -> None:
    (ref,) = await replay_case(IRREGULAR, tmp_path)
    requests = [ref.request_text(record) for record in ref.records(kind=None)]
    notice = SealPrompts().irregular_reset_line.format(cells=4)

    assert ref.expected["state"] == "won"
    assert ref.action_ids() == ["ACTION1", "ACTION6", "RESET", "ACTION1", "ACTION1", "ACTION1"]
    # Told once, on the prompt after the RESET, and not before it.
    told = next(index for index, text in enumerate(requests) if notice in text)
    assert told > 0 and all(text.count(notice) == 1 for text in requests[told:])
    (event,) = span_events(replay_spans(tmp_path), "seal/irregular_reset")
    assert event["values"] == {"cells": 4}


async def test_regenerate_seal_scripted_irregular_reset(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend); gated
    because it rewrites committed files. Validated by immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_script(), on_exhausted="raise")
    await regenerate_case(IRREGULAR, backend=backend, tmp_path=tmp_path)
