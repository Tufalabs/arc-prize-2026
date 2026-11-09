"""The diff image (``SealAgent.diff_image``): what a move's frame is compared with, and the image beside the board.

What the kernel compares a move's frame with (`diff_image_baseline`), on the
harness's own turns: framediff's choice where it described the frame; else the
frame before a win or a loss, a checked move's render unless it matched
(energy aside, the pixel tolerance not), and the frame before any other move;
nothing after a RESET, or after an UNDO back to an earlier frame of the level. In a live kernel, the payload after a call, once per new
frame, and nothing where nothing changed. On the host, the caption and the
image after the board image, the cells the move did not change in a navy no
game colour uses; and the switch refused where the board image does not ride
every prompt.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest
import taaf.game
import ullm.conv

from platypus.llm import frame_image
from seal import SealAgent
from seal.diff_image import UNCHANGED_RGB
from seal.harness import _repl_state as rs
from seal.harness.game import Action, Frame, Turn
from seal.repl import GameREPL
from seal.state import FramePayload

# --- what a move's frame is compared with -----------------------------------------


def _board(col: int, *, size: int = 8) -> np.ndarray:
    """A ``size`` x ``size`` board with a 2x2 blue block at (2, col)."""
    grid = np.zeros((size, size), dtype=np.int8)
    grid[2:4, col : col + 2] = 9
    return grid


def _turns(
    move: str = "RIGHT",
    *,
    outcome: str = "playing",
    predicted: np.ndarray | None = None,
    levels: int = 2,
) -> list[Turn]:
    """A level's turns: the seed at column 1, earlier moves keeping the board, and ``move`` to column 3."""
    turns = [Turn(None, "playing", Frame(_board(1)))]
    turns += [Turn(Action("UP"), "playing", Frame(_board(1))) for _ in range(levels - 2)]
    turns.append(Turn(Action(move), outcome, Frame(_board(3)), predicted_frame=predicted))
    return turns


def _choice(turns: list[Turn], record: dict[str, Any] | None = None, **flags: bool) -> tuple[str, np.ndarray] | None:
    return rs.diff_image_baseline(
        turns, record, energy_loose=flags.get("energy_loose", False), tolerated=flags.get("tolerated", False)
    )


def test_a_reset_gets_no_image_whatever_framediff_did() -> None:
    record = {"event": "frame", "reason": "unpredicted", "chosen": "previous", "shown": True}
    assert _choice(_turns("RESET"), record) is None


@pytest.mark.parametrize("record", [None, {"event": "frame", "reason": "unpredicted", "shown": False, "repeats": 0}])
def test_an_undo_back_to_an_earlier_frame_gets_no_image(record: dict[str, Any] | None) -> None:
    """The settled delta names such a frame rather than describing it, as after a RESET; with it off too."""
    turns = _turns() + [Turn(Action("UNDO"), "playing", Frame(_board(1)))]
    assert _choice(turns, record) is None


def test_an_undo_to_a_new_board_is_a_move_like_any_other() -> None:
    """An UNDO whose board matches no earlier frame of the level (a meter still ticking, say) shows what changed."""
    turns = _turns() + [Turn(Action("UNDO"), "playing", Frame(_board(2)))]
    chosen = _choice(turns)
    assert chosen is not None and chosen[0] == "previous" and np.array_equal(chosen[1], turns[-2].frame.grid)


@pytest.mark.parametrize("reason", ["predicted", "energy_reconciled"])
def test_a_frame_framediff_found_predicted_gets_no_image(reason: str) -> None:
    turns = _turns(predicted=_board(3))
    assert _choice(turns, {"event": "frame", "reason": reason}, energy_loose=reason == "energy_reconciled") is None


def test_framediffs_choice_is_the_images() -> None:
    """Whatever framediff explained the frame from, the image compares against, the level start included."""
    render = _board(2)
    turns = _turns(predicted=render, levels=3)
    chosen = {
        kind: _choice(turns, {"event": "frame", "reason": "mispredicted", "chosen": kind, "shown": True})
        for kind in ("prediction", "previous", "level_start")
    }
    assert chosen["prediction"] is not None and chosen["prediction"][0] == "prediction"
    assert chosen["prediction"][1] is render
    assert chosen["previous"] is not None and np.array_equal(chosen["previous"][1], turns[-2].frame.grid)
    assert chosen["level_start"] is not None and np.array_equal(chosen["level_start"][1], turns[0].frame.grid)
    tolerated = _choice(
        turns, {"event": "frame", "reason": "tolerated", "chosen": "render", "shown": True}, tolerated=True
    )
    assert tolerated is not None and tolerated[0] == "render" and tolerated[1] is render


def test_a_block_over_its_budget_leaves_the_image_its_own_rule() -> None:
    """A frame framediff found no concise description of: the model read no block, so the image does not take
    the baseline the block would have used (a win screen cheapest from the level start, say)."""
    turns = _turns(outcome="won", levels=3)
    over = {"event": "frame", "reason": "won", "chosen": "level_start", "shown": False}
    chosen = _choice(turns, over)
    assert chosen is not None and chosen[0] == "previous" and np.array_equal(chosen[1], turns[-2].frame.grid)


@pytest.mark.parametrize("outcome", ["won", "lost"])
def test_without_framediff_a_win_or_a_loss_is_against_the_frame_before(outcome: str) -> None:
    """No world model renders a win's or a loss's frame, whatever it predicted before."""
    turns = _turns(outcome=outcome, predicted=_board(2))
    chosen = _choice(turns)
    assert chosen is not None and chosen[0] == "previous" and np.array_equal(chosen[1], turns[-2].frame.grid)


def test_without_framediff_a_checked_move_is_against_its_render_unless_it_matched() -> None:
    wrong, right = _board(2), _board(3)
    wildcard = right.copy()
    wildcard[7, :] = -1
    assert _choice(_turns(predicted=right)) is None
    assert _choice(_turns(predicted=wildcard)) is None, "a wildcard always matches"
    assert _choice(_turns(predicted=wrong), energy_loose=True) is None, "energy alone is no miss"
    tolerated = _choice(_turns(predicted=wrong), energy_loose=True, tolerated=True)
    assert tolerated is not None and tolerated[0] == "render", "pixels let through are shown, energy or not"
    missed = _choice(_turns(predicted=wrong))
    assert missed is not None and missed[0] == "prediction" and missed[1] is wrong


def test_without_framediff_or_a_render_a_move_is_against_the_frame_before() -> None:
    turns = _turns()
    chosen = _choice(turns)
    assert chosen is not None and chosen[0] == "previous" and np.array_equal(chosen[1], turns[-2].frame.grid)


# --- in a live kernel ---------------------------------------------------------------

_HOOK = "import runtime._repl_state as _s\nprint(_s.after_call())\ndel _s\n"


def _payload(grid: np.ndarray, step: int) -> dict[str, Any]:
    """A frame as the host sends it."""
    return {"grid_b64": base64.b64encode(grid.tobytes()).decode(), "shape": list(grid.shape), "step": step, "level": 1}


#: The kernel's boards are full size: framediff reads only those.
_SIZE = 64


class _Level:
    """A kernel over one level whose moves set the block's column from a script: RIGHT moves it two columns,
    UP leaves the board as it is, and RESET puts it back."""

    def __init__(self, *, settled_delta: bool, reflection: bool = False) -> None:
        self.col = 1
        self.reflection = reflection
        self.turns: list[dict[str, Any]] = [
            {"action": "", "outcome": "playing", "animation": [_payload(_board(1, size=_SIZE), 0)]}
        ]
        self.tool = GameREPL(
            timeout=30.0,
            reset_before_each_call=False,
            use_world_model="none",
            settled_delta=settled_delta,
            animation_description="none",
            diff_image=True,
        )
        self.tool.host_functions = {"get_state": self._state, "action": self._action}

    def _state(self) -> dict[str, Any]:
        return {
            "level": 1,
            "turns": list(self.turns),
            "available_actions": ["UP", "RIGHT", "LEFT"],
            "reflection": self.reflection,
        }

    def _action(self, normalized: list[dict[str, Any]]) -> dict[str, Any]:
        name = normalized[0]["action"]
        self.col = {"RIGHT": self.col + 2, "RESET": 1}.get(name, self.col)
        step = len(self.turns)
        # LEFT wins the level where it stands, with the block moved left one column.
        outcome = "won" if name == "LEFT" else "playing"
        self.col -= name == "LEFT"
        self.turns.append(
            {"action": name, "outcome": outcome, "animation": [_payload(_board(self.col, size=_SIZE), step)]}
        )
        return {"result": {"executed": True}, "state": self._state()}

    async def after(self, code: str) -> dict[str, Any] | None:
        """The call, then what the host collects after it, as the host does (a harness cell, whose output
        no cap cuts): the diff image's payload."""
        await self.tool._execute({"code": code})
        output, failed = await self.tool.run_harness(_HOOK, timeout=30.0)
        assert not failed, output
        return json.loads(output.strip().splitlines()[-1])["diff_image"]


def _changed(payload: dict[str, Any]) -> np.ndarray:
    shape = tuple(payload["shape"])
    return np.frombuffer(base64.b64decode(payload["changed_b64"]), dtype=np.uint8).reshape(shape).astype(bool)


@pytest.mark.parametrize("settled_delta", [False, True])
async def test_a_move_gets_one_image_of_what_it_changed(settled_delta: bool) -> None:
    """Without a world model the frame before the move is the baseline, framediff's or not; once per frame, and
    none for a call that played nothing, a move that changed nothing, or a RESET."""
    level = _Level(settled_delta=settled_delta)
    try:
        assert await level.after("print(1)") is None, "the seed turn alone is a level's first frame"
        payload = await level.after("submit_action([Action('RIGHT')])")
        assert payload is not None
        assert (payload["kind"], payload["turn_index"], payload["step"], payload["won"]) == ("previous", 1, 1, False)
        assert payload["move"] == "Action('RIGHT')"
        assert np.array_equal(_changed(payload), _board(1, size=_SIZE) != _board(3, size=_SIZE))
        frame = np.frombuffer(base64.b64decode(payload["frame_b64"]), dtype=np.int8).reshape(_SIZE, _SIZE)
        assert np.array_equal(frame, _board(3, size=_SIZE))
        assert await level.after("print(2)") is None, "no new frame since"
        assert await level.after("submit_action([Action('UP')])") is None, "nothing changed"
        assert await level.after("submit_action([Action('RESET')])") is None, "a RESET restores, it does not move"
    finally:
        await level.tool.clear_state()


@pytest.mark.parametrize("reflection", [False, True])
async def test_a_winning_move_gets_an_image_only_for_the_reflection(reflection: bool) -> None:
    """The win's frame is shown where the reflection reads it; with none, the next prompt is the new level's,
    and no image of the old level's win screen comes with it, as no settled delta does."""
    level = _Level(settled_delta=True, reflection=reflection)
    try:
        payload = await level.after("submit_action([Action('LEFT')])")
        if reflection:
            assert payload is not None and payload["won"] and payload["kind"] == "previous"
        else:
            assert payload is None
    finally:
        await level.tool.clear_state()


# --- on the host ---------------------------------------------------------------------


def _pending(*, won: bool = False, kind: str = "previous") -> dict[str, Any]:
    """The kernel's payload for a move from column 1 to column 3."""
    changed = _board(1) != _board(3)
    return {
        "kind": kind,
        "turn_index": 1,
        "step": 7,
        "move": "Action('RIGHT')",
        "won": won,
        "shape": [8, 8],
        "frame_b64": base64.b64encode(_board(3).tobytes()).decode(),
        "changed_b64": base64.b64encode(changed.astype(np.uint8).tobytes()).decode(),
    }


def _board_entry() -> FramePayload:
    grid = _board(3)
    return FramePayload(grid_b64=base64.b64encode(grid.tobytes()).decode(), shape=[8, 8], step=7, level=1)


def test_the_image_follows_the_board_once() -> None:
    """Captioned with the move and its baseline; the changed cells in the board image's own colours, the rest
    navy; and on one prompt only."""
    agent = SealAgent(diff_image=True)
    agent.diff_image_pending = _pending(kind="prediction")
    content = agent._user_prompt_element("text", _board_entry()).content  # pyright: ignore[reportPrivateUsage]
    assert isinstance(content, ullm.conv.ListContent)
    *_, board_label, board, caption, diff = content.elements
    assert isinstance(board_label, ullm.conv.StringContent) and board_label.text.endswith("Current grid image:")
    assert isinstance(caption, ullm.conv.StringContent)
    assert caption.text == agent.prompts.diff_image_caption.format(
        move="Action('RIGHT')", step=7, baseline="your mispredicted frame"
    )
    assert isinstance(board, ullm.conv.ImageContent) and isinstance(diff, ullm.conv.ImageContent)
    board_rgb, diff_rgb = np.asarray(board.data), np.asarray(diff.data)
    assert diff_rgb.shape == board_rgb.shape
    scale = agent.multimodal_upscale
    changed = (_board(1) != _board(3)).repeat(scale, axis=0).repeat(scale, axis=1)
    assert np.array_equal(diff_rgb[changed], board_rgb[changed])
    assert (diff_rgb[~changed] == UNCHANGED_RGB).all()
    again = agent._user_prompt_element("text", _board_entry()).content  # pyright: ignore[reportPrivateUsage]
    assert isinstance(again, ullm.conv.ListContent) and len(again.elements) == 2, "the board alone"


def test_the_move_that_won_says_so() -> None:
    agent = SealAgent(diff_image=True)
    agent.diff_image_pending = _pending(won=True)
    content = agent._user_prompt_element("text", _board_entry()).content  # pyright: ignore[reportPrivateUsage]
    assert isinstance(content, ullm.conv.ListContent)
    caption = content.elements[-2]
    assert isinstance(caption, ullm.conv.StringContent)
    assert caption.text == agent.prompts.diff_image_caption_won.format(
        move="Action('RIGHT')", step=7, baseline="the previous frame"
    )


def test_off_no_image() -> None:
    agent = SealAgent()
    agent.diff_image_pending = _pending()
    content = agent._user_prompt_element("text", _board_entry()).content  # pyright: ignore[reportPrivateUsage]
    assert isinstance(content, ullm.conv.ListContent)
    assert not any(isinstance(e, ullm.conv.StringContent) and "Diff image" in e.text for e in content.elements)


def test_navy_is_no_game_colour() -> None:
    palette = np.asarray(frame_image(taaf.game.Frame(data=np.arange(16, dtype=np.int8).reshape(4, 4))).data)
    assert tuple(UNCHANGED_RGB) not in {tuple(int(v) for v in rgb) for rgb in palette.reshape(-1, 3)}


@pytest.mark.parametrize("mode", ["", "level_start"])
def test_refused_where_the_board_image_does_not_ride_every_prompt(mode: str) -> None:
    with pytest.raises(ValueError, match="diff_image=True requires multimodal_context='current_grid'"):
        SealAgent(diff_image=True, multimodal_context=mode).prepare_for_backend()
