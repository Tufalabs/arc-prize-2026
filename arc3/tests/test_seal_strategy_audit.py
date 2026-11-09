"""``strategy_audit`` at ``audit_level_tokens``: a step back from a level that has run long.

Checked before every model call: once the level's spend passes a threshold, a
turn that is not for acting asks the model to reconsider, with advice, the
actions it has not tried, and two questions. After a call that played no move
it interrupts the turn, which then goes on with one line; after a call that
played moves it follows the next move's status prompt. A threshold the audit's
own spend passes counts as fired, and the reflection turn's spend is not
counted. No summary runs inside it: one that falls due comes with the call
after it, and keeps the move's own status prompt, not the audit's. Played in a
real game, on a real kernel, through scripted replies.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import taaf.benchmark
import ullm.backend
import ullm.conv
import ullm.factory

from platypus import Action, LLMAgentSolver, LLMConfig, RunConfig
from seal import SealAgent
from seal.prompts import SealPrompts
from seal.state import FramePayload, TimelineEntry
from test_seal_winning_turn import MODEL, WinGame, _python, _Script

SURFACE = SealPrompts()
INTRO = SURFACE.audit_intro.split("{level}")[0]

LOOK = "print('LEVEL', history[-1].frame.level, len(history))"
ACT = "submit_action([Action('DOWN')])"
WIN = "open('world_model.py', 'w').write({model!r})\nsubmit_action([Action('UP'), Action('UP')])\n".format(model=MODEL)
FINISH = "submit_action([Action('UP'), Action('UP')])"
AUDIT_CELL = "try:\n    submit_action(Action('UP'))\nexcept ValueError as error:\n    print('REFUSED', error)\n"
ANSWERS = "(1) Two UPs advance the level; nothing else is known. (2) Play UP twice."
SUMMARY = "SUMMARY: two UPs win a level."

#: The window, reply cap, audit's extra and engine ceiling of the compaction
#: test: the summary's trigger past 18,488 of prompt, the audit's past 28,488.
WINDOW, REPLY_CAP, EXTRA, CEILING = 20_000, 1_000, 10_000, 40_000


def _text(text: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=5),
    )


class _Counting(_Script):
    """Counts a prompt as a context the audit fills: small before it, past the
    summary's trigger once its prompt is in, small again once a summary has
    rebuilt it."""

    async def _token_count(self, sampling_params: Any, conversation: ullm.conv.Conversation) -> int | None:
        """The prompt's weight, as above."""
        del sampling_params
        text = conversation.to_string()
        if "[context compacted" in text and SURFACE.compaction_request_head[:30] not in text:
            return 2_000
        if INTRO in text:
            return 19_000
        return 1_000


async def _play(
    tmp_path: Path, replies: list[ullm.conv.LLMResponse], backend_cls: type[_Script] = _Script, **overrides: Any
) -> Any:
    """One game of WinGame on ``replies``; returns its run. Each scripted reply
    bills five completion tokens."""
    _Script.last_user = []
    _Script.texts = []
    backend = backend_cls(responses=replies, on_exhausted="repeat_last")
    settings: dict[str, Any] = {
        "derive_max_model_len": False,
        "derive_max_num_seqs": False,
        "derive_max_pixels": False,
        "reply_cap_tokens": 4096,
        "use_world_model": "optional",
        "use_search": False,
        "strategy_audit": "verbose",
        # WinGame's win would be followed by a reflection turn, the script's
        # replies answering it; the audit alone is under test.
        "level_reflection": "none",
        "llm_config": LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        "run_config": RunConfig(rng_seed=1, process_mode="inprocess"),
    }
    settings.update(overrides)
    benchmark = taaf.benchmark.Benchmark(
        games=[WinGame()], solver=LLMAgentSolver(agent=SealAgent(**settings)), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=120)
    (run,) = benchmark.game_runs
    return run


def _audits() -> list[int]:
    """The requests whose last user message is an audit prompt."""
    return [i for i, text in enumerate(_Script.last_user) if text.startswith(INTRO)]


async def test_a_threshold_passed_mid_turn_interrupts_it(tmp_path: Path) -> None:
    """Three calls of five tokens pass 12: the audit runs before the fourth,
    and the turn goes on with one line."""
    replies = [
        _python(LOOK),
        _python(LOOK),
        _python(LOOK),
        _python(AUDIT_CELL),
        _text(ANSWERS),
        _python(WIN),
        _python(LOOK),
        _python(FINISH),
    ]
    run = await _play(tmp_path, replies, audit_level_tokens=(12,))
    assert run.state == "won"
    assert _audits() == [3, 4]
    audit = _Script.last_user[3]
    assert audit.startswith(SURFACE.audit_intro.format(level=1))
    for line in SURFACE.audit_advice:
        assert f"- {line}" in audit
    assert "- Not yet tried on this level: UP, DOWN; UP and DOWN have not been tried in this game at all." in audit
    assert SURFACE.audit_questions in audit
    # The level's opening board and the current one, after the text.
    assert f"{SURFACE.audit_answer_line}\n{SURFACE.no_action_turn_line}\n\nLevel start image:\n[image:" in audit
    assert "\nCurrent grid image:\n[image:" in audit
    # In the audit: no move.
    assert f"REFUSED {SURFACE.no_action_turn_refusal}" in _Script.texts[4]
    # The answer ends it, and the turn goes on.
    assert ANSWERS in _Script.texts[5]
    assert _Script.last_user[5] == SURFACE.audit_continue_line


async def test_a_threshold_its_own_spend_passes_does_not_fire_again(tmp_path: Path) -> None:
    """The audit's two calls carry the level from 15 to 25 tokens, past 20:
    that threshold counts as fired, so no second audit follows it."""
    replies = [
        _python(LOOK),
        _python(LOOK),
        _python(LOOK),
        _python(AUDIT_CELL),
        _text(ANSWERS),
        _python(LOOK),
        _python(WIN),
        _python(FINISH),
    ]
    run = await _play(tmp_path, replies, audit_level_tokens=(12, 20))
    assert run.state == "won"
    assert _audits() == [3, 4]


async def test_after_a_call_that_played_a_move_it_follows_the_next_status_prompt(tmp_path: Path) -> None:
    """The move's one call passes 4 and plays DOWN, ending the move: the
    audit comes before the next move's first call, after its status prompt.
    Level 2 is won in its first call, before it has spent anything."""
    replies = [_python(ACT), _text(ANSWERS), _python(WIN), _python(FINISH)]
    run = await _play(tmp_path, replies, audit_level_tokens=(4,))
    assert run.state == "won"
    assert _audits() == [1]
    request = _Script.texts[1]
    assert request.index("The code executed 1 action in the previous sequence.") < request.index(INTRO)
    assert _Script.last_user[2] == SURFACE.audit_continue_line


async def test_the_reflection_turn_s_spend_does_not_count(tmp_path: Path) -> None:
    """Level 1 is won in one call; the reflection then spends 20 tokens,
    charged to level 2, past its threshold of 12, and no audit follows."""
    replies = [
        _python(WIN),
        _python(LOOK),
        _python(LOOK),
        _python(LOOK),
        _text(ANSWERS),
        _python(LOOK),
        _python(FINISH),
    ]
    run = await _play(tmp_path, replies, level_reflection="verbose", audit_level_tokens=(12,))
    assert run.state == "won"
    assert _audits() == []


async def test_a_summary_due_in_it_waits_and_keeps_the_move_s_prompt(tmp_path: Path) -> None:
    """The audit's context passes the summary's trigger: no summary runs in
    it. The one at the call after it is written with the audit and its line
    in view, and the conversation it rebuilds keeps the move's own status
    prompt, not the audit's."""
    replies = [
        _python(LOOK),
        _python(LOOK),
        _python(AUDIT_CELL),
        _text(ANSWERS),
        _text(SUMMARY),
        _python(WIN),
        _python(LOOK),
        _python(FINISH),
    ]
    run = await _play(
        tmp_path,
        replies,
        _Counting,
        audit_level_tokens=(8,),
        reply_cap_tokens=REPLY_CAP,
        context_window_tokens=WINDOW,
        audit_extra_window_tokens=EXTRA,
        engine_ceiling_tokens=CEILING,
        token_native_context=True,
    )
    assert run.state == "won"
    head = SURFACE.compaction_request_head[:30]
    assert [head in text for text in _Script.texts] == [False, False, False, False, True, False, False, False]
    summary_request = _Script.texts[4]
    assert ANSWERS in summary_request and SURFACE.audit_continue_line in summary_request
    rebuilt = _Script.texts[5]
    assert "No actions have been executed yet." in rebuilt
    assert INTRO not in rebuilt


def _frame(fill: int) -> FramePayload:
    grid = np.full((4, 4), fill, dtype=np.uint8)
    grid[0, 0] = 5
    grid[3, 3] = 9
    return FramePayload(grid_b64=base64.b64encode(grid.tobytes()).decode("ascii"), shape=[4, 4], step=0, level=1)


def test_a_click_s_colour_is_tried_and_the_rest_named() -> None:
    """In a game with MOUSE, the colours on the board no click hit: the level
    clicked colour 5, so 0 and 9 are named, as never clicked in the game too."""
    agent = SealAgent(game_action_names=["RESET", "UP", "DOWN", "MOUSE"], multimodal_context="")
    board = _frame(0)
    agent.history_entries = [
        TimelineEntry(action=None, frame=board, animation=[board], level=1),
        TimelineEntry(action=Action("ACTION6", row=0, col=0), frame=board, animation=[board], level=1),
        TimelineEntry(action=Action("ACTION1"), frame=board, animation=[board], level=1),
    ]
    text = agent._build_audit_prompt_element().content.to_string()  # pyright: ignore[reportPrivateUsage]
    assert (
        "- Not yet tried on this level: DOWN; DOWN has not been tried in this game at all. "
        "Colors on the board that no click has hit on this level: 0, 9; 0 and 9 have never been clicked in this game."
    ) in text


def test_an_offered_undo_is_not_named_untried() -> None:
    """Under ``expose_undo`` UNDO is among the game's actions, and still left
    out of what the audit names, as RESET is."""
    agent = SealAgent(
        game_action_names=["RESET", "UP", "DOWN", "UNDO"],
        use_world_model="none",
        expose_undo=True,
        multimodal_context="",
    )
    board = _frame(0)
    agent.history_entries = [
        TimelineEntry(action=None, frame=board, animation=[board], level=1),
        TimelineEntry(action=Action("ACTION1"), frame=board, animation=[board], level=1),
    ]
    text = agent._build_audit_prompt_element().content.to_string()  # pyright: ignore[reportPrivateUsage]
    assert "- Not yet tried on this level: DOWN; DOWN has not been tried in this game at all." in text
    assert "UNDO" not in text


@pytest.mark.parametrize("thresholds", [(0,), (20, 10), (10, 10)])
def test_its_thresholds_are_positive_and_ascending(thresholds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="audit_level_tokens"):
        SealAgent(audit_level_tokens=thresholds)


def test_its_extra_must_leave_a_summary_room_in_the_engine() -> None:
    """The audit runs past the window by its extra, and the summary after it
    still has to fit the engine's window."""
    agent = SealAgent(
        audit_level_tokens=(75_000,),
        strategy_audit="verbose",
        level_reflection="none",
        reply_cap_tokens=REPLY_CAP,
        context_window_tokens=WINDOW,
        audit_extra_window_tokens=EXTRA,
    )
    agent._require_summary_fits(CEILING)  # pyright: ignore[reportPrivateUsage]
    agent.audit_extra_window_tokens = EXTRA + 1_000
    with pytest.raises(ValueError, match="audit_extra_window_tokens"):
        agent._require_summary_fits(CEILING)  # pyright: ignore[reportPrivateUsage]


def test_its_mode_needs_thresholds() -> None:
    with pytest.raises(ValueError, match="strategy_audit='concise' needs audit_level_tokens"):
        SealAgent(strategy_audit="concise", audit_level_tokens=()).prepare_for_backend()


async def test_concise_asks_for_a_brief_reflection_and_no_questions(tmp_path: Path) -> None:
    replies = [_python(LOOK), _python(LOOK), _python(LOOK), _text(ANSWERS), _python(WIN), _python(FINISH)]
    run = await _play(tmp_path, replies, audit_level_tokens=(12,), strategy_audit="concise")
    assert run.state == "won"
    (index,) = [i for i, text in enumerate(_Script.last_user) if text.startswith(INTRO)]
    audit = _Script.last_user[index]
    assert audit.startswith(SURFACE.audit_concise_intro.format(level=1))
    assert SURFACE.audit_questions not in audit
    assert f"{SURFACE.concise_answer_line}\n{SURFACE.no_action_turn_line}" in audit
    assert all(f"- {line}" in audit for line in SURFACE.audit_advice)
