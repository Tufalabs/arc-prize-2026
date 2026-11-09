"""``level_reflection``: a turn after a move that wins a level, before the next level's first.

Its prompt reports the move as any turn's does, over the win screen, then asks
for a reflection on the level, answered in a reply without a tool call. The
kernel still holds the finished level, and ``submit_action`` is refused. No
summary runs inside it: its context may run past the window by
``reflection_extra_window_tokens``, and past that the turn ends, saying so, and
the summary comes with the next level's first call. The next level's prompt then
opens at the level change, without repeating the report. None after the game's
win. Played in a real game, on a real kernel, through scripted replies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest
import taaf.benchmark
import ullm.backend
import ullm.conv
import ullm.factory

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from seal import SealAgent
from seal.prompts import SealPrompts
from test_seal_winning_turn import BLIND_MODEL, MODEL, WIN_SCREEN, WinGame, _python, _Script

SURFACE = SealPrompts()

#: Wins level 1, and reads nothing after.
WIN = "open('world_model.py', 'w').write({model!r})\nsubmit_action([Action('UP'), Action('UP')])\n"

#: A call in the reflection: the finished level, and an action refused.
REFLECT = (
    "print('REFLECTED', len(history), history[-1].outcome, int(history[-1].frame.grid[0, 0]))\n"
    "try:\n"
    "    submit_action(Action('UP'))\n"
    "except ValueError as error:\n"
    "    print('REFUSED', error)\n"
)
ANSWERS = "(1) Two UPs win a level. (2) None. (3) Read the board's fill first."
LOOK = "print('LEVEL', history[-1].frame.level, len(history))"
FINISH = "submit_action([Action('UP'), Action('UP')])"
SUMMARY = "SUMMARY: two UPs win a level."

#: The window, reply cap, reflection's extra and engine ceiling of the
#: compaction tests: the summary's trigger past 18,488 of prompt, the
#: reflection's past 28,488.
WINDOW, REPLY_CAP, EXTRA, CEILING = 20_000, 1_000, 10_000, 40_000


def _text(text: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=5),
    )


class _Counting(_Script):
    """Counts a prompt as a context the reflection fills: small before the
    win, past the summary's trigger once the reflection's prompt is in, and,
    under ``overflow``, past the reflection's own trigger once it has run a
    call; small again once a summary has rebuilt it."""

    overflow: ClassVar[bool] = False

    async def _token_count(self, sampling_params: Any, conversation: ullm.conv.Conversation) -> int | None:
        """The prompt's weight, as above."""
        del sampling_params
        text = conversation.to_string()
        if "[context compacted" in text and SURFACE.compaction_request_head[:30] not in text:
            return 2_000
        if type(self).overflow and "REFLECTED 3" in text:
            return 31_000
        if "You have won level" in text:
            return 19_000
        return 1_000


def _agent(backend: Any, **overrides: Any) -> SealAgent:
    settings: dict[str, Any] = {
        "derive_max_model_len": False,
        "derive_max_num_seqs": False,
        "derive_max_pixels": False,
        "reply_cap_tokens": 4096,
        "use_world_model": "optional",
        "use_search": False,
        "level_reflection": "verbose",
        "strategy_audit": "none",
        "llm_config": LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        "run_config": RunConfig(rng_seed=1, process_mode="inprocess"),
    }
    settings.update(overrides)
    return SealAgent(**settings)


async def _play(
    tmp_path: Path, replies: list[ullm.conv.LLMResponse], backend_cls: type[_Script] = _Script, **overrides: Any
) -> Any:
    """One game of WinGame on ``replies``; returns its run."""
    _Script.last_user = []
    _Script.texts = []
    backend = backend_cls(responses=replies, on_exhausted="repeat_last")
    benchmark = taaf.benchmark.Benchmark(
        games=[WinGame()], solver=LLMAgentSolver(agent=_agent(backend, **overrides)), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=120)
    (run,) = benchmark.game_runs
    return run


def _is_summary(text: str) -> bool:
    return SURFACE.compaction_request_head[:30] in text


async def test_the_turn_after_a_win_reflects_on_it(tmp_path: Path) -> None:
    replies = [_python(WIN.format(model=MODEL)), _python(REFLECT), _text(ANSWERS), _python(LOOK), _python(FINISH)]
    run = await _play(tmp_path, replies, animation_description="auto")
    assert run.state == "won"
    reflection = _Script.last_user[1]
    # The move reported, the win, the win screen as a delta and the winning
    # animation as a block, then the reflection asked for.
    assert reflection.startswith("The code executed 2 actions in the previous sequence.\nExecuted actions:\n- UP (")
    assert "history[2].animation_frames" in reflection
    assert "You have won level 1!" in reflection
    assert f"g[0:64, 0:64] = {WIN_SCREEN}" in reflection
    assert "The animation of history[2] (1 frames)" in reflection
    for fragment in (
        SURFACE.reflection_intro,
        SURFACE.reflection_world_model_sentence.strip(),
        SURFACE.reflection_questions.strip(),
        SURFACE.reflection_answer_line,
        SURFACE.no_action_turn_line,
    ):
        assert fragment in reflection
    # The model predicted the win, so nothing says it did not.
    assert SURFACE.reflection_win_missed_line not in reflection
    # Nothing is played in the turn: no status line.
    assert "Current state:" not in reflection
    # In the turn: the finished level, and no move.
    assert "REFLECTED 3 won 13" in _Script.texts[2]
    assert f"REFUSED {SURFACE.no_action_turn_refusal}" in _Script.texts[2]
    # The answer ends the turn; the next level opens at the level change,
    # without the report again.
    level_two = _Script.last_user[3]
    assert ANSWERS in _Script.texts[3]
    assert "You have progressed to a new level!" in level_two
    assert "The code executed" not in level_two
    assert "LEVEL 2 1" in _Script.texts[4]


async def test_a_win_the_world_model_missed_is_said(tmp_path: Path) -> None:
    replies = [_python(WIN.format(model=BLIND_MODEL)), _text(ANSWERS), _python(LOOK), _python(FINISH)]
    run = await _play(tmp_path, replies)
    assert run.state == "won"
    assert SURFACE.reflection_win_missed_line in _Script.last_user[1]


async def test_none_follows_the_games_win(tmp_path: Path) -> None:
    replies = [_python(WIN.format(model=MODEL)), _text(ANSWERS), _python(FINISH), _text("unused")]
    run = await _play(tmp_path, replies)
    assert run.state == "won"
    # Level 1's reflection, then level 2 won in one call: no request follows.
    assert len(_Script.texts) == 3
    assert not any("You have won level 2!" in text for text in _Script.texts)


def _summarizing() -> dict[str, Any]:
    return {
        "reply_cap_tokens": REPLY_CAP,
        "context_window_tokens": WINDOW,
        "reflection_extra_window_tokens": EXTRA,
        # The audit's extra would count against the ceiling too.
        "strategy_audit": "none",
        "engine_ceiling_tokens": CEILING,
        "token_native_context": True,
    }


async def test_a_summary_due_in_it_waits_for_the_next_level(tmp_path: Path) -> None:
    """The reflection's context passes the summary's trigger: no summary runs
    in the turn, and the one that runs at the next level's first call is
    written with the reflection and the next level's prompt in view."""
    _Counting.overflow = False
    replies = [
        _python(WIN.format(model=MODEL)),
        _python(REFLECT),
        _text(ANSWERS),
        _text(SUMMARY),
        _python(LOOK),
        _python(FINISH),
    ]
    run = await _play(tmp_path, replies, _Counting, **_summarizing())
    assert run.state == "won"
    assert [_is_summary(text) for text in _Script.texts] == [False, False, False, True, False, False]
    summary_request = _Script.texts[3]
    assert ANSWERS in summary_request
    assert summary_request.index(ANSWERS) < summary_request.index("You have progressed to a new level!")
    assert "LEVEL 2 1" in _Script.texts[5]


async def test_past_its_extra_it_ends_and_the_summary_follows(tmp_path: Path) -> None:
    """The reflection's context passes the window and its extra: the turn
    ends before its next call, and the summary runs at the next level's
    first, over the note and that level's prompt."""
    _Counting.overflow = True
    try:
        replies = [_python(WIN.format(model=MODEL)), _python(REFLECT), _text(SUMMARY), _python(LOOK), _python(FINISH)]
        run = await _play(tmp_path, replies, _Counting, **_summarizing())
    finally:
        _Counting.overflow = False
    assert run.state == "won"
    assert [_is_summary(text) for text in _Script.texts] == [False, False, True, False, False]
    summary_request = _Script.texts[2]
    note = summary_request.index(SURFACE.reflection_cut_note)
    assert note < summary_request.index("You have progressed to a new level!")
    assert "LEVEL 2 1" in _Script.texts[4]


def test_the_journal_refuses_it() -> None:
    with pytest.raises(ValueError, match="level_reflection='verbose' requires journal=False"):
        SealAgent(level_reflection="verbose", journal=True).prepare_for_backend()


def test_its_extra_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="reflection_extra_window_tokens"):
        SealAgent(reflection_extra_window_tokens=-1)


def test_its_extra_must_leave_a_summary_room_in_the_engine() -> None:
    """The turn runs past the window by its extra, and the summary after it
    still has to fit the engine's window."""
    fits = SealAgent(level_reflection="verbose", **_summarizing())
    fits._require_summary_fits(CEILING)  # pyright: ignore[reportPrivateUsage]
    too_wide = SealAgent(level_reflection="verbose", **_summarizing())
    too_wide.reflection_extra_window_tokens = EXTRA + 1_000
    with pytest.raises(ValueError, match="reflection_extra_window_tokens"):
        too_wide._require_summary_fits(CEILING)  # pyright: ignore[reportPrivateUsage]


async def test_concise_asks_for_a_brief_reflection_and_no_questions(tmp_path: Path) -> None:
    replies = [_python(WIN.format(model=MODEL)), _text(ANSWERS), _python(LOOK), _python(FINISH)]
    run = await _play(tmp_path, replies, level_reflection="concise")
    assert run.state == "won"
    reflection = _Script.last_user[1]
    assert "You have won level 1!" in reflection
    assert f"{SURFACE.reflection_concise}{SURFACE.reflection_world_model_sentence}" in reflection
    assert SURFACE.reflection_questions.strip() not in reflection and SURFACE.reflection_intro not in reflection
    assert f"{SURFACE.concise_answer_line}\n{SURFACE.no_action_turn_line}" in reflection
