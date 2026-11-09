"""``suggest_moves_after_tokens``: a note on every tool result once a game has
thought that long without a move.

Counted on the game's own meter from the last move, summary, or reflection or
audit end; a call that moved has just reset it, and a side turn, which cannot
move, gets no note. Played in a real game, on a real kernel, through scripted
replies of five completion tokens each.
"""

from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import Any

import pytest
import taaf.benchmark
import ullm.backend
import ullm.conv
import ullm.factory

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from seal import SealAgent
from seal.prompts import SealPrompts
from test_seal_winning_turn import MODEL, WinGame, _python, _Script

SURFACE = SealPrompts()
NOTE_HEAD = SURFACE.suggest_moves_note.split("{tokens")[0]

LOOK = "print('LEVEL', history[-1].frame.level, len(history))"
WIN = "open('world_model.py', 'w').write({model!r})\nsubmit_action([Action('UP'), Action('UP')])\n".format(model=MODEL)
FINISH = "submit_action([Action('UP'), Action('UP')])"
ANSWERS = "Two UPs advance the level."


def _note(tokens: int) -> str:
    return SURFACE.suggest_moves_note.format(tokens=f"{tokens:,}")


def _text(text: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=5),
    )


async def _play(tmp_path: Path, replies: list[ullm.conv.LLMResponse], **overrides: Any) -> Any:
    """One game of WinGame on ``replies``; returns its run."""
    _Script.last_user = []
    _Script.texts = []
    backend = _Script(responses=replies, on_exhausted="repeat_last")
    settings: dict[str, Any] = {
        "derive_max_model_len": False,
        "derive_max_num_seqs": False,
        "derive_max_pixels": False,
        "reply_cap_tokens": 4096,
        "use_world_model": "optional",
        "use_search": False,
        "strategy_audit": "none",
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


async def test_every_result_past_the_threshold_says_so_until_a_move(tmp_path: Path) -> None:
    """The second call reaches 10 tokens without a move and the third 15, each
    result saying so; the call that moves resets the count, so its result goes
    without, and level 2's first call is 5 past the move."""
    replies = [_python(LOOK), _python(LOOK), _python(LOOK), _python(WIN), _python(LOOK), _python(FINISH)]
    run = await _play(tmp_path, replies, suggest_moves_after_tokens=10)
    assert run.state == "won"
    texts = _Script.texts
    assert NOTE_HEAD not in texts[1]
    assert texts[2].count(NOTE_HEAD) == 1 and _note(10) in texts[2]
    assert texts[3].count(NOTE_HEAD) == 2 and _note(15) in texts[3]
    assert all(_note(tokens) not in text for text in texts for tokens in (0, 5, 20, 25))


async def test_off_by_default(tmp_path: Path) -> None:
    replies = [_python(LOOK), _python(LOOK), _python(LOOK), _python(WIN), _python(FINISH)]
    run = await _play(tmp_path, replies)
    assert run.state == "won"
    assert not any(NOTE_HEAD in text for text in _Script.texts)


async def test_a_side_turn_gets_none_and_its_end_resets_the_count(tmp_path: Path) -> None:
    """Level 1 is won in its first call; the reflection's two calls pass the
    threshold of 5 without a note, since it cannot move, and level 2's first
    call is counted from the reflection's end."""
    replies = [_python(WIN), _python(LOOK), _python(LOOK), _text(ANSWERS), _python(LOOK), _python(FINISH)]
    run = await _play(tmp_path, replies, level_reflection="verbose", suggest_moves_after_tokens=5)
    assert run.state == "won"
    texts = _Script.texts
    # Requests 1-3 are the reflection's, the last carrying both its results.
    assert ANSWERS not in texts[3] and texts[3].count("LEVEL") >= 2
    assert not any(NOTE_HEAD in text for text in texts[:4])
    assert texts[5].count(NOTE_HEAD) == 1 and _note(5) in texts[5]


def test_a_restored_agent_counts_from_its_restore() -> None:
    """The game's meter starts again at zero on a restore, so the mark does."""
    agent = SealAgent(suggest_moves_after_tokens=10)
    agent._moveless_since = 12_345  # pyright: ignore[reportPrivateUsage]
    restored = pickle.loads(pickle.dumps(agent))
    assert restored._moveless_since == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("tokens", [-1, 2.5, True])
def test_its_threshold_is_a_whole_number(tokens: Any) -> None:
    with pytest.raises(ValueError, match="suggest_moves_after_tokens"):
        SealAgent(suggest_moves_after_tokens=tokens)
