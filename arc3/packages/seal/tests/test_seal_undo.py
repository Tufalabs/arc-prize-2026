"""``expose_undo``: UNDO offered as the duck offers it, on a seal without a world
model alone.

Named among the game's actions only when offered and the game has it, refused
by the host otherwise, and refused at ``prepare_for_backend`` beside a world
model, which has no contract for it. Played in a real game by the
``seal_scripted_undo`` reference (``tests/test_seal_scripted_counter.py``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from seal import SealAgent


def _observed(names: list[str]) -> Any:
    return SimpleNamespace(available_actions=names)


def test_undo_is_named_only_when_offered_and_the_game_has_it() -> None:
    with_undo = _observed(["RESET", "ACTION1", "ACTION7"])
    off, on = SealAgent(use_world_model="none"), SealAgent(use_world_model="none", expose_undo=True)

    assert "UNDO" not in off._game_action_names(with_undo)  # pyright: ignore[reportPrivateUsage]
    assert "UNDO" in on._game_action_names(with_undo)  # pyright: ignore[reportPrivateUsage]
    assert "UNDO" not in on._game_action_names(_observed(["RESET", "ACTION1"]))  # pyright: ignore[reportPrivateUsage]


def test_undo_is_refused_unless_offered() -> None:
    off, on = SealAgent(use_world_model="none"), SealAgent(use_world_model="none", expose_undo=True)

    _, error = off._stage_actions([{"action": "UNDO"}])  # pyright: ignore[reportPrivateUsage]
    staged, none = on._stage_actions([{"action": "UNDO"}])  # pyright: ignore[reportPrivateUsage]
    assert error == off.prompts.stage_error_undo_disabled.format(index=1)
    assert none is None and [a.name for a in staged] == ["ACTION7"]


@pytest.mark.parametrize("mode", ["optional", "mandatory"])
def test_a_world_model_refuses_undo(mode: str) -> None:
    with pytest.raises(ValueError, match="expose_undo=True requires use_world_model='none'"):
        SealAgent(use_world_model=mode, expose_undo=True).prepare_for_backend()  # pyright: ignore[reportArgumentType]
    SealAgent(use_world_model="none", expose_undo=True).prepare_for_backend()
