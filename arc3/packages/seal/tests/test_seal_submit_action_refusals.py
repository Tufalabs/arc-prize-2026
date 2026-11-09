"""What ``submit_action`` says when a batch is written wrong, in the kernel.

A batch that is no Action or list of Actions is refused, ending with the game's actions, as the host's
refusals of a name or a click do (``test_seal_live_actions``). A ``plan`` keyword is refused with where it
belongs where the ``python`` tool takes one, and as Python refuses any unknown keyword where it does not; the
signature ``help()`` shows has ``actions`` alone.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from seal.harness import game as harness_game


def _submit_action(*, actions: list[str]) -> Any:
    def submit(_normalized: list[dict[str, Any]]) -> Any:
        raise AssertionError("a refused batch reaches no host")

    return harness_game._make_submit_action(submit, lambda _state: None, None, lambda: actions)  # pyright: ignore[reportPrivateUsage]


def test_a_batch_that_is_no_action_names_the_games_actions() -> None:
    with pytest.raises(TypeError) as raised:
        _submit_action(actions=["RESET", "UP", "MOUSE"])(5)
    assert str(raised.value).endswith(" This game's actions: RESET, UP, MOUSE.")
    with pytest.raises(ValueError, match=r"requires at least one action\. This game's actions: RESET, UP\.$"):
        _submit_action(actions=["RESET", "UP"])([])


@pytest.mark.parametrize("has_plan", [False, True])
def test_a_plan_keyword_is_refused_and_unseen(monkeypatch: pytest.MonkeyPatch, has_plan: bool) -> None:
    monkeypatch.setattr(harness_game, "_has_plan", has_plan)
    submit_action = _submit_action(actions=["UP"])
    expected = (
        harness_game.PLAN_ON_SUBMIT_ACTION if has_plan else "submit_action() got an unexpected keyword argument 'plan'"
    )
    with pytest.raises(TypeError) as raised:
        submit_action([harness_game.Action("UP")], plan="go up")
    assert str(raised.value) == expected
    assert str(inspect.signature(submit_action)) == "(actions)"
