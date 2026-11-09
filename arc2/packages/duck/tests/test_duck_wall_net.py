"""The wall net is on everywhere but Kaggle, and that contrast is deliberate.

The net exists for the failure the token budget structurally cannot see: a
game spending its hours inside the kernel generates almost no tokens, so
``max_tokens_per_game`` never retires it and only the job's walltime does —
which cancels the game rather than scoring it (`README.md`, "Run budgets").

Kaggle is the one deployment that turns it off, because there the
submission's own clock is already the only budget and cutting a game short
would forfeit an answer the arena would otherwise have scored. Forgetting the
cap and choosing to drop it produce the same value, so what is pinned here is
the difference between the two rather than either number: a retune of the cap
leaves these passing, and losing the contrast does not.
"""

from __future__ import annotations

import pytest
from duck import DuckAgent, presets


def test_a_duck_run_config_carries_the_net_by_default() -> None:
    """Every preset builds its RunConfig through here so the cap has one
    definition — a preset that built its own and forgot would fail silently,
    as no cap at all."""
    assert presets.duck_run_config().max_runtime_s_per_game is not None


def test_a_duck_built_by_hand_is_bounded_too() -> None:
    """The agent's own default is the same helper, so a duck assembled
    without a preset is not the one configuration that can run forever."""
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).run_config.max_runtime_s_per_game
        is not None
    )


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_kaggle_is_the_one_deployment_without_it() -> None:
    """And it says so by passing None through the same helper, which reads as
    the deliberate choice it is rather than as an omission."""
    kaggle_config = DuckAgent.build(environment="kaggle-4xl4").run_config
    served_config = DuckAgent.build().run_config

    assert kaggle_config.max_runtime_s_per_game is None
    assert served_config.max_runtime_s_per_game is not None
    assert served_config.max_runtime_s_per_game != kaggle_config.max_runtime_s_per_game


def test_a_deployment_can_still_choose_its_own_net() -> None:
    """The helper supplies a default, not a policy: a caller that knows its own
    deadline overrides it, which is how Kaggle turns it off in the first
    place."""
    assert presets.duck_run_config(max_runtime_s_per_game=60.0).max_runtime_s_per_game == 60.0
    assert presets.duck_run_config(max_runtime_s_per_game=None).max_runtime_s_per_game is None
