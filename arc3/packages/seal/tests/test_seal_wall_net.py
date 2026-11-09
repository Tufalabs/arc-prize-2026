"""The wall net is on everywhere but Kaggle, and that contrast is deliberate.

The net exists for the failure the token allowance structurally cannot see: a
game spending its hours on actions issued from inside the kernel generates no
tokens, so the allowance never retires it and only the job's walltime does —
which cancels the game rather than scoring it (``design.md``, "Run budgets";
the reading is in ``evidence.md``).

Kaggle is the one deployment that turns it off, because there the arena's
clock is already the only budget and cutting a game short would forfeit
levels it was still being scored on. Forgetting the cap and choosing to drop
it produce the same value, so what is pinned here is the difference between
the two rather than either number: a retune of the cap leaves these passing,
and losing the contrast does not.
"""

from __future__ import annotations

from seal import presets
from seal.agent import SealAgent


def test_a_seal_run_config_carries_the_net_by_default() -> None:
    """Every preset builds its RunConfig through here so the cap has one
    definition — a preset that built its own and forgot would fail silently,
    as no cap at all."""
    assert presets.seal_run_config().max_runtime_s_per_game is not None


def test_kaggle_is_the_one_deployment_without_it() -> None:
    """And it says so by passing None through the same helper, which reads as
    the deliberate choice it is rather than as an omission."""
    kaggle_config = SealAgent.build(environment="kaggle-rtx").run_config
    standard = SealAgent.build().run_config

    assert kaggle_config.max_runtime_s_per_game is None
    assert standard.max_runtime_s_per_game != kaggle_config.max_runtime_s_per_game


def test_a_deployment_can_still_choose_its_own_net() -> None:
    """The helper supplies a default, not a policy: a caller that knows its own
    deadline overrides it, which is how Kaggle turns it off in the first
    place."""
    assert presets.seal_run_config(max_runtime_s_per_game=60.0).max_runtime_s_per_game == 60.0
    assert presets.seal_run_config(max_runtime_s_per_game=None).max_runtime_s_per_game is None
