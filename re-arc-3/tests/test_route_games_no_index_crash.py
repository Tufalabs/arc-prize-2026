from __future__ import annotations

from contextlib import suppress

from arcengine import GameAction

from re_arc import EnvSampler


def test_stack_equalizer_repeated_clicks_do_not_raise_with_transition_reward():
    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id="stack_equalizer", seed=0)

    try:
        _ = env.reset()
        for _ in range(40):
            _, _, done, _ = env.step(GameAction.ACTION6, data={"x": 46, "y": 58})
            if done:
                break
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            with suppress(Exception):
                close_fn()
