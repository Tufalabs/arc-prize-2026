from __future__ import annotations

from re_arc.dsl import resolve_action, unpack_step_result
from re_arc.env_sampler import EnvSampler


def test_bottomless_pit_fall_animates_across_frames() -> None:
    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id="bottomless_pit-0001", seed=0)
    try:
        env.reset()
        env.step(resolve_action(env, 4), data={})

        observation, _reward, _done, _info = unpack_step_result(env.step(resolve_action(env, 4), data={}))

        assert observation.state.name == "NOT_FINISHED"
        assert len(observation.frame) > 1
        assert len({tuple(layer.ravel()) for layer in observation.frame}) > 1
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
