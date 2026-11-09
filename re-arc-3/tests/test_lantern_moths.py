from __future__ import annotations

from re_arc.env_sampler import EnvSampler


def test_lantern_moths_level_5_moths_in_matching_boxes_are_solved() -> None:
    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id="lantern_moths-0001", seed=0)
    try:
        env.reset()
        game = env._game
        game.set_level(4)

        for moth in game.moths.values():
            flower = next(flower for flower in game.flowers if flower["color"] == moth["color"])
            moth["pos"] = tuple(flower["pos"])
            moth["awake"] = True

        assert game._is_solved()
        game._advance_moths()

        for moth in game.moths.values():
            assert game._matching_flower_at(tuple(moth["pos"]), str(moth["color"])) is not None
            assert moth["awake"] is False
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
