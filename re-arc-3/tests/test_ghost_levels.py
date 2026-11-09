from __future__ import annotations

from arcengine import GameState

from re_arc.env_sampler import EnvSampler


def test_ghost_level_6_green_opener_is_not_direct_shortcut() -> None:
    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id="ghost-0001", seed=0)
    try:
        env.reset()
        game = env._game
        game.set_level(5)
        game._state = GameState.NOT_FINISHED
        game.live_pos = game.spec.plates["green"][0]
        game.ghosts = [None for _ in range(game.spec.max_ghosts)]
        gate = game.spec.gates["green"][0]

        assert abs(game.live_pos[0] - gate[0]) + abs(game.live_pos[1] - gate[1]) > 1

        game._handle_move(3)

        assert game.live_pos == game.spec.plates["green"][0]
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
