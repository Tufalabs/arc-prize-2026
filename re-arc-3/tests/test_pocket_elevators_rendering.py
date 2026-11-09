from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_pocket_elevators_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "re_arc"
        / "environment_files"
        / "pocket_elevators"
        / "0001"
        / "pocket_elevators.py"
    )
    spec = importlib.util.spec_from_file_location("test_pocket_elevators_env", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_empty_socket_is_a_hollow_stone_silhouette() -> None:
    env = _load_pocket_elevators_module()
    game = env.PocketElevators()
    game.on_set_level(game._levels[2])

    socket = (5, 4)
    x, y = game._cell_rect(socket)
    pixels = game._board.pixels

    for yy, row in enumerate(env.SOCKET_SHAPE):
        for xx, on in enumerate(row):
            if on:
                assert int(pixels[y + 1 + yy, x + 1 + xx]) == env.COLOR_STONE
    assert int(pixels[y + 2, x + 2]) == env.COLOR_H1


def test_filled_socket_flashes_around_the_stone() -> None:
    env = _load_pocket_elevators_module()
    game = env.PocketElevators()
    game.on_set_level(game._levels[2])

    socket = (5, 4)
    game.stones = {socket}
    game.heights[socket] = 0
    game._sync_visuals(socket_flash=True)
    x, y = game._cell_rect(socket)
    pixels = game._board.pixels

    assert int(pixels[y + 1, x + 2]) == env.COLOR_MINER
    assert int(pixels[y + 2, x + 2]) == env.COLOR_STONE
