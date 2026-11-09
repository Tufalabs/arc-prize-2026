"""The example world models a game starts with.

Each is the complete ``world_model.py`` of a small game — a square recoloured
by LEFT/RIGHT, and tiles flipped by clicking — written into the working
directory at game start, the one for a game with MOUSE or without it. They
show the contract working, and they fail on any real game, which is what
keeps exploration open until the model writes its own.

The files are data, not code: every byte is what the model reads, so they
are excluded from formatting, linting and type checking. The games they
model exist as TAAF games in ``tests/test_seal_seeds.py``, which holds each
example to ``test_world_model()`` over a played sequence.
"""

from __future__ import annotations

from importlib import resources


def seed_source(mouse: bool) -> str:
    """The example ``world_model.py`` for a game with MOUSE, or without."""
    name = "world_model_mouse.py" if mouse else "world_model_keyboard.py"
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
