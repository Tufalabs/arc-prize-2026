# ruff: noqa: E501

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

# Sprite tag constants.
TAG_STATUS_PANEL = "status_panel"
TAG_TARGET_SLOT = "target_slot"
TAG_WALL = "wall"
TAG_SELECTED_TELEPORT = "selected_teleport"
TAG_PLAYER = "player"
TAG_TELEPORT = "teleport"

# Level data keys.
LEVEL_DATA_TIMER_MAX_STEPS = "timer_max_steps"
LEVEL_DATA_INITIAL_TELEPORT_INDEX = "initial_teleport_index"

# Create sprites dictionary with all sprite definitions
# fmt: off
sprites = {
    # Status Panel
    "sta": Sprite(
        pixels=[
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        name="sta",
        visible=True,
        collidable=True,
        tags=[TAG_STATUS_PANEL],
    ),
    "krg": Sprite(
        pixels=[
            [11],
        ],
        name="krg",
        visible=True,
        collidable=True,
        layer=3,
    ),
    # Target
    "tgt": Sprite(
        pixels=[
            [4, 1, 4, 1, 4],
            [1, 4, 1, 4, 1],
            [4, 1, 4, 1, 4],
            [1, 4, 1, 4, 1],
            [4, 1, 4, 1, 4],
        ],
        name="tgt",
        visible=True,
        collidable=False,
        tags=[TAG_TARGET_SLOT],
    ),
    "itg": Sprite(
        pixels=[
            [1, 4, 1, 4, 1],
            [4, 1, 4, 1, 4],
            [1, 4, 1, 4, 1],
            [4, 1, 4, 1, 4],
            [1, 4, 1, 4, 1],
        ],
        name="itg",
        visible=True,
        collidable=False,
        tags=[TAG_TARGET_SLOT],
    ),
    # Game Hull
    "bgd": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
        ],
        name="bgd",
        visible=True,
        collidable=True,
        layer=-3,
    ),
    "wal": Sprite(
        pixels=[
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
        ],
        name="wal",
        visible=True,
        collidable=True,
        tags=[TAG_WALL],
        layer=-5,
    ),
    "bck": Sprite(
        pixels=[
            [4, 4, 4],
            [4, 4, 4],
            [4, 4, 4],
        ],
        name="bck",
        visible=True,
        collidable=True,
        tags=[TAG_WALL],
        layer=-5,
    ),
    # Distraction
    "eff": Sprite(
        pixels=[
            [0, -1, 0, -1, 0],
            [-1, -1, -1, -1, -1],
            [0, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1],
        ],
        name="eff",
        visible=True,
        collidable=True,
        tags=[TAG_TELEPORT],
    ),
     "flp": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, 0],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0],
        ],
        name="flp",
        visible=True,
        collidable=True,
        tags=[TAG_TELEPORT],
    ),
    # Teleporters
    "ver": Sprite(
        pixels=[
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
        ],
        name="ver",
        visible=True,
        collidable=True,
        tags=[TAG_TELEPORT],
    ),
    "veru": Sprite(
        pixels=[
            [-1, -1, 15, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
        ],
        name="veru",
        visible=True,
        collidable=False,
    ),
    "verd": Sprite(
        pixels=[
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 15, -1, -1],
        ],
        name="verd",
        visible=True,
        collidable=False,
    ),
    "hor": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [0, -1, 0, -1, 0],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        name="hor",
        visible=True,
        collidable=True,
        tags=[TAG_TELEPORT],
    ),
    "horl": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [15, -1, 0, -1, 0],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        name="horl",
        visible=True,
        collidable=False,
    ),
    "horr": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [0, -1, 0, -1, 15],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        name="horr",
        visible=True,
        collidable=False,
    ),
    "dgr": Sprite(
        pixels=[
            [-1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1],
        ],
        name="dgr",
        visible=True,
        collidable=True,
        tags=[TAG_TELEPORT],
    ),
    "dgl": Sprite(
        pixels=[
            [0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0],
        ],
        name="dgl",
        visible=True,
        collidable=True,
        tags=[TAG_TELEPORT],
    ),
    "rup": Sprite(
        pixels=[
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1],
        ],
        name="rup",
        visible=True,
        collidable=True,
        tags=[TAG_TELEPORT],
    ),
    "ldn": Sprite(
        pixels=[
            [-1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1],
        ],
        name="ldn",
        visible=True,
        collidable=True,
        tags=[TAG_TELEPORT],
    ),
    "kdj": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        name="kdj",
        visible=True,
        collidable=True,
        tags=[TAG_SELECTED_TELEPORT],
        layer=10,
    ),
    # Players
    "pch": Sprite(
        pixels=[
            [10, 10, 10, 10, 10],
            [10, 12, 10, 12, 10],
            [10, 10, 10, 10, 10],
            [12, 10, 10, 10, 12],
            [10, 12, 12, 12, 10],
        ],
        name="pch",
        visible=True,
        collidable=True,
        tags=[TAG_PLAYER],
    ),
    "pcm": Sprite(
        pixels=[
            [10, 10, 10, 10, 10],
            [10, 13, 10, 13, 10],
            [10, 10, 10, 10, 10],
            [13, 13, 13, 13, 13],
            [10, 10, 10, 10, 10],
        ],
        name="pcm",
        visible=True,
        collidable=True,
        tags=[TAG_PLAYER],
    ),
    "pcd": Sprite(
        pixels=[
            [10, 10, 10, 10, 10],
            [10, 8, 10, 8, 10],
            [10, 10, 10, 10, 10],
            [10, 8, 8, 8, 10],
            [8, 10, 10, 10, 8],
        ],
        name="pcd",
        visible=True,
        collidable=True,
        tags=[TAG_PLAYER],
    ),
}

# Teleport routing matrix:
# - First key is the teleport sprite name (`ver`, `hor`, `dgr`, `dgl`).
# - Inner key is the incoming unit move direction (dx, dy) from the action
#   that entered the teleport tile.
# - Value is the final jump delta in grid cells, always 2 cells in this table.
# - The player can jump over walls, but landing on a wall is rejected.
#
# Example (`ver`):
# - Enter while moving right   `(1, 0)` -> jump right 2 cells `(2, 0)`.
# - Enter while moving left  `(-1, 0)` -> jump left  2 cells `(-2, 0)`.
# - Enter while moving down   `(0, 1)` -> jump down 2 cells `(0, 2)`.
# - Enter while moving up     `(0, -1)` -> jump up   2 cells `(0, -2)`.
teleport_matrix = {
    "ver": {
        (-1, 0): (0, 0),
        (1, 0): (0, 0),
        (0, -1): (0, -2),
        (0, 1): (0, 2),
        # if entered diagonally, only move vertically
        (1, -1): (0, -2),
        (-1, -1): (0, -2),
        (1, 1): (0, 2),
        (-1, 1): (0, 2),
    },
    "hor": {
        (-1, 0): (-2, 0),
        (1, 0): (2, 0),
        (0, -1): (0, 0),
        (0, 1): (0, 0),
        # if entered diagonally, only move horizontally
        (1, -1): (2, 0),
        (-1, -1): (-2, 0),
        (1, 1): (2, 0),
        (-1, 1): (-2, 0),
    },
    "dgr": {
        (-1, 0): (-2, 2),
        (1, 0): (2, -2),
        (0, -1): (2, -2),
        (0, 1): (-2, 2),
        # if entered diagonally, prioritize vertical movement
        (1, -1): (2, -2),
        (-1, -1): (2, -2),
        (1, 1): (-2, 2),
        (-1, 1): (-2, 2),
    },
    "dgl": {
        (-1, 0): (-2, -2),
        (1, 0): (2, 2),
        (0, -1): (-2, -2),
        (0, 1): (2, 2),
        # if entered diagonally, prioritize horizontal movement
        (1, -1): (-2, -2),
        (-1, -1): (-2, -2),
        (1, 1): (2, 2),
        (-1, 1): (2, 2),
    },
    "rup": {
        (-1, 0): (1, -2),
        (1, 0): (1, -2),
        (0, -1): (1, -2),
        (0, 1): (1, -2),
        # if entered diagonally, do the same
        (1, -1): (1, -2),
        (-1, -1): (1, -2),
        (1, 1): (1, -2),
        (-1, 1): (1, -2),
    },
    "ldn": {
        (-1, 0): (-1, 2),
        (1, 0): (-1, 2),
        (0, -1): (-1, 2),
        (0, 1): (-1, 2),
        # if entered diagonally, do the same
        (1, -1): (-1, 2),
        (-1, -1): (-1, 2),
        (1, 1): (-1, 2),
        (-1, 1): (-1, 2),
    },
}


# Create levels array with all level definitions
levels = [
    # krg
    Level(
        sprites=[
            # Status
            sprites["sta"].clone().set_position(54, 0),
            sprites["kdj"].clone().set_position(57, 2),
            # Target
            sprites["tgt"].clone().set_position(47, 17),
            # Background
            sprites["bgd"].clone(),
            # player start position
            sprites["pch"].clone().set_position(2, 57),
            # Teleports
            sprites["ver"].clone().set_position(2, 27),
            sprites["hor"].clone().set_position(27, 37),
            sprites["dgr"].clone().set_position(12, 47),
            sprites["dgl"].clone().set_position(42, 47),
            sprites["hor"].clone().set_position(17, 12),
            sprites["dgl"].clone().set_position(42, 22),
            # useless teleports
            # T
            sprites["ver"].clone().set_position(18, 55),
            sprites["hor"].clone().set_position(18, 53),
            # U
            sprites["ver"].clone().set_position(22, 55).color_remap(0, 5),
            sprites["ver"].clone().set_position(26, 55).color_remap(0, 5),
            sprites["hor"].clone().set_position(24, 57).color_remap(0, 5),
            # F
            sprites["eff"].clone().set_position(30, 55),
            # A
            sprites["ldn"].clone().set_position(33, 55).color_remap(0, 5),
            sprites["flp"].clone().set_position(35, 55).color_remap(0, 5),
            # Wall around the level
            sprites["wal"].clone().set_position(-3, 7),
            sprites["wal"].clone().set_position(2, 7),
            sprites["wal"].clone().set_position(7, 7),
            sprites["wal"].clone().set_position(12, 7),
            sprites["wal"].clone().set_position(17, 7),
            sprites["wal"].clone().set_position(22, 7),
            sprites["wal"].clone().set_position(27, 7),
            sprites["wal"].clone().set_position(32, 7),
            sprites["wal"].clone().set_position(37, 7),
            sprites["wal"].clone().set_position(42, 7),
            sprites["wal"].clone().set_position(47, 7),
            sprites["wal"].clone().set_position(52, 7),
            sprites["wal"].clone().set_position(-3, 12),
            sprites["wal"].clone().set_position(-3, 17),
            sprites["wal"].clone().set_position(-3, 22),
            sprites["wal"].clone().set_position(-3, 27),
            sprites["wal"].clone().set_position(-3, 32),
            sprites["wal"].clone().set_position(-3, 37),
            sprites["wal"].clone().set_position(-3, 42),
            sprites["wal"].clone().set_position(-3, 47),
            sprites["wal"].clone().set_position(-3, 52),
            sprites["wal"].clone().set_position(-3, 57),
            sprites["wal"].clone().set_position(-3, 62),
            sprites["wal"].clone().set_position(2, 62),
            sprites["wal"].clone().set_position(7, 62),
            sprites["wal"].clone().set_position(12, 62),
            sprites["wal"].clone().set_position(17, 62),
            sprites["wal"].clone().set_position(22, 62),
            sprites["wal"].clone().set_position(27, 62),
            sprites["wal"].clone().set_position(32, 62),
            sprites["wal"].clone().set_position(37, 62),
            sprites["wal"].clone().set_position(42, 62),
            sprites["wal"].clone().set_position(47, 62),
            sprites["wal"].clone().set_position(52, 62),
            sprites["wal"].clone().set_position(57, 62),
            sprites["wal"].clone().set_position(57, 7),
            sprites["wal"].clone().set_position(57, 12),
            sprites["wal"].clone().set_position(57, 17),
            sprites["wal"].clone().set_position(57, 22),
            sprites["wal"].clone().set_position(57, 27),
            sprites["wal"].clone().set_position(57, 32),
            sprites["wal"].clone().set_position(57, 37),
            sprites["wal"].clone().set_position(57, 42),
            sprites["wal"].clone().set_position(57, 47),
            sprites["wal"].clone().set_position(57, 52),
            sprites["wal"].clone().set_position(57, 57),
            # Additional walls
            sprites["wal"].clone().set_position(12, 57),
            sprites["wal"].clone().set_position(12, 52),
            sprites["wal"].clone().set_position(17, 47),
            sprites["wal"].clone().set_position(22, 47),
            sprites["wal"].clone().set_position(27, 47),
            sprites["wal"].clone().set_position(32, 47),
            sprites["wal"].clone().set_position(37, 47),
            sprites["wal"].clone().set_position(42, 52),
            sprites["wal"].clone().set_position(42, 57),
            sprites["wal"].clone().set_position(2, 12),
            sprites["wal"].clone().set_position(52, 12),
            sprites["wal"].clone().set_position(12, 22),
            sprites["wal"].clone().set_position(12, 27),
            sprites["wal"].clone().set_position(12, 32),
            sprites["wal"].clone().set_position(17, 22),
            sprites["wal"].clone().set_position(22, 22),
            sprites["wal"].clone().set_position(17, 27),
        ],
        grid_size=(64, 64),
        data={
            LEVEL_DATA_TIMER_MAX_STEPS: 12,
            LEVEL_DATA_INITIAL_TELEPORT_INDEX: -1,
        },
        name="krg",
    ),
    # bgd
    Level(
         sprites=[
            # Status
            sprites["sta"].clone().set_position(54, 0),
            sprites["kdj"].clone().set_position(57, 2),
            # Target
            sprites["tgt"].clone().set_position(2, 12),
            # Background
            sprites["bgd"].clone(),
            # player start position
            sprites["pch"].clone().set_position(47, 57),
            # Teleports
            sprites["dgl"].clone().set_position(47, 27).color_remap(0, 5),
            sprites["hor"].clone().set_position(37, 17).color_remap(0, 5),
            sprites["rup"].clone().set_position(27, 32).color_remap(0, 5),
            sprites["ldn"].clone().set_position(27, 47).color_remap(0, 5),
            sprites["dgl"].clone().set_position(17, 22).color_remap(0, 5),
            sprites["ver"].clone().set_position(17, 32).color_remap(0, 5),
            sprites["dgr"].clone().set_position(7, 42).color_remap(0, 5),
            sprites["rup"].clone().set_position(2, 52).color_remap(0, 5),
            # useless teleports
            sprites["dgl"].clone().set_position(52, 12),
            sprites["dgr"].clone().set_position(52, 12),
            # Wall around the level
            sprites["wal"].clone().set_position(-3, 7),
            sprites["wal"].clone().set_position(2, 7),
            sprites["wal"].clone().set_position(7, 7),
            sprites["wal"].clone().set_position(12, 7),
            sprites["wal"].clone().set_position(17, 7),
            sprites["wal"].clone().set_position(22, 7),
            sprites["wal"].clone().set_position(27, 7),
            sprites["wal"].clone().set_position(32, 7),
            sprites["wal"].clone().set_position(37, 7),
            sprites["wal"].clone().set_position(42, 7),
            sprites["wal"].clone().set_position(47, 7),
            sprites["wal"].clone().set_position(52, 7),
            sprites["wal"].clone().set_position(-3, 12),
            sprites["wal"].clone().set_position(-3, 17),
            sprites["wal"].clone().set_position(-3, 22),
            sprites["wal"].clone().set_position(-3, 27),
            sprites["wal"].clone().set_position(-3, 32),
            sprites["wal"].clone().set_position(-3, 37),
            sprites["wal"].clone().set_position(-3, 42),
            sprites["wal"].clone().set_position(-3, 47),
            sprites["wal"].clone().set_position(-3, 52),
            sprites["wal"].clone().set_position(-3, 57),
            sprites["wal"].clone().set_position(-3, 62),
            sprites["wal"].clone().set_position(2, 62),
            sprites["wal"].clone().set_position(7, 62),
            sprites["wal"].clone().set_position(12, 62),
            sprites["wal"].clone().set_position(17, 62),
            sprites["wal"].clone().set_position(22, 62),
            sprites["wal"].clone().set_position(27, 62),
            sprites["wal"].clone().set_position(32, 62),
            sprites["wal"].clone().set_position(37, 62),
            sprites["wal"].clone().set_position(42, 62),
            sprites["wal"].clone().set_position(47, 62),
            sprites["wal"].clone().set_position(52, 62),
            sprites["wal"].clone().set_position(57, 62),
            sprites["wal"].clone().set_position(57, 7),
            sprites["wal"].clone().set_position(57, 12),
            sprites["wal"].clone().set_position(57, 17),
            sprites["wal"].clone().set_position(57, 22),
            sprites["wal"].clone().set_position(57, 27),
            sprites["wal"].clone().set_position(57, 32),
            sprites["wal"].clone().set_position(57, 37),
            sprites["wal"].clone().set_position(57, 42),
            sprites["wal"].clone().set_position(57, 47),
            sprites["wal"].clone().set_position(57, 52),
            sprites["wal"].clone().set_position(57, 57),
            # Additional walls
            sprites["wal"].clone().set_position(52, 17),
            sprites["wal"].clone().set_position(47, 17),
            sprites["wal"].clone().set_position(47, 12),
            sprites["wal"].clone().set_position(2, 17),
            sprites["wal"].clone().set_position(2, 22),
            sprites["wal"].clone().set_position(2, 27),
            sprites["wal"].clone().set_position(2, 32),
            sprites["wal"].clone().set_position(2, 37),
            sprites["wal"].clone().set_position(2, 42),
            sprites["wal"].clone().set_position(2, 47),
            sprites["wal"].clone().set_position(2, 57),
            sprites["wal"].clone().set_position(22, 12),
            sprites["wal"].clone().set_position(27, 12),
            sprites["wal"].clone().set_position(32, 12),
            sprites["wal"].clone().set_position(37, 12),
            sprites["wal"].clone().set_position(42, 12),
            sprites["wal"].clone().set_position(42, 27),
            sprites["wal"].clone().set_position(42, 32),
            sprites["wal"].clone().set_position(42, 37),
            sprites["wal"].clone().set_position(42, 42),
            sprites["wal"].clone().set_position(42, 47),
            sprites["wal"].clone().set_position(42, 52),
            sprites["wal"].clone().set_position(42, 57),
            sprites["wal"].clone().set_position(47, 47),
            sprites["wal"].clone().set_position(47, 42),
            sprites["wal"].clone().set_position(47, 37),
            sprites["wal"].clone().set_position(22, 17),
            sprites["wal"].clone().set_position(22, 22),
            sprites["wal"].clone().set_position(22, 27),
            sprites["wal"].clone().set_position(22, 32),
            sprites["wal"].clone().set_position(22, 37),
            sprites["wal"].clone().set_position(22, 42),
            sprites["wal"].clone().set_position(22, 47),
            sprites["wal"].clone().set_position(22, 52),

        ],
        grid_size=(64, 64),
        data={
            LEVEL_DATA_TIMER_MAX_STEPS: 22,
            LEVEL_DATA_INITIAL_TELEPORT_INDEX: -1,
        },
        name="bgd",
    ),
    # puq
    Level(
        sprites=[
            # Status
            sprites["sta"].clone().set_position(54, 0),
            sprites["kdj"].clone().set_position(57, 2),
            # Target
            sprites["tgt"].clone().set_position(27, 32),
            # Background
            sprites["bgd"].clone(),
            # player start position
            sprites["pch"].clone().set_position(27, 42),
            # Teleports
            sprites["ldn"].clone().set_position(17, 47),
            sprites["rup"].clone().set_position(12, 57),
            sprites["ldn"].clone().set_position(42, 22),
            sprites["rup"].clone().set_position(37, 32),

            sprites["dgl"].clone().set_position(42, 47),
            sprites["dgr"].clone().set_position(42, 57),
            sprites["hor"].clone().set_position(32, 57),

            sprites["dgl"].clone().set_position(27, 22),

            sprites["ver"].clone().set_position(2, 37),
            sprites["ver"].clone().set_position(2, 47),

            sprites["hor"].clone().set_position(17, 12),
            sprites["hor"].clone().set_position(12, 32),

            # Wall around the level
            sprites["wal"].clone().set_position(-3, 7),
            sprites["wal"].clone().set_position(2, 7),
            sprites["wal"].clone().set_position(7, 7),
            sprites["wal"].clone().set_position(12, 7),
            sprites["wal"].clone().set_position(17, 7),
            sprites["wal"].clone().set_position(22, 7),
            sprites["wal"].clone().set_position(27, 7),
            sprites["wal"].clone().set_position(32, 7),
            sprites["wal"].clone().set_position(37, 7),
            sprites["wal"].clone().set_position(42, 7),
            sprites["wal"].clone().set_position(47, 7),
            sprites["wal"].clone().set_position(52, 7),
            sprites["wal"].clone().set_position(-3, 12),
            sprites["wal"].clone().set_position(-3, 17),
            sprites["wal"].clone().set_position(-3, 22),
            sprites["wal"].clone().set_position(-3, 27),
            sprites["wal"].clone().set_position(-3, 32),
            sprites["wal"].clone().set_position(-3, 37),
            sprites["wal"].clone().set_position(-3, 42),
            sprites["wal"].clone().set_position(-3, 47),
            sprites["wal"].clone().set_position(-3, 52),
            sprites["wal"].clone().set_position(-3, 57),
            sprites["wal"].clone().set_position(-3, 62),
            sprites["wal"].clone().set_position(2, 62),
            sprites["wal"].clone().set_position(7, 62),
            sprites["wal"].clone().set_position(12, 62),
            sprites["wal"].clone().set_position(17, 62),
            sprites["wal"].clone().set_position(22, 62),
            sprites["wal"].clone().set_position(27, 62),
            sprites["wal"].clone().set_position(32, 62),
            sprites["wal"].clone().set_position(37, 62),
            sprites["wal"].clone().set_position(42, 62),
            sprites["wal"].clone().set_position(47, 62),
            sprites["wal"].clone().set_position(52, 62),
            sprites["wal"].clone().set_position(57, 62),
            sprites["wal"].clone().set_position(57, 7),
            sprites["wal"].clone().set_position(57, 12),
            sprites["wal"].clone().set_position(57, 17),
            sprites["wal"].clone().set_position(57, 22),
            sprites["wal"].clone().set_position(57, 27),
            sprites["wal"].clone().set_position(57, 32),
            sprites["wal"].clone().set_position(57, 37),
            sprites["wal"].clone().set_position(57, 42),
            sprites["wal"].clone().set_position(57, 47),
            sprites["wal"].clone().set_position(57, 52),
            sprites["wal"].clone().set_position(57, 57),
            # Additional walls
            sprites["wal"].clone().set_position(17, 42),
            sprites["wal"].clone().set_position(12, 42),
            sprites["wal"].clone().set_position(12, 52),

            sprites["wal"].clone().set_position(17, 27),
            sprites["wal"].clone().set_position(12, 27),
            sprites["wal"].clone().set_position(17, 17),
            sprites["wal"].clone().set_position(12, 17),
            sprites["wal"].clone().set_position(22, 17),

            sprites["wal"].clone().set_position(32, 17),
            sprites["wal"].clone().set_position(37, 27),
            sprites["wal"].clone().set_position(37, 17),
            sprites["wal"].clone().set_position(42, 17),

            sprites["wal"].clone().set_position(37, 42),
            sprites["wal"].clone().set_position(42, 42),
            sprites["wal"].clone().set_position(37, 52),
            sprites["wal"].clone().set_position(42, 52),

            sprites["wal"].clone().set_position(7, 37),
            sprites["wal"].clone().set_position(12, 37),
            sprites["wal"].clone().set_position(17, 37),
            sprites["wal"].clone().set_position(22, 37),
            sprites["wal"].clone().set_position(27, 37),
            sprites["wal"].clone().set_position(32, 37),
            sprites["wal"].clone().set_position(37, 37),
            sprites["wal"].clone().set_position(42, 37),
            sprites["wal"].clone().set_position(47, 37),
        ],
        grid_size=(64, 64),
        data={
            LEVEL_DATA_TIMER_MAX_STEPS: 25,
            LEVEL_DATA_INITIAL_TELEPORT_INDEX: -1,
        },
        name="puq",
    ),
    # tmx
    Level(
        sprites=[
            # Status
            sprites["sta"].clone().set_position(54, 0),
            sprites["kdj"].clone().set_position(57, 2),
            # Target
            sprites["tgt"].clone().set_position(12, 42),
            # Background
            sprites["bgd"].clone(),
            # player start position
            sprites["pch"].clone().set_position(2, 12),
            # Teleports
            sprites["dgl"].clone().set_position(2, 22).color_remap(0, 5),
            sprites["dgr"].clone().set_position(2, 42).color_remap(0, 5),
            sprites["ldn"].clone().set_position(17, 22).color_remap(0, 5),
            sprites["dgl"].clone().set_position(37, 27).color_remap(0, 5),
            sprites["ver"].clone().set_position(22, 37).color_remap(0, 5),
            sprites["ver"].clone().set_position(32, 37).color_remap(0, 5),
            sprites["ver"].clone().set_position(47, 47).color_remap(0, 5),
            sprites["dgl"].clone().set_position(42, 57).color_remap(0, 5),
            sprites["ldn"].clone().set_position(17, 47).color_remap(0, 5),
            sprites["rup"].clone().set_position(2, 57).color_remap(0, 5),
            # Wall around the level
            sprites["wal"].clone().set_position(-3, 7),
            sprites["wal"].clone().set_position(2, 7),
            sprites["wal"].clone().set_position(7, 7),
            sprites["wal"].clone().set_position(12, 7),
            sprites["wal"].clone().set_position(17, 7),
            sprites["wal"].clone().set_position(22, 7),
            sprites["wal"].clone().set_position(27, 7),
            sprites["wal"].clone().set_position(32, 7),
            sprites["wal"].clone().set_position(37, 7),
            sprites["wal"].clone().set_position(42, 7),
            sprites["wal"].clone().set_position(47, 7),
            sprites["wal"].clone().set_position(52, 7),
            sprites["wal"].clone().set_position(-3, 12),
            sprites["wal"].clone().set_position(-3, 17),
            sprites["wal"].clone().set_position(-3, 22),
            sprites["wal"].clone().set_position(-3, 27),
            sprites["wal"].clone().set_position(-3, 32),
            sprites["wal"].clone().set_position(-3, 37),
            sprites["wal"].clone().set_position(-3, 42),
            sprites["wal"].clone().set_position(-3, 47),
            sprites["wal"].clone().set_position(-3, 52),
            sprites["wal"].clone().set_position(-3, 57),
            sprites["wal"].clone().set_position(-3, 62),
            sprites["wal"].clone().set_position(2, 62),
            sprites["wal"].clone().set_position(7, 62),
            sprites["wal"].clone().set_position(12, 62),
            sprites["wal"].clone().set_position(17, 62),
            sprites["wal"].clone().set_position(22, 62),
            sprites["wal"].clone().set_position(27, 62),
            sprites["wal"].clone().set_position(32, 62),
            sprites["wal"].clone().set_position(37, 62),
            sprites["wal"].clone().set_position(42, 62),
            sprites["wal"].clone().set_position(47, 62),
            sprites["wal"].clone().set_position(52, 62),
            sprites["wal"].clone().set_position(57, 62),
            sprites["wal"].clone().set_position(57, 7),
            sprites["wal"].clone().set_position(57, 12),
            sprites["wal"].clone().set_position(57, 17),
            sprites["wal"].clone().set_position(57, 22),
            sprites["wal"].clone().set_position(57, 27),
            sprites["wal"].clone().set_position(57, 32),
            sprites["wal"].clone().set_position(57, 37),
            sprites["wal"].clone().set_position(57, 42),
            sprites["wal"].clone().set_position(57, 47),
            sprites["wal"].clone().set_position(57, 52),
            sprites["wal"].clone().set_position(57, 57),
            # Additional walls
            sprites["wal"].clone().set_position(7, 17),
            sprites["wal"].clone().set_position(12, 17),
            sprites["wal"].clone().set_position(7, 22),
            sprites["wal"].clone().set_position(12, 22),
            sprites["wal"].clone().set_position(7, 27),
            sprites["wal"].clone().set_position(12, 27),
            sprites["wal"].clone().set_position(7, 32),
            sprites["wal"].clone().set_position(7, 37),
            sprites["wal"].clone().set_position(12, 37),
            sprites["wal"].clone().set_position(7, 42),
            sprites["wal"].clone().set_position(17, 42),
            sprites["wal"].clone().set_position(22, 42),

            sprites["wal"].clone().set_position(2, 52),
            sprites["wal"].clone().set_position(7, 52),
            sprites["wal"].clone().set_position(12, 52),
            sprites["wal"].clone().set_position(17, 52),
            sprites["wal"].clone().set_position(22, 52),
            sprites["wal"].clone().set_position(27, 52),
            sprites["wal"].clone().set_position(32, 52),
            sprites["wal"].clone().set_position(37, 52),
            sprites["wal"].clone().set_position(42, 52),
            sprites["wal"].clone().set_position(47, 52),

            sprites["wal"].clone().set_position(52, 57),
            sprites["wal"].clone().set_position(52, 52),
            sprites["wal"].clone().set_position(52, 47),
            sprites["wal"].clone().set_position(52, 42),
            sprites["wal"].clone().set_position(52, 37),
            sprites["wal"].clone().set_position(52, 32),
            sprites["wal"].clone().set_position(52, 27),
            sprites["wal"].clone().set_position(52, 22),
            sprites["wal"].clone().set_position(52, 17),
            sprites["wal"].clone().set_position(52, 12),

            sprites["wal"].clone().set_position(27, 12),
            sprites["wal"].clone().set_position(22, 12),
            sprites["wal"].clone().set_position(17, 27),
            sprites["wal"].clone().set_position(22, 27),
            sprites["wal"].clone().set_position(27, 27),
            sprites["wal"].clone().set_position(32, 27),
            sprites["wal"].clone().set_position(17, 32),
            sprites["wal"].clone().set_position(17, 37),

            sprites["wal"].clone().set_position(37, 22),
            sprites["wal"].clone().set_position(42, 22),
            sprites["wal"].clone().set_position(37, 17),
            sprites["wal"].clone().set_position(42, 17),
            sprites["wal"].clone().set_position(37, 47),
            sprites["wal"].clone().set_position(42, 47),
            sprites["wal"].clone().set_position(37, 42),
            sprites["wal"].clone().set_position(42, 42),
            sprites["wal"].clone().set_position(37, 37),
            sprites["wal"].clone().set_position(42, 37),

            sprites["wal"].clone().set_position(32, 42),
            sprites["wal"].clone().set_position(27, 42),
            sprites["wal"].clone().set_position(27, 47),
            sprites["wal"].clone().set_position(27, 57),
        ],
        grid_size=(64, 64),
        data={
            LEVEL_DATA_TIMER_MAX_STEPS: 30,
            LEVEL_DATA_INITIAL_TELEPORT_INDEX: -1,
        },
        name="tmx",
    ),
    # lyd
    Level(
        sprites=[
            # Status
            sprites["sta"].clone().set_position(54, 0),
            sprites["kdj"].clone().set_position(57, 2),
            # Target
            sprites["tgt"].clone().set_position(27, 52),
            # Background
            sprites["bgd"].clone(),
            # player start position
            sprites["pch"].clone().set_position(52, 52),
            # Teleports
            sprites["dgl"].clone().set_position(52, 22),
            sprites["dgr"].clone().set_position(42, 32),
            sprites["ver"].clone().set_position(32, 22),
            sprites["dgl"].clone().set_position(42, 47),

            sprites["dgl"].clone().set_position(7, 47),
            sprites["hor"].clone().set_position(17, 57),
            sprites["ver"].clone().set_position(17, 37),
            sprites["ver"].clone().set_position(7, 22),

            # Wall around the level
            sprites["wal"].clone().set_position(-3, 7),
            sprites["wal"].clone().set_position(2, 7),
            sprites["wal"].clone().set_position(7, 7),
            sprites["wal"].clone().set_position(12, 7),
            sprites["wal"].clone().set_position(17, 7),
            sprites["wal"].clone().set_position(22, 7),
            sprites["wal"].clone().set_position(27, 7),
            sprites["wal"].clone().set_position(32, 7),
            sprites["wal"].clone().set_position(37, 7),
            sprites["wal"].clone().set_position(42, 7),
            sprites["wal"].clone().set_position(47, 7),
            sprites["wal"].clone().set_position(52, 7),
            sprites["wal"].clone().set_position(-3, 12),
            sprites["wal"].clone().set_position(-3, 17),
            sprites["wal"].clone().set_position(-3, 22),
            sprites["wal"].clone().set_position(-3, 27),
            sprites["wal"].clone().set_position(-3, 32),
            sprites["wal"].clone().set_position(-3, 37),
            sprites["wal"].clone().set_position(-3, 42),
            sprites["wal"].clone().set_position(-3, 47),
            sprites["wal"].clone().set_position(-3, 52),
            sprites["wal"].clone().set_position(-3, 57),
            sprites["wal"].clone().set_position(-3, 62),
            sprites["wal"].clone().set_position(2, 62),
            sprites["wal"].clone().set_position(7, 62),
            sprites["wal"].clone().set_position(12, 62),
            sprites["wal"].clone().set_position(17, 62),
            sprites["wal"].clone().set_position(22, 62),
            sprites["wal"].clone().set_position(27, 62),
            sprites["wal"].clone().set_position(32, 62),
            sprites["wal"].clone().set_position(37, 62),
            sprites["wal"].clone().set_position(42, 62),
            sprites["wal"].clone().set_position(47, 62),
            sprites["wal"].clone().set_position(52, 62),
            sprites["wal"].clone().set_position(57, 62),
            sprites["wal"].clone().set_position(57, 7),
            sprites["wal"].clone().set_position(57, 12),
            sprites["wal"].clone().set_position(57, 17),
            sprites["wal"].clone().set_position(57, 22),
            sprites["wal"].clone().set_position(57, 27),
            sprites["wal"].clone().set_position(57, 32),
            sprites["wal"].clone().set_position(57, 37),
            sprites["wal"].clone().set_position(57, 42),
            sprites["wal"].clone().set_position(57, 47),
            sprites["wal"].clone().set_position(57, 52),
            sprites["wal"].clone().set_position(57, 57),
            # Additional walls
            sprites["wal"].clone().set_position(2, 12),
            sprites["wal"].clone().set_position(2, 17),
            sprites["wal"].clone().set_position(2, 22),
            sprites["wal"].clone().set_position(2, 27),
            sprites["wal"].clone().set_position(2, 32),
            sprites["wal"].clone().set_position(2, 37),
            sprites["wal"].clone().set_position(2, 42),
            sprites["wal"].clone().set_position(2, 47),
            sprites["wal"].clone().set_position(2, 52),
            sprites["wal"].clone().set_position(2, 57),

            sprites["wal"].clone().set_position(12, 12),
            sprites["wal"].clone().set_position(12, 17),
            sprites["wal"].clone().set_position(12, 22),
            sprites["wal"].clone().set_position(12, 27),
            sprites["wal"].clone().set_position(12, 32),
            sprites["wal"].clone().set_position(12, 37),
            sprites["wal"].clone().set_position(12, 42),
            sprites["wal"].clone().set_position(12, 47),
            sprites["wal"].clone().set_position(12, 52),
            sprites["wal"].clone().set_position(12, 57),

            sprites["wal"].clone().set_position(22, 12),
            sprites["wal"].clone().set_position(22, 17),
            sprites["wal"].clone().set_position(22, 22),
            sprites["wal"].clone().set_position(22, 27),
            sprites["wal"].clone().set_position(22, 32),
            sprites["wal"].clone().set_position(22, 37),
            sprites["wal"].clone().set_position(22, 42),
            sprites["wal"].clone().set_position(22, 47),
            sprites["wal"].clone().set_position(22, 52),
            sprites["wal"].clone().set_position(22, 57),

            sprites["wal"].clone().set_position(47, 12),
            sprites["wal"].clone().set_position(47, 17),
            sprites["wal"].clone().set_position(47, 22),
            sprites["wal"].clone().set_position(47, 27),
            sprites["wal"].clone().set_position(47, 32),
            sprites["wal"].clone().set_position(47, 37),
            sprites["wal"].clone().set_position(47, 42),
            sprites["wal"].clone().set_position(47, 47),
            sprites["wal"].clone().set_position(47, 52),
            sprites["wal"].clone().set_position(47, 57),

            sprites["wal"].clone().set_position(37, 12),
            sprites["wal"].clone().set_position(37, 17),
            sprites["wal"].clone().set_position(37, 22),
            sprites["wal"].clone().set_position(37, 27),
            sprites["wal"].clone().set_position(37, 32),
            sprites["wal"].clone().set_position(37, 37),
            sprites["wal"].clone().set_position(37, 42),
            sprites["wal"].clone().set_position(37, 47),
            sprites["wal"].clone().set_position(37, 52),
            sprites["wal"].clone().set_position(37, 57),

            sprites["wal"].clone().set_position(42, 17),
            sprites["wal"].clone().set_position(32, 17),
            sprites["wal"].clone().set_position(27, 12),
            
            sprites["wal"].clone().set_position(7, 27),
            sprites["wal"].clone().set_position(7, 37),
            sprites["wal"].clone().set_position(27, 37),
            sprites["wal"].clone().set_position(32, 32),
            sprites["wal"].clone().set_position(17, 42),
            sprites["wal"].clone().set_position(27, 47),
            sprites["wal"].clone().set_position(32, 52),
            sprites["wal"].clone().set_position(42, 57),
        ],
        grid_size=(64, 64),
        data={
            LEVEL_DATA_TIMER_MAX_STEPS: 20,
            LEVEL_DATA_INITIAL_TELEPORT_INDEX: 0,
        },
        name="lyd",
    ),
    # zba
    Level(
        sprites=[
            # Status
            sprites["sta"].clone().set_position(54, 0),
            sprites["kdj"].clone().set_position(57, 2),
            # Target
            sprites["tgt"].clone().set_position(2, 37),
            sprites["itg"].clone().set_position(57, 2).color_remap(1, 3),
            # Background
            sprites["bgd"].clone(),
            # player start position
            sprites["pch"].clone().set_position(27, 57),
            # Teleports
            sprites["ldn"].clone().set_position(42, 37).color_remap(0, 5),
            sprites["rup"].clone().set_position(37, 47).color_remap(0, 5),
            sprites["ldn"].clone().set_position(17, 37).color_remap(0, 5),
            sprites["rup"].clone().set_position(12, 47).color_remap(0, 5),
            sprites["ldn"].clone().set_position(37, 17).color_remap(0, 5),
            sprites["dgl"].clone().set_position(32, 27).color_remap(0, 5),
            sprites["dgl"].clone().set_position(47, 27).color_remap(0, 5),
            sprites["ver"].clone().set_position(12, 17).color_remap(0, 5),
            sprites["ldn"].clone().set_position(12, 27).color_remap(0, 5),
            sprites["hor"].clone().set_position(17, 57).color_remap(0, 5),
            # Wall around the level
            sprites["wal"].clone().set_position(-3, 7),
            sprites["wal"].clone().set_position(2, 7),
            sprites["wal"].clone().set_position(7, 7),
            sprites["wal"].clone().set_position(12, 7),
            sprites["wal"].clone().set_position(17, 7),
            sprites["wal"].clone().set_position(22, 7),
            sprites["wal"].clone().set_position(27, 7),
            sprites["wal"].clone().set_position(32, 7),
            sprites["wal"].clone().set_position(37, 7),
            sprites["wal"].clone().set_position(42, 7),
            sprites["wal"].clone().set_position(47, 7),
            sprites["wal"].clone().set_position(52, 7),
            sprites["wal"].clone().set_position(-3, 12),
            sprites["wal"].clone().set_position(-3, 17),
            sprites["wal"].clone().set_position(-3, 22),
            sprites["wal"].clone().set_position(-3, 27),
            sprites["wal"].clone().set_position(-3, 32),
            sprites["wal"].clone().set_position(-3, 37),
            sprites["wal"].clone().set_position(-3, 42),
            sprites["wal"].clone().set_position(-3, 47),
            sprites["wal"].clone().set_position(-3, 52),
            sprites["wal"].clone().set_position(-3, 57),
            sprites["wal"].clone().set_position(-3, 62),
            sprites["wal"].clone().set_position(2, 62),
            sprites["wal"].clone().set_position(7, 62),
            sprites["wal"].clone().set_position(12, 62),
            sprites["wal"].clone().set_position(17, 62),
            sprites["wal"].clone().set_position(22, 62),
            sprites["wal"].clone().set_position(27, 62),
            sprites["wal"].clone().set_position(32, 62),
            sprites["wal"].clone().set_position(37, 62),
            sprites["wal"].clone().set_position(42, 62),
            sprites["wal"].clone().set_position(47, 62),
            sprites["wal"].clone().set_position(52, 62),
            sprites["wal"].clone().set_position(57, 62),
            sprites["wal"].clone().set_position(57, 7),
            sprites["wal"].clone().set_position(57, 12),
            sprites["wal"].clone().set_position(57, 17),
            sprites["wal"].clone().set_position(57, 22),
            sprites["wal"].clone().set_position(57, 27),
            sprites["wal"].clone().set_position(57, 32),
            sprites["wal"].clone().set_position(57, 37),
            sprites["wal"].clone().set_position(57, 42),
            sprites["wal"].clone().set_position(57, 47),
            sprites["wal"].clone().set_position(57, 52),
            sprites["wal"].clone().set_position(57, 57),
            # Additional walls
            sprites["wal"].clone().set_position(2, 22),
            sprites["wal"].clone().set_position(7, 22),
            sprites["wal"].clone().set_position(12, 22),
            sprites["wal"].clone().set_position(17, 22),
            sprites["wal"].clone().set_position(22, 22),
            sprites["wal"].clone().set_position(27, 22),
            sprites["wal"].clone().set_position(32, 22),
            sprites["wal"].clone().set_position(37, 22),
            sprites["wal"].clone().set_position(42, 22),
            sprites["wal"].clone().set_position(47, 22),
            sprites["wal"].clone().set_position(52, 22),

            sprites["wal"].clone().set_position(2, 32),
            sprites["wal"].clone().set_position(7, 32),
            sprites["wal"].clone().set_position(12, 32),
            sprites["wal"].clone().set_position(17, 32),
            sprites["wal"].clone().set_position(22, 32),
            sprites["wal"].clone().set_position(27, 32),
            sprites["wal"].clone().set_position(32, 32),
            sprites["wal"].clone().set_position(37, 32),
            sprites["wal"].clone().set_position(42, 32),
            sprites["wal"].clone().set_position(47, 32),
            sprites["wal"].clone().set_position(52, 32),

            sprites["wal"].clone().set_position(2, 52),
            sprites["wal"].clone().set_position(7, 52),
            sprites["wal"].clone().set_position(12, 52),
            sprites["wal"].clone().set_position(17, 52),
            sprites["wal"].clone().set_position(22, 52),
            sprites["wal"].clone().set_position(27, 52),
            sprites["wal"].clone().set_position(32, 52),
            sprites["wal"].clone().set_position(37, 52),
            sprites["wal"].clone().set_position(42, 52),
            sprites["wal"].clone().set_position(47, 52),
            sprites["wal"].clone().set_position(52, 52),

            sprites["wal"].clone().set_position(2, 47),
            sprites["wal"].clone().set_position(7, 47),
            sprites["wal"].clone().set_position(17, 42),
            sprites["wal"].clone().set_position(37, 42),

            sprites["wal"].clone().set_position(17, 27),
            sprites["wal"].clone().set_position(37, 27),
        ],
        grid_size=(64, 64),
        data={
            LEVEL_DATA_TIMER_MAX_STEPS: 28,
            LEVEL_DATA_INITIAL_TELEPORT_INDEX: 2,
        },
        name="zba",
    ),
]
# fmt: on


BACKGROUND_COLOR = 3
PADDING_COLOR = 3
TIMER_BAR_COLUMN = 61
TIMER_BAR_TOP = 12
TIMER_BAR_HEIGHT = 40


class Js26Hud(RenderableUserDisplay):
    debug_points: list[tuple[int, int]]

    def __init__(self, game: "Js26", max_timer_steps: int):
        self.game = game
        self.max_timer_steps = max_timer_steps
        self.timer_steps_remaining = max_timer_steps

    def set_timer_steps_remaining(self, timer_steps_remaining: int) -> None:
        self.timer_steps_remaining = max(0, min(timer_steps_remaining, self.max_timer_steps))

    def consume_timer_step(self) -> bool:
        if self.timer_steps_remaining >= 0:
            self.timer_steps_remaining -= 1
        return self.timer_steps_remaining >= 0

    def reset_timer(self) -> None:
        self.timer_steps_remaining = self.max_timer_steps

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_timer_steps == 0 or self.game.pending_respawn_flash:
            return frame

        fill_ratio = max(0.0, min(1.0, float(self.timer_steps_remaining) / float(self.max_timer_steps)))
        filled_pixels = round(fill_ratio * TIMER_BAR_HEIGHT)
        filled_start_row = TIMER_BAR_HEIGHT - filled_pixels
        for timer_pixel_index in range(TIMER_BAR_HEIGHT):
            timer_row = TIMER_BAR_TOP + timer_pixel_index
            frame[timer_row, TIMER_BAR_COLUMN : TIMER_BAR_COLUMN + 2] = (
                11 if timer_pixel_index >= filled_start_row else 3
            )

        for life_index in range(3):
            life_x = 61
            life_y = 54 + 3 * life_index
            for life_row_offset in range(2):
                frame[life_y + life_row_offset, life_x : life_x + 2] = (
                    8 if self.game.lives_remaining > life_index else 3
                )
        return frame


class Js26(ARCBaseGame):
    def __init__(self) -> None:
        initial_timer_steps = levels[0].get_data(LEVEL_DATA_TIMER_MAX_STEPS) if levels else 0
        max_timer_steps = initial_timer_steps if initial_timer_steps else 0
        self.hud = Js26Hud(self, max_timer_steps)
        self.teleport_templates = [
            [sprites["horl"], (-2, 0)],
            [sprites["horr"], (2, 0)],
            [sprites["veru"], (0, -2)],
            [sprites["verd"], (0, 2)],
            [sprites["kdj"], None],
        ]

        super().__init__(
            "js26", levels, Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR, [self.hud]), False, 1, [1, 2, 3, 4, 5]
        )

    def _sync_timer_from_level_data(self) -> None:
        level_timer_steps = self.current_level.get_data(LEVEL_DATA_TIMER_MAX_STEPS)
        self.hud.max_timer_steps = level_timer_steps if level_timer_steps is not None else 0
        self.hud.reset_timer()

    def on_set_level(self, level: Level) -> None:
        self.level = level
        self.player_sprite = self.current_level.get_sprites_by_tag(TAG_PLAYER)[0]
        self.selected_teleport_display_sprite = self.current_level.get_sprites_by_tag(TAG_SELECTED_TELEPORT)[0]
        self.status_panel_sprite = self.current_level.get_sprites_by_tag(TAG_STATUS_PANEL)[0]
        self.target_slot_sprites = self.current_level.get_sprites_by_tag(TAG_TARGET_SLOT)
        self.target_solved_flags = [False] * len(self.target_slot_sprites)
        self.cell_width, self.cell_height = self._infer_cell_size()

        self._sync_timer_from_level_data()

        self._reset_selected_teleport_from_level_data()
        self.respawn_flash_sprite = sprites["krg"].clone()
        self.current_level.add_sprite(self.respawn_flash_sprite)
        self.respawn_flash_sprite.set_visible(False)
        self.lives_remaining = 3
        self._sync_player_sprite_from_lives()
        self.removed_target_slot_sprites: list[Sprite] = []
        self.pending_respawn_flash = False
        self.pending_error_flash_reset = False
        self.teleport_recursion_overflow = False
        self.player_spawn_x = self.player_sprite.x
        self.player_spawn_y = self.player_sprite.y

    def _sync_player_sprite_from_lives(self) -> None:
        if self.lives_remaining >= 3:
            sprite_name = "pch"
        elif self.lives_remaining == 2:
            sprite_name = "pcm"
        else:
            sprite_name = "pcd"
        self.player_sprite.pixels = sprites[sprite_name].pixels.copy()

    def _infer_cell_size(self) -> tuple[int, int]:
        if self.target_slot_sprites:
            slot_pixels = self.target_slot_sprites[0].render()
            return slot_pixels.shape[1], slot_pixels.shape[0]
        player_pixels = self.player_sprite.render()
        return player_pixels.shape[1], player_pixels.shape[0]

    def _get_overlapping_sprites(self, left: int, top: int, width: int, height: int) -> list[Sprite]:
        all_sprites = self.current_level._sprites
        right = left + width
        bottom = top + height
        overlapping_sprites: list[Sprite] = []
        for sprite in all_sprites:
            if sprite == self.player_sprite:
                continue
            sprite_pixels = sprite.render()
            sprite_right = sprite.x + sprite_pixels.shape[1]
            sprite_bottom = sprite.y + sprite_pixels.shape[0]
            if sprite.x < right and sprite_right > left and sprite.y < bottom and sprite_bottom > top:
                overlapping_sprites.append(sprite)
        return overlapping_sprites

    def _resolve_teleport(
        self,
        next_x: int,
        next_y: int,
        move_x: int,
        move_y: int,
        collided_sprites: list[Sprite],
        recursion_depth: int = 0,
    ) -> tuple[int, int]:
        if recursion_depth >= 4:
            self.teleport_recursion_overflow = True
            return next_x, next_y

        # Teleport routing tables are keyed by unit-step direction.
        lookup_move_x = 0 if move_x == 0 else (1 if move_x > 0 else -1)
        lookup_move_y = 0 if move_y == 0 else (1 if move_y > 0 else -1)

        teleport_sprite = None
        for collided_sprite in collided_sprites:
            if collided_sprite.tags is not None and TAG_TELEPORT in collided_sprite.tags:
                teleport_sprite = collided_sprite
                break
        if teleport_sprite is None:
            return next_x, next_y

        teleport_delta = teleport_matrix.get(teleport_sprite.name, {}).get((lookup_move_x, lookup_move_y))
        if teleport_delta is None:
            return next_x, next_y

        destination_x = next_x + teleport_delta[0] * self.cell_width
        destination_y = next_y + teleport_delta[1] * self.cell_height
        destination_collisions = self._get_overlapping_sprites(
            destination_x, destination_y, self.cell_width, self.cell_height
        )
        for destination_sprite in destination_collisions:
            if (
                destination_sprite.tags is not None
                and TAG_WALL in destination_sprite.tags
                and TAG_TELEPORT not in destination_sprite.tags
            ):
                return next_x, next_y

        destination_has_teleport = False
        for destination_sprite in destination_collisions:
            if destination_sprite.tags is not None and TAG_TELEPORT in destination_sprite.tags:
                destination_has_teleport = True
                break
        if destination_has_teleport:
            recurse_move_x = 0 if teleport_delta[0] == 0 else (1 if teleport_delta[0] > 0 else -1)
            recurse_move_y = 0 if teleport_delta[1] == 0 else (1 if teleport_delta[1] > 0 else -1)
            return self._resolve_teleport(
                destination_x,
                destination_y,
                recurse_move_x,
                recurse_move_y,
                destination_collisions,
                recursion_depth + 1,
            )

        return destination_x, destination_y

    def _timeout(self) -> None:
        self.lives_remaining -= 1
        if self.lives_remaining == 0:
            self.lose()
            self.complete_action()
            return
        self._sync_player_sprite_from_lives()

        self.respawn_flash_sprite.set_visible(True)
        self.respawn_flash_sprite.set_scale(64)
        self.respawn_flash_sprite.set_position(0, 0)
        self.selected_teleport_display_sprite.set_visible(False)

        self.pending_respawn_flash = True
        self.target_solved_flags = [False] * len(self.target_slot_sprites)
        self.player_sprite.set_position(self.player_spawn_x, self.player_spawn_y)
        self._reset_selected_teleport_from_level_data()
        for target_slot_sprite in self.removed_target_slot_sprites:
            self.current_level.add_sprite(target_slot_sprite)
        self.removed_target_slot_sprites = []
        self.hud.set_timer_steps_remaining(self.hud.max_timer_steps)

    def step(self) -> None:
        if self.pending_respawn_flash:
            self.respawn_flash_sprite.set_visible(False)
            self.selected_teleport_display_sprite.set_visible(True)
            self.pending_respawn_flash = False
            self.complete_action()
            return

        if self.pending_error_flash_reset:
            self.status_panel_sprite.color_remap(None, 5)
            self.pending_error_flash_reset = False
            self.complete_action()
            return

        self.teleport_recursion_overflow = False

        move_x = 0
        move_y = 0
        has_movement_action = False
        if self.action.id.value == 1:
            move_y = -1
            has_movement_action = True
        elif self.action.id.value == 2:
            move_y = 1
            has_movement_action = True
        elif self.action.id.value == 3:
            move_x = -1
            has_movement_action = True
        elif self.action.id.value == 4:
            move_x = 1
            has_movement_action = True
        elif self.action.id.value == 5:
            move_xy = self.selected_teleport_action
            if move_xy is None:
                has_movement_action = False
            else:
                move_x, move_y = move_xy
                has_movement_action = True

        if not has_movement_action:
            self.complete_action()
            return

        if self.hud.timer_steps_remaining <= 0:
            self._timeout()
            return

        next_x = self.player_sprite.x + move_x * self.cell_width
        next_y = self.player_sprite.y + move_y * self.cell_height
        collided_sprites = self._get_overlapping_sprites(next_x, next_y, self.cell_width, self.cell_height)

        blocked_by_wall = False
        for collided_sprite in collided_sprites:
            if collided_sprite.tags is None:
                break
            if TAG_WALL in collided_sprite.tags:
                blocked_by_wall = True
                break

        if not blocked_by_wall:
            resolved_x, resolved_y = self._resolve_teleport(next_x, next_y, move_x, move_y, collided_sprites)
            self.player_sprite.set_position(resolved_x, resolved_y)

        if self.teleport_recursion_overflow:
            self._timeout()
            return

        if self._update_target_completion_and_check_win():
            self.next_level()
            self.complete_action()
            return

        if not self.hud.consume_timer_step():
            self._timeout()
            return
        self.complete_action()

    def _reset_selected_teleport_from_level_data(self) -> None:
        self.selected_teleport_index = self.current_level.get_data(LEVEL_DATA_INITIAL_TELEPORT_INDEX)
        self.selected_teleport_template_sprite = self.teleport_templates[self.selected_teleport_index][0]
        self.selected_teleport_action = self.teleport_templates[self.selected_teleport_index][1]
        self.selected_teleport_display_sprite.pixels = self.selected_teleport_template_sprite.pixels.copy()

    def _update_target_completion_and_check_win(self) -> bool:
        for target_index, target_slot_sprite in enumerate(self.target_slot_sprites):
            if (
                not self.target_solved_flags[target_index]
                and self.player_sprite.x == target_slot_sprite.x
                and self.player_sprite.y == target_slot_sprite.y
            ):
                self.target_solved_flags[target_index] = True
                self.removed_target_slot_sprites.append(self.target_slot_sprites[target_index])

                self.current_level.remove_sprite(self.target_slot_sprites[target_index])

        for target_solved in self.target_solved_flags:
            if not target_solved:
                return False
        return True
