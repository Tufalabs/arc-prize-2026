from __future__ import annotations

from contextlib import suppress

import numpy as np
from tests._worker_game import discover_generated_game

from re_arc import EnvSampler

MIN_COLOR = 0
MAX_COLOR = 15
STEPS_PER_GAME = 20


def _iter_observation_layers(observation):
    frame = getattr(observation, "frame", None)
    assert frame is not None, "Observation has no frame payload."
    if isinstance(frame, list):
        for idx, layer in enumerate(frame):
            if layer is None:
                continue
            arr = np.asarray(layer)
            assert arr.size > 0, f"Observation frame layer {idx} is empty."
            yield idx, arr
        return
    arr = np.asarray(frame)
    assert arr.size > 0, "Observation frame is empty."
    yield 0, arr


def test_generated_game_frame_colors_stay_in_valid_range():
    generated_game = discover_generated_game()
    game_id = str(generated_game["game_id"])
    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id=game_id, seed=0)

    try:
        observation = env.reset()
        for step_idx in range(STEPS_PER_GAME + 1):
            for layer_idx, colors in _iter_observation_layers(observation):
                min_color = int(colors.min())
                max_color = int(colors.max())
                assert np.issubdtype(colors.dtype, np.integer), (
                    f"{game_id}: step={step_idx} layer={layer_idx} dtype must be integer, got {colors.dtype!r}."
                )
                assert MIN_COLOR <= min_color <= MAX_COLOR, (
                    f"{game_id}: step={step_idx} layer={layer_idx} min color out of bounds. Observed min={min_color}."
                )
                assert MIN_COLOR <= max_color <= MAX_COLOR, (
                    f"{game_id}: step={step_idx} layer={layer_idx} max color out of bounds. Observed max={max_color}."
                )
            if step_idx == STEPS_PER_GAME:
                break
            action = env.action_space[0]
            step_result = env.step(action)
            observation = step_result[0] if isinstance(step_result, tuple) else step_result
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            with suppress(Exception):
                close_fn()
