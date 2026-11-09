from __future__ import annotations

import numpy as np
import pytest

from re_arc import EnvSampler, list_game_ids

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


@pytest.mark.parametrize("game_id", list(list_game_ids()))
def test_environment_frame_colors_stay_in_valid_range(game_id: str):
    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id=game_id, seed=0)

    observation = env.reset()
    for step_idx in range(STEPS_PER_GAME + 1):
        for layer_idx, colors in _iter_observation_layers(observation):
            min_color = int(colors.min())
            max_color = int(colors.max())
            assert np.issubdtype(colors.dtype, np.integer), (
                f"{game_id}: step={step_idx} layer={layer_idx} dtype must be integer, got {colors.dtype!r}."
            )
            assert MIN_COLOR <= min_color <= MAX_COLOR, (
                f"{game_id}: step={step_idx} layer={layer_idx} min color out of bounds "
                f"[{MIN_COLOR}, {MAX_COLOR}]. Observed min={min_color}."
            )
            assert MIN_COLOR <= max_color <= MAX_COLOR, (
                f"{game_id}: step={step_idx} layer={layer_idx} max color out of bounds "
                f"[{MIN_COLOR}, {MAX_COLOR}]. Observed max={max_color}."
            )
        if step_idx == STEPS_PER_GAME:
            break
        action = env.action_space[0]
        data = {"x": 0, "y": 0} if int(getattr(action, "value", -1)) == 6 else {}
        step_result = env.step(action, data=data)
        observation = step_result[0] if isinstance(step_result, tuple) else step_result
