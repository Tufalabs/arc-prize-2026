from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import GameAction

from re_arc import AugmentationConfig
from re_arc.env_sampler import _AugmentedEnv, _EpisodeTransform, _ObservationAugmenter


def _identity_colors() -> np.ndarray:
    return np.arange(16, dtype=np.int16)


def test_augmentation_config_coerce_from_dict():
    config = AugmentationConfig.coerce({"rotation": False})
    assert config.color_permutation is True
    assert config.rotation is False
    assert config.flip_lr is True


def test_augmentation_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown augmentation_config keys"):
        AugmentationConfig.coerce({"unknown": True})


def test_augmenter_all_components_disabled_is_identity():
    augmenter = _ObservationAugmenter(
        random.Random(0),
        enabled=True,
        config=AugmentationConfig(color_permutation=False, rotation=False, flip_lr=False),
    )

    augmenter.new_episode()
    assert augmenter.is_effective is False
    assert np.array_equal(augmenter._transform.color_map, _identity_colors())
    assert augmenter._transform.rot_k == 0
    assert augmenter._transform.flip_lr is False

    frame = np.array([[1, 2], [3, 4]], dtype=np.int16)
    observation = SimpleNamespace(frame=frame)
    assert augmenter.transform_observation(observation) is observation
    assert augmenter.inverse_delta(1, -1) == (1, -1)
    assert augmenter.inverse_point(2, 3, height=5, width=7) == (2, 3)


def test_augmenter_respects_selective_flags():
    augmenter = _ObservationAugmenter(
        random.Random(0), enabled=True, config=AugmentationConfig(color_permutation=False, rotation=True, flip_lr=True)
    )
    augmenter.new_episode()

    assert augmenter.is_effective is True
    assert np.array_equal(augmenter._transform.color_map, _identity_colors())
    assert augmenter._transform.rot_k == 3
    assert augmenter._transform.flip_lr is True


def test_augmenter_can_keep_only_color_permutation():
    augmenter = _ObservationAugmenter(
        random.Random(0), enabled=True, config=AugmentationConfig(color_permutation=True, rotation=False, flip_lr=False)
    )
    augmenter.new_episode()

    assert augmenter.is_effective is True
    assert augmenter._transform.rot_k == 0
    assert augmenter._transform.flip_lr is False
    assert not np.array_equal(augmenter._transform.color_map, _identity_colors())


def test_forward_delta_inverts_inverse_delta_for_all_episode_transforms():
    augmenter = _ObservationAugmenter(
        random.Random(0), enabled=True, config=AugmentationConfig(color_permutation=False, rotation=True, flip_lr=True)
    )

    for rot_k in range(4):
        for flip_lr in (False, True):
            augmenter._transform = _EpisodeTransform(color_map=_identity_colors(), rot_k=rot_k, flip_lr=flip_lr)
            for dx_base, dy_base in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                dx_t, dy_t = augmenter.forward_delta(dx_base, dy_base)
                assert augmenter.inverse_delta(dx_t, dy_t) == (dx_base, dy_base)


class _StubAugmentedEnvBase:
    def __init__(self, action_space):
        self.action_space = tuple(action_space)
        self.step_calls: list[tuple[int, dict[str, int | float | str | bool | None]]] = []
        self._observation = SimpleNamespace(frame=np.zeros((4, 4), dtype=np.int16), game_id="stub-0001")

    def reset(self):
        return self._observation

    def step(self, action, data=None):
        self.step_calls.append((int(action.value), dict(data or {})))
        return self._observation


def test_augmented_env_remaps_available_move_actions_for_rotation():
    base_env = _StubAugmentedEnvBase((GameAction.ACTION3, GameAction.ACTION4))
    augmenter = _ObservationAugmenter(
        random.Random(0), enabled=True, config=AugmentationConfig(color_permutation=False, rotation=True, flip_lr=False)
    )
    augmenter._transform = _EpisodeTransform(color_map=_identity_colors(), rot_k=1, flip_lr=False)
    env = _AugmentedEnv(base_env, augmenter, initial_transform_ready=True, game_id="stub-0001")

    env.reset()

    assert tuple(int(action.value) for action in env.action_space) == (
        int(GameAction.ACTION2.value),
        int(GameAction.ACTION1.value),
    )


def test_augmented_env_remaps_available_move_actions_for_flip():
    base_env = _StubAugmentedEnvBase((GameAction.ACTION3, GameAction.ACTION4))
    augmenter = _ObservationAugmenter(
        random.Random(0), enabled=True, config=AugmentationConfig(color_permutation=False, rotation=False, flip_lr=True)
    )
    augmenter._transform = _EpisodeTransform(color_map=_identity_colors(), rot_k=0, flip_lr=True)
    env = _AugmentedEnv(base_env, augmenter, initial_transform_ready=True, game_id="stub-0001")

    env.reset()

    assert tuple(int(action.value) for action in env.action_space) == (
        int(GameAction.ACTION4.value),
        int(GameAction.ACTION3.value),
    )


def test_augmented_env_maps_augmented_move_actions_back_to_base_env():
    base_env = _StubAugmentedEnvBase((GameAction.ACTION3, GameAction.ACTION4))
    augmenter = _ObservationAugmenter(
        random.Random(0), enabled=True, config=AugmentationConfig(color_permutation=False, rotation=True, flip_lr=False)
    )
    augmenter._transform = _EpisodeTransform(color_map=_identity_colors(), rot_k=1, flip_lr=False)
    env = _AugmentedEnv(base_env, augmenter, initial_transform_ready=True, game_id="stub-0001")

    env.reset()
    env.step(GameAction.ACTION2)
    env.step(GameAction.ACTION1)

    assert [action_id for action_id, _ in base_env.step_calls] == [
        int(GameAction.ACTION3.value),
        int(GameAction.ACTION4.value),
    ]


def test_envsampler_make_augmentation_is_deterministic_under_concurrency():
    """sampler.make(game_id=g, seed=s) must produce the same augmentation regardless
    of concurrent calls with other seeds.

    Regression: previously the augmenter drew from sampler._rng (shared), so concurrent
    make() calls interleaved on a single RNG and the resulting transform depended on
    thread scheduling. Now make() seeds a per-call RNG from resolved_seed.
    """
    from concurrent.futures import ThreadPoolExecutor

    from re_arc.env_sampler import EnvSampler, _AugmentedEnv

    sampler = EnvSampler(include=["ar25-0c556536"], augment=True, seed=99)

    def transform_for(seed: int) -> tuple[tuple[int, ...], int, bool]:
        env = sampler.make(game_id="ar25-0c556536", seed=seed)
        env.reset()  # triggers _AugmentedEnv.new_episode() which draws the transform
        # Unwrap TransitionRewardEnv -> _AugmentedEnv if needed.
        inner = getattr(env, "_env", env)
        while not isinstance(inner, _AugmentedEnv):
            nxt = getattr(inner, "_env", None)
            if nxt is None or nxt is inner:
                pytest.fail("Could not find _AugmentedEnv in wrapped env stack")
            inner = nxt
        t = inner._augmenter._transform
        return (tuple(int(c) for c in t.color_map), int(t.rot_k), bool(t.flip_lr))

    seeds = [1, 2, 3, 4, 5] * 6  # repeats so determinism shows up
    baseline = {s: transform_for(s) for s in set(seeds)}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(transform_for, seeds))

    for seed, got in zip(seeds, results, strict=True):
        assert got == baseline[seed], f"seed={seed} produced a different transform under concurrent make()"
