from __future__ import annotations

from contextlib import suppress

from re_arc.cli import (
    _augmentation_config_from_config,
    _build_env_sampler,
    _create_env_and_initial_observation,
    _WebSession,
)
from re_arc.env_sampler import _AugmentedEnv


def _close_env(env) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        with suppress(Exception):
            close_fn()


def _close_web_session(session) -> None:
    env = getattr(session, "_env", None)
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        with suppress(Exception):
            close_fn()


def test_build_env_sampler_reads_selective_augmentation_config() -> None:
    config = {
        "AUGMENT": "true",
        "AUGMENT_COLOR_PERMUTATION": "false",
        "AUGMENT_ROTATION": "true",
        "AUGMENT_FLIP_LR": "false",
    }

    augmentation_config = _augmentation_config_from_config(config)
    assert augmentation_config.color_permutation is False
    assert augmentation_config.rotation is True
    assert augmentation_config.flip_lr is False

    env = _build_env_sampler(config, seed=0, augment=True, include=["maze-0001"]).make("maze-0001", seed=0)
    try:
        assert isinstance(env._env, _AugmentedEnv)
        assert env._env._augmenter._config == augmentation_config
    finally:
        _close_env(env)


def test_create_env_and_initial_observation_uses_sampler_for_augmented_specific_game() -> None:
    config = {
        "AUGMENT": "true",
        "AUGMENT_COLOR_PERMUTATION": "false",
        "AUGMENT_ROTATION": "true",
        "AUGMENT_FLIP_LR": "false",
    }

    env = None
    try:
        env, observation, selected_game_id = _create_env_and_initial_observation(
            config=config, game_id="maze-0001", seed=0, renderer=None
        )

        assert selected_game_id == "maze-0001"
        assert observation is not None
        assert isinstance(env._env, _AugmentedEnv)
        assert env._env._augmenter._config.color_permutation is False
        assert env._env._augmenter._config.rotation is True
        assert env._env._augmenter._config.flip_lr is False
    finally:
        _close_env(env)


def test_create_env_and_initial_observation_ignores_dataset_filters_for_augmented_specific_game() -> None:
    config = {"AUGMENT": "true", "DATASET": "eval"}

    env = None
    try:
        env, observation, selected_game_id = _create_env_and_initial_observation(
            config=config, game_id="easy_taps-0001", seed=0, renderer=None
        )

        assert selected_game_id == "easy_taps-0001"
        assert observation is not None
        assert isinstance(env._env, _AugmentedEnv)
    finally:
        _close_env(env)


def test_web_session_can_select_specific_game_outside_dataset_filter_when_augmented() -> None:
    config = {"AUGMENT": "true", "DATASET": "eval"}

    session = None
    try:
        session = _WebSession(config=config, game_id="random", seed=0)

        options = session.game_options()
        assert "easy_taps-0001" in options["games"]

        state = session.select_game("easy_taps-0001")

        assert state["requested_game_id"] == "easy_taps-0001"
        assert state["game_id"] == "easy_taps-0001"
    finally:
        _close_web_session(session)
