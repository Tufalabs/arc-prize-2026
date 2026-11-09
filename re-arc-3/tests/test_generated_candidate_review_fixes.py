from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from re_arc import EnvSampler
from re_arc.dsl import resolve_action, unpack_step_result
from re_arc.dsl.precomputed_actions import load_precomputed_actions

CANDIDATE_GAME_IDS = (
    "compass_weave-0001",
    "corner_cart-0001",
    "counterweight_court-0001",
    "plank_relay-0001",
    "prism_paths-0001",
    "vector_wake-0001",
)
UNDO_CASES = (
    ("compass_weave-0001", 4),
    ("corner_cart-0001", 3),
    ("counterweight_court-0001", 4),
    ("plank_relay-0001", 5),
    ("vector_wake-0001", 4),
)
ROOT = Path(__file__).resolve().parents[1]


def _game(env: Any) -> Any:
    return env._env._game


def _board_state(game_id: str, game: Any) -> tuple[object, ...]:
    if game_id == "compass_weave-0001":
        return game.position, game.orientation, tuple(game.seal_cut)
    if game_id == "corner_cart-0001":
        return game.elbow, game.orientation, game.lit_mask
    if game_id == "counterweight_court-0001":
        return (
            game.player,
            tuple(sorted(game.stones.items())),
            tuple(sorted((key, tuple(value)) for key, value in game.pair_tiers.items())),
        )
    if game_id == "plank_relay-0001":
        return game.player_position, game.player_facing, game.plank_mode, tuple(game.dropped_plank_cells)
    if game_id == "vector_wake-0001":
        return (
            game.position,
            tuple(game.checkpoint_active),
            game.heading,
            game.speed,
            tuple(game.last_sweep),
            tuple(game.wake_cells),
        )
    raise AssertionError(f"Missing board-state adapter for {game_id}")


def _resource_used(game_id: str, game: Any) -> int:
    if game_id == "compass_weave-0001":
        return int(game.steps_used)
    if game_id == "corner_cart-0001":
        return int(game.step_count)
    if game_id == "counterweight_court-0001":
        return int(game.steps_used)
    if game_id == "plank_relay-0001":
        return int(game.step_count)
    if game_id == "vector_wake-0001":
        return int(game.step_budget - game.remaining_steps)
    raise AssertionError(f"Missing resource adapter for {game_id}")


@pytest.mark.parametrize(("game_id", "mutation_action"), UNDO_CASES)
def test_action7_undo_rewinds_board_without_refunding_turns(game_id: str, mutation_action: int) -> None:
    env = EnvSampler(augment=False, seed=0).make(game_id=game_id, seed=0)
    observation = env.reset()
    game = _game(env)
    before = _board_state(game_id, game)
    resource_before = _resource_used(game_id, game)

    observation, _reward, _done, _info = unpack_step_result(env.step(resolve_action(env, mutation_action)))
    assert _board_state(game_id, game) != before
    assert 7 in {int(getattr(action, "value", action)) for action in observation.available_actions}

    unpack_step_result(env.step(resolve_action(env, 7)))

    assert _board_state(game_id, game) == before
    assert _resource_used(game_id, game) == resource_before + 2


def _first_level_solution(game_id: str) -> list[tuple[int, dict[str, object]]]:
    payload = load_precomputed_actions(game_id)
    assert payload is not None
    env = EnvSampler(augment=False, seed=0).make(game_id=game_id, seed=0)
    observation = env.reset()
    prefix: list[tuple[int, dict[str, object]]] = []
    for action_id, action_data in payload["actions"]:
        prefix.append((int(action_id), dict(action_data or {})))
        observation, _reward, _done, _info = unpack_step_result(
            env.step(resolve_action(env, int(action_id)), data=dict(action_data or {}))
        )
        if int(observation.levels_completed) == 1:
            return prefix
    raise AssertionError(f"Precomputed trace never completed level 1 for {game_id}")


@pytest.mark.parametrize("game_id", ["counterweight_court-0001", "vector_wake-0001"])
def test_winning_action_on_final_budget_step_advances_level(game_id: str) -> None:
    solution = _first_level_solution(game_id)
    env = EnvSampler(augment=False, seed=0).make(game_id=game_id, seed=0)
    observation = env.reset()
    for action_id, action_data in solution[:-1]:
        observation, _reward, _done, _info = unpack_step_result(
            env.step(resolve_action(env, action_id), data=action_data)
        )
        assert int(observation.levels_completed) == 0

    game = _game(env)
    if game_id == "counterweight_court-0001":
        game.steps_used = game.step_budget - 1
    else:
        game.remaining_steps = 1

    final_action, final_data = solution[-1]
    observation, _reward, _done, _info = unpack_step_result(
        env.step(resolve_action(env, final_action), data=final_data)
    )
    assert int(observation.levels_completed) == 1


def test_candidate_metadata_marks_review_without_claiming_gameplay_evaluation() -> None:
    for game_id in CANDIDATE_GAME_IDS:
        slug, variant = game_id.rsplit("-", 1)
        manifest = json.loads(
            (ROOT / "re_arc/environment_files" / slug / variant / "manifest.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (ROOT / "re_arc/environment_files" / slug / variant / "metadata.json").read_text(encoding="utf-8")
        )
        assert 7 in manifest["available_actions"]
        assert "undo" in manifest["action_set"]
        assert "codex-reviewed" in metadata["tags"]
        assert "codex-evaluated" not in metadata["tags"]
