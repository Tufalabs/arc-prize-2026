from __future__ import annotations

import ast
from contextlib import suppress
from pathlib import Path

from re_arc import EnvSampler
from re_arc.cli import _create_env_and_initial_observation, _load_config
from re_arc.dsl import create_dsl_agent, is_terminal, resolve_action, unpack_step_result

MAX_EPISODE_RETURN = 1.0001
MAX_STEPS_PER_EPISODE = 1000
DSL_MAX_ACTIONS = 1200
ENVIRONMENT_FILES_DIR = Path(__file__).resolve().parents[1] / "re_arc" / "environment_files"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.env"


def _state_name(observation) -> str:
    state = getattr(observation, "state", None)
    return str(getattr(state, "name", state))


def _int_observation_attr(observation, attr: str, *, game_id: str) -> int:
    raw_value = getattr(observation, attr, None)
    assert raw_value is not None, f"{game_id}: observation is missing {attr!r}."
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{game_id}: observation {attr}={raw_value!r} is not an integer.") from exc


def _played_scorecard_score(sampler: EnvSampler, game_id: str) -> float:
    scorecard = sampler._arcade.get_scorecard()
    env_list = scorecard.find_environment(game_id) if scorecard is not None else None
    played_runs = (
        [run for run in env_list.runs if str(getattr(getattr(run, "state", None), "name", "")) == "WIN"]
        if env_list is not None
        else []
    )
    assert played_runs, f"{game_id}: arc_agi scorecard has no completed WIN run."
    return float(played_runs[-1].score)


def _iter_mutations_of_engine_score(source: str, *, filename: str) -> list[tuple[int, str]]:
    tree = ast.parse(source, filename=filename)
    violations: list[tuple[int, str]] = []

    def _is_self_score_target(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "_score"
        )

    def _walk_targets(node: ast.AST):
        if isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                yield from _walk_targets(item)
            return
        yield node

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for candidate in _walk_targets(target):
                    if _is_self_score_target(candidate):
                        violations.append((node.lineno, "assignment"))
        elif isinstance(node, ast.AugAssign):
            if _is_self_score_target(node.target):
                violations.append((node.lineno, "augmented assignment"))
        elif isinstance(node, ast.AnnAssign):
            if _is_self_score_target(node.target):
                violations.append((node.lineno, "annotated assignment"))
        elif isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "_score"
        ):
            violations.append((node.lineno, "setattr"))

    return violations


def test_environment_games_do_not_mutate_engine_score_directly():
    failures: list[str] = []
    for py_file in sorted(ENVIRONMENT_FILES_DIR.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        violations = _iter_mutations_of_engine_score(source, filename=str(py_file))
        for lineno, kind in violations:
            rel = py_file.relative_to(Path(__file__).resolve().parents[1])
            failures.append(f"{rel}:{lineno} ({kind})")

    assert not failures, (
        "Environment files must not mutate ARCBaseGame._score directly. "
        "Use a separate internal counter (for example `_route_score`) and "
        "advance level progress with `next_level()`.\n" + "\n".join(failures)
    )


def test_transition_episode_return_for_shortest_route_challenge_is_bounded():
    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id="shortest_route_challenge", seed=0)

    try:
        observation = env.reset()
        agent = create_dsl_agent("shortest_route_challenge")
        agent.reset_episode()
        agent.observe(observation)

        total_reward = 0.0
        for _ in range(MAX_STEPS_PER_EPISODE):
            if is_terminal(observation):
                break

            action_id, action_data = agent.next_action(env, observation)
            action = resolve_action(env, action_id)
            agent.record_action(action_id)
            step_result = env.step(action, data=action_data or {})
            observation, reward, done, _ = unpack_step_result(step_result)
            agent.observe(observation)

            total_reward += float(reward)
            assert total_reward <= MAX_EPISODE_RETURN, (
                "Transition-reward return exceeded bound for shortest_route_challenge: "
                f"{total_reward:.6f} > {MAX_EPISODE_RETURN:.6f}"
            )
            if done:
                break

        state_name = str(getattr(getattr(observation, "state", None), "name", None))
        assert state_name == "WIN", f"shortest_route_challenge should end in WIN, got {state_name!r}"
        assert 0.999 <= total_reward <= MAX_EPISODE_RETURN, (
            "shortest_route_challenge should produce a single bounded level-transition reward. "
            f"Observed total_reward={total_reward:.6f}."
        )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            with suppress(Exception):
                close_fn()


def test_configured_dsl_game_reaches_unit_reward():
    config = _load_config(CONFIG_PATH)
    configured_game_id = str(config.get("GAME_ID") or "").strip()
    assert configured_game_id, f"{CONFIG_PATH} must define GAME_ID."
    assert configured_game_id.lower() != "random", (
        "config.env GAME_ID must be a concrete game id for deterministic DSL reward checks."
    )

    env = None
    try:
        env, observation, selected_game_id = _create_env_and_initial_observation(
            config=config, game_id=configured_game_id, seed=None, renderer=None
        )
        agent = create_dsl_agent(selected_game_id)
        agent.reset_episode()
        agent.observe(observation)

        total_reward = 0.0
        for _ in range(DSL_MAX_ACTIONS):
            if is_terminal(observation):
                break

            action_id, action_data = agent.next_action(env, observation)
            action = resolve_action(env, action_id)
            agent.record_action(action_id)
            step_result = env.step(action, data=action_data or {})
            observation, reward, done, _ = unpack_step_result(step_result)
            agent.observe(observation)

            total_reward += float(reward)
            assert total_reward <= MAX_EPISODE_RETURN, (
                f"Configured GAME_ID={selected_game_id!r} exceeded reward bound: "
                f"{total_reward:.6f} > {MAX_EPISODE_RETURN:.6f}."
            )
            if done:
                break

        final_state = str(getattr(getattr(observation, "state", None), "name", None))

        assert final_state == "WIN", (
            f"Configured GAME_ID={selected_game_id!r} should finish in WIN; got {final_state!r}."
        )
        assert 0.999 <= total_reward <= 1.001, (
            f"Configured GAME_ID={selected_game_id!r} "
            "should end with total reward 1.0. "
            f"Observed total_reward={total_reward:.6f}."
        )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            with suppress(Exception):
                close_fn()


def test_configured_dsl_game_level_transitions_and_scorecard_reach_full_credit():
    config = _load_config(CONFIG_PATH)
    game_id = str(config.get("GAME_ID") or "").strip()
    assert game_id, f"{CONFIG_PATH} must define GAME_ID."
    assert game_id.lower() != "random", (
        "config.env GAME_ID must be a concrete game id for deterministic DSL transition checks."
    )

    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id=game_id, seed=0)

    try:
        observation = env.reset()
        win_levels = _int_observation_attr(observation, "win_levels", game_id=game_id)
        previous_completed = _int_observation_attr(observation, "levels_completed", game_id=game_id)
        observed_level_transitions = 0
        agent = create_dsl_agent(game_id)
        agent.reset_episode()
        agent.observe(observation)

        for _ in range(DSL_MAX_ACTIONS):
            if is_terminal(observation):
                break

            action_id, action_data = agent.next_action(env, observation)
            action = resolve_action(env, action_id)
            agent.record_action(action_id)
            step_result = env.step(action, data=action_data or {})
            observation, _reward, done, _ = unpack_step_result(step_result)
            agent.observe(observation)

            completed = _int_observation_attr(observation, "levels_completed", game_id=game_id)
            if completed > previous_completed:
                is_terminal_transition = completed >= win_levels
                frame_count = len(getattr(observation, "frame", []) or [])
                assert is_terminal_transition or frame_count >= 2, (
                    f"{game_id}: intermediate level transition "
                    f"({previous_completed}->{completed}) emitted only {frame_count} frame(s); expected at least 2. "
                    "Use `next_level()` and let arcengine render the transition instead of calling `set_level()` "
                    "manually inside `step()`."
                )
                observed_level_transitions += completed - previous_completed
            previous_completed = completed
            if done:
                break

        final_state = _state_name(observation)
        assert final_state == "WIN", f"{game_id}: expected WIN, got {final_state!r}."
        assert observed_level_transitions >= win_levels, (
            f"{game_id}: terminal WIN reached with only "
            f"{observed_level_transitions}/{win_levels} completed level transition(s). "
            "A final bare `self.win()` does not credit the solved level; use `self.next_level()`."
        )

        engine_score = _played_scorecard_score(sampler, game_id)
        assert 99.999 <= engine_score <= 100.001, (
            f"{game_id}: arc_agi scorecard score={engine_score:.4f}; expected 100.0."
        )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            with suppress(Exception):
                close_fn()
        with suppress(Exception):
            sampler._arcade.close_scorecard()
