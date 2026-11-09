from __future__ import annotations

import os
import signal
import sys
from contextlib import contextmanager, suppress

import pytest
from arcengine import GameAction
from tqdm import tqdm

from re_arc import EnvSampler
from re_arc.dsl import create_dsl_agent, is_terminal, resolve_action, unpack_step_result
from re_arc.dsl.core import frame_signature
from re_arc.dsl.precomputed_actions import load_precomputed_actions, metadata_baseline_actions, precomputed_actions_path
from re_arc.env_sampler import PACKAGE_ENVIRONMENTS_DIR, _discover_game_metadata_from_dir

MAX_STEPS_PER_GAME = 1000
MAX_ENV_STEP_MS = 10.0
STEP_TIME_ASSERTION_EXCLUDED_GAMES = {"vc33-9851e02b", "ls20-9607627b"}
DSL_REPLAY_EXCLUDED_GAMES = set()
MIN_LEVELS_PER_GAME = 1
MAX_LEVELS_PER_GAME = 10
MAX_EPISODE_RETURN = 1.0001
REPLAY_TIMEOUT_PADDING_SECONDS = float(os.environ.get("DSL_REPLAY_TIMEOUT_PADDING_SECONDS", "1.0"))
REPLAY_TIMEOUT_FLOOR_SECONDS = float(os.environ.get("DSL_REPLAY_TIMEOUT_FLOOR_SECONDS", "1.0"))


@contextmanager
def _per_game_timeout(game_id: str, timeout_seconds: float):
    if timeout_seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _on_alarm(_signum, _frame):
        raise TimeoutError(f"{game_id}: exceeded {timeout_seconds:.3f}s wall-clock timeout.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _state_name(observation) -> str:
    state = getattr(observation, "state", None)
    return str(getattr(state, "name", state))


def _replay_timeout_seconds(action_count: int) -> float:
    step_budget = (MAX_ENV_STEP_MS / 1000.0) * max(1, int(action_count))
    return max(REPLAY_TIMEOUT_FLOOR_SECONDS, step_budget + REPLAY_TIMEOUT_PADDING_SECONDS)


def _official_game_ids() -> set[str]:
    metadata = _discover_game_metadata_from_dir(PACKAGE_ENVIRONMENTS_DIR)
    return {gid for gid, tags in metadata.items() if "official" in tags}


# The env layer always advertises RESET on every observation, so it is
# excluded from the fixed-action-set check below.
ENV_LAYER_MANAGED_ACTION_IDS = frozenset({int(GameAction.RESET.value)})


def _available_action_ids(observation) -> frozenset[int]:
    actions = getattr(observation, "available_actions", None) or []
    return frozenset(int(getattr(action, "value", action)) for action in actions) - ENV_LAYER_MANAGED_ACTION_IDS


def test_dsl_agents_complete_all_games_under_constraints():
    sampler = EnvSampler(augment=False, seed=0)
    game_ids = sampler.game_ids
    assert game_ids, "No games discovered by EnvSampler."

    official_ids = _official_game_ids()
    failures: list[str] = []
    game_iterator = tqdm(game_ids, desc="DSL games", unit="game", leave=False, disable=not sys.stderr.isatty())

    for game_id in game_iterator:
        env = None
        try:
            # The official games' replays and solutions are not shipped here.
            if game_id in DSL_REPLAY_EXCLUDED_GAMES or game_id in official_ids:
                continue

            try:
                _agent = create_dsl_agent(game_id)
            except ValueError:
                failures.append(f"{game_id}: no DSL agent registered.")
                continue

            precomputed = load_precomputed_actions(game_id)
            if precomputed is None:
                failures.append(
                    f"{game_id}: missing precomputed actions at "
                    f"{precomputed_actions_path(game_id)}. "
                    "Run `make precompute_dsl_actions`."
                )
                continue

            precomputed_actions = list(precomputed.get("actions", []))
            timeout_seconds = _replay_timeout_seconds(len(precomputed_actions))

            with _per_game_timeout(game_id, timeout_seconds):
                env = sampler.make(game_id=game_id, seed=0)
                observation = env.reset()
                # real games keep a fixed action set for the whole game
                baseline_available_actions = _available_action_ids(observation)

                raw_level_count = getattr(observation, "win_levels", None)
                try:
                    level_count = int(raw_level_count)
                except (TypeError, ValueError):
                    failures.append(
                        f"{game_id}: invalid win_levels={raw_level_count!r}; "
                        "expected an integer in "
                        f"[{MIN_LEVELS_PER_GAME}, {MAX_LEVELS_PER_GAME}]."
                    )
                    continue

                if not (MIN_LEVELS_PER_GAME <= level_count <= MAX_LEVELS_PER_GAME):
                    failures.append(
                        f"{game_id}: level count is {level_count}; "
                        f"expected [{MIN_LEVELS_PER_GAME}, {MAX_LEVELS_PER_GAME}]."
                    )

                expected_signature = str(precomputed.get("initial_observation_signature") or "").strip()
                if expected_signature:
                    observed_signature = frame_signature(observation)
                    if observed_signature != expected_signature:
                        failures.append(
                            f"{game_id}: precomputed trace is stale "
                            "(initial signature mismatch). "
                            "Run `make precompute_dsl_actions --force`."
                        )
                        continue

                expected_baseline_actions = metadata_baseline_actions(game_id)
                if len(expected_baseline_actions) != level_count:
                    failures.append(
                        f"{game_id}: metadata baseline_actions has len={len(expected_baseline_actions)}; "
                        f"expected {level_count}. Run `make baseline-actions`."
                    )
                    continue

                actions = 0
                episode_reward = 0.0
                done = False
                actions_in_level = 0
                observed_baseline_actions: list[int] = []
                raw_prev_level = getattr(observation, "levels_completed", None)
                try:
                    prev_level = int(raw_prev_level)
                except (TypeError, ValueError):
                    failures.append(f"{game_id}: invalid levels_completed={raw_prev_level!r}; expected an integer.")
                    continue

                for action_id, action_data in precomputed_actions:
                    if is_terminal(observation):
                        done = True
                        break

                    action = resolve_action(env, int(action_id))
                    step_result = env.step(action, data=dict(action_data or {}))

                    observation, reward, step_done, _ = unpack_step_result(step_result)
                    current_available_actions = _available_action_ids(observation)
                    if current_available_actions != baseline_available_actions:
                        failures.append(
                            f"{game_id}: available_actions changed during play "
                            f"(after action {actions + 1}, level {prev_level}): "
                            f"{sorted(baseline_available_actions)} -> {sorted(current_available_actions)}."
                        )
                        break
                    episode_reward += float(reward)
                    if episode_reward > MAX_EPISODE_RETURN:
                        failures.append(
                            f"{game_id}: cumulative reward {episode_reward:.6f} exceeded {MAX_EPISODE_RETURN:.6f}."
                        )
                    actions_in_level += 1
                    raw_new_level = getattr(observation, "levels_completed", None)
                    try:
                        new_level = int(raw_new_level)
                    except (TypeError, ValueError):
                        failures.append(f"{game_id}: invalid levels_completed={raw_new_level!r}; expected an integer.")
                        break
                    if new_level > prev_level:
                        # Intermediate level transitions must emit ≥2 frames
                        # (old level's final + new level's initial). Games
                        # that bypass arcengine's auto-transition — by
                        # calling `set_level()` themselves inside `step()`
                        # — collapse the transition into 1 frame, breaking
                        # downstream consumers. Terminal transitions are
                        # exempt: arcengine's action loop exits on WIN with
                        # 1 frame and wrappers handle that case separately.
                        is_terminal_transition = new_level >= level_count
                        n_frames = len(getattr(observation, "frame", []) or [])
                        if not is_terminal_transition and n_frames < 2:
                            failures.append(
                                f"{game_id}: intermediate level transition "
                                f"(level {prev_level}→{new_level}) emitted "
                                f"only {n_frames} frame(s); expected ≥2 (old "
                                f"level's final frame + new level's initial "
                                f"frame). The game's `step()` likely sets "
                                f"`_next_level=False` and calls "
                                f"`self.set_level(idx+1)` manually, "
                                f"suppressing arcengine's automatic "
                                f"transition render. Drop the manual "
                                f"transition and let `self.next_level()` "
                                f"flow through arcengine's action loop."
                            )
                        solved_now = new_level - prev_level
                        for solved_idx in range(solved_now):
                            if len(observed_baseline_actions) >= level_count:
                                break
                            observed_baseline_actions.append(actions_in_level if solved_idx == 0 else 1)
                        actions_in_level = 0
                    elif new_level < prev_level:
                        actions_in_level = 0
                    prev_level = new_level
                    actions += 1
                    if step_done:
                        done = True
                        break

                final_state = _state_name(observation)
                if not done and not is_terminal(observation):
                    failures.append(
                        f"{game_id}: precomputed trace exhausted after {actions} actions before terminal state."
                    )
                elif final_state != "WIN":
                    failures.append(
                        f"{game_id}: final state is {final_state} after {actions} replayed actions; expected WIN."
                    )
                elif len(observed_baseline_actions) < level_count:
                    # Bare `self.win()` on the final level (instead of
                    # `self.next_level()`, which both increments the score
                    # AND calls win() automatically on the last level)
                    # produces a terminal WIN with `levels_completed <
                    # win_levels`. arc_agi's scorecard
                    # (EnvironmentScoreCalculator) only credits levels that
                    # had an explicit level transition, so the un-transitioned
                    # final level scores 0 — the player wins the episode but
                    # the engine reports no reward for the level they actually
                    # solved. Always use `self.next_level()` for terminal level
                    # transitions.
                    failures.append(
                        f"{game_id}: terminal WIN reached with only "
                        f"{len(observed_baseline_actions)}/{level_count} level "
                        f"transitions emitted. The game's terminal step uses "
                        f"`self.win()` instead of `self.next_level()`; the "
                        f"un-transitioned level scores 0 in arc_agi's "
                        f"scorecard so the player gets no credit for solving "
                        f"it. Replace the final `self.win()` with "
                        f"`self.next_level()` (which calls win() automatically "
                        f"on the last level)."
                    )
                elif not (0.999 <= episode_reward <= 1.001):
                    failures.append(f"{game_id}: WIN episode should sum to reward 1.0; got {episode_reward:.6f}.")

                elif game_id not in official_ids and observed_baseline_actions != expected_baseline_actions:
                    failures.append(
                        f"{game_id}: metadata baseline_actions={expected_baseline_actions} "
                        f"does not match replay-derived per-level actions={observed_baseline_actions}. "
                        "Run `make baseline-actions`."
                    )
                else:
                    # All structural checks passed; verify the arc_agi scorecard
                    # gives full credit. Per-level actions match the baseline so
                    # arc_agi's score formula should yield exactly 100.0; a
                    # lower score means the engine isn't crediting one or more
                    # levels (typically a bare `self.win()` terminal that
                    # bypasses next_level(), already caught by the earlier
                    # transitions check, but kept here as belt-and-suspenders
                    # against any future scoring-formula divergence).
                    scorecard = sampler._arcade.get_scorecard()
                    env_list = scorecard.find_environment(game_id) if scorecard is not None else None
                    # EnvSampler.make() uses the arcade's default scorecard
                    # without a fresh per-game scorecard_id, so the env_list
                    # tends to carry a leading empty/pre-reset run alongside
                    # the played run. Pick the played run by state == WIN.
                    played_runs = (
                        [r for r in env_list.runs if str(getattr(getattr(r, "state", None), "name", "")) == "WIN"]
                        if env_list is not None
                        else []
                    )
                    if not played_runs:
                        failures.append(f"{game_id}: arc_agi scorecard has no completed (WIN) run for this game.")
                    else:
                        engine_score = played_runs[-1].score
                        if not (99.999 <= engine_score <= 100.001):
                            failures.append(
                                f"{game_id}: arc_agi scorecard score={engine_score:.4f}, "
                                f"expected 100.0 (per-level actions match baseline)."
                            )
        except AssertionError:
            raise
        except TimeoutError as exc:
            raise AssertionError(str(exc)) from exc
        except Exception as exc:
            failures.append(f"{game_id}: raised {type(exc).__name__}: {exc}")
        finally:
            if env is not None:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    with suppress(Exception):
                        close_fn()
            # Reset the default scorecard so each game starts with a clean
            # slate; otherwise runs accumulate across games and find_environment
            # can return stale runs from earlier iterations.
            with suppress(Exception):
                sampler._arcade.close_scorecard()

    if failures:
        pytest.fail("DSL validation failures:\n- " + "\n- ".join(failures))
