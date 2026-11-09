from __future__ import annotations

import argparse
import os
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .dsl import is_terminal, resolve_action, unpack_step_result
from .env_sampler import EnvSampler

DEFAULT_MAX_BASELINE_STEPS = 120
SINGLE_ACTION_ENV = "BASELINE_SINGLE_ACTION_ID"

PolicyFn = Callable[[list[int], int], int]


@dataclass(frozen=True)
class BaselineEpisodeResult:
    game_id: str
    policy_name: str
    final_state: str
    steps: int
    solved: bool


def _state_name(observation: Any) -> str:
    state = getattr(observation, "state", None)
    return str(getattr(state, "name", state))


def _available_action_ids(env: Any) -> list[int]:
    return sorted(int(action.value) for action in env.action_space)


def _action_data(action_id: int) -> dict[str, int]:
    if action_id == 6:
        return {"x": -1, "y": -1}
    return {}


def _policy_noop_like(available: list[int], step_idx: int) -> int:
    _ = step_idx
    return 6 if 6 in available else available[0]


def _policy_repeat_first(available: list[int], step_idx: int) -> int:
    _ = step_idx
    return available[0]


def _policy_repeat_up_or_first(available: list[int], step_idx: int) -> int:
    _ = step_idx
    return 1 if 1 in available else available[0]


def _policy_repeat_configured_single(available: list[int], step_idx: int) -> int:
    _ = step_idx
    raw = os.environ.get(SINGLE_ACTION_ENV, "").strip()
    try:
        configured = int(raw) if raw else 1
    except ValueError:
        configured = 1
    return configured if configured in available else available[0]


def _policy_cycle_moves(available: list[int], step_idx: int) -> int:
    moves = [action_id for action_id in (1, 4, 2, 3) if action_id in available]
    if not moves:
        return available[0]
    return moves[step_idx % len(moves)]


def _policy_cycle_all(available: list[int], step_idx: int) -> int:
    return available[step_idx % len(available)]


def _policy_seeded_random(available: list[int], step_idx: int) -> int:
    rng = random.Random(step_idx + 1337)
    return available[rng.randrange(len(available))]


POLICIES: Mapping[str, PolicyFn] = {
    "noop_like": _policy_noop_like,
    "repeat_first": _policy_repeat_first,
    "repeat_configured_single": _policy_repeat_configured_single,
    "repeat_up_or_first": _policy_repeat_up_or_first,
    "cycle_moves": _policy_cycle_moves,
    "cycle_all": _policy_cycle_all,
    "seeded_random": _policy_seeded_random,
}


def run_baseline_episode(
    game_id: str, policy_name: str, *, max_steps: int = DEFAULT_MAX_BASELINE_STEPS, seed: int = 0
) -> BaselineEpisodeResult:
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy {policy_name!r}. Expected one of: {', '.join(sorted(POLICIES))}")
    env = EnvSampler(augment=False, seed=seed).make(game_id=game_id, seed=seed)
    observation = env.reset()
    available = _available_action_ids(env)

    if not available:
        raise RuntimeError(f"{game_id}: environment has no available actions")

    policy = POLICIES[policy_name]
    steps = 0
    while steps < max_steps and not is_terminal(observation):
        action_id = int(policy(available, steps))
        if action_id not in available:
            raise RuntimeError(
                f"{game_id}: policy {policy_name} produced unavailable action {action_id}; available={available}"
            )
        action = resolve_action(env, action_id)
        observation, _, done, _ = unpack_step_result(env.step(action, data=_action_data(action_id)))
        steps += 1
        if done:
            break

    final_state = _state_name(observation)
    return BaselineEpisodeResult(
        game_id=game_id, policy_name=policy_name, final_state=final_state, steps=steps, solved=(final_state == "WIN")
    )


def run_baseline_suite(
    game_id: str,
    *,
    policy_names: Iterable[str] | None = None,
    max_steps: int = DEFAULT_MAX_BASELINE_STEPS,
    seed: int = 0,
) -> list[BaselineEpisodeResult]:
    selected_policies = sorted(policy_names) if policy_names is not None else sorted(POLICIES)
    return [
        run_baseline_episode(game_id, policy_name, max_steps=max_steps, seed=seed) for policy_name in selected_policies
    ]


def run_many_games(
    game_ids: Iterable[str],
    *,
    policy_names: Iterable[str] | None = None,
    max_steps: int = DEFAULT_MAX_BASELINE_STEPS,
    seed: int = 0,
) -> list[BaselineEpisodeResult]:
    all_results: list[BaselineEpisodeResult] = []
    for game_id in game_ids:
        all_results.extend(run_baseline_suite(game_id, policy_names=policy_names, max_steps=max_steps, seed=seed))
    return all_results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run anti-triviality baseline robustness checks for one or more environments.")
    )
    parser.add_argument(
        "--game", action="append", default=[], help="Target game id (repeatable), e.g. --game push-0001"
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        choices=sorted(POLICIES),
        help="Limit to one or more policies (repeatable).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_BASELINE_STEPS,
        help=f"Maximum steps per episode (default: {DEFAULT_MAX_BASELINE_STEPS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used for environment construction and policy sampling."
    )
    parser.add_argument(
        "--single-action-id",
        type=int,
        default=None,
        help=f"Override {SINGLE_ACTION_ENV} for repeat_configured_single policy.",
    )
    return parser.parse_args(argv)


def _resolve_game_ids(args: argparse.Namespace) -> list[str]:
    game_ids = [str(game_id).strip() for game_id in args.game if str(game_id).strip()]
    deduped = list(dict.fromkeys(game_ids))
    if not deduped:
        raise ValueError("No game ids selected. Use --game <id>.")
    return deduped


def _print_results(results: list[BaselineEpisodeResult]) -> None:
    for result in results:
        verdict = "PASS" if not result.solved else "FAIL"
        print(
            f"{verdict} game={result.game_id} policy={result.policy_name} "
            f"state={result.final_state} steps={result.steps}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    game_ids = _resolve_game_ids(args)
    policy_names = args.policy or None

    previous_single_action = os.environ.get(SINGLE_ACTION_ENV)
    if args.single_action_id is not None:
        os.environ[SINGLE_ACTION_ENV] = str(int(args.single_action_id))

    try:
        results = run_many_games(
            game_ids, policy_names=policy_names, max_steps=int(args.max_steps), seed=int(args.seed)
        )
    finally:
        if args.single_action_id is not None:
            if previous_single_action is None:
                os.environ.pop(SINGLE_ACTION_ENV, None)
            else:
                os.environ[SINGLE_ACTION_ENV] = previous_single_action

    _print_results(results)
    solved = [result for result in results if result.solved]
    if solved:
        print()
        print("Baseline robustness failed: trivial policy reached WIN.")
        for result in solved:
            print(f"- game={result.game_id} policy={result.policy_name} steps={result.steps}")
        return 1

    print()
    print("Baseline robustness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
