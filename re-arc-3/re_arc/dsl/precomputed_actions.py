from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..env_sampler import default_environments_dir

PRECOMPUTED_ACTIONS_SCHEMA_VERSION = 1


def default_precomputed_actions_dir() -> Path:
    return Path(__file__).resolve().parent / "precomputed_actions"


def precomputed_actions_path(game_id: str, *, directory: str | Path | None = None) -> Path:
    root = Path(directory) if directory is not None else default_precomputed_actions_dir()
    return root / f"{str(game_id).strip()}.json"


def encode_actions(actions: list[tuple[int, dict[str, int]]]) -> list[list[Any]]:
    encoded: list[list[Any]] = []
    for action_id, action_data in actions:
        payload: dict[str, int] = {}
        for key, value in dict(action_data or {}).items():
            payload[str(key)] = int(value)
        encoded.append([int(action_id), payload])
    return encoded


def decode_actions(raw_actions: Any) -> list[tuple[int, dict[str, int]]]:
    if not isinstance(raw_actions, list):
        raise ValueError("precomputed actions must be a list.")
    out: list[tuple[int, dict[str, int]]] = []
    for idx, item in enumerate(raw_actions):
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], dict):
            raise ValueError(f"invalid precomputed action entry at index {idx}: {item!r}")
        action_id = int(item[0])
        payload = {str(k): int(v) for k, v in item[1].items()}
        out.append((action_id, payload))
    return out


def load_precomputed_actions(game_id: str, *, directory: str | Path | None = None) -> dict[str, Any] | None:
    path = precomputed_actions_path(game_id, directory=directory)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != PRECOMPUTED_ACTIONS_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version={raw.get('schema_version')!r}")
    if str(raw.get("game_id", "")).strip() != str(game_id).strip():
        raise ValueError(f"{path}: game_id mismatch expected={game_id!r} got={raw.get('game_id')!r}")
    raw["actions"] = decode_actions(raw.get("actions", []))
    raw["_path"] = path
    return raw


def write_precomputed_actions(record: dict[str, Any], *, directory: str | Path | None = None) -> Path:
    game_id = str(record.get("game_id") or "").strip()
    if not game_id:
        raise ValueError("precomputed record must include non-empty game_id.")

    payload = dict(record)
    payload["schema_version"] = PRECOMPUTED_ACTIONS_SCHEMA_VERSION
    payload["game_id"] = game_id
    payload["actions"] = encode_actions(list(payload.get("actions", [])))

    path = precomputed_actions_path(game_id, directory=directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _normalize_game_id(value: str) -> str:
    return str(value).strip().lower()


def metadata_index(*, environments_dir: str | Path | None = None) -> dict[str, Path]:
    base = Path(environments_dir or default_environments_dir()).expanduser()
    discovered: dict[str, Path] = {}
    for metadata_path in sorted(base.rglob("metadata.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        game_id = str(payload.get("game_id") or "").strip()
        if game_id:
            discovered[_normalize_game_id(game_id)] = metadata_path
    return discovered


def metadata_path_for_game(game_id: str, *, environments_dir: str | Path | None = None) -> Path:
    normalized = _normalize_game_id(game_id)
    discovered = metadata_index(environments_dir=environments_dir)
    try:
        return discovered[normalized]
    except KeyError as exc:
        raise FileNotFoundError(f"Could not find metadata.json for game_id={game_id!r}.") from exc


def metadata_baseline_actions(game_id: str, *, environments_dir: str | Path | None = None) -> list[int]:
    metadata_path = metadata_path_for_game(game_id, environments_dir=environments_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = metadata.get("baseline_actions")
    if not isinstance(raw, list):
        raise ValueError(f"{game_id}: metadata baseline_actions must be a list, got {type(raw).__name__}.")
    return [int(value) for value in raw]


def derive_baseline_actions_from_precomputed(
    game_id: str, *, sampler: Any, seed: int = 0, precomputed: dict[str, Any] | None = None
) -> list[int]:
    from re_arc.dsl import resolve_action, unpack_step_result

    record = precomputed if precomputed is not None else load_precomputed_actions(game_id)
    if record is None:
        raise FileNotFoundError(f"{game_id}: missing precomputed actions.")

    env = sampler.make(game_id=game_id, seed=seed)
    try:
        observation = env.reset()
        total_levels = int(getattr(observation, "win_levels", 0) or 0)
        if total_levels <= 0:
            raise ValueError(f"{game_id}: expected positive win_levels, got {total_levels}.")

        prev_level = int(getattr(observation, "levels_completed", 0) or 0)
        actions_in_level = 0
        level_actions: list[int] = []

        for action_id, action_data in list(record.get("actions", [])):
            if getattr(getattr(observation, "state", None), "name", getattr(observation, "state", None)) == "WIN":
                break

            action = resolve_action(env, int(action_id))
            observation, _, done, _ = unpack_step_result(env.step(action, data=dict(action_data or {})))
            actions_in_level += 1

            new_level = int(getattr(observation, "levels_completed", prev_level) or prev_level)
            if new_level > prev_level:
                solved_now = new_level - prev_level
                for solved_idx in range(solved_now):
                    if len(level_actions) >= total_levels:
                        break
                    level_actions.append(actions_in_level if solved_idx == 0 else 1)
                actions_in_level = 0
            elif new_level < prev_level:
                actions_in_level = 0

            prev_level = new_level
            if done:
                break

        final_state = _state_name(observation).upper()
        if final_state != "WIN":
            raise RuntimeError(f"{game_id}: expected WIN while deriving baseline actions, got {final_state!r}.")

        if len(level_actions) < total_levels and actions_in_level > 0:
            level_actions.append(actions_in_level)

        if len(level_actions) != total_levels:
            raise RuntimeError(
                f"{game_id}: derived {len(level_actions)} baseline levels, expected {total_levels}: {level_actions!r}"
            )

        return [max(1, int(value)) for value in level_actions]
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            with suppress(Exception):
                close_fn()


@dataclass(frozen=True)
class BaselineActionSyncResult:
    game_id: str
    metadata_path: Path
    previous_actions: list[int]
    derived_actions: list[int]
    changed: bool


def sync_baseline_actions(
    *, config: dict[str, str], sampler: Any, game_ids: list[str], seed: int = 0, check_only: bool = False
) -> tuple[list[BaselineActionSyncResult], list[str]]:
    environments_dir = config.get("ENVIRONMENTS_DIR") or default_environments_dir()
    results: list[BaselineActionSyncResult] = []
    failures: list[str] = []

    for game_id in game_ids:
        try:
            metadata_path = metadata_path_for_game(game_id, environments_dir=environments_dir)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            if "official" in (metadata.get("tags") or []):
                continue

            existing = metadata.get("baseline_actions")
            if not isinstance(existing, list):
                raise ValueError(f"{game_id}: metadata baseline_actions must be a list, got {type(existing).__name__}.")

            derived = derive_baseline_actions_from_precomputed(game_id, sampler=sampler, seed=seed)
            changed = [int(value) for value in existing] != derived
            if changed and not check_only:
                metadata["baseline_actions"] = derived
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

            results.append(
                BaselineActionSyncResult(
                    game_id=game_id,
                    metadata_path=metadata_path,
                    previous_actions=[int(value) for value in existing],
                    derived_actions=derived,
                    changed=changed,
                )
            )
        except Exception as exc:
            failures.append(f"{game_id}: {type(exc).__name__}: {exc}")

    return results, failures


_WORKER_SAMPLER: Any | None = None


def _worker_init(config_path: str, seed: int) -> None:
    global _WORKER_SAMPLER
    from re_arc.cli import _build_env_sampler, _load_config

    config = _load_config(config_path)
    _WORKER_SAMPLER = _build_env_sampler(config, seed=seed, augment=False)


def _worker_run_game(game_id: str, seed: int, max_actions: int) -> tuple[str, str, int, float]:
    global _WORKER_SAMPLER
    if _WORKER_SAMPLER is None:
        raise RuntimeError("worker sampler was not initialized")

    record = _generate_record_for_game(sampler=_WORKER_SAMPLER, game_id=game_id, seed=seed, max_actions=max_actions)
    output_path = write_precomputed_actions(record)
    return (game_id, str(output_path), int(record["num_actions"]), float(record["episode_reward"]))


def _state_name(observation: Any) -> str:
    state = getattr(observation, "state", None)
    return str(getattr(state, "name", state))


def _generate_record_for_game(*, sampler: Any, game_id: str, seed: int, max_actions: int) -> dict[str, Any]:
    from re_arc.dsl import create_dsl_agent, is_terminal, resolve_action, unpack_step_result
    from re_arc.dsl.core import frame_signature

    env = sampler.make(game_id=game_id, seed=seed)
    if env is None:
        raise RuntimeError(f"failed to create environment for {game_id}")

    try:
        observation = env.reset()
        agent = create_dsl_agent(game_id)
        agent.reset_episode()
        agent.observe(observation)

        initial_signature = frame_signature(observation)
        initial_state_name = _state_name(observation)
        actions: list[tuple[int, dict[str, int]]] = []
        episode_reward = 0.0

        for _ in range(max_actions):
            if is_terminal(observation):
                break

            action_id, action_data = agent.next_action(env, observation)
            action = resolve_action(env, action_id)
            action_payload = dict(action_data or {})
            agent.record_action(action_id)

            observation, reward, done, _info = unpack_step_result(env.step(action, data=action_payload))
            episode_reward += float(reward)
            actions.append((int(action_id), action_payload))
            agent.observe(observation)
            if done:
                break

        final_state = _state_name(observation)
        if final_state != "WIN":
            raise RuntimeError(
                f"{game_id}: expected WIN while precomputing, got {final_state!r} after {len(actions)} actions."
            )

        return {
            "game_id": game_id,
            "seed": int(seed),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "generator_agent": agent.__class__.__name__,
            "initial_observation_signature": initial_signature,
            "initial_state": initial_state_name,
            "actions": actions,
            "num_actions": len(actions),
            "terminal_state": final_state,
            "episode_reward": float(episode_reward),
            "solved_levels": int(agent.solved_levels),
            "total_levels": (None if agent.total_levels is None else int(agent.total_levels)),
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            with suppress(Exception):
                close_fn()


def main() -> int:
    from re_arc.cli import _build_env_sampler, _load_config

    parser = argparse.ArgumentParser(
        description=("Generate per-game precomputed DSL actions for deterministic replay tests.")
    )
    parser.add_argument("--config", default="config.env", help="Path to config.env")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when generating action traces.")
    parser.add_argument(
        "--max-actions", type=int, default=12000, help="Maximum actions to allow per game while generating traces."
    )
    parser.add_argument(
        "--game", action="append", default=[], help="Optional game_id to process. May be provided multiple times."
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate traces even when the output file already exists."
    )
    parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes to use (default: 1).")
    parser.add_argument(
        "--sync-baseline-actions",
        action="store_true",
        help="Also sync metadata baseline_actions from deterministic replay after traces are available.",
    )
    parser.add_argument(
        "--baseline-actions-only",
        action="store_true",
        help="Skip trace generation and only sync/check metadata baseline_actions using existing traces.",
    )
    parser.add_argument(
        "--check-baseline-actions",
        action="store_true",
        help="Check metadata baseline_actions without writing changes. Exits non-zero when mismatches are found.",
    )
    args = parser.parse_args()

    if args.max_actions <= 0:
        raise ValueError("--max-actions must be > 0.")
    if int(args.jobs) <= 0:
        raise ValueError("--jobs must be > 0.")

    config = _load_config(args.config)
    sampler = _build_env_sampler(config, seed=args.seed, augment=False)

    game_ids = [str(g).strip() for g in (args.game or []) if str(g).strip()]
    if not game_ids:
        from re_arc.dsl import create_dsl_agent

        all_ids = list(sampler.game_ids)
        game_ids = []
        for gid in all_ids:
            try:
                create_dsl_agent(gid)
                game_ids.append(gid)
            except ValueError:
                print(f"[skip] {gid}: no DSL agent")

    if not game_ids:
        print("No games discovered; nothing to precompute.")
        return 1

    failures: list[str] = []
    generated = 0
    skipped = 0
    pending_game_ids: list[str] = []

    if not args.baseline_actions_only:
        for game_id in game_ids:
            output_path = precomputed_actions_path(game_id)
            if output_path.exists() and not args.force:
                skipped += 1
                print(f"[skip] {game_id}: {output_path}")
                continue
            pending_game_ids.append(game_id)

        if int(args.jobs) == 1:
            for game_id in pending_game_ids:
                try:
                    print(f"[run ] {game_id}")
                    record = _generate_record_for_game(
                        sampler=sampler, game_id=game_id, seed=args.seed, max_actions=args.max_actions
                    )
                    output_path = write_precomputed_actions(record)
                    generated += 1
                    print(
                        f"[save] {game_id}: {output_path} "
                        f"({record['num_actions']} actions, "
                        f"reward={record['episode_reward']:.6f})"
                    )
                except Exception as exc:  # pragma: no cover - best-effort tooling command
                    failures.append(f"{game_id}: {type(exc).__name__}: {exc}")
                    print(f"[fail] {game_id}: {type(exc).__name__}: {exc}")
        else:
            for game_id in pending_game_ids:
                print(f"[run ] {game_id}")

            with ProcessPoolExecutor(
                max_workers=int(args.jobs), initializer=_worker_init, initargs=(str(args.config), int(args.seed))
            ) as pool:
                future_to_game = {
                    pool.submit(_worker_run_game, game_id, int(args.seed), int(args.max_actions)): game_id
                    for game_id in pending_game_ids
                }

                for future in as_completed(future_to_game):
                    game_id = future_to_game[future]
                    try:
                        _gid, output_path_str, num_actions, episode_reward = future.result()
                        generated += 1
                        print(
                            f"[save] {game_id}: {output_path_str} ({num_actions} actions, reward={episode_reward:.6f})"
                        )
                    except Exception as exc:  # pragma: no cover - best-effort tooling command
                        failures.append(f"{game_id}: {type(exc).__name__}: {exc}")
                        print(f"[fail] {game_id}: {type(exc).__name__}: {exc}")

        print(f"Done. generated={generated} skipped={skipped} failed={len(failures)} total={len(game_ids)}")

    run_baseline_sync = bool(args.sync_baseline_actions or args.baseline_actions_only or args.check_baseline_actions)
    baseline_check_failed = False
    if run_baseline_sync:
        baseline_results, baseline_failures = sync_baseline_actions(
            config=config,
            sampler=sampler,
            game_ids=game_ids,
            seed=int(args.seed),
            check_only=bool(args.check_baseline_actions),
        )
        failures.extend(baseline_failures)
        for item in baseline_failures:
            print(f"[fail] {item}")

        changed = [result for result in baseline_results if result.changed]
        for result in baseline_results:
            status = (
                "mismatch" if result.changed and args.check_baseline_actions else "update" if result.changed else "ok"
            )
            suffix = f"baseline_actions={result.derived_actions}"
            if result.changed and args.check_baseline_actions:
                suffix = (
                    f"current={result.previous_actions} derived={result.derived_actions} path={result.metadata_path}"
                )
            elif result.changed:
                suffix = f"{suffix} path={result.metadata_path}"
            print(f"[{status}] {result.game_id}: {suffix}")

        if args.check_baseline_actions:
            print(
                f"Checked {len(baseline_results)} game(s); mismatches={len(changed)} failures={len(baseline_failures)}."
            )
            baseline_check_failed = bool(changed)
        else:
            print(
                f"Baseline sync done. updated={len(changed)} unchanged={len(baseline_results) - len(changed)} "
                f"failed={len(baseline_failures)}."
            )

    if failures:
        print("Failures:")
        for item in failures:
            print(f"- {item}")
        return 1
    if baseline_check_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
