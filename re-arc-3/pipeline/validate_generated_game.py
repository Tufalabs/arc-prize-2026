from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

GENERATED_TEST_PATHS: tuple[str, ...] = (
    "tests/test_generated_game_score_contract.py",
    "tests/test_generated_game_color_contract.py",
    "tests/test_generated_game_reset_contract.py",
)
MAX_ACTIONS = "1200"


@dataclass(frozen=True)
class GeneratedGame:
    game_id: str
    slug: str
    metadata_path: Path
    env_files: tuple[Path, ...]
    agent_path: Path


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object.")
    return payload


def _discover_game(root: Path, game_id: str) -> GeneratedGame:
    environment_root = root / "re_arc" / "environment_files"
    agents_root = root / "re_arc" / "dsl" / "agents"
    matches: list[tuple[str, Path, dict[str, object]]] = []

    for metadata_path in sorted(environment_root.glob("*/**/metadata.json")):
        payload = _load_json(metadata_path)
        if str(payload.get("game_id") or "").strip() == game_id:
            matches.append((metadata_path.relative_to(environment_root).parts[0], metadata_path, payload))

    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one metadata.json for {game_id!r}, found {len(matches)}.")

    slug, metadata_path, _payload = matches[0]
    env_dir = environment_root / slug
    env_files = tuple(sorted(path for path in env_dir.rglob("*.py") if path.is_file()))
    if not env_files:
        raise RuntimeError(f"{slug}: no generated environment Python files found under {env_dir}.")

    agent_path = agents_root / f"{slug}.py"
    if not agent_path.is_file():
        raise RuntimeError(f"{slug}: expected DSL agent at {agent_path}.")

    return GeneratedGame(
        game_id=game_id, slug=slug, metadata_path=metadata_path, env_files=env_files, agent_path=agent_path
    )


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def _run(cmd: Sequence[str], *, root: Path) -> None:
    print(f"[validate] {_format_cmd(cmd)}", flush=True)
    result = subprocess.run(list(cmd), cwd=root, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {_format_cmd(cmd)}")


def _iter_mutations_of_engine_score(source: str, *, filename: str) -> list[tuple[int, str, str]]:
    tree = ast.parse(source, filename=filename)
    violations: list[tuple[int, str, str]] = []

    def _is_forbidden_self_score_target(node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in {"_score", "_win_score"}:
                return node.attr
        return None

    def _walk_targets(node: ast.AST) -> Iterator[ast.AST]:
        if isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                yield from _walk_targets(item)
            return
        yield node

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for candidate in _walk_targets(target):
                    attr = _is_forbidden_self_score_target(candidate)
                    if attr:
                        violations.append((node.lineno, attr, "assignment"))
        elif isinstance(node, ast.AugAssign):
            attr = _is_forbidden_self_score_target(node.target)
            if attr:
                violations.append((node.lineno, attr, "augmented assignment"))
        elif isinstance(node, ast.AnnAssign):
            attr = _is_forbidden_self_score_target(node.target)
            if attr:
                violations.append((node.lineno, attr, "annotated assignment"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in {"_score", "_win_score"}
        ):
            violations.append((node.lineno, str(node.args[1].value), "setattr"))

    return violations


def _check_no_direct_engine_score_mutation(game: GeneratedGame) -> None:
    failures: list[str] = []
    for py_file in game.env_files:
        source = py_file.read_text(encoding="utf-8")
        for lineno, attr, kind in _iter_mutations_of_engine_score(source, filename=str(py_file)):
            failures.append(f"{py_file}:{lineno}: direct self.{attr} {kind}")
    if failures:
        joined = "\n".join(failures)
        raise RuntimeError(
            "Generated environment files must not mutate ARCBaseGame score fields directly. "
            "Use game-specific counters and `next_level()` for solved-level progression.\n"
            f"{joined}"
        )


def _check_baseline_actions(game: GeneratedGame) -> None:
    payload = _load_json(game.metadata_path)
    baseline_actions = payload.get("baseline_actions")
    if not isinstance(baseline_actions, list) or not baseline_actions:
        raise RuntimeError(f"{game.metadata_path}: baseline_actions must be a non-empty list of winning action counts.")
    bad_values = [
        value for value in baseline_actions if not isinstance(value, int) or isinstance(value, bool) or value <= 0
    ]
    if bad_values:
        raise RuntimeError(f"{game.metadata_path}: baseline_actions contains invalid values: {bad_values!r}.")


def _existing_test_paths(root: Path) -> list[str]:
    missing = [path for path in GENERATED_TEST_PATHS if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"Worker checkout is missing generated-game test files: {missing!r}.")
    return list(GENERATED_TEST_PATHS)


def validate_generated_game(game_id: str) -> None:
    root = _repo_root()
    game = _discover_game(root, game_id)
    files_to_lint = [str(path.relative_to(root)) for path in (*game.env_files, game.agent_path)]

    print(f"[validate] game={game.game_id} slug={game.slug}", flush=True)
    _check_no_direct_engine_score_mutation(game)
    _check_baseline_actions(game)
    _run([sys.executable, "-m", "ruff", "check", *files_to_lint, "--select", "I", "--fix"], root=root)
    _run([sys.executable, "-m", "ruff", "format", *files_to_lint], root=root)
    _run([sys.executable, "-m", "ruff", "check", *files_to_lint], root=root)
    _run([sys.executable, "-m", "pytest", "-q", *_existing_test_paths(root)], root=root)
    _run([sys.executable, "-m", "re_arc.baseline_robustness", "--game", game.game_id], root=root)
    _run(
        [
            sys.executable,
            "-m",
            "re_arc",
            "--config",
            "config.env",
            "--game",
            game.game_id,
            "--policy",
            "dsl",
            "--max-actions",
            MAX_ACTIONS,
        ],
        root=root,
    )
    print("[validate] generated game validation passed", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate one generated worker game using worker-local checks.")
    parser.add_argument("--game", required=True, help="Generated game id, for example ls20_locksmith_01-0001.")
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="Accepted for compatibility; validation still runs the worker-local runtime checks.",
    )
    args = parser.parse_args(argv)

    try:
        validate_generated_game(str(args.game))
    except Exception as exc:
        print(f"[validate] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
