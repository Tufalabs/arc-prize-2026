from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "prepare_profile"
DEFAULT_PREPARE_PYTEST_ARGS = ["-q", "--ignore=tests/test_reset_restores_initial_frame.py"]

TEST_FILE_EXPLANATIONS = {
    "tests/test_augmentation_config.py": (
        "Checks augmentation config parsing, validation, defaults, and env wrapper behavior."
    ),
    "tests/test_cli_augmentation_config.py": "Checks CLI handling of augmentation flags and config overrides.",
    "tests/test_cli_replay.py": "Checks replay/GIF CLI flows, replay JSON writing, and error handling.",
    "tests/test_codex_batch.py": "Checks batch Codex job loading, prompting, output handling, and progress display.",
    "tests/test_dsl_agents_all_games.py": (
        "Runs every DSL agent and verifies games can be solved within replay and timing constraints."
    ),
    "tests/test_env_sampler_datasets.py": (
        "Checks dataset selection and validation in EnvSampler and CLI dataset flags."
    ),
    "tests/test_env_sampler_tags.py": "Checks tag filters and tag validation for game sampling.",
    "tests/test_environment_color_contract.py": (
        "Samples every game and checks rendered color values stay inside the supported palette."
    ),
    "tests/test_environment_score_contract.py": (
        "Checks reward/score contracts, solved-level progression, and forbidden score mutations."
    ),
    "tests/test_identify_the_agent.py": "Checks behavior and solver coverage for the identify-the-agent environment.",
    "tests/test_pipeline_cli.py": (
        "Checks pipeline CLI config resolution, local/slurm execution wiring, and result ordering."
    ),
    "tests/test_pipeline_config.py": "Checks pipeline run config parsing and defaults.",
    "tests/test_pipeline_idea_loading.py": "Checks game idea JSON loading, slug generation, and validation errors.",
    "tests/test_pipeline_implementation_stage.py": (
        "Checks worker package creation, implementation runners, artifacts, and failure status updates."
    ),
    "tests/test_pipeline_review.py": (
        "Checks replay review manifest generation, copy/import behavior, and review validation."
    ),
    "tests/test_pipeline_smoke.py": (
        "Runs small end-to-end pipeline smoke tests with fake spec, implementation, and artifact runners."
    ),
    "tests/test_random_action_smoke.py": "Optional random-action smoke test for environment robustness when enabled.",
    "tests/test_random_unsolvable_stress.py": (
        "Optional random stress test that looks for unsolvable or crashing generated states when enabled."
    ),
    "tests/test_reset_restores_initial_frame.py": "Checks every game resets back to its initial frame after exercise.",
    "tests/test_reward_env.py": "Checks transition reward wrapper behavior and episode reward accounting.",
    "tests/test_route_games_no_index_crash.py": (
        "Checks route-style games do not crash with representative indexing paths."
    ),
}


@dataclass(frozen=True)
class PhaseResult:
    name: str
    command: list[str]
    duration_seconds: float
    returncode: int


class PytestDurationPlugin:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._outcomes: dict[str, str] = {}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None) -> Any:
        del nextitem
        started = time.perf_counter()
        outcome = yield
        outcome.get_result()
        duration = time.perf_counter() - started
        path = str(Path(str(item.path)).relative_to(ROOT))
        self.items.append(
            {
                "nodeid": item.nodeid,
                "path": path,
                "duration_seconds": duration,
                "outcome": self._outcomes.get(item.nodeid, "unknown"),
            }
        )

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call":
            self._outcomes[report.nodeid] = report.outcome
        elif report.when == "setup" and report.skipped:
            self._outcomes[report.nodeid] = "skipped"


def _format_duration(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _run_phase(name: str, command: list[str]) -> PhaseResult:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, check=False)
    duration = time.perf_counter() - started
    return PhaseResult(name=name, command=command, duration_seconds=duration, returncode=result.returncode)


def _run_pytest() -> tuple[PhaseResult, list[dict[str, Any]]]:
    plugin = PytestDurationPlugin()
    started = time.perf_counter()
    returncode = pytest.main(DEFAULT_PREPARE_PYTEST_ARGS, plugins=[plugin])
    duration = time.perf_counter() - started
    phase = PhaseResult(
        name="pytest",
        command=[str(PYTHON), "-m", "pytest", *DEFAULT_PREPARE_PYTEST_ARGS],
        duration_seconds=duration,
        returncode=int(returncode),
    )
    return phase, plugin.items


def _test_file_rows(items: list[dict[str, Any]], pytest_duration: float) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "duration_seconds": 0.0})
    for item in items:
        row = by_path[str(item["path"])]
        row["count"] += 1
        row["duration_seconds"] += float(item["duration_seconds"])

    measured_item_total = sum(float(item["duration_seconds"]) for item in items)
    rows = [
        {
            "path": path,
            "count": row["count"],
            "duration_seconds": row["duration_seconds"],
            "explanation": TEST_FILE_EXPLANATIONS.get(path, "No explanation recorded."),
        }
        for path, row in by_path.items()
    ]
    rows.sort(key=lambda row: float(row["duration_seconds"]), reverse=True)
    rows.append(
        {
            "path": "pytest collection/session overhead",
            "count": 0,
            "duration_seconds": max(0.0, pytest_duration - measured_item_total),
            "explanation": "Time spent collecting tests, loading modules/plugins, and pytest session setup/teardown.",
        }
    )
    return rows


def _write_csv(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["nodeid", "path", "outcome", "duration_seconds"])
        writer.writeheader()
        for item in sorted(items, key=lambda row: float(row["duration_seconds"]), reverse=True):
            writer.writerow(
                {
                    "nodeid": item["nodeid"],
                    "path": item["path"],
                    "outcome": item["outcome"],
                    "duration_seconds": f"{float(item['duration_seconds']):.6f}",
                }
            )


def _write_markdown(
    path: Path,
    *,
    phases: list[PhaseResult],
    test_file_rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    prepare_total = sum(phase.duration_seconds for phase in phases)
    pytest_total = next(phase.duration_seconds for phase in phases if phase.name == "pytest")
    test_file_total = sum(float(row["duration_seconds"]) for row in test_file_rows)
    lines = [
        "# Prepare Profile",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Phase Timings",
        "",
        "| Phase | Duration | Command |",
        "| --- | ---: | --- |",
    ]
    for phase in phases:
        command = " ".join(phase.command)
        lines.append(f"| {phase.name} | {_format_duration(phase.duration_seconds)} | `{command}` |")
    lines.extend(
        [
            f"| **prepare total** | **{_format_duration(prepare_total)}** | Sum of measured phases |",
            "",
            "## Test File Timings",
            "",
            f"These rows sum to the measured pytest phase: {_format_duration(test_file_total)} "
            f"(pytest phase: {_format_duration(pytest_total)}).",
            "",
            "| Test File | Tests | Duration | What It Checks |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in test_file_rows:
        lines.append(
            f"| `{row['path']}` | {row['count']} | {_format_duration(float(row['duration_seconds']))} | "
            f"{row['explanation']} |"
        )
    lines.extend(
        [
            "",
            "## Slowest Individual Tests",
            "",
            f"Full per-test CSV: `{csv_path.relative_to(ROOT)}`",
            f"Machine-readable report: `{json_path.relative_to(ROOT)}`",
            "",
            "| Test | Outcome | Duration |",
            "| --- | --- | ---: |",
        ]
    )
    for item in sorted(items, key=lambda row: float(row["duration_seconds"]), reverse=True)[:40]:
        lines.append(
            f"| `{item['nodeid']}` | {item['outcome']} | {_format_duration(float(item['duration_seconds']))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the same phases run by `make prepare`.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    markdown_path = output_dir / f"prepare_profile_{stamp}.md"
    json_path = output_dir / f"prepare_profile_{stamp}.json"
    csv_path = output_dir / f"prepare_profile_{stamp}.tests.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = [
        _run_phase(
            "uv sync", ["uv", "sync", "--python", str(PYTHON), "--locked", "--extra", "dev", "--no-install-project"]
        ),
        _run_phase(
            "ruff import fix",
            [
                str(PYTHON),
                "-m",
                "ruff",
                "check",
                "re_arc/",
                "pipeline/",
                "scripts/",
                "tests/",
                "--select",
                "I",
                "--fix",
            ],
        ),
        _run_phase("ruff format", [str(PYTHON), "-m", "ruff", "format", "re_arc/", "pipeline/", "scripts/", "tests/"]),
        _run_phase("ruff check", [str(PYTHON), "-m", "ruff", "check", "re_arc/", "pipeline/", "scripts/", "tests/"]),
        _run_phase("mypy", [str(PYTHON), "-m", "mypy", "re_arc/", "pipeline/"]),
    ]
    pytest_phase, items = _run_pytest()
    phases.append(pytest_phase)

    test_file_rows = _test_file_rows(items, pytest_phase.duration_seconds)
    _write_csv(csv_path, items)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "prepare_total_seconds": sum(phase.duration_seconds for phase in phases),
        "phases": [phase.__dict__ for phase in phases],
        "test_files": test_file_rows,
        "tests": items,
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(
        markdown_path, phases=phases, test_file_rows=test_file_rows, items=items, csv_path=csv_path, json_path=json_path
    )

    print(f"Wrote prepare profile: {markdown_path.relative_to(ROOT)}")
    print(f"Wrote per-test CSV: {csv_path.relative_to(ROOT)}")
    return 0 if all(phase.returncode == 0 for phase in phases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
