from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from pipeline.codex_batch import BatchJob, main, run_batch


def test_run_batch_writes_prompt_logs_and_metadata(tmp_path: Path) -> None:
    jobs = [
        {"game": "alternating_helpers", "folder": "re_arc/environment_files/alternating_helpers/0001"},
        {"game": "snake", "folder": "re_arc/environment_files/snake/0001"},
    ]
    prompt_template = "Update {game} inside {folder}."
    captured_calls: list[dict[str, object]] = []

    def fake_runner(
        cmd: Sequence[str], cwd: Path, log_path: Path, output_path: Path
    ) -> subprocess.CompletedProcess[str]:
        captured_calls.append(
            {"cmd": list(cmd), "cwd": str(cwd), "log_path": str(log_path), "output_path": str(output_path)}
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("codex log\n", encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("done\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "done\n", "")

    results = run_batch(
        jobs=[BatchJob(index=index, variables=job) for index, job in enumerate(jobs, start=1)],
        prompt_template=prompt_template,
        output_dir=tmp_path / "logs",
        base_dir=tmp_path,
        max_workers=1,
        codex_runner=fake_runner,
    )

    assert len(results) == 2
    assert [result["game"] for result in results] == ["alternating_helpers", "snake"]

    first_prompt = (tmp_path / "logs" / "001-alternating_helpers" / "prompt.txt").read_text(encoding="utf-8")
    assert first_prompt == "Update alternating_helpers inside re_arc/environment_files/alternating_helpers/0001."

    first_metadata = json.loads(
        (tmp_path / "logs" / "001-alternating_helpers" / "run.json").read_text(encoding="utf-8")
    )
    assert first_metadata["status"] == "completed"
    assert first_metadata["exit_code"] == 0
    assert first_metadata["folder"] == str(
        (tmp_path / "re_arc" / "environment_files" / "alternating_helpers" / "0001").resolve()
    )

    assert captured_calls[0]["cwd"] == str(
        (tmp_path / "re_arc" / "environment_files" / "alternating_helpers" / "0001").resolve()
    )
    assert captured_calls[1]["cwd"] == str((tmp_path / "re_arc" / "environment_files" / "snake" / "0001").resolve())


def test_main_loads_csv_jobs_and_writes_summary(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    jobs_path = tmp_path / "jobs.csv"
    jobs_path.write_text(
        "game,folder\nalternating_helpers,re_arc/environment_files/alternating_helpers/0001\n", encoding="utf-8"
    )
    output_dir = tmp_path / "batch-output"
    captured: dict[str, object] = {}

    def fake_run_batch(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return [
            {
                "job_index": 1,
                "game": "alternating_helpers",
                "folder": str((tmp_path / "re_arc" / "environment_files" / "alternating_helpers" / "0001").resolve()),
                "exit_code": 0,
                "status": "completed",
            }
        ]

    monkeypatch.setattr("pipeline.codex_batch.run_batch", fake_run_batch)

    assert (
        main(
            [
                "--jobs",
                str(jobs_path),
                "--prompt",
                "Fix {game} in {folder}.",
                "--output-dir",
                str(output_dir),
                "--base-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert captured["prompt_template"] == "Fix {game} in {folder}."
    assert captured["base_dir"] == tmp_path.resolve()
    assert captured["max_workers"] == 1
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["job_count"] == 1
    assert summary["jobs"][0]["game"] == "alternating_helpers"


def test_run_batch_stops_after_failure_by_default(tmp_path: Path) -> None:
    jobs = [
        BatchJob(index=1, variables={"game": "alternating_helpers", "folder": "game-a"}),
        BatchJob(index=2, variables={"game": "snake", "folder": "game-b"}),
    ]
    calls: list[str] = []

    def fake_runner(
        cmd: Sequence[str], cwd: Path, log_path: Path, output_path: Path
    ) -> subprocess.CompletedProcess[str]:
        calls.append(str(cwd))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("failed\n", encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 1, "", "failed\n")

    with pytest.raises(RuntimeError, match="Codex failed for game"):
        run_batch(
            jobs=jobs,
            prompt_template="Fix {game}.",
            output_dir=tmp_path / "logs",
            base_dir=tmp_path,
            max_workers=1,
            codex_runner=fake_runner,
        )

    assert calls == [str((tmp_path / "game-a").resolve())]


def test_main_requires_exactly_one_prompt_source(tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.csv"
    jobs_path.write_text(
        "game,folder\nalternating_helpers,re_arc/environment_files/alternating_helpers/0001\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="exactly one of --prompt or --prompt-file"):
        main(["--jobs", str(jobs_path), "--prompt", "x", "--prompt-file", "prompt.txt"])


def test_main_rejects_non_csv_jobs_file(tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Use \\.csv"):
        main(["--jobs", str(jobs_path), "--prompt", "Fix {game}."])


def test_run_batch_parallel_mode_runs_all_jobs_and_keeps_order(tmp_path: Path) -> None:
    jobs = [
        BatchJob(index=1, variables={"game": "one", "folder": "game-a"}),
        BatchJob(index=2, variables={"game": "two", "folder": "game-b"}),
    ]
    started: list[str] = []
    release = threading.Event()

    def fake_runner(
        cmd: Sequence[str], cwd: Path, log_path: Path, output_path: Path
    ) -> subprocess.CompletedProcess[str]:
        started.append(cwd.name)
        release.wait(timeout=1.0)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("done\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "done\n", "")

    worker = threading.Thread(target=release.set, daemon=True)
    worker.start()
    time.sleep(0.01)

    results = run_batch(
        jobs=jobs,
        prompt_template="Fix {game}.",
        output_dir=tmp_path / "logs",
        base_dir=tmp_path,
        max_workers=2,
        codex_runner=fake_runner,
    )

    assert sorted(started) == ["game-a", "game-b"]
    assert [result["job_index"] for result in results] == [1, 2]


def test_main_passes_max_workers_to_run_batch(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    jobs_path = tmp_path / "jobs.csv"
    jobs_path.write_text("game,folder\nalternating_helpers,game-a\n", encoding="utf-8")
    output_dir = tmp_path / "batch-output"
    captured: dict[str, object] = {}

    def fake_run_batch(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return [{"job_index": 1, "game": "alternating_helpers", "folder": str(tmp_path / "game-a"), "exit_code": 0}]

    monkeypatch.setattr("pipeline.codex_batch.run_batch", fake_run_batch)

    assert (
        main(
            [
                "--jobs",
                str(jobs_path),
                "--prompt",
                "Fix {game}.",
                "--output-dir",
                str(output_dir),
                "--base-dir",
                str(tmp_path),
                "--max-workers",
                "3",
            ]
        )
        == 0
    )

    assert captured["max_workers"] == 3


def test_run_batch_writes_progress_bar(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = [BatchJob(index=1, variables={"game": "one", "folder": "game-a"})]

    def fake_runner(
        cmd: Sequence[str], _cwd: Path, log_path: Path, output_path: Path
    ) -> subprocess.CompletedProcess[str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("done\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(cmd), 0, "done\n", "")

    run_batch(
        jobs=jobs,
        prompt_template="Fix {game}.",
        output_dir=tmp_path / "logs",
        base_dir=tmp_path,
        max_workers=1,
        codex_runner=fake_runner,
    )

    assert "0/1" in capsys.readouterr().err
