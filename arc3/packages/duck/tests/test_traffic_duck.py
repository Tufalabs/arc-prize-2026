"""The traffic benchmark's duck (``duck.traffic``): the workload rides the
Kaggle bundle as a payload file, is read before the engine would load, is
found through the kernel runner's mount map, and replays only on an engine
that serves ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ullm.traffic

from duck.agent import DuckAgent
from duck.traffic import TrafficDuck

pytest.importorskip("duck.presets", reason="needs the vllm extra (linux)")

ENGINE, ENVIRONMENT = "qwen3.8-flash-next", "kaggle-rtx"


def _build(**overrides: Any) -> TrafficDuck:
    """The column's traffic duck, with ``overrides`` on top."""
    return TrafficDuck.build(engine=ENGINE, environment=ENVIRONMENT, **overrides)


def _ids(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _call(
    input_ids: np.ndarray, output_ids: np.ndarray, finished_at: float, *, opens: bool = False
) -> ullm.traffic.RecordedCall:
    return ullm.traffic.RecordedCall(
        input_ids=input_ids,
        output_ids=output_ids,
        images=(),
        finished_at=finished_at,
        continued=not opens,
        opens=opens,
        after=None,
    )


def _workload_file(path: Path) -> Path:
    """One slot's recording of three calls, cut between the first two, so the
    lane holds a history at the snapshot and two calls after it."""
    first_in, first_out = _ids(1, 2, 3, 4), _ids(5, 6)
    second_in, second_out = np.concatenate([first_in, first_out, _ids(7, 8)]), _ids(9, 10, 11)
    third_in = np.concatenate([second_in, second_out, _ids(12)])
    run = ullm.traffic.RecordedRun(
        "a",
        (
            _call(first_in, first_out, 10.0, opens=True),
            _call(second_in, second_out, 20.0),
            _call(third_in, _ids(13, 14), 30.0),
        ),
    )
    workload = ullm.traffic.build_workload([run], {}, horizon_tokens=1, snapshot=15.0)
    return workload.save(path)


def _factory(agent: Any) -> Any:
    return agent.llm_config.backend_factory


def test_build_carries_the_workload_in_the_kaggle_bundle_and_nothing_else_of_its_own(tmp_path: Path) -> None:
    """The column's engine, tier and concurrency are what is measured, so the
    benchmark's duck differs from the column's duck by the payload alone."""
    workload = _workload_file(tmp_path / "workload.b64")
    agent = _build(workload=str(workload))
    plain = DuckAgent.build(engine=ENGINE, environment=ENVIRONMENT)
    assert str(workload) in agent.run_config.kaggle_bundle_files
    assert str(workload) not in plain.run_config.kaggle_bundle_files
    assert agent.run_config.max_concurrent == plain.run_config.max_concurrent
    assert repr(_factory(agent).engine_args) == repr(_factory(plain).engine_args)


def test_prepare_reads_the_workload_before_the_engine_would_load(tmp_path: Path) -> None:
    """A missing or unnamed workload fails here, not once the engine is up."""
    workload = _workload_file(tmp_path / "workload.b64")
    _build(workload=str(workload)).prepare_for_backend()
    with pytest.raises(FileNotFoundError):
        _build(workload=str(tmp_path / "missing.b64")).prepare_for_backend()
    with pytest.raises(ValueError, match="names no workload"):
        _build().prepare_for_backend()


def test_a_kernel_finds_the_bundles_copy_through_the_runners_mount_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Kaggle the workload was named by a path that exists only where the
    deploy was staged; the copy is in the source bundle's payload, under the
    mount the runner publishes for the first attached source."""
    mount = tmp_path / "mount"
    (mount / "payload").mkdir(parents=True)
    _workload_file(mount / "payload" / "workload.b64")
    monkeypatch.chdir(tmp_path)  # the bare name resolves to nothing here
    monkeypatch.setenv("TAAF_KAGGLE_INPUT_PATHS", json.dumps({"owner/src": str(mount)}))
    monkeypatch.setenv("TAAF_KAGGLE_DATASET_SOURCES", json.dumps(["owner/src"]))
    _build(workload="workload.b64").prepare_for_backend()
    monkeypatch.setenv("TAAF_KAGGLE_INPUT_PATHS", json.dumps({"owner/src": str(tmp_path / "elsewhere")}))
    with pytest.raises(FileNotFoundError, match="payload"):
        _build(workload="workload.b64").prepare_for_backend()


def test_a_replay_refuses_an_engine_that_cannot_serve_ids(tmp_path: Path) -> None:
    """The replay serves id prompts, which only the in-process engine can."""
    agent = _build(workload=str(_workload_file(tmp_path / "workload.b64")))
    # Refused before the first await, so the coroutine is driven by hand: an
    # event loop made for it would outlive the test with its self-pipe.
    replay = agent.run_on_host(object(), None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="serves id prompts"):
        replay.send(None)
