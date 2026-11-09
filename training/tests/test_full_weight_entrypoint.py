from __future__ import annotations

from types import SimpleNamespace

import pytest

from training import full_weight_entrypoint
from training.policy_training_backend import FullWeightPolicyTrainingBackend


class _Runtime:
    def __init__(self, *, control_group, events: list[object]) -> None:  # noqa: ANN001
        events.append(("runtime", control_group))
        self.events = events
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.events.append("close")
            self.closed = True

    def serve(self) -> None:
        self.events.append("serve")


def _install_distributed_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int,
    weight_mode: str = "full_weight",
) -> list[object]:
    events: list[object] = []
    config = SimpleNamespace(
        training=SimpleNamespace(
            weight_mode=weight_mode,
            full_weight=SimpleNamespace(device_ids=[2, 3]),
        ),
    )
    monkeypatch.setattr(
        full_weight_entrypoint.ExperimentConfig,
        "load",
        lambda _path: config,
    )
    monkeypatch.setattr(
        full_weight_entrypoint.torch.cuda,
        "set_device",
        lambda device: events.append(("set_device", device)),
    )
    monkeypatch.setattr(
        full_weight_entrypoint.dist,
        "init_process_group",
        lambda backend, **kwargs: events.append(("init", backend, kwargs)),
    )
    monkeypatch.setattr(
        full_weight_entrypoint.dist,
        "new_group",
        lambda **kwargs: events.append(("new_group", kwargs)) or "control",
    )
    monkeypatch.setattr(full_weight_entrypoint.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(
        full_weight_entrypoint.dist,
        "barrier",
        lambda **kwargs: events.append(("barrier", kwargs)),
    )
    monkeypatch.setattr(
        full_weight_entrypoint.dist,
        "destroy_process_group",
        lambda: events.append("destroy"),
    )
    monkeypatch.setattr(
        full_weight_entrypoint,
        "FullWeightDistributedRuntime",
        lambda **kwargs: _Runtime(events=events, **kwargs),
    )
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    return events


def test_rank_zero_injects_backend_and_closes_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_distributed_fakes(monkeypatch, rank=0)

    def fake_run_training(config_path, *, backend_factory) -> None:  # noqa: ANN001
        events.append(("run_training", config_path))
        backend = backend_factory(SimpleNamespace())
        assert isinstance(backend, FullWeightPolicyTrainingBackend)

    monkeypatch.setattr(full_weight_entrypoint, "run_training", fake_run_training)

    full_weight_entrypoint.run("config.json")

    assert events[:4] == [
        ("set_device", 0),
        ("init", "nccl", {"timeout": full_weight_entrypoint._NCCL_TIMEOUT}),
        (
            "new_group",
            {"backend": "gloo", "timeout": full_weight_entrypoint._CONTROL_TIMEOUT},
        ),
        ("runtime", "control"),
    ]
    assert events[4:] == [
        ("run_training", "config.json"),
        "close",
        ("barrier", {"group": "control"}),
        "destroy",
    ]


def test_nonzero_rank_only_serves_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _install_distributed_fakes(monkeypatch, rank=1)
    monkeypatch.setattr(
        full_weight_entrypoint,
        "run_training",
        lambda *_args, **_kwargs: pytest.fail("worker rank entered the CLI"),
    )

    full_weight_entrypoint.run("config.json")

    assert "serve" in events
    assert "close" not in events
    assert events[-2:] == [("barrier", {"group": "control"}), "destroy"]


def test_failed_rank_does_not_enter_cleanup_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_distributed_fakes(monkeypatch, rank=0)

    def fail(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        raise RuntimeError("training failed")

    monkeypatch.setattr(full_weight_entrypoint, "run_training", fail)

    with pytest.raises(RuntimeError, match="training failed"):
        full_weight_entrypoint.run("config.json")

    assert not any(
        event == "destroy" or (isinstance(event, tuple) and event[0] == "barrier")
        for event in events
    )


def test_failed_final_barrier_does_not_destroy_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_distributed_fakes(monkeypatch, rank=0)
    monkeypatch.setattr(
        full_weight_entrypoint,
        "run_training",
        lambda *_args, **_kwargs: None,
    )

    def fail_barrier(**kwargs) -> None:  # noqa: ANN003
        events.append(("barrier", kwargs))
        raise RuntimeError("peer failed")

    monkeypatch.setattr(full_weight_entrypoint.dist, "barrier", fail_barrier)

    with pytest.raises(RuntimeError, match="peer failed"):
        full_weight_entrypoint.run("config.json")

    assert "destroy" not in events


@pytest.mark.parametrize("requested_mode", ["full_weight", "flash"])
def test_entrypoint_rejects_lora_before_distributed_init(
    monkeypatch: pytest.MonkeyPatch, requested_mode: str,
) -> None:
    events = _install_distributed_fakes(
        monkeypatch,
        rank=0,
        weight_mode="lora",
    )

    with pytest.raises(RuntimeError, match=f"requires training.weight_mode={requested_mode!r}"):
        full_weight_entrypoint.run("config.json", weight_mode=requested_mode)

    assert events == []


@pytest.mark.parametrize(
    ("name", "value"),
    [("WORLD_SIZE", "1"), ("LOCAL_WORLD_SIZE", "1"), ("LOCAL_RANK", "2")],
)
def test_entrypoint_requires_configured_single_node_topology(monkeypatch, name, value):
    events = _install_distributed_fakes(monkeypatch, rank=0)
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="topology must match"):
        full_weight_entrypoint.run("config.json")
    assert events == []


def test_rank_zero_defers_sigterm_to_a_safe_boundary_and_unwinds(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A real TERM must not kill rank zero mid-run: the CLI's own boundary ends it."""
    import os
    import signal

    from training.shutdown import raise_if_shutdown_requested

    events = _install_distributed_fakes(monkeypatch, rank=0)
    marker = tmp_path / "cleanup-ran"

    def run_training_under_term(config_path, *, backend_factory) -> None:  # noqa: ANN001
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            # The handler only records the request; the safe boundary raises.
            events.append("survived-term")
            raise_if_shutdown_requested()
            pytest.fail("the deferred shutdown was not raised at the boundary")
        finally:
            marker.touch()

    monkeypatch.setattr(full_weight_entrypoint, "run_training", run_training_under_term)

    assert full_weight_entrypoint.run("config.json") == 0

    assert marker.exists()
    assert events[4:] == [
        "survived-term",
        "close",
        ("barrier", {"group": "control"}),
        "destroy",
    ]
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_rank_zero_segment_yield_returns_the_continuation_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training.shutdown import SEGMENT_YIELD_EXIT_CODE, SegmentYield

    events = _install_distributed_fakes(monkeypatch, rank=0)

    def yield_segment(config_path, *, backend_factory) -> None:  # noqa: ANN001
        raise SegmentYield("segment budget")

    monkeypatch.setattr(full_weight_entrypoint, "run_training", yield_segment)
    monkeypatch.setattr(full_weight_entrypoint.sys, "argv", ["x", "--config", "c.json"])

    assert full_weight_entrypoint.run("config.json") == SEGMENT_YIELD_EXIT_CODE
    assert events[4:] == ["close", ("barrier", {"group": "control"}), "destroy"]
    with pytest.raises(SystemExit) as exit_info:
        full_weight_entrypoint.main()
    assert exit_info.value.code == SEGMENT_YIELD_EXIT_CODE


def test_peer_rank_keeps_serving_through_sigterm_until_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import signal

    events = _install_distributed_fakes(monkeypatch, rank=1)

    def serve_under_term(self) -> None:  # noqa: ANN001
        os.kill(os.getpid(), signal.SIGTERM)
        # Still alive: rank zero's EXIT command ends this loop, not the TERM.
        self.events.append("serve-after-term")

    monkeypatch.setattr(_Runtime, "serve", serve_under_term)
    monkeypatch.setattr(
        full_weight_entrypoint,
        "run_training",
        lambda *_args, **_kwargs: pytest.fail("worker rank entered the CLI"),
    )

    assert full_weight_entrypoint.run("config.json") == 0
    assert events[4:] == [
        "serve-after-term",
        ("barrier", {"group": "control"}),
        "destroy",
    ]
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
