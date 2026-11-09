"""Engine profiling: armed at start, opened in timed windows, never fatal.

Covers what can be checked without an engine -- the profiler configuration the
factory hands vLLM, the refusals, and the window schedule against a stand-in
trigger. Whether vLLM records and writes a trace is the engine's, seen on a run.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest

vllm = pytest.importorskip("vllm", reason="needs the vllm extra (linux)")

import ullm.backends.vllm_factory as vf  # noqa: E402
import ullm.backends.vllmx  # noqa: E402


def _factory(**fields: Any) -> vf.EngineFactory:
    return vf.EngineFactory(
        engine_args=vllm.AsyncEngineArgs(model="facebook/opt-125m"),
        backend_cls=ullm.backends.vllmx.VLLM,
        **fields,
    )


def test_off_arms_nothing() -> None:
    factory = _factory()
    factory._apply_profile(factory.engine_args)
    assert factory.engine_args.profiler_config is None or factory.engine_args.profiler_config.profiler is None


def test_armed_under_the_job_directory(tmp_path: Path) -> None:
    factory = _factory(profile=vf.EngineProfile(iterations=7), job_dir=tmp_path)
    factory._apply_profile(factory.engine_args)
    config = factory.engine_args.profiler_config
    assert config.profiler == "torch"
    assert config.torch_profiler_dir == str(tmp_path / "engine_profile")
    assert (tmp_path / "engine_profile").is_dir()
    assert config.max_iterations == 7, "the engine closes each window itself"
    assert config.ignore_frontend, "the front end's trace would span every window"
    assert not config.torch_profiler_with_stack and not config.torch_profiler_record_shapes


def test_a_callers_own_profiler_is_refused(tmp_path: Path) -> None:
    from vllm.config import ProfilerConfig

    factory = _factory(profile=vf.EngineProfile(), job_dir=tmp_path)
    factory.engine_args.profiler_config = ProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path))
    with pytest.raises(ValueError, match="already names a profiler"):
        factory._apply_profile(factory.engine_args)


def test_no_directory_is_refused() -> None:
    factory = _factory(profile=vf.EngineProfile())
    with pytest.raises(ValueError, match="job directory"):
        factory._apply_profile(factory.engine_args)


def _metrics(usage: float, running: float) -> Any:
    import ullm.engine_monitor as monitor

    m = monitor.EngineMetrics()
    m.scalars[monitor.KV_CACHE_USAGE] = usage
    m.scalars[monitor.REQUESTS_RUNNING] = running
    return m


async def _run(profile: vf.EngineProfile, readings: list[Any], fail_on: Collection[int] = frozenset()) -> list[int]:
    """Drive the watcher over a scripted metrics sequence; return the poll index
    at which each window opened."""
    opened: list[int] = []
    polls = {"n": 0}

    async def scenario() -> None:
        factory = _factory(profile=profile)

        async def read_metrics() -> Any:
            i = polls["n"]
            polls["n"] += 1
            if i >= len(readings):
                await asyncio.sleep(3600)
            reading = readings[i]
            if isinstance(reading, BaseException):
                raise reading
            return reading

        async def open_window() -> None:
            attempt = polls["n"] - 1
            if attempt in fail_on:
                raise RuntimeError("profiler already active")
            opened.append(attempt)

        factory._open_profile_windows(open_window, read_metrics)
        task = factory._profile_task
        assert task is not None
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        await factory._close_profile_windows()
        assert factory._profile_task is None

    await scenario()
    return opened


async def test_a_window_waits_for_the_saturated_phase() -> None:
    profile = vf.EngineProfile(not_before_s=0.0, windows=1, poll_s=0.001, spacing_s=0.0, kv_cache_above=0.85)
    readings = [_metrics(0.3, 40), _metrics(0.84, 40), _metrics(0.9, 0), _metrics(0.86, 38)]
    assert await _run(profile, readings) == [3], "below the threshold, or idle, is not the phase"


async def test_later_windows_keep_their_spacing() -> None:
    times: list[float] = []

    async def scenario() -> None:
        factory = _factory(
            profile=vf.EngineProfile(not_before_s=0.0, windows=2, poll_s=0.001, spacing_s=0.2, kv_cache_above=0.85)
        )
        loop = asyncio.get_running_loop()

        async def read_metrics() -> Any:
            return _metrics(0.9, 30)

        async def open_window() -> None:
            times.append(loop.time())

        factory._open_profile_windows(open_window, read_metrics)
        assert factory._profile_task is not None
        await asyncio.wait_for(factory._profile_task, timeout=5.0)
        await factory._close_profile_windows()

    await scenario()
    assert len(times) == 2
    assert times[1] - times[0] >= 0.2, "the second window did not wait out the spacing"


async def test_a_delay_moves_the_window_of_a_watch_already_running() -> None:
    times: list[float] = []

    async def scenario() -> None:
        factory = _factory(
            profile=vf.EngineProfile(not_before_s=0.0, windows=1, poll_s=0.001, spacing_s=0.0, kv_cache_above=0.85)
        )
        loop = asyncio.get_running_loop()
        started = loop.time()

        async def read_metrics() -> Any:
            return _metrics(0.9, 30)

        async def open_window() -> None:
            times.append(loop.time() - started)

        # The watch starts, then a caller's untimed phase moves its window on
        # (the traffic replay's decode gate): the watcher reads the delay live.
        factory._open_profile_windows(open_window, read_metrics)
        factory.delay_profile_windows(0.3)
        assert factory._profile_task is not None
        await asyncio.wait_for(factory._profile_task, timeout=5.0)
        await factory._close_profile_windows()

    await scenario()
    assert len(times) == 1 and times[0] >= 0.3, "the window opened before the delay"


async def test_a_hold_keeps_the_window_shut_through_a_phase_longer_than_its_floor() -> None:
    times: list[float] = []
    lifted: list[float] = []

    async def scenario() -> None:
        factory = _factory(
            profile=vf.EngineProfile(not_before_s=0.03, windows=1, poll_s=0.001, spacing_s=0.0, kv_cache_above=0.85)
        )
        loop = asyncio.get_running_loop()
        started = loop.time()

        async def read_metrics() -> Any:
            return _metrics(0.9, 30)

        async def open_window() -> None:
            times.append(loop.time() - started)

        # The watch is running when a caller's untimed phase starts (the
        # traffic replay's decode gate) and outlasts the window's floor: the
        # window stays shut until the hold lifts, then waits its floor again.
        factory._open_profile_windows(open_window, read_metrics)
        await asyncio.sleep(0.01)
        factory.hold_profile_windows()
        await asyncio.sleep(0.1)
        assert times == [], "a window opened inside the held phase"
        lifted.append(loop.time() - started)
        factory.delay_profile_windows(lifted[0])
        assert factory._profile_task is not None
        await asyncio.wait_for(factory._profile_task, timeout=5.0)
        await factory._close_profile_windows()

    await scenario()
    assert len(times) == 1 and times[0] >= lifted[0] + 0.03, "the window did not wait its floor past the phase"


async def test_a_failed_open_or_a_metrics_hiccup_is_not_fatal() -> None:
    profile = vf.EngineProfile(not_before_s=0.0, windows=2, poll_s=0.001, spacing_s=0.0, kv_cache_above=0.85)
    readings = [RuntimeError("scrape failed"), _metrics(0.9, 30), _metrics(0.9, 30), _metrics(0.9, 30)]
    assert await _run(profile, readings, fail_on={1}) == [2, 3]


async def test_no_metrics_means_no_windows() -> None:
    import ullm.engine_monitor as monitor

    profile = vf.EngineProfile(not_before_s=0.0, windows=1, poll_s=0.001)
    assert await _run(profile, [monitor.MetricsUnavailable("404"), _metrics(0.9, 30)]) == []


async def test_close_cancels_the_watch() -> None:
    profile = vf.EngineProfile(not_before_s=0.0, windows=1, poll_s=3600.0)
    assert await _run(profile, [_metrics(0.9, 30)]) == []


async def test_traces_are_packed_once_written_and_again_at_close(tmp_path: Path) -> None:
    """One archive beside the directory, rebuilt after each window's trace has
    stopped growing, and once more at shutdown."""
    import tarfile

    directory = tmp_path / "engine_profile"
    profile = vf.EngineProfile(not_before_s=0.0, windows=1, poll_s=0.001, spacing_s=0.0, directory=str(directory))

    async def scenario() -> None:
        factory = _factory(profile=profile)
        factory._apply_profile(factory.engine_args)

        async def read_metrics() -> Any:
            return _metrics(0.9, 30)

        async def open_window() -> None:
            # The engine writes the trace later, from its own process.
            async def write_later() -> None:
                await asyncio.sleep(0.05)
                (directory / "rank0.1.pt.trace.json.gz").write_bytes(b"x" * 1000)

            asyncio.get_running_loop().create_task(write_later())

        factory._open_profile_windows(open_window, read_metrics)
        assert factory._profile_task is not None
        await asyncio.wait_for(factory._profile_task, timeout=30.0)
        with tarfile.open(tmp_path / "engine_profile.tar") as archive:
            assert archive.getnames() == ["engine_profile", "engine_profile/rank0.1.pt.trace.json.gz"]

        # A second trace landing after the watcher is done still reaches the archive.
        (directory / "rank0.2.pt.trace.json.gz").write_bytes(b"y" * 10)
        await factory._close_profile_windows()
        with tarfile.open(tmp_path / "engine_profile.tar") as archive:
            assert "engine_profile/rank0.2.pt.trace.json.gz" in archive.getnames()
        assert not (tmp_path / "engine_profile.tar.partial").exists()

    await scenario()


async def test_no_traces_no_archive(tmp_path: Path) -> None:
    factory = _factory(profile=vf.EngineProfile(directory=str(tmp_path / "engine_profile")))
    factory._apply_profile(factory.engine_args)
    await factory._close_profile_windows()
    assert not (tmp_path / "engine_profile.tar").exists()


async def test_an_archive_failure_does_not_end_the_watch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Packing is a convenience; a tar that cannot be written must not cost
    the windows still to come."""
    directory = tmp_path / "engine_profile"
    profile = vf.EngineProfile(not_before_s=0.0, windows=2, poll_s=0.001, spacing_s=0.0, directory=str(directory))

    def broken_bundle(self: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(vf.LocalEngineFields, "_bundle_profile", broken_bundle)
    opened = 0

    async def scenario() -> None:
        nonlocal opened
        factory = _factory(profile=profile)
        factory._apply_profile(factory.engine_args)

        async def read_metrics() -> Any:
            return _metrics(0.9, 30)

        async def open_window() -> None:
            nonlocal opened
            opened += 1
            trace = directory / f"rank0.{opened}.pt.trace.json.gz"

            async def write_later() -> None:
                await asyncio.sleep(0.05)
                trace.write_bytes(b"x" * 100)

            asyncio.get_running_loop().create_task(write_later())

        factory._open_profile_windows(open_window, read_metrics)
        assert factory._profile_task is not None
        await asyncio.wait_for(factory._profile_task, timeout=30.0)
        await factory._close_profile_windows()

    await scenario()
    assert opened == 2, "the second window was still watched for after the first archive failed"
    assert not (tmp_path / "engine_profile.tar").exists()


async def test_no_window_before_the_floor_whatever_the_pool_says() -> None:
    """A small pool can cross its threshold on the first wave of prompts, while
    kernels still compile; the floor keeps that out of every trace."""
    times: list[float] = []

    async def scenario() -> None:
        factory = _factory(profile=vf.EngineProfile(windows=1, poll_s=0.001, not_before_s=0.2))
        loop = asyncio.get_running_loop()
        start = loop.time()

        async def read_metrics() -> Any:
            return _metrics(0.99, 15)

        async def open_window() -> None:
            times.append(loop.time() - start)

        factory._open_profile_windows(open_window, read_metrics)
        assert factory._profile_task is not None
        await asyncio.wait_for(factory._profile_task, timeout=5.0)
        await factory._close_profile_windows()

    await scenario()
    assert len(times) == 1 and times[0] >= 0.2


def test_the_real_backend_carries_the_profile_delay_and_drops_it_when_pickled() -> None:
    from test_response_injection import TOK, _FakeEngine

    # The backend is frozen once built: the delay is a declared field the
    # starting factory assigns, not a new attribute (which the freeze refuses).
    backend = ullm.backends.vllmx.VLLM(engine=_FakeEngine(TOK))  # type: ignore[arg-type]
    delays: list[float] = []
    backend.delay_profile_windows = delays.append
    backend.hold_profile_windows = lambda: delays.append(0.0)
    assert backend.delay_profile_windows is not None and backend.hold_profile_windows is not None
    backend.hold_profile_windows()
    backend.delay_profile_windows(5.0)
    assert delays == [0.0, 5.0]
    state = backend.__getstate__()
    assert state["delay_profile_windows"] is None and state["hold_profile_windows"] is None
