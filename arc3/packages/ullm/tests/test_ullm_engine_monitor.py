"""Engine-status monitor: line construction, prometheus-text parsing,
the monitor loop's lifecycle, and the line saying how an engine process
ended."""

import asyncio
import multiprocessing
import os
import re
import signal
import time
import types
from typing import Optional

import pytest
from ullm import engine_monitor
from ullm.engine_monitor import (
    KV_GPU_PARKED_SHARE,
    KV_REQUESTS_SHARE,
    EngineMetrics,
    EngineStatusMonitor,
    MetricsUnavailable,
    format_status_line,
    metrics_from_prometheus_text,
)


def _snapshot(
    *,
    running: float = 0,
    waiting: float = 0,
    kv: float = 0.0,
    gen: float = 0,
    prompt: float = 0,
    prompt_cached: float = 0,
    prefix_q: float = 0,
    prefix_h: float = 0,
    preempt: float = 0,
    reasons: Optional[dict[str, float]] = None,
) -> EngineMetrics:
    return EngineMetrics(
        scalars={
            "vllm:num_requests_running": running,
            "vllm:num_requests_waiting": waiting,
            "vllm:kv_cache_usage_perc": kv,
            "vllm:generation_tokens": gen,
            "vllm:prompt_tokens": prompt,
            "vllm:prompt_tokens_cached": prompt_cached,
            "vllm:prefix_cache_queries": prefix_q,
            "vllm:prefix_cache_hits": prefix_h,
            "vllm:num_preemptions": preempt,
        },
        waiting_by_reason=reasons or {},
    )


def test_format_line_interval_math() -> None:
    previous = _snapshot(gen=1000, prompt=5000, prefix_q=100, prefix_h=90, preempt=1)
    current = _snapshot(running=12, waiting=0, kv=0.372, gen=7000, prompt=65000, prefix_q=300, prefix_h=140, preempt=1)
    line = format_status_line(current, previous, dt=60.0)
    assert line is not None
    # Interval figures use deltas, not cumulative counters.
    assert "gen=100tok/s" in line
    assert "prompt=1000tok/s" in line
    # Interval hit rate: (140-90)/(300-100), NOT 140/300.
    assert "prefix_hit=25.0%" in line
    assert "preempt=0" in line
    # Gauges are instantaneous; kv usage is a 0..1 fraction rendered as %.
    assert "running=12" in line
    assert "kv_cache=37.2%" in line


def test_format_line_separates_computed_prompt_tokens() -> None:
    previous = _snapshot(prompt=5000, prompt_cached=4000)
    current = _snapshot(prompt=65000, prompt_cached=52000)
    line = format_status_line(current, previous, dt=60.0)
    assert line is not None
    # prompt= stays vLLM's total (1000/s), new= is what the GPU computed:
    # (60000 - 48000)/60 = 200/s. A monitor reporting only the total reads
    # 5x the real prefill at this hit rate.
    assert "prompt=1000tok/s(new=200)" in line
    # Nothing cached -> the two coincide, so the suffix is not printed.
    nothing_cached = format_status_line(_snapshot(prompt=65000), _snapshot(prompt=5000), dt=60.0)
    assert nothing_cached is not None and "prompt=1000tok/s " in nothing_cached + " "
    assert "new=" not in nothing_cached


def test_format_line_reports_draft_acceptance_only_with_a_drafter() -> None:
    """Interval acceptance: (accepted delta)/(draft-token delta), and accepted tokens per draft."""
    previous = _snapshot(gen=1000)
    current = _snapshot(gen=7000)
    for snap, drafts, draft_tokens, accepted in ((previous, 100, 200, 100), (current, 400, 800, 460)):
        snap.scalars["vllm:spec_decode_num_drafts"] = drafts
        snap.scalars["vllm:spec_decode_num_draft_tokens"] = draft_tokens
        snap.scalars["vllm:spec_decode_num_accepted_tokens"] = accepted
    line = format_status_line(current, previous, dt=60.0)
    assert line is not None
    # (460-100)/(800-200) accepted; (460-100)/(400-100) per draft.
    assert "accept=60.0%(1.20/draft)" in line
    # Per position: of the 300 drafts in the interval, the first token was
    # accepted in 240 and the second in 120.
    previous.accepted_by_position.update({"0": 50, "1": 50})
    current.accepted_by_position.update({"0": 290, "1": 170})
    by_pos = format_status_line(current, previous, dt=60.0)
    assert by_pos is not None and "accept=60.0%(1.20/draft,pos 80%/40%)" in by_pos
    plain = format_status_line(_snapshot(gen=7000), _snapshot(gen=1000), dt=60.0)
    assert plain is not None and "accept=" not in plain


def test_format_line_reports_block_lifetime_means() -> None:
    previous = _snapshot()
    previous.scalars.update(
        {
            "vllm:kv_block_idle_before_evict_seconds_sum": 100.0,
            "vllm:kv_block_idle_before_evict_seconds_count": 10.0,
        }
    )
    current = _snapshot()
    current.scalars.update(
        {
            # 40 more blocks evicted, 800 s of idle time between them -> 20 s mean
            # for the interval, NOT the 22.5 s lifetime average.
            "vllm:kv_block_idle_before_evict_seconds_sum": 900.0,
            "vllm:kv_block_idle_before_evict_seconds_count": 50.0,
        }
    )
    line = format_status_line(current, previous, dt=60.0)
    assert line is not None and "idle_evict=20.0s" in line
    # An engine without kv_cache_metrics exports neither histogram; the fields
    # are then absent rather than reported as zero.
    plain = format_status_line(_snapshot(running=1), _snapshot(), dt=60.0)
    assert plain is not None and "idle_evict" not in plain and "reuse_gap" not in plain


def test_format_line_first_tick_and_reasons() -> None:
    current = _snapshot(
        running=30, waiting=5, kv=0.4, gen=1234, prefix_q=10, prefix_h=5, reasons={"capacity": 4.0, "deferred": 1.0}
    )
    line = format_status_line(current, None, None)
    assert line is not None
    # First tick: cumulative label, totals not rates.
    assert "cum:" in line
    assert "gen=1234tok" in line and "tok/s" not in line
    # Waiting-reason breakdown surfaces (the 40%-ceiling diagnosis signal).
    assert "waiting=5(capacity:4,deferred:1)" in line
    assert format_status_line(EngineMetrics(), None, None) is None


def test_prometheus_text_parsing() -> None:
    # Build the exposition with the real library — the format a vLLM
    # /metrics endpoint actually serves (counters expose `_total` samples
    # plus a separate `_created` gauge family) rather than a hand-written
    # approximation of it.
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

    registry = CollectorRegistry()
    running = Gauge("vllm:num_requests_running", "d", ["engine"], registry=registry)
    running.labels(engine="0").set(3)
    running.labels(engine="1").set(2)
    queries = Counter("vllm:prefix_cache_queries", "d", ["engine"], registry=registry)
    queries.labels(engine="0").inc(500)
    by_reason = Gauge("vllm:num_requests_waiting_by_reason", "d", ["engine", "reason"], registry=registry)
    by_reason.labels(engine="0", reason="capacity").set(7)
    queue_time = Histogram("vllm:request_queue_time_seconds", "d", registry=registry)
    queue_time.observe(0.5)
    Gauge("other_metric", "d", registry=registry).set(1)

    metrics = metrics_from_prometheus_text(generate_latest(registry).decode())
    # Summed across engines; `_total` stripped back to the family name;
    # `_created` timestamp gauges, histograms, and non-vllm families skipped.
    assert metrics.scalars["vllm:num_requests_running"] == 5.0
    assert metrics.scalars["vllm:prefix_cache_queries"] == 500.0
    assert metrics.waiting_by_reason == {"capacity": 7.0}
    assert "other_metric" not in metrics.scalars
    assert not any("_created" in name or "queue_time" in name for name in metrics.scalars)


@pytest.mark.asyncio
async def test_monitor_prints_deltas_and_stops(capsys: pytest.CaptureFixture[str]) -> None:
    tick = 0

    async def source() -> EngineMetrics:
        nonlocal tick
        tick += 1
        return _snapshot(running=1, gen=1000.0 * tick, prefix_q=10.0 * tick, prefix_h=10.0 * tick)

    monitor = EngineStatusMonitor(source=source, interval_s=0.01, label="test-engine")
    monitor.start()
    while tick < 2:
        await asyncio.sleep(0.005)
    await monitor.stop()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[test-engine]")]
    assert len(lines) >= 2
    assert "cum:" in lines[0]
    # Later ticks report the interval; 100% hit rate from the deltas.
    assert re.search(r"gen=\d+tok/s", lines[1])
    assert "prefix_hit=100.0%" in lines[1]
    # stop() is idempotent and the task is truly gone.
    await monitor.stop()
    assert monitor._task is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_monitor_stops_on_unavailable(capsys: pytest.CaptureFixture[str]) -> None:
    calls = 0

    async def source() -> EngineMetrics:
        nonlocal calls
        calls += 1
        raise MetricsUnavailable("endpoint is not a vLLM server")

    monitor = EngineStatusMonitor(source=source, interval_s=0.01, label="oai")
    monitor.start()
    await asyncio.sleep(0.05)
    await monitor.stop()
    out = capsys.readouterr().out
    assert "status monitor stopped: endpoint is not a vLLM server" in out
    assert calls == 1  # one notice, no retry spam


# --- GPU path: the sources against a live engine ---------------------------
#
# The unit tests above pin the formatting/parsing contract; these verify the
# contract holds against what vLLM actually exposes — metric names move
# between vLLM versions, and only a live engine catches that.


async def _run_one_prompt(backend: object) -> None:
    import ullm.conv
    import ullm.utils

    tm = ullm.utils.make_turn_manager(backend, reasoning_effort="low", max_tokens=32)  # type: ignore[arg-type]
    tm.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="Say OK.")))
    await tm.run()


async def _exercise_live_monitor(
    source: object, label: str, backend: object, capsys: pytest.CaptureFixture[str]
) -> list[str]:
    """Run a fast-ticking monitor against a live engine while a prompt goes
    through; return the monitor's actual printed lines (re-printed so they
    land in the job log — run pytest with ``-rP`` to see them)."""
    from ullm.engine_monitor import EngineStatusMonitor

    monitor = EngineStatusMonitor(source=source, interval_s=1.0, label=label)  # type: ignore[arg-type]
    monitor.start()
    await _run_one_prompt(backend)
    await asyncio.sleep(3.5)
    await monitor.stop()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(f"[{label}]")]
    print(f"--- monitor lines as captured ({label}) ---")
    for line in lines:
        print(line)
    return lines


def _assert_monitor_lines(lines: list[str]) -> None:
    assert len(lines) >= 2, f"expected >=2 monitor ticks, got {lines}"
    # First tick has no previous snapshot -> cumulative; later ticks report
    # the interval. Gauges present on every line.
    assert "cum:" in lines[0]
    assert all("running=" in ln and "kv_cache=" in ln and "waiting=" in ln for ln in lines)
    assert any("tok/s" in ln for ln in lines[1:])


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_gpu_direct_engine_monitor(_vllm_direct_backend: object, capsys: pytest.CaptureFixture[str]) -> None:
    from ullm.engine_monitor import inprocess_source

    lines = await _exercise_live_monitor(inprocess_source, "engine", _vllm_direct_backend, capsys)
    _assert_monitor_lines(lines)
    current = await inprocess_source()
    # The scheduler-relevant gauges exist and the counters moved.
    assert "vllm:num_requests_running" in current.scalars
    assert "vllm:num_requests_waiting" in current.scalars
    assert "vllm:kv_cache_usage_perc" in current.scalars
    assert current.scalars["vllm:generation_tokens"] > 0


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="session")
async def test_gpu_server_metrics_monitor(_vllm_server_backend: object, capsys: pytest.CaptureFixture[str]) -> None:
    from ullm.engine_monitor import http_source

    base_url: str = getattr(_vllm_server_backend, "base_url")
    source = http_source(base_url.removesuffix("/v1") + "/metrics")
    lines = await _exercise_live_monitor(source, "server", _vllm_server_backend, capsys)
    _assert_monitor_lines(lines)
    current = await source()
    assert current.scalars["vllm:generation_tokens"] > 0


def test_a_killed_engine_process_says_how_it_ended(capfd: pytest.CaptureFixture[str]) -> None:
    """vLLM logs only that an engine core died; the exit status is what says
    whether the OOM killer did it (SIGKILL) or it crashed. The line is written
    to the stdout descriptor, so it is read there."""
    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,), name="EngineCore")
    process.start()
    manager = types.SimpleNamespace(processes=[process])
    engine = types.SimpleNamespace(
        engine_core=types.SimpleNamespace(resources=types.SimpleNamespace(engine_manager=manager))
    )
    assert engine_monitor.watch_engine_exits(engine)
    assert process.pid is not None
    os.kill(process.pid, signal.SIGKILL)
    out, deadline = "", time.monotonic() + 30
    while "ended:" not in out and time.monotonic() < deadline:
        time.sleep(0.1)
        out += capfd.readouterr().out
    assert f"[engine] EngineCore pid {process.pid} ended: killed by SIGKILL (-9)" in out
    assert not engine_monitor.watch_engine_exits(object()), "nothing to watch is not an error"


def test_format_line_splits_the_kv_share_when_threads_park_on_the_gpu() -> None:
    """With threads parked on the GPU the pool's share is split into what the
    requests hold and what only parked threads hold; without them, the line
    gives the pool's share alone."""
    snapshot = _snapshot(running=3)
    snapshot.scalars["vllm:kv_cache_usage_perc"] = 0.7
    plain = format_status_line(snapshot, None, None)
    assert plain is not None and "kv_cache=70.0% |" in plain
    snapshot.scalars[KV_REQUESTS_SHARE] = 0.52
    snapshot.scalars[KV_GPU_PARKED_SHARE] = 0.18
    line = format_status_line(snapshot, None, None)
    assert line is not None and "kv_cache=70.0%(running:52.0%,parked:18.0%) |" in line
