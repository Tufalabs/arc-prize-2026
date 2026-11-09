"""Periodic engine-status line, printed to stdout.

Reads vLLM's metrics and formats the line ourselves — deliberately not a
dispatch to vLLM's own logger (``do_log_stats``): the point is proving
the scheduler-relevant numbers (running/waiting requests, KV-cache
usage, prefix-cache hit rate, preemptions) are accessible to our code at
the backend layer. Two sources:

- ``inprocess_source`` — the frontend-side prometheus registry snapshot
  (``vllm.v1.metrics.reader``), for an in-process ``AsyncLLMEngine``;
- ``http_source`` — a served ``/metrics`` endpoint, for vLLM's
  OpenAI-compatible server (in-process or external). An endpoint without
  vllm metrics stops the monitor with one notice line.

Gauges are instantaneous; counter-derived figures (tok/s, hit rates,
preemptions) are per-interval deltas.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing.connection
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

_RUNNING = "vllm:num_requests_running"
_WAITING = "vllm:num_requests_waiting"
_WAITING_BY_REASON = "vllm:num_requests_waiting_by_reason"
_KV_USAGE = "vllm:kv_cache_usage_perc"
# With threads parked on the GPU (the victim offload connector's
# ``gpu_park``), the pool's share the requests hold and the share only parked
# threads hold: parked threads count as used, so the engine's own share no
# longer says how full the running set is. Published by that connector.
KV_REQUESTS_SHARE = "vllm:kv_cache_requests_perc"
KV_GPU_PARKED_SHARE = "vllm:kv_cache_gpu_parked_perc"
# Public for readers that act on the same numbers this line prints.
REQUESTS_RUNNING = _RUNNING
KV_CACHE_USAGE = _KV_USAGE
_PREEMPTIONS = "vllm:num_preemptions"
_PREFIX_QUERIES = "vllm:prefix_cache_queries"
_PREFIX_HITS = "vllm:prefix_cache_hits"
# The KV connector's own cache (e.g. host RAM under an offload connector),
# consulted for what the GPU cache missed. Reported separately because the
# two answer different questions: the local rate is how much of the prompt
# never left the GPU, the external one is how much of the REST came back
# without being recomputed.
_EXTERNAL_PREFIX_QUERIES = "vllm:external_prefix_cache_queries"
_EXTERNAL_PREFIX_HITS = "vllm:external_prefix_cache_hits"
# Bytes moved to and from a connector's own store (OffloadingConnector
# reports these; SimpleCPUOffloadConnector reports nothing). Store volume is
# what says whether blocks are reaching the offload cache at all — an
# external hit rate of zero reads the same whether they were never written
# or written and evicted.
_OFFLOAD_STORE_BYTES = "vllm:kv_offload_store_bytes"
_OFFLOAD_LOAD_BYTES = "vllm:kv_offload_load_bytes"
# Speculative decoding, exported only with a drafter configured. Acceptance
# is accepted draft tokens over proposed ones; what a draft step adds on top
# of its own token is accepted tokens per draft.
_SPEC_DRAFTS = "vllm:spec_decode_num_drafts"
_SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
_SPEC_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"
# Accepted tokens per draft position, labelled by position: the share of
# drafts whose k-th token was accepted, which says whether a further draft
# token still earns its place.
_SPEC_ACCEPTED_BY_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"
_MM_QUERIES = "vllm:mm_cache_queries"
_MM_HITS = "vllm:mm_cache_hits"
_GENERATION_TOKENS = "vllm:generation_tokens"
_PROMPT_TOKENS = "vllm:prompt_tokens"
_PROMPT_TOKENS_CACHED = "vllm:prompt_tokens_cached"
# Block-lifetime histograms, present only under engine arg kv_cache_metrics
# (sampled at kv_cache_metrics_sample). ``idle_evict`` is how long a block sat
# unreferenced before being recycled — the window a cached prefix has to
# survive; ``reuse_gap`` is how long between a block's uses — the gap it
# actually has to survive. Comparing the two says whether prefix loss is cache
# pressure or something else.
_BLOCK_IDLE_BEFORE_EVICT = "vllm:kv_block_idle_before_evict_seconds"
_BLOCK_REUSE_GAP = "vllm:kv_block_reuse_gap_seconds"
# vLLM exports dozens of latency histograms; only these two are folded into
# a snapshot, so it stays the handful of scheduler numbers it reports.
_HISTOGRAMS = (_BLOCK_IDLE_BEFORE_EVICT, _BLOCK_REUSE_GAP)


class MetricsUnavailable(Exception):
    """Raised by a source when metrics will never appear (e.g. a non-vLLM
    HTTP endpoint). Stops the monitor with a single notice line."""


@dataclass
class EngineMetrics:
    """Flattened snapshot: scalar gauges/counters summed across label sets
    (multiple engines / DP ranks), plus the waiting-reason breakdown."""

    scalars: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    waiting_by_reason: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    accepted_by_position: dict[str, float] = field(default_factory=lambda: dict[str, float]())

    def _add(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Fold one scraped sample into the tally, keeping the labelled breakdowns
        separate from the plain scalars.
        """
        if name == _WAITING_BY_REASON:
            reason = labels.get("reason", "?")
            self.waiting_by_reason[reason] = self.waiting_by_reason.get(reason, 0.0) + value
            return
        if name == _SPEC_ACCEPTED_BY_POS:
            position = labels.get("position", "?")
            self.accepted_by_position[position] = self.accepted_by_position.get(position, 0.0) + value
            return
        self.scalars[name] = self.scalars.get(name, 0.0) + value


async def inprocess_source() -> EngineMetrics:
    """Snapshot the in-process prometheus registry (frontend side — works
    while the EngineCore runs in its own subprocess)."""
    from vllm.v1.metrics import reader

    metrics = EngineMetrics()
    for metric in reader.get_metrics_snapshot():
        if isinstance(metric, (reader.Gauge, reader.Counter)):
            metrics._add(metric.name, float(metric.value), metric.labels)
        elif isinstance(metric, reader.Vector) and metric.name == _SPEC_ACCEPTED_BY_POS:
            # The reader folds the per-position counter into one vector
            # indexed by position; the text exposition keeps it labelled.
            for position, value in enumerate(metric.values):
                metrics._add(metric.name, float(value), {**metric.labels, "position": str(position)})
        elif isinstance(metric, reader.Histogram) and metric.name in _HISTOGRAMS:
            # Kept as the two running totals a mean needs, so the interval
            # mean is a delta ratio like every other rate here.
            metrics._add(f"{metric.name}_sum", float(metric.sum), metric.labels)
            metrics._add(f"{metric.name}_count", float(metric.count), metric.labels)
    return metrics


def http_source(metrics_url: str) -> Callable[[], Awaitable[EngineMetrics]]:
    """Source reading a prometheus ``/metrics`` endpoint (vLLM's
    OpenAI-compatible server). 404 or a vllm-metric-free exposition means
    the endpoint is not a vLLM server -> ``MetricsUnavailable``."""

    async def fetch() -> EngineMetrics:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(metrics_url)
        if response.status_code == 404:
            raise MetricsUnavailable(f"{metrics_url} returned 404 (not a vLLM server?)")
        response.raise_for_status()
        metrics = metrics_from_prometheus_text(response.text)
        if not metrics.scalars:
            raise MetricsUnavailable(f"{metrics_url} exposes no vllm metrics")
        return metrics

    return fetch


def metrics_from_prometheus_text(text: str) -> EngineMetrics:
    """Parse a Prometheus exposition body into the metrics we report."""
    from prometheus_client.parser import text_string_to_metric_families

    metrics = EngineMetrics()
    for family in text_string_to_metric_families(text):
        # Counters expose as `# TYPE x_total counter` (family name comes back
        # stripped, samples keep the suffix); their `x_created` timestamps
        # parse as separate gauge families and must not pollute the snapshot.
        if not family.name.startswith("vllm:") or family.name.endswith("_created"):
            continue
        if family.type == "histogram":
            if family.name not in _HISTOGRAMS:
                continue
            # Only the running totals a mean needs; buckets are ignored.
            for sample in family.samples:
                if sample.name.endswith(("_sum", "_count")):
                    metrics._add(sample.name, float(sample.value), dict(sample.labels))
            continue
        if family.type not in ("gauge", "counter"):
            continue
        for sample in family.samples:
            name = sample.name.removesuffix("_total")
            if name == family.name:
                metrics._add(name, float(sample.value), dict(sample.labels))
    return metrics


def _rate(current: EngineMetrics, previous: EngineMetrics, name: str, dt: float) -> float:
    """Per-second rate of a counter between two samples."""
    return (current.scalars.get(name, 0.0) - previous.scalars.get(name, 0.0)) / dt


def _histogram_mean(current: EngineMetrics, previous: Optional[EngineMetrics], name: str) -> Optional[float]:
    """Mean of the observations a histogram recorded over the interval, or None
    when it recorded none (or the engine never exported it)."""
    prev_scalars = previous.scalars if previous is not None else dict[str, float]()
    count = current.scalars.get(f"{name}_count", 0.0) - prev_scalars.get(f"{name}_count", 0.0)
    if count <= 0:
        return None
    return (current.scalars.get(f"{name}_sum", 0.0) - prev_scalars.get(f"{name}_sum", 0.0)) / count


def _hit_pct(
    current: EngineMetrics, previous: Optional[EngineMetrics], hits_name: str, queries_name: str
) -> Optional[float]:
    """Hit rate over the interval, or ``None`` when nothing was queried in it.
    Interval-scoped on purpose: a cumulative ratio stops moving once a run
    is long enough, which is when it stops being informative.
    """
    prev_scalars = previous.scalars if previous is not None else dict[str, float]()
    queries = current.scalars.get(queries_name, 0.0) - prev_scalars.get(queries_name, 0.0)
    if queries <= 0:
        return None
    hits = current.scalars.get(hits_name, 0.0) - prev_scalars.get(hits_name, 0.0)
    return 100.0 * hits / queries


def format_status_line(current: EngineMetrics, previous: Optional[EngineMetrics], dt: Optional[float]) -> Optional[str]:
    """The status line, or None when no vllm metrics are visible (engine
    built with ``disable_log_stats``, or not started yet). With no
    previous snapshot the counter-derived figures cover the whole engine
    lifetime (labelled ``cum:``); with one, the interval since it.

    ``prompt`` is vLLM's own total, which COUNTS prefix-cache hits — at a
    high hit rate it can read several times the prefill the GPU actually
    ran. The ``new=`` figure beside it is that total minus
    ``prompt_tokens_cached``, i.e. the tokens really computed, and is the
    one to compare against prefill capacity; it is omitted when nothing
    was cached, where the two coincide. ``prompt_tokens_cached`` counts
    local AND external hits, so ``new=`` stays the honest measure of
    recomputation whether a block came from the GPU or from a connector."""
    if not current.scalars:
        return None
    waiting = current.scalars.get(_WAITING, 0.0)
    waiting_text = f"waiting={waiting:.0f}"
    reasons = {k: v for k, v in current.waiting_by_reason.items() if v > 0}
    if waiting > 0 and reasons:
        breakdown = ",".join(f"{k}:{v:.0f}" for k, v in sorted(reasons.items()))
        waiting_text += f"({breakdown})"
    parts = [
        # Wall clock, so the line can be placed against the engine's own log
        # without hunting for the nearest timestamped line.
        time.strftime("%H:%M:%S"),
        f"running={current.scalars.get(_RUNNING, 0.0):.0f}",
        waiting_text,
        f"kv_cache={100.0 * current.scalars.get(_KV_USAGE, 0.0):.1f}%"
        + (
            f"(running:{100.0 * current.scalars.get(KV_REQUESTS_SHARE, 0.0):.1f}%,"
            f"parked:{100.0 * current.scalars[KV_GPU_PARKED_SHARE]:.1f}%)"
            if KV_GPU_PARKED_SHARE in current.scalars
            else ""
        ),
    ]

    if previous is not None and dt is not None and dt > 0:
        parts.append(f"| {dt:.0f}s:")
        parts.append(f"gen={_rate(current, previous, _GENERATION_TOKENS, dt):.0f}tok/s")
        prompt = _rate(current, previous, _PROMPT_TOKENS, dt)
        cached = _rate(current, previous, _PROMPT_TOKENS_CACHED, dt)
        parts.append(f"prompt={prompt:.0f}tok/s" + (f"(new={prompt - cached:.0f})" if cached > 0 else ""))
        preemptions = current.scalars.get(_PREEMPTIONS, 0.0) - previous.scalars.get(_PREEMPTIONS, 0.0)
    else:
        parts.append("| cum:")
        parts.append(f"gen={current.scalars.get(_GENERATION_TOKENS, 0.0):.0f}tok")
        prompt = current.scalars.get(_PROMPT_TOKENS, 0.0)
        cached = current.scalars.get(_PROMPT_TOKENS_CACHED, 0.0)
        parts.append(f"prompt={prompt:.0f}tok" + (f"(new={prompt - cached:.0f})" if cached > 0 else ""))
        preemptions = current.scalars.get(_PREEMPTIONS, 0.0)
    prefix_hit = _hit_pct(current, previous, _PREFIX_HITS, _PREFIX_QUERIES)
    if prefix_hit is not None:
        parts.append(f"prefix_hit={prefix_hit:.1f}%")
    if previous is not None and dt is not None and dt > 0:
        stored = _rate(current, previous, _OFFLOAD_STORE_BYTES, dt)
        loaded = _rate(current, previous, _OFFLOAD_LOAD_BYTES, dt)
        if stored or loaded:
            parts.append(f"offload=+{stored / 1e6:.0f}/-{loaded / 1e6:.0f}MB/s")
    external_hit = _hit_pct(current, previous, _EXTERNAL_PREFIX_HITS, _EXTERNAL_PREFIX_QUERIES)
    if external_hit is not None:
        # Only present with a KV connector configured, and measured against
        # what the LOCAL cache missed — so it is not comparable to
        # prefix_hit, and low local + high external means the connector is
        # carrying the run rather than failing it.
        parts.append(f"ext_hit={external_hit:.1f}%")
    mm_hit = _hit_pct(current, previous, _MM_HITS, _MM_QUERIES)
    if mm_hit is not None:
        parts.append(f"mm_hit={mm_hit:.1f}%")
    accepted = _hit_pct(current, previous, _SPEC_ACCEPTED_TOKENS, _SPEC_DRAFT_TOKENS)
    if accepted is not None:
        prev_scalars = previous.scalars if previous is not None else dict[str, float]()
        drafts = current.scalars.get(_SPEC_DRAFTS, 0.0) - prev_scalars.get(_SPEC_DRAFTS, 0.0)
        accepted_tokens = current.scalars.get(_SPEC_ACCEPTED_TOKENS, 0.0) - prev_scalars.get(_SPEC_ACCEPTED_TOKENS, 0.0)
        per_draft = f"{accepted_tokens / drafts:.2f}/draft" if drafts > 0 else ""
        prev_by_pos = previous.accepted_by_position if previous is not None else dict[str, float]()
        by_pos = (
            "/".join(
                f"{100.0 * (current.accepted_by_position[pos] - prev_by_pos.get(pos, 0.0)) / drafts:.0f}%"
                for pos in sorted(current.accepted_by_position, key=int)
            )
            if drafts > 0 and current.accepted_by_position
            else ""
        )
        detail = ",".join(part for part in (per_draft, f"pos {by_pos}" if by_pos else "") if part)
        parts.append(f"accept={accepted:.1f}%" + (f"({detail})" if detail else ""))
    parts.append(f"preempt={preemptions:.0f}")
    idle = _histogram_mean(current, previous, _BLOCK_IDLE_BEFORE_EVICT)
    if idle is not None:
        parts.append(f"idle_evict={idle:.1f}s")
    gap = _histogram_mean(current, previous, _BLOCK_REUSE_GAP)
    if gap is not None:
        parts.append(f"reuse_gap={gap:.1f}s")
    return " ".join(parts)


@dataclass
class EngineStatusMonitor:
    """Prints one self-constructed status line every ``interval_s``.

    ``start()``/``stop()`` from the owning factory. A tick failure prints
    its traceback and ends the loop (visible, never fatal to the run);
    ``MetricsUnavailable`` ends it with a single notice instead."""

    source: Callable[[], Awaitable[EngineMetrics]] = field(kw_only=True)
    interval_s: float = 60.0
    label: str = "engine"
    _task: Optional["asyncio.Task[None]"] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Begin reporting."""
        assert self._task is None, "monitor already started"
        self._task = asyncio.create_task(self._loop(), name=f"engine-status-{self.label}")

    async def stop(self) -> None:
        """Stop reporting and wait for the loop to unwind."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        """Scrape and report on the interval. Never fatal: engine status is
        reporting, and a broken scrape must not take a run down.
        """
        previous: Optional[EngineMetrics] = None
        previous_t: Optional[float] = None
        notified_empty = False
        try:
            while True:
                await asyncio.sleep(self.interval_s)
                current = await self.source()
                now = time.monotonic()
                dt = None if previous_t is None else now - previous_t
                line = format_status_line(current, previous, dt)
                if line is None:
                    if not notified_empty:
                        print(f"[{self.label}] no vllm metrics visible yet (disable_log_stats set?)", flush=True)
                        notified_empty = True
                    continue
                print(f"[{self.label}] {line}", flush=True)
                previous, previous_t = current, now
        except asyncio.CancelledError:
            raise
        except MetricsUnavailable as e:
            print(f"[{self.label}] status monitor stopped: {e}", flush=True)
        except BaseException:
            print(f"[{self.label}] status monitor died:", flush=True)
            traceback.print_exc()


def start_monitor(
    source: Callable[[], Awaitable[EngineMetrics]], *, interval_s: float = 60.0, label: str = "engine"
) -> EngineStatusMonitor:
    """Start a status monitor over ``source`` and return it."""
    monitor = EngineStatusMonitor(source=source, interval_s=interval_s, label=label)
    monitor.start()
    return monitor


# ---------------------------------------------------------------------------
# How an engine process ended
# ---------------------------------------------------------------------------
#
# vLLM notices an engine core dying (its liveness monitor) and shuts the rest
# down, but never says how the process ended. That is the one fact that tells
# an OOM kill (SIGKILL, with the cgroup's oom_kill count going up) from a
# native crash (SIGSEGV, SIGABRT) from an exception (a positive exit code,
# after a logged traceback).

# How long to wait for a dead process's exit status: a process holding a GPU
# context and a large host tier can take seconds to finish tearing down after
# the event that ended it.
_EXIT_STATUS_WAIT_S = 120.0


def describe_exit(exitcode: Optional[int]) -> str:
    """An engine process's exit code as what ended it."""
    if exitcode is None:
        return f"exit status unknown after {_EXIT_STATUS_WAIT_S:.0f}s"
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = f"signal {-exitcode}"
        text = f"killed by {name} ({exitcode})"
        if -exitcode == signal.SIGKILL:
            text += "; the OOM killer sends SIGKILL, and the [mem] cgroup line's oom kills says whether it did"
        return text
    return f"exit code {exitcode}"


def watch_engine_exits(engine: Any) -> bool:
    """Log how each of ``engine``'s core processes ends, from a daemon
    thread. Returns False, doing nothing, when the engine exposes no
    processes to watch (an in-process engine, or a vLLM without them)."""
    manager = getattr(getattr(getattr(engine, "engine_core", None), "resources", None), "engine_manager", None)
    processes = list(getattr(manager, "processes", None) or [])
    if not processes:
        return False
    threading.Thread(target=_report_exits, args=(processes,), name="ullm-engine-exits", daemon=True).start()
    return True


def _report_exits(processes: list[Any]) -> None:
    """Wait for each process to end and say how it did.

    The line goes straight to the stdout file descriptor rather than through
    ``print``: an engine commonly ends while the interpreter shuts down, and a
    daemon thread that holds ``sys.stdout``'s buffer lock when the
    interpreter flushes it aborts the whole process (``Fatal Python error:
    _enter_buffered_busy``). A raw write holds no Python lock."""
    pending = {process.sentinel: process for process in processes}
    while pending:
        for sentinel in multiprocessing.connection.wait(list(pending)):
            process = pending.pop(sentinel)
            deadline = time.monotonic() + _EXIT_STATUS_WAIT_S
            while process.exitcode is None and time.monotonic() < deadline:
                process.join(1.0)
            line = f"[engine] {process.name} pid {process.pid} ended: {describe_exit(process.exitcode)}\n"
            with contextlib.suppress(OSError, ValueError):
                os.write(sys.__stdout__.fileno() if sys.__stdout__ is not None else 1, line.encode())
