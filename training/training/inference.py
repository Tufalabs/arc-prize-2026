from __future__ import annotations

import copy
import json
import logging
import os
import queue
import signal
import socket
import subprocess
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit

import requests

from training.config import ExperimentConfig, require_server_endpoints, validate_training_sampling
from training.contracts import RolloutTrajectory
from training.game_sampling import sample_training_games
from training.rolling_seed import RollingSeed
from training.run_failure import record_run_failure, raise_if_run_failed
from training.run_artifacts import (
    read_run_launch_time,
    read_text_tail,
)
from training.source_snapshot import verify_runtime_fingerprint
from training.server_endpoints import collector_server_log_paths
from training.vllm_runtime import native_url
from training.policy_identity import (
    PUBLISHED_IDENTITY_FIELD_SETS,
    full_weight_identity,
    published_identity,
)


log = logging.getLogger(__name__)

_STATIC_LORA_ALIAS = "policy"
_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS = 180
_ADAPTER_READY_FILE = "adapter_config.json"
_ADAPTER_READY_TIMEOUT = 30.0
_ADAPTER_READY_POLL = 0.2
_SERVER_LOG_ERROR_TAIL_LINES = 60


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _wait_for_adapter_ready(
    adapter_path: Path, ready_name: str = _ADAPTER_READY_FILE
) -> None:
    ready_file = adapter_path / ready_name
    deadline = time.monotonic() + _ADAPTER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if ready_file.exists():
            return
        time.sleep(_ADAPTER_READY_POLL)
    raise RuntimeError(f"Publication at {adapter_path} is missing {ready_name}")


def _validated_adapter_identity(adapter: Any, model_id: str) -> dict[str, Any]:
    if (
        not isinstance(adapter, dict)
        or set(adapter)
        not in (
            {"name", "id", "path", "server_instance_id"},
            *(
                {"name", "id", "path", "server_instance_id"} | published
                for published in PUBLISHED_IDENTITY_FIELD_SETS
            ),
        )
        or adapter["name"] != model_id
        or type(adapter["id"]) is not int
        or adapter["id"] < 0
        or not isinstance(adapter["server_instance_id"], str)
        or not adapter["server_instance_id"]
        or (adapter["id"] == 0 and adapter["path"] is not None)
        or (
            adapter["id"] > 0
            and (
                not isinstance(adapter["path"], str)
                or not Path(adapter["path"]).is_absolute()
            )
        )
    ):
        raise RuntimeError(
            f"native engine returned invalid adapter identity for {model_id!r}"
        )
    published_identity(adapter)
    return adapter


def _exported_lora_census(census: Any, adapter_path: Path) -> dict[str, Any]:
    """The gateway's LoRA census must apply exactly what the runtime export wrote.

    The census maps each adapter module name (``q_proj``) to the served layer
    that applies it (``applied``) and counts the adapter's modules of that
    name (``modules``); the gateway has refused the load had any module been
    left unapplied (training/lora_census.py). Every exported target module
    must be applied, nothing else, and every exported tensor (an A and a B
    factor per module).
    """
    from training.runtime_lora import read_runtime_export

    exported, tensors = read_runtime_export(adapter_path)
    if (
        not isinstance(census, dict)
        or set(census) != {"applied", "modules"}
        or not isinstance(census["applied"], dict)
        or not isinstance(census["modules"], dict)
        or set(census["applied"]) != set(census["modules"])
        or not all(isinstance(layer, str) and layer for layer in census["applied"].values())
        or not all(type(count) is int and count > 0 for count in census["modules"].values())
    ):
        raise RuntimeError(
            f"native engine returned no valid LoRA module census for {adapter_path}: "
            f"{census!r}"[:500]
        )
    applied = set(census["applied"])
    modules = sum(census["modules"].values())
    if applied != set(exported) or 2 * modules != tensors:
        raise RuntimeError(
            f"native engine applies LoRA modules {sorted(applied)} ({modules} modules) "
            f"of {adapter_path}, but its runtime export wrote {sorted(exported)} "
            f"({tensors} tensors): missing {sorted(set(exported) - applied)}, "
            f"unexpected {sorted(applied - set(exported))}"
        )
    return census


def _lora_census_summary(census: dict[str, Any]) -> str:
    return ", ".join(
        f"{module}->{layer} x{census['modules'][module]}"
        for module, layer in census["applied"].items()
    )


def _behavior_policy_snapshot(
    adapter: dict[str, Any], *, model: str, revision: str | None
) -> dict[str, Any]:
    """Identify identical policy weights independently of the serving process."""
    import hashlib

    published = published_identity(adapter)
    if published is not None:
        return {"model": model, "revision": revision, **published}
    if adapter["id"] == 0 or not adapter["path"]:
        raise RuntimeError("Training snapshots require an immutable published policy")
    path = Path(adapter["path"])
    fingerprints = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        with (path / name).open("rb") as stream:
            fingerprints[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"model": model, "revision": revision, "adapter_files": fingerprints}


def find_available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# Deterministic pre-launched server port contract, shared with the job
# script: server index i listens on base_port + SERVER_PORT_STRIDE * i,
# where base_port comes from the configured inference.base_url. Servers are
# connect-only here, so every port must be one the launcher can compute on
# its own — never a discovered free port that no launcher owns. The stride
# keeps each server's +5000 capture gateway (_platypus_gateway) clear of
# every neighboring server and gateway port.
SERVER_PORT_STRIDE = 100


def derive_server_base_url(base_url: str, server_index: int) -> str:
    """URL of pre-launched server ``server_index`` under the port contract.

    Index 0 returns ``base_url`` unchanged (single-server configs stay
    byte-identical); higher indices shift the port by the stride.
    """
    index = int(server_index)
    if index < 0:
        raise ValueError(f"server_index must be >= 0, got {server_index}")
    if index == 0:
        return base_url
    base_port = _base_url_port(base_url)
    if base_port is None:
        raise ValueError(
            f"cannot derive server {index}'s port: base_url {base_url!r} "
            "carries no explicit port"
        )
    return replace_base_url_port(base_url, base_port + SERVER_PORT_STRIDE * index)


def replace_base_url_port(base_url: str, port: int) -> str:
    parsed = urlsplit(base_url)
    host = parsed.hostname or "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    scheme = parsed.scheme or "http"
    path = parsed.path
    return urlunsplit(
        (scheme, f"{userinfo}{host}:{port}", path, parsed.query, parsed.fragment)
    )


def _per_server_configs(config: ExperimentConfig) -> list[ExperimentConfig]:
    """One config per connect-only vLLM server (inference.server_gpus), each
    on its deterministic pre-launched port (derive_server_base_url): the job
    script launches one server per entry on exactly these ports. Explicit
    base_urls replace this derivation, including server zero."""
    require_server_endpoints(config.inference)
    urls = config.inference.base_urls
    gpus = [
        str(gpu).strip() for gpu in config.inference.server_gpus if str(gpu).strip()
    ]
    if not gpus:
        return [config]
    server_configs: list[ExperimentConfig] = []
    for server_index, _gpu in enumerate(gpus):
        server_config = copy.deepcopy(config)
        if urls:
            server_config.inference.base_url = urls[server_index]
        elif server_index > 0:
            server_config.inference.base_url = derive_server_base_url(
                config.inference.base_url, server_index
            )
        server_configs.append(server_config)
    return server_configs


def _base_url_port(base_url: str) -> int | None:
    try:
        return urlsplit(base_url).port
    except ValueError:
        return None


def parse_prefix_cache_counters(metrics_text: str) -> tuple[float, float] | None:
    """Sum (hits, queries) prefix-cache counters from a Prometheus payload.

    Metric names vary across vLLM versions, so counters are matched by the
    'prefix_cache' substring — but ONLY true counters: '*_created' series are
    unix TIMESTAMPS (Prometheus counter metadata), and summing them drove the
    reported hit rate to ~0.999 regardless of actual cache behavior
    (1.788e9 dominates both sides of the ratio; the real rate on a fresh
    server measures ~0.91, matching vLLM's own logged number)."""
    hits = 0.0
    queries = 0.0
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if "prefix_cache" not in name or "rate" in name:
            continue
        bare_name = name.partition("{")[0]
        if bare_name.endswith("_created"):
            continue
        try:
            numeric = float(value.strip().split()[0])
        except (ValueError, IndexError):
            continue
        if "queries" in name:
            queries += numeric
        elif "hit" in name:
            hits += numeric
    if queries <= 0:
        return None
    return hits, queries


def _seed_text_to_int(seed_text: str) -> int:
    """Numeric seed texts pass through; anything else hashes stably."""
    if seed_text.isdigit():
        return int(seed_text)
    return zlib.crc32(seed_text.encode("utf-8"))


def _platypus_window_seed(window_id: str, rollout_seed: str | None) -> int:
    """Window-level RNG seed handed to the platypus worker.

    The worker's per-game seeds come from the spec's ``slot_seeds`` when the
    window is seeded; this value seeds everything else (and every game of an
    UNSEEDED window, offset by game index). A window WITH a rollout_seed
    carries a deliberately fixed seed (official eval replays the same
    layouts every cycle; explicit ARC3_ROLLOUT_SEED overrides): the window
    id must not perturb it. Unseeded training windows draw fresh layouts
    per window.
    """
    if rollout_seed is not None:
        return _seed_text_to_int(str(rollout_seed))
    return zlib.crc32(window_id.encode("utf-8"))


def _platypus_slot_seeds(slots: list[_RolloutSlot]) -> list[int] | None:
    """Ordered per-slot layout seeds for the worker spec, or None for an
    unseeded window (the worker then derives per-game seeds from the
    window seed).

    The worker keys the harness's per-game seed by each slot's external id
    (game#s<i>), one entry per slot, and that seed drives the decode
    seed of every model call the slot makes: copies of one game get their
    OWN seed (training rolls one child seed per slot per window; official
    eval derives one per game, copy and cycle; ARC3_ROLLOUT_SEED sets every
    slot alike). The official games' layouts do not depend on it.
    """
    if all(slot.rollout_seed is None for slot in slots):
        return None
    seeds: list[int] = []
    for slot in slots:
        if slot.rollout_seed is None:
            raise RuntimeError(
                f"slot {slot.slot_index} ({slot.game_id}) is unseeded in a "
                "window with scheduled seeds; a seeded window must seed "
                "every slot"
            )
        seeds.append(_seed_text_to_int(str(slot.rollout_seed)))
    return seeds


def split_slots_into_balanced_chunks(
    scheduled_games: list[str], server_count: int
) -> list[list[int]]:
    """Distribute full-game slots across servers, retaining their original indices.

    Keep complete balanced passes when possible. For arbitrary repeat counts,
    spread copies across servers and balance the remaining slots by load.
    Unique harness slot IDs allow repeated games within a child window.
    """
    if server_count <= 0:
        raise ValueError(f"server_count must be positive, got {server_count}")
    if not scheduled_games:
        return [[] for _ in range(server_count)]
    counts: dict[str, int] = {}
    for game_id in scheduled_games:
        counts[game_id] = counts.get(game_id, 0) + 1
    copies = set(counts.values())
    k = max(copies)
    if len(copies) == 1 and k % server_count == 0:
        per_server = k // server_count
        chunks: list[list[int]] = [[] for _ in range(server_count)]
        occurrence: dict[str, int] = {}
        for slot_index, game_id in enumerate(scheduled_games):
            seen = occurrence.get(game_id, 0)
            occurrence[game_id] = seen + 1
            chunks[seen // per_server].append(slot_index)
        return chunks
    chunks = [[] for _ in range(server_count)]
    held: list[set[str]] = [set() for _ in range(server_count)]
    # Deal repeated games first, spreading their copies when possible before
    # assigning any additional copies to the least-loaded server.
    order = sorted(
        range(len(scheduled_games)),
        key=lambda index: (-counts[scheduled_games[index]], index),
    )
    for slot_index in order:
        game_id = scheduled_games[slot_index]
        candidates = [
            server for server in range(server_count) if game_id not in held[server]
        ] or list(range(server_count))
        target = min(candidates, key=lambda server: (len(chunks[server]), server))
        chunks[target].append(slot_index)
        held[target].add(game_id)
    for chunk in chunks:
        chunk.sort()
    return chunks


@dataclass(slots=True)
class CollectedWindow:
    window_id: str
    window_dir: Path
    rollouts: list[RolloutTrajectory]
    policy_iteration: int
    schedule_index: int = 0
    collection_duration_seconds: float = 0.0
    launched_rollouts: int = 0
    zero_action_rollouts: int = 0
    failed_processes: int = 0
    artifact_rollouts: int = 0
    # Games whose agent raised (status CRASHED), scored on what they reached.
    crashed_rollouts: int = 0


@dataclass(slots=True)
class _RolloutSlot:
    slot_index: int
    game_id: str
    rollout_seed: str | None


def _terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    terminate_timeout_seconds: float = 10.0,
    kill_timeout_seconds: float = 10.0,
) -> bool:
    if process.poll() is not None:
        return True

    _signal_process_tree(process, signal.SIGTERM)
    return _wait_for_process_tree_after_sigterm(
        process,
        terminate_timeout_seconds=terminate_timeout_seconds,
        kill_timeout_seconds=kill_timeout_seconds,
    )


def _wait_for_process_tree_after_sigterm(
    process: subprocess.Popen[str],
    *,
    terminate_timeout_seconds: float = 10.0,
    kill_timeout_seconds: float = 10.0,
) -> bool:
    """Wait for a process tree that has already received SIGTERM.

    Kept separate from signal delivery so watchdog and shutdown paths never
    send a second SIGTERM while a runner is flushing its partial trajectory.
    """
    if process.poll() is not None:
        return True
    try:
        process.wait(timeout=terminate_timeout_seconds)
        return True
    except subprocess.TimeoutExpired:
        pass

    _signal_process_tree(process, signal.SIGKILL)

    try:
        process.wait(timeout=kill_timeout_seconds)
        return True
    except subprocess.TimeoutExpired:
        return False


def _signal_process_tree(process: subprocess.Popen[str], sig: signal.Signals) -> bool:
    if process.poll() is not None:
        return True

    pid = getattr(process, "pid", None)
    try:
        if pid is None:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        else:
            os.killpg(pid, sig)
    except ProcessLookupError:
        return True
    except OSError:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    return True


def _rollout_status_counts(rollouts: Iterable[RolloutTrajectory]) -> dict[str, int]:
    """Rollouts by terminal status. ``crashed`` counts games whose agent
    raised (``CRASHED``: the harness contained the error and scored what the
    game had reached); ``errors`` runs the harness never finished (``ERROR``)."""
    counts = {
        "wins": 0,
        "timeouts": 0,
        "context_exhausted": 0,
        "crashed": 0,
        "errors": 0,
        "other": 0,
    }
    for rollout in rollouts:
        status = (
            str(
                rollout.termination_reason
                or (rollout.metadata or {}).get("status")
                or ""
            )
            .strip()
            .upper()
        )
        if status == "COMPLETED_RUN":
            counts["wins"] += 1
        elif status == "TIMEOUT":
            counts["timeouts"] += 1
        elif status == "CONTEXT_EXHAUSTED":
            counts["context_exhausted"] += 1
        elif status == "CRASHED":
            counts["crashed"] += 1
        elif status == "ERROR":
            counts["errors"] += 1
        else:
            counts["other"] += 1
    return counts


def _platypus_window_outcomes(
    window_id: str, rollouts: list[RolloutTrajectory], *, log_path: Path
) -> dict[str, int]:
    """Count a collected window's zero-action, ERROR and CRASHED rollouts, log
    them, and refuse a window in which no run ended any other way.

    A crashed game (its agent raised; the harness's ``crash_game`` policy)
    does not stop the window: it is scored on what it reached and logged
    with the tail of its traceback. A window where every run crashed or
    errored has nothing to train on and stops the run.
    """
    zero_action = sum(
        1
        for rollout in rollouts
        if int(rollout.metadata.get("run_total_actions", 0)) == 0
    )
    errors = sum(
        1 for rollout in rollouts if rollout.metadata.get("status") == "ERROR"
    )
    crashed = [
        rollout for rollout in rollouts if rollout.metadata.get("status") == "CRASHED"
    ]
    log.info(
        "platypus window %s complete: rollouts=%d zero_action=%d errors=%d crashed=%d",
        window_id,
        len(rollouts),
        zero_action,
        errors,
        len(crashed),
    )
    for rollout in crashed:
        note = rollout.metadata.get("crash_note_tail") or "(no harness note)"
        log.warning(
            "platypus window %s: run %s (%s) crashed and was scored on what it "
            "reached (return %.4f); harness note tail: %s",
            window_id,
            rollout.metadata.get("run_id"),
            rollout.task_id,
            rollout.trajectory_reward,
            note.strip(),
        )
    if rollouts and errors + len(crashed) == len(rollouts):
        tail = read_text_tail(log_path, lines=25)
        raise RuntimeError(
            f"platypus window {window_id}: every run crashed ({errors} errored, "
            f"{len(crashed)} crashed of {len(rollouts)}); refusing to continue "
            f"on an empty window. worker.log tail:\n{tail}"
        )
    return {"zero_action": zero_action, "errors": errors, "crashed": len(crashed)}


@dataclass(slots=True)
class _PlatypusWorker:
    """One persistent harness worker per collector (see
    scripts/platypus_rollout_worker.py ``serve``): importing the harness
    costs minutes, so it is paid once per run, not once per window."""

    process: subprocess.Popen[str]
    serve_dir: Path
    log_path: Path
    log_handle: Any


def _wait_for_window_done(
    process: Any,
    done_path: Path,
    *,
    deadline: float,
    poll_seconds: float = 2.0,
    check_health: Callable[[], None] | None = None,
) -> str:
    """Block until the worker answers a window: ``"done"`` (done file
    present), ``"exited"`` (worker process gone first), or ``"watchdog"``
    (deadline passed). The done file is the authority when both happen."""
    while True:
        if check_health is not None:
            check_health()
        if done_path.exists():
            return "done"
        if process.poll() is not None:
            return "done" if done_path.exists() else "exited"
        if time.monotonic() >= deadline:
            return "watchdog"
        time.sleep(poll_seconds)


class RolloutCollector:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        repo_root: Path,
        server_log_path: Path | None = None,
        run_root: Path | None = None,
        failure_root: Path | None = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.server_log_path = server_log_path
        self._runtime_fingerprint_warning_emitted = False
        self._platypus_worker: _PlatypusWorker | None = None
        self._stopping_window = False
        # Anchor for inference.max_runtime_minutes_schedule: the run's
        # run_launched_at marker lives in this directory (written once at
        # first trainer start). None disables the schedule (base
        # max_runtime_minutes applies).
        self.run_root = run_root
        self.failure_root = failure_root if failure_root is not None else run_root
        self._runtime_schedule_warning_emitted = False

    def _effective_max_runtime_minutes(self, window_kind: str) -> float | None:
        """Resolve the per-rollout runtime for this window.

        Train windows follow inference.max_runtime_minutes_schedule — a
        piecewise-LINEAR ramp over wall-clock hours since the recorded run
        launch: between neighboring schedule points the minutes interpolate
        continuously, beyond the last point the final value holds, and before
        the first point (or when the launch marker is unreadable) the base
        inference.max_runtime_minutes applies. Eval windows and schedule-less
        configs always use the base value.
        """
        base_minutes = self.config.inference.max_runtime_minutes
        schedule = self.config.inference.max_runtime_minutes_schedule
        if not schedule or str(window_kind).strip().lower() == "eval":
            return base_minutes
        launch_time = read_run_launch_time(self.run_root)
        if launch_time is None:
            if not self._runtime_schedule_warning_emitted:
                log.warning(
                    "inference.max_runtime_minutes_schedule is set but no "
                    "run_launched_at marker is readable (run_root=%s); using "
                    "base max_runtime_minutes=%s",
                    self.run_root,
                    base_minutes,
                )
                self._runtime_schedule_warning_emitted = True
            return base_minutes
        elapsed_hours = max(
            0.0, (datetime.now() - launch_time).total_seconds() / 3600.0
        )
        if elapsed_hours < float(schedule[0][0]):
            minutes: float | None = base_minutes
        else:
            # Strictly increasing hours are enforced at config parse, so
            # every interpolation span is positive.
            minutes = float(schedule[-1][1])
            for (hours_low, minutes_low), (hours_high, minutes_high) in zip(
                schedule, schedule[1:]
            ):
                if elapsed_hours < float(hours_high):
                    fraction = (elapsed_hours - float(hours_low)) / (
                        float(hours_high) - float(hours_low)
                    )
                    minutes = float(minutes_low) + fraction * (
                        float(minutes_high) - float(minutes_low)
                    )
                    break
        log.info(
            "runtime schedule: elapsed=%.1fh since launch -> "
            "max_runtime_minutes=%.2f (base %s) for this %s window",
            elapsed_hours,
            minutes if minutes is not None else float("nan"),
            base_minutes,
            window_kind,
        )
        return minutes

    def _scheduled_games_for_window(
        self,
        *,
        slot_count: int,
        policy_iteration: int,
        schedule_index: int | None = None,
        games: list[str] | None = None,
    ) -> list[str]:
        games = (
            games
            if games is not None
            else (self.config.inference.games or [self.config.inference.game])
        )
        if slot_count <= 0:
            return []
        if not games:
            return [self.config.inference.game] * slot_count

        if schedule_index is None:
            schedule_index = int(policy_iteration) + 1
        schedule_index = max(0, int(schedule_index))
        return sample_training_games(
            games,
            slot_count=slot_count,
            seed=self.config.training.seed,
            schedule_index=schedule_index,
        )

    def _rollout_seed_override(self) -> str | None:
        explicit_seed = str(os.environ.get("ARC3_ROLLOUT_SEED", "")).strip()
        return explicit_seed or None

    def _rollout_slots(
        self,
        *,
        policy_iteration: int,
        schedule_index: int | None = None,
        scheduled_games: list[str] | None = None,
        scheduled_rollout_seeds: list[str | int | None] | None = None,
    ) -> list[_RolloutSlot]:
        seed_override = self._rollout_seed_override()
        if scheduled_games is None:
            scheduled_games = self._scheduled_games_for_window(
                slot_count=self.config.parallel_rollouts,
                policy_iteration=policy_iteration,
                schedule_index=schedule_index,
            )
        if scheduled_rollout_seeds is not None and len(scheduled_rollout_seeds) != len(
            scheduled_games
        ):
            raise ValueError(
                "scheduled_rollout_seeds must have the same length as scheduled_games."
            )
        slots: list[_RolloutSlot] = []
        for slot_index, game_id in enumerate(scheduled_games):
            rollout_seed = seed_override
            if rollout_seed is None and scheduled_rollout_seeds is not None:
                raw_seed = scheduled_rollout_seeds[slot_index]
                rollout_seed = None if raw_seed is None else str(raw_seed)
            slots.append(
                _RolloutSlot(
                    slot_index=slot_index,
                    game_id=game_id,
                    rollout_seed=rollout_seed,
                )
            )
        return slots

    def _server_api_key(self) -> str:
        return str(os.environ.get("SERVER_API_KEY", "") or "").strip()

    def _server_auth_headers(self) -> dict[str, str]:
        api_key = self._server_api_key()
        if not api_key:
            return {}
        return {"Authorization": f"Bearer {api_key}"}

    def _metrics_url(self) -> str:
        parsed = urlsplit(self.config.inference.base_url)
        return f"{parsed.scheme}://{parsed.netloc}/metrics"

    def prefix_cache_counters(self) -> tuple[float, float] | None:
        """Cumulative (hits, queries) from the vLLM Prometheus endpoint, or
        None when the metrics are unavailable."""
        try:
            response = requests.get(
                self._metrics_url(),
                headers=self._server_auth_headers(),
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException:
            return None
        return parse_prefix_cache_counters(response.text)

    def _health_url(self) -> str:
        return native_url(self.config.inference.base_url, "/health")

    def _models_url(self) -> str:
        return native_url(self.config.inference.base_url, "/models")

    def _server_is_ready(self) -> bool:
        try:
            response = requests.get(
                self._health_url(), headers=self._server_auth_headers(), timeout=5
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _load_lora_url(self) -> str:
        return native_url(self.config.inference.base_url, "/adapters/load")

    def _unload_lora_url(self) -> str:
        return native_url(self.config.inference.base_url, "/adapters/unload")

    def _server_log_tail(self, *, lines: int = _SERVER_LOG_ERROR_TAIL_LINES) -> str:
        if self.server_log_path is None or not self.server_log_path.exists():
            return ""
        try:
            return read_text_tail(self.server_log_path, lines=lines)
        except OSError:
            return ""

    def _with_server_log_tail(self, message: str) -> str:
        """``message`` plus the server's recent log when one is readable; the
        log only enriches an error, nothing requires it."""
        tail = self._server_log_tail()
        return f"{message}\nRecent server log tail:\n{tail}" if tail else message

    def runtime_model_alias_available(self, model_id: str) -> bool:
        try:
            response = requests.get(
                self._models_url(), headers=self._server_auth_headers(), timeout=5
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return False

        models = payload.get("models")
        if not isinstance(models, list):
            return False
        for item in models:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).strip() == model_id:
                return True
        return False

    def ensure_server(self) -> None:
        """Connect-only: the vLLM server is launched by the job script; this
        process verifies its actual runtime sampling configuration too."""
        try:
            raise_if_run_failed(getattr(self, "failure_root", None))
            if self._server_is_ready():
                from training.vllm_runtime import validate_server_runtime

                self._runtime_config = validate_server_runtime(
                    self.config.inference.base_url,
                    headers=self._server_auth_headers(),
                    score_centering_top_k=self.config.training.score_centering_top_k
                    or 0,
                    speculative=self.config.inference.speculative_decoding.identity(),
                )
                return
            raise RuntimeError(
                f"Inference server is not reachable at {self._health_url()}."
            )
        except Exception as error:
            record_run_failure(getattr(self, "failure_root", None), error)
            raise

    def pin_full_weight_identity(self, identity: dict[str, Any]) -> None:
        """Bind future waves to the publication confirmed by native controls."""
        model_id = self.config.inference.model_id or self.config.training.model_name
        validated = _validated_adapter_identity(identity, model_id)
        if full_weight_identity(validated) is None:
            raise RuntimeError("Full-weight publication acknowledgement lacks identity")
        self._loaded_full_weight_identity = dict(validated)

    def adapter_identity(self, model_id: str) -> dict[str, Any]:
        """Pin the engine's actual adapter and instance before each window."""
        response = requests.get(
            native_url(self.config.inference.base_url, "/metadata"),
            params={"model": model_id},
            headers=self._server_auth_headers(),
            timeout=10,
        )
        response.raise_for_status()
        adapter = response.json().get("adapter")
        adapter = _validated_adapter_identity(adapter, model_id)
        if self.config.training.weight_mode == "full_weight":
            expected = getattr(self, "_loaded_full_weight_identity", None)
            if expected is None or adapter != expected:
                raise RuntimeError(
                    "native engine differs from the confirmed full-weight publication"
                )
        elif self.config.training.weight_mode == "flash":
            expected = getattr(self, "_loaded_policy_identity", None)
            if expected is None or adapter != expected:
                raise RuntimeError(
                    f"native engine serves {adapter}, not the applied merged policy "
                    f"weights {expected}"
                )
        else:
            expected = getattr(self, "_loaded_runtime_adapter_identity", None)
        if (
            expected is not None
            and expected["name"] == model_id
            and adapter != expected
        ):
            raise RuntimeError(
                "native engine adapter changed after the trainer loaded it"
            )
        return adapter

    def load_runtime_adapter(
        self, *, adapter_name: str, adapter_path: Path
    ) -> dict[str, Any]:
        """Load the runtime export at ``adapter_path`` under ``adapter_name``.

        One control RPC. The service refuses an adapter with any module the
        served model would not apply, then loads a fresh LoRA ID, retires the
        old ID and publishes the alias; a failed engine update poisons the
        serving process. The response's LoRA census must apply exactly the
        modules the export wrote (``loaded_runtime_adapter_census``).
        """
        response = requests.post(
            self._load_lora_url(),
            json={"name": adapter_name, "path": str(adapter_path)},
            headers=self._server_auth_headers(),
            timeout=_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to load runtime LoRA adapter {adapter_name!r}: "
                f"{response.status_code} {response.text}"
            )
        payload = response.json()
        identity = _validated_adapter_identity(payload.get("adapter"), adapter_name)
        if identity["id"] == 0 or identity["path"] != str(adapter_path):
            raise RuntimeError(
                "native engine loaded a different adapter path than requested"
            )
        census = _exported_lora_census(payload.get("lora_census"), Path(adapter_path))
        self._loaded_runtime_adapter_identity = identity
        self._loaded_runtime_adapter_path = str(adapter_path)
        self.loaded_runtime_adapter_census = census
        return identity

    def warm_runtime_adapter(
        self, *, adapter_name: str, expected_identity: dict[str, Any]
    ) -> float:
        """Pay first-scheduled LoRA latency on the trainer control plane.

        The response names the adapter that served the draw (the gateway has
        checked it against the engine's own LoRA request); it must be
        ``expected_identity``, the one ``load_runtime_adapter`` returned.
        """
        started_at = time.perf_counter()
        response = requests.post(
            native_url(self.config.inference.base_url, "/adapters/warm"),
            json={"name": adapter_name},
            headers=self._server_auth_headers(),
            timeout=_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - started_at
        if response.status_code != 200:
            raise RuntimeError(
                self._with_server_log_tail(
                    f"Runtime LoRA alias {adapter_name!r} did not serve a warm-up "
                    f"draw: HTTP {response.status_code} {response.text[:300]}"
                )
            )
        served = _validated_adapter_identity(
            response.json().get("adapter"), adapter_name
        )
        if served != expected_identity:
            raise RuntimeError(
                self._with_server_log_tail(
                    f"Runtime LoRA alias {adapter_name!r} served its warm-up draw "
                    f"with adapter {served}, not the loaded {expected_identity}"
                )
            )
        return elapsed

    def confirm_policy_weights(
        self, identity: dict[str, Any], *, label: dict[str, Any], model_id: str
    ) -> None:
        """Pin the Flash publication this server confirmed when it resumed.

        ``identity`` is the base name's identity the server returned from
        /weights/resume after every worker verified each tensor's checksum
        against the publication's manifest (training/flash_policy_weights.py);
        it must name ``label`` on the same server instance as before. Then
        ``/policy_weights/info`` and a warm-up draw must serve it too. A
        restarted server (back on the checkpoint's weights) is fatal.
        """
        base_url = self.config.inference.base_url
        served = _validated_adapter_identity(identity, model_id)
        if served["id"] != 0 or published_identity(served) != label:
            raise RuntimeError(
                f"Inference server {base_url} resumed {served}, not the published policy weights {label}"
            )
        instance = served["server_instance_id"]
        previous = getattr(self, "_applied_policy_weights", None)
        if previous is not None and previous["server_instance_id"] != instance:
            raise RuntimeError(
                f"Inference server {base_url} restarted: it served the previous policy "
                f"weights as {previous['server_instance_id']}, these as {instance}"
            )
        self._applied_policy_weights = {
            "sha256": label["policy_sha256"],
            "iteration": label["policy_iteration"],
            "server_instance_id": instance,
        }
        self._loaded_policy_identity = None
        self.verify_policy_weights()
        warm_seconds = self.warm_policy_weights(model_id=model_id)
        log.info(
            "server %s serves the merged policy weights of trainer iteration %s (identity %s): "
            "warm-up %.2fs",
            base_url,
            label["policy_iteration"],
            label["policy_sha256"],
            warm_seconds,
        )

    def verify_policy_weights(self) -> None:
        """The server must still serve the publication it confirmed, as the same instance."""
        expected = getattr(self, "_applied_policy_weights", None)
        base_url = self.config.inference.base_url
        if expected is None:
            raise RuntimeError(f"No merged policy weights were published to {base_url}")
        response = requests.get(
            native_url(base_url, "/policy_weights/info"),
            headers=self._server_auth_headers(),
            timeout=_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise RuntimeError(
                self._with_server_log_tail(
                    f"Policy weights info from {base_url} failed: "
                    f"HTTP {response.status_code} {response.text[:300]}"
                )
            )
        served = response.json()
        if (
            served.get("sha256") != expected["sha256"]
            or served.get("iteration") != expected["iteration"]
            or served.get("server_instance_id") != expected["server_instance_id"]
        ):
            raise RuntimeError(
                f"Fatal policy weights mismatch on {base_url}: it serves "
                f"{served.get('sha256')} (iteration {served.get('iteration')}) on instance "
                f"{served.get('server_instance_id')}, not the published {expected['sha256']} "
                f"(iteration {expected['iteration']}) on {expected['server_instance_id']}"
            )

    def warm_policy_weights(self, *, model_id: str) -> float:
        """A one-token draw under the base name, served with the published policy weights.

        Pays the first scheduled draw's latency on the trainer control plane
        and pins the identity every window's draws must carry.
        """
        expected = getattr(self, "_applied_policy_weights", None)
        base_url = self.config.inference.base_url
        if expected is None:
            raise RuntimeError(f"No merged policy weights were published to {base_url}")
        started_at = time.perf_counter()
        response = requests.post(
            native_url(base_url, "/adapters/warm"),
            json={"name": model_id},
            headers=self._server_auth_headers(),
            timeout=_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - started_at
        if response.status_code != 200:
            raise RuntimeError(
                self._with_server_log_tail(
                    f"{model_id!r} did not serve a warm-up draw on {base_url}: "
                    f"HTTP {response.status_code} {response.text[:300]}"
                )
            )
        served = _validated_adapter_identity(response.json().get("adapter"), model_id)
        publication = published_identity(served)
        if (
            served["id"] != 0
            or served["server_instance_id"] != expected["server_instance_id"]
            or publication
            != {
                "kind": "policy_weights",
                "policy_sha256": expected["sha256"],
                "policy_iteration": expected["iteration"],
            }
        ):
            raise RuntimeError(
                self._with_server_log_tail(
                    f"{model_id!r} served its warm-up draw on {base_url} as {served}, not "
                    f"with the published policy weights {expected}"
                )
            )
        self._loaded_policy_identity = served
        return elapsed

    def unload_runtime_adapter(self, *, adapter_name: str) -> None:
        response = requests.post(
            self._unload_lora_url(),
            json={"name": adapter_name},
            headers=self._server_auth_headers(),
            timeout=_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code == 404:
            return
        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to unload runtime LoRA adapter {adapter_name!r}: "
                f"{response.status_code} {response.text}"
            )

    def _platypus_gateway(self):
        """The serving-edge capture gateway, started once per collector.

        Listens on the upstream port + 5000 (the harness gets THIS url as
        its base_url) and proxies to the configured vLLM server.
        """
        gateway = getattr(self, "_rollout_gateway", None)
        if gateway is None:
            from urllib.parse import urlparse

            from training.rollout_gateway import CaptureGateway

            upstream = self.config.inference.base_url
            parsed = urlparse(upstream)
            if parsed.port is None:
                raise RuntimeError(
                    f"cannot derive a gateway port from inference.base_url {upstream!r}"
                )
            gateway = CaptureGateway(
                upstream_base_url=upstream,
                port=parsed.port + 5000,
                on_contract_violation=lambda error: record_run_failure(
                    self.failure_root, error
                ),
            )
            gateway.start()
            log.info(
                "rollout gateway listening on %s (upstream %s)",
                gateway.base_url,
                upstream,
            )
            self._rollout_gateway = gateway
        return gateway

    def _collector_id(self) -> str:
        """Stable per-collector identity: the upstream server's port.

        Multi-server collection clones this collector once per server config
        (distinct ``inference.base_url`` ports), so the port separates their
        worker queues and logs on the shared run root.
        """
        port = _base_url_port(self.config.inference.base_url)
        return f"port-{port}" if port is not None else "local"

    def _ensure_platypus_worker(self, durable_log_root: Path) -> _PlatypusWorker:
        worker = self._platypus_worker
        if worker is not None and worker.process.poll() is None:
            return worker
        if worker is not None:
            self._close_platypus_worker(worker)
        inference_config = self.config.inference
        worker_python = Path(inference_config.platypus_venv) / "bin" / "python"
        worker_script = self.repo_root / "scripts" / "platypus_rollout_worker.py"
        # One serve directory PER COLLECTOR: under multi-server collection the
        # collectors run concurrently and window ids have second resolution,
        # so a shared queue would let one worker consume or overwrite the
        # other's spec/done files.
        serve_dir = (
            self.run_root if self.run_root is not None else durable_log_root
        ) / f"platypus_worker_serve_{self._collector_id()}"
        serve_dir.mkdir(parents=True, exist_ok=True)
        (serve_dir / "STOP").unlink(missing_ok=True)
        log_path = durable_log_root / "worker.log"
        env = os.environ.copy()
        # The worker is pure orchestration + HTTP; keep it off the GPUs.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONUNBUFFERED"] = "1"
        log_handle = open(log_path, "a", encoding="utf-8")
        process = subprocess.Popen(
            [str(worker_python), str(worker_script), "--serve", str(serve_dir)],
            cwd=str(self.repo_root),
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.info(
            "started persistent platypus worker pid=%s (serve_dir=%s, log=%s)",
            process.pid,
            serve_dir,
            log_path,
        )
        self._platypus_worker = _PlatypusWorker(
            process=process,
            serve_dir=serve_dir,
            log_path=log_path,
            log_handle=log_handle,
        )
        return self._platypus_worker

    def _close_platypus_worker(self, worker: _PlatypusWorker) -> None:
        if worker.process.poll() is None:
            _terminate_process_tree(worker.process)
        try:
            worker.log_handle.close()
        except OSError:
            pass
        if self._platypus_worker is worker:
            self._platypus_worker = None

    def stop_platypus_worker(self) -> None:
        """Cooperative stop (STOP file, drained queue), then terminate."""
        self._stopping_window = True
        worker = self._platypus_worker
        if worker is None:
            return
        try:
            (worker.serve_dir / "STOP").touch()
            try:
                worker.process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                pass
        finally:
            self._close_platypus_worker(worker)

    def abort_window(self) -> None:
        """Terminate a sibling window after a fatal run failure."""
        self._stopping_window = True
        worker = self._platypus_worker
        if worker is not None:
            self._close_platypus_worker(worker)

    def _collect_window_platypus(
        self,
        *,
        window_id: str,
        window_dir: Path,
        slots: list[_RolloutSlot],
        per_rollout_runtime_seconds: float | None,
        policy_iteration: int,
        schedule_index: int,
        model_id: str | None,
        collection_started_at: float,
        window_kind: str = "train",
    ) -> CollectedWindow:
        """Collect one window through the platypus harness.

        One benchmark worker process (running in the platypus venv) hosts
        every slot concurrently, with all of its model traffic routed
        through the trainer's rollout gateway (training/rollout_gateway.py):
        the gateway records token ids + sampled logprobs at the serving
        edge, the worker POSTs each run's outcome + step map when the
        benchmark ends, and training.platypus_rollout joins the two in
        memory. No harness artifact files cross the trainer boundary; the
        platypus job dir is a viewer/debug surface only. The legacy
        per-rollout subprocess path, replay buffer, and per-process salvage
        do not apply here: platypus owns per-game containment, and a
        worker-level watchdog owns the window bound.
        """
        from harness_support import augmentation
        from training import platypus_rollout
        from training.config import resolve_context_contract
        from training.rollout_gateway import GatewayError, SamplingPolicy

        gateway = self._platypus_gateway()
        # A previous window must have consumed every captured draw and result.
        stale = gateway.drain_window()
        if stale.captures or stale.results:
            raise GatewayError(
                "rollout gateway held unexpected evidence before a new window: "
                f"{len(stale.captures)} captures, {len(stale.results)} results"
            )

        if not model_id:
            raise RuntimeError(
                "A concrete served model alias is required for platypus rollout collection."
            )
        inference_config = self.config.inference
        context = resolve_context_contract(inference_config)
        if self.config.training.weight_mode == "flash":
            self.verify_policy_weights()
        gateway.set_policy(
            SamplingPolicy(
                require_progress=True,
                score_centering_top_k=(self.config.training.score_centering_top_k or 0)
                if window_kind == "train"
                else 0,
                vocab_size=getattr(self, "_runtime_config", {}).get("vocab_size", 0),
                require_lora=self.config.training.weight_mode == "lora",
                require_full_weight=self.config.training.weight_mode == "full_weight",
                require_policy_weights=self.config.training.weight_mode == "flash",
                model=model_id,
                expected_adapter=self.adapter_identity(model_id),
                temperature=float(inference_config.agent_temperature),
                top_p=float(inference_config.agent_top_p),
                top_k=int(inference_config.agent_top_k),
                context=context,
                max_runtime_seconds=per_rollout_runtime_seconds,
                malformed_call_penalty=self.config.training.malformed_call_penalty,
                max_episode_penalty=self.config.training.max_episode_penalty,
            )
        )
        job_dir = window_dir / "platypus"
        job_dir.mkdir(parents=True, exist_ok=True)
        # Diagnostics must survive the node: window dirs live on node-local
        # scratch, so the worker log goes to the durable run root — one log
        # root per collector so concurrent collectors don't interleave.
        durable_log_root = (
            self.run_root / f"platypus_worker_logs_{self._collector_id()}"
            if self.run_root is not None
            else job_dir
        )
        durable_log_root.mkdir(parents=True, exist_ok=True)
        from training.harness import get_trainer_support, harness_engine
        from training.multimodal import checkpoint_source

        harness_support = get_trainer_support(inference_config.environment)
        # Flash serves a local checkpoint under its alias: the worker's
        # tokenizer comes from that directory, never from the Hub.
        tokenizer_source, _tokenizer_revision, _local = checkpoint_source(self.config.training)
        spec: dict[str, Any] = {
            "environment": inference_config.environment,
            "dataset": inference_config.dataset,
            "multimodal_context": inference_config.agent_multimodal_context,
            "behavior_policy_snapshot": _behavior_policy_snapshot(
                gateway._policy.expected_adapter,
                model=self.config.training.model_name,
                revision=self.config.training.model_revision,
            )
            if harness_support.REQUIRES_BEHAVIOR_POLICY_SNAPSHOT
            else None,
            "serving_adapter": gateway._policy.expected_adapter,
            "window_id": window_id,
            "job_dir": str(job_dir),
            "games": [slot.game_id for slot in slots],
            "rng_seed": _platypus_window_seed(
                window_id, slots[0].rollout_seed if slots else None
            ),
            # Ordered per-slot decode seeds (None for unseeded windows); the
            # worker keys them by unique external slot IDs.
            "slot_seeds": _platypus_slot_seeds(slots),
            # Training windows only: one augmentation per puzzle, derived from
            # (run seed, window index, puzzle) by the worker and re-derived
            # below (harness_support/augmentation.py). Evaluation is never
            # augmented.
            "augmentation": augmentation.window_spec(
                inference_config.augmentation,
                environment=inference_config.environment,
                run_seed=self.config.training.seed,
                window=schedule_index,
                kind=window_kind,
            ),
            "max_concurrent": min(
                len(slots), inference_config.max_concurrent_episodes or len(slots)
            ),
            "max_runtime_s_per_game": per_rollout_runtime_seconds,
            "base_url": gateway.base_url,
            "results_url": gateway.results_url,
            "served_model": model_id,
            "tokenizer_id": tokenizer_source,
            "model_revision": self.config.training.model_revision,
            "engine": harness_engine(self.config.training),
            "max_episode_tokens": inference_config.agent_max_episode_tokens,
            "reasoning_effort": inference_config.agent_reasoning_effort or None,
            "context_window_tokens": context.context_window_tokens,
            "compact_to_tokens": context.reset_prompt_ceiling_tokens,
            "compaction_trigger_tokens": context.compaction_trigger_tokens,
            # Pass the resolved margin explicitly: preset changes must not
            # silently move the trainer-owned context threshold.
            "safety_margin_tokens": context.safety_margin_tokens,
            # Every sampled draw is a step, including malformed model replies.
            "malformed_retries": 0,
            "accept_malformed_output": True,
            "malformed_call_penalty": self.config.training.malformed_call_penalty,
            # ARC-2: the most an episode's penalties take from its reward
            # (harness_support/arc2/results.py).
            "max_episode_penalty": self.config.training.max_episode_penalty,
            # ARC-2: an empty retained reply may end the episode as a give-up.
            "terminate_on_empty_reply": bool(
                inference_config.agent_terminate_on_empty_reply
            ),
            "max_completion_tokens": context.max_completion_tokens,
            "max_summary_tokens": context.max_summary_completion_tokens,
            "summary_forbidden_token_ids": list(context.summary_forbidden_token_ids),
            "max_served_sequence_tokens": context.max_served_sequence_tokens,
            # The trainer scores at T=1.0 with the admitted bias: rollouts must
            # sample from the distribution PPO assumes, so the configured
            # sampling travels to the harness explicitly (the duck's own
            # profile is 0.8 for the ARC-2 duck, 0.6 for the ARC-3 duck, and would be silently off-policy).
            "sampling": {
                "temperature": float(inference_config.agent_temperature),
                "top_p": float(inference_config.agent_top_p),
                "top_k": (
                    int(inference_config.agent_top_k)
                    if inference_config.agent_top_k > 0
                    else -1
                ),
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "min_p": 0.0,
                "min_tokens": 0,
            },
        }
        # A missed worker deadline is an infrastructure failure, even if
        # some episodes already reported. Never retry or train a survivor subset.
        worker = self._ensure_platypus_worker(durable_log_root)
        log_path = worker.log_path
        # Viewer/debug copy; the worker consumes the serve-dir copy.
        (job_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        spec_path = worker.serve_dir / f"{window_id}.spec.json"
        staging = spec_path.with_suffix(".tmp")
        staging.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        staging.replace(spec_path)
        done_path = worker.serve_dir / f"{window_id}.done.json"
        log.info(
            "collecting platypus rollout window %s (slots=%d, runtime=%ss, served_model=%s, worker_pid=%s)",
            window_id,
            len(slots),
            per_rollout_runtime_seconds,
            model_id,
            worker.process.pid,
        )
        # Queued episodes receive their own full runtime after admission.
        # Bound all admission waves, plus setup and trace finalization.
        budget = float(per_rollout_runtime_seconds or 900.0)
        waves = (len(slots) + spec["max_concurrent"] - 1) // spec["max_concurrent"]
        deadline = time.monotonic() + waves * budget + 900.0

        def check_health() -> None:
            raise_if_run_failed(self.failure_root)
            gateway.raise_if_invalid()
            if getattr(self, "_stopping_window", False):
                raise InterruptedError("rollout collection was deliberately stopped")

        try:
            outcome = _wait_for_window_done(
                worker.process,
                done_path,
                deadline=deadline,
                check_health=check_health,
            )
        except (RuntimeError, InterruptedError):
            self._close_platypus_worker(worker)
            raise
        if outcome == "watchdog":
            self._close_platypus_worker(worker)
            tail = read_text_tail(log_path, lines=40)
            raise RuntimeError(
                f"FATAL: platypus window {window_id} exceeded its worker watchdog; "
                f"no retry or partial-window training is allowed. worker.log tail:\n{tail}"
            )
        if outcome == "exited":
            return_code = worker.process.poll()
            self._close_platypus_worker(worker)
            if getattr(self, "_stopping_window", False):
                raise InterruptedError("rollout collection was deliberately stopped")
            tail = read_text_tail(log_path, lines=40)
            raise RuntimeError(
                f"platypus rollout worker exited with {return_code} during window "
                f"{window_id}; worker.log tail:\n{tail}"
            )
        status = json.loads(done_path.read_text(encoding="utf-8"))
        if status.get("status") != "ok":
            tail = read_text_tail(log_path, lines=40)
            raise RuntimeError(
                f"platypus rollout worker failed window {window_id}: "
                f"{status.get('error')}; worker.log tail:\n{tail}"
            )

        window_capture = gateway.drain_window(
            require_results=True, check_health=check_health
        )
        if len(window_capture.results) != len(slots):
            raise RuntimeError(
                f"platypus window {window_id}: harness reported "
                f"{len(window_capture.results)} run results for {len(slots)} "
                "slots — the results callback is incomplete"
            )
        rollouts = platypus_rollout.build_window_trajectories(
            window_capture,
            policy_iteration=policy_iteration,
            max_runtime_seconds=per_rollout_runtime_seconds,
            game_ids=[slot.game_id for slot in slots],
            include_void_attempts=True,
        )
        try:
            augmentation.verify_rollouts(rollouts, spec["augmentation"])
        except ValueError as error:
            raise GatewayError(f"platypus window {window_id}: {error}") from error
        outcomes = _platypus_window_outcomes(window_id, rollouts, log_path=log_path)
        return CollectedWindow(
            window_id=window_id,
            window_dir=window_dir,
            rollouts=rollouts,
            policy_iteration=policy_iteration,
            schedule_index=schedule_index,
            collection_duration_seconds=time.perf_counter() - collection_started_at,
            launched_rollouts=len(slots),
            zero_action_rollouts=outcomes["zero_action"],
            failed_processes=outcomes["errors"],
            artifact_rollouts=len(rollouts),
            crashed_rollouts=outcomes["crashed"],
        )

    def collect_window(
        self,
        *,
        output_root: Path,
        model_id: str | None = None,
        policy_iteration: int = -1,
        schedule_index: int | None = None,
        scheduled_games: list[str] | None = None,
        scheduled_rollout_seeds: list[str | int | None] | None = None,
        window_kind: str = "train",
    ) -> CollectedWindow:
        try:
            raise_if_run_failed(self.failure_root)
            if str(window_kind).strip().lower() == "train":
                validate_training_sampling(self.config.inference)
            self._verify_runtime_fingerprint()

            collection_started_at = time.perf_counter()
            window_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            window_dir = output_root / f"window_{window_id}"

            effective_max_runtime_minutes = self._effective_max_runtime_minutes(
                window_kind
            )
            per_rollout_runtime_seconds = (
                None
                if effective_max_runtime_minutes is None
                else max(0.0, float(effective_max_runtime_minutes) * 60.0)
            )
            slots = self._rollout_slots(
                policy_iteration=policy_iteration,
                schedule_index=schedule_index,
                scheduled_games=scheduled_games,
                scheduled_rollout_seeds=scheduled_rollout_seeds,
            )
            return self._collect_window_platypus(
                window_id=window_id,
                window_dir=window_dir,
                slots=slots,
                per_rollout_runtime_seconds=per_rollout_runtime_seconds,
                policy_iteration=policy_iteration,
                schedule_index=(
                    max(0, int(schedule_index))
                    if schedule_index is not None
                    else max(0, int(policy_iteration) + 1)
                ),
                model_id=model_id,
                collection_started_at=collection_started_at,
                window_kind=str(window_kind).strip().lower(),
            )
        except Exception as error:
            if isinstance(error, InterruptedError) and getattr(
                self, "_stopping_window", False
            ):
                raise
            record_run_failure(self.failure_root, error)
            raise

    def _verify_runtime_fingerprint(self) -> None:
        fingerprint_path = str(
            os.environ.get("ARC3_RUNTIME_FINGERPRINT_PATH") or ""
        ).strip()
        if not fingerprint_path:
            return
        try:
            verify_runtime_fingerprint(fingerprint_path)
        except RuntimeError:
            if _env_flag_enabled("ARC3_STRICT_RUNTIME_FINGERPRINT"):
                raise
            if not self._runtime_fingerprint_warning_emitted:
                log.warning(
                    "runtime Python environment fingerprint changed after this run started; "
                    "continuing because frozen source snapshots isolate run code. "
                    "Set ARC3_STRICT_RUNTIME_FINGERPRINT=1 to restore fail-fast behavior.",
                    exc_info=True,
                )
                self._runtime_fingerprint_warning_emitted = True


def sync_collector_runtime_adapter(
    collector: RolloutCollector,
    *,
    policy_path: Path,
    policy_iteration: int,
    model_id: str | None = None,
) -> None:
    """Publish the runtime LoRA export at ``policy_path`` to one inference server.

    Flash never publishes by path: the learner broadcasts merged policy
    weights over NCCL (training/flash_backend.py). LoRA hot-loads the
    adapter under the static alias, confirming the load over HTTP alone: the
    load response names a fresh adapter at ``policy_path`` and carries a LoRA
    census that applies exactly the modules the runtime export wrote (the
    server refuses an adapter with any module its model would not apply),
    ``/models`` lists the alias, and the warm-up draw is served under that
    same identity. The census is logged once per publication. The server log
    is never required. The server is connect-only. A failed load propagates
    as a fatal error; it is never retried against a potentially changed
    engine."""
    if collector.config.training.weight_mode == "flash":
        raise RuntimeError(
            "Flash publishes merged policy weights over NCCL at a rollout boundary, never by path"
        )
    loaded = collector.load_runtime_adapter(
        adapter_name=_STATIC_LORA_ALIAS,
        adapter_path=policy_path,
    )
    if not collector.runtime_model_alias_available(_STATIC_LORA_ALIAS):
        raise RuntimeError(
            collector._with_server_log_tail(
                f"Runtime LoRA alias {_STATIC_LORA_ALIAS!r} is not listed by "
                f"{collector._models_url()} after a reported adapter load."
            )
        )
    warm_seconds = collector.warm_runtime_adapter(
        adapter_name=_STATIC_LORA_ALIAS, expected_identity=loaded
    )
    log.info(
        "runtime adapter %s (id %s, %s) served its warm-up completion for trainer iteration %s "
        "in %.2fs; %s applies its LoRA modules as %s",
        _STATIC_LORA_ALIAS,
        loaded["id"],
        loaded["path"],
        policy_iteration,
        warm_seconds,
        collector.config.inference.base_url,
        _lora_census_summary(collector.loaded_runtime_adapter_census),
    )


class ContinuousRolloutWorker:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        repo_root: Path,
        output_root: Path,
        run_root: Path | None = None,
    ) -> None:
        self.config = config
        # Durable per-run state lives under run_root; output_root may sit on
        # node-local scratch (resolve_rollout_windows_root) where anything
        # left behind is lost when the job ends.
        self.run_root = run_root if run_root is not None else output_root.parent
        log_paths = collector_server_log_paths(config, self.run_root)
        self.server_log_path = log_paths[0]
        self.collectors = [
            RolloutCollector(
                server_config,
                repo_root=repo_root,
                server_log_path=log_paths[server_index],
                run_root=self.run_root,
            )
            for server_index, server_config in enumerate(_per_server_configs(config))
        ]
        self.collector = self.collectors[0]
        self.output_root = output_root
        self._buffer: queue.Queue[CollectedWindow] = queue.Queue(
            maxsize=config.rollout_buffer_size
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._policy_ready = threading.Condition(self._lock)
        self._last_collected_policy_iteration: int | None = None
        self._last_consumed_policy_iteration: int | None = None
        self._collection_lock = threading.Lock()
        self._pending_policy_path: Path | None = None
        self._pending_policy_iteration = -1
        self._active_policy_path: Path | None = None
        self._published_policy_ready = False
        self._active_policy_iteration = -1
        self._active_model_id: str | None = None
        self._next_schedule_index = 0
        # Overlapped collection starts the next window as soon as a slot frees.
        # The trainer caps the count so no window is collected after the
        # final iteration only to be dropped at shutdown.
        self._window_limit: int | None = None
        self._windows_produced = 0
        self._estimated_window_ready_at: float | None = None
        self._error: BaseException | None = None
        self._server_started = False
        # Higher-level rolling seed: rolls a fresh iteration seed for every
        # train/eval window and splits per-rollout child seeds from it, so each
        # iteration sees new re_arc instance seeds (JAX rolling-key style).
        self._rolling_seed = RollingSeed(int(self.config.training.seed))

    @property
    def server_log_paths(self) -> list[Path]:
        return [
            collector.server_log_path
            for collector in self.collectors
            if collector.server_log_path is not None
        ]

    def _base_model_id(self) -> str | None:
        explicit_model_id = str(self.config.inference.model_id or "").strip()
        if explicit_model_id:
            return explicit_model_id
        raise RuntimeError(
            "inference.model_id must be set so rollout collection uses the "
            "exact served model id."
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="rollout-worker", daemon=True
        )
        self._thread.start()

    def limit_windows(self, count: int) -> None:
        """Collect at most ``count`` more training windows, then stop.

        Each iteration consumes one window, so the trainer passes the number
        of iterations left; without a limit the overlapped schedule would
        start one more window after the final iteration that nothing
        consumes.
        """
        if type(count) is not int or count < 0:
            raise ValueError("window limit must be a nonnegative integer")
        with self._lock:
            self._window_limit = self._windows_produced + count

    def _window_limit_reached(self) -> bool:
        with self._lock:
            return (
                self._window_limit is not None
                and self._windows_produced >= self._window_limit
            )

    def stop(self) -> None:
        self._stop_event.set()
        with self._policy_ready:
            self._policy_ready.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for collector in self.collectors:
            collector.stop_platypus_worker()

    def _run_on_all_collectors(self, action_name: str, action: Any) -> None:
        """Run action(collector) on every collector concurrently; raise the
        first failure after all have finished."""
        if len(self.collectors) == 1:
            action(self.collectors[0])
            return

        def guarded_action(collector: RolloutCollector) -> None:
            try:
                action(collector)
            except Exception as error:
                record_run_failure(self.run_root, error)
                raise

        with ThreadPoolExecutor(
            max_workers=len(self.collectors),
            thread_name_prefix=f"collector-{action_name}",
        ) as executor:
            futures = [
                executor.submit(guarded_action, collector)
                for collector in self.collectors
            ]
            errors: list[BaseException] = []
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{action_name} failed on {len(errors)}/{len(self.collectors)} "
                "inference servers."
            ) from errors[0]

    def prepare_servers(self) -> None:
        """Verify native servers before publishing the initial full policy."""
        if self._thread is not None:
            raise RuntimeError("Prepare servers before starting collection")
        with self._collection_lock:
            self.raise_if_failed()
            self._ensure_server_ready()

    def activate_policy_at_boundary(
        self,
        *,
        policy_iteration: int,
        activate: Callable[[], None],
        deadline_epoch: float | None = None,
    ) -> bool:
        """Publish over NCCL only after every episode in the active wave ends.

        Full weights and Flash's merged policy weights both reach the servers
        from the trainer (``activate``), never by path.
        """
        if self.config.training.weight_mode not in ("full_weight", "flash"):
            raise RuntimeError("External weight publication requires full_weight or flash mode")
        while True:
            self.raise_if_failed()
            if self._stop_event.is_set():
                raise InterruptedError("Collection stopped before weight publication")
            if deadline_epoch is not None and time.time() >= deadline_epoch:
                return False
            if self._collection_lock.acquire(timeout=1.0):
                break
        try:
            self.raise_if_failed()
            if not self._server_started:
                raise RuntimeError("Prepare servers before weight publication")
            if deadline_epoch is not None and time.time() >= deadline_epoch:
                return False
            activate()
            model_id = self._base_model_id()
            with self._policy_ready:
                self._active_policy_path = None
                self._active_policy_iteration = int(policy_iteration)
                self._active_model_id = model_id
                self._published_policy_ready = True
                self._policy_ready.notify_all()
            return True
        except BaseException as error:
            with self._policy_ready:
                self._error = error
                self._stop_event.set()
                self._policy_ready.notify_all()
            record_run_failure(self.run_root, error)
            raise
        finally:
            self._collection_lock.release()

    def publish_policy(self, policy_path: Path, *, policy_iteration: int) -> None:
        error: RuntimeError | None = None
        with self._policy_ready:
            label = "Synchronous rollout" if self._synchronous_collection else "Rollout"
            if type(policy_iteration) is not int or policy_iteration < -1:
                error = RuntimeError(
                    "A rollout policy version must be an integer at least -1."
                )
            elif self._pending_policy_path is not None:
                previous = self._pending_policy_iteration
                if policy_iteration == previous:
                    if policy_path != self._pending_policy_path:
                        error = RuntimeError(
                            f"{label} policy identity changed for version {previous}."
                        )
                elif policy_iteration != previous + 1 or (
                    self._synchronous_collection
                    and self._last_consumed_policy_iteration != previous
                ):
                    error = RuntimeError(
                        f"{label} policy publication requires the immediately next "
                        "version (and consuming the previous window when synchronous): "
                        f"pending={previous}, consumed={self._last_consumed_policy_iteration}, "
                        f"published={policy_iteration}."
                    )
            if error is None:
                self._pending_policy_path = policy_path
                self._pending_policy_iteration = policy_iteration
            else:
                # A swallowed publication error must still terminate collection
                # and be visible to the trainer's normal health checks.
                self._error = error
            self._policy_ready.notify_all()
        if error is not None:
            record_run_failure(self.run_root, error)
            raise error

    @property
    def _synchronous_collection(self) -> bool:
        training = getattr(getattr(self, "config", None), "training", None)
        return getattr(training, "rollout_schedule", "overlapped") == "synchronous"

    def _wait_for_next_policy(self) -> bool:
        """Do not prefetch while the trainer updates the preceding wave.

        Queue capacity alone does not provide this guarantee: consuming a
        queued window frees its slot before training begins. A new policy
        publication, rather than an empty queue, releases the next wave.
        """
        with self._policy_ready:
            self._estimated_window_ready_at = None
            while not self._stop_event.is_set():
                if self._error is not None:
                    raise self._error
                if self.config.training.weight_mode in ("full_weight", "flash"):
                    ready = self._published_policy_ready
                    iteration = self._active_policy_iteration
                else:
                    ready = self._pending_policy_path is not None
                    iteration = self._pending_policy_iteration
                if ready and (
                    self._last_collected_policy_iteration is None
                    or iteration > self._last_collected_policy_iteration
                ):
                    return True
                self._policy_ready.wait()
            return False

    def set_next_schedule_index(self, schedule_index: int) -> None:
        self._next_schedule_index = max(0, int(schedule_index))

    @property
    def next_schedule_index(self) -> int:
        return self._next_schedule_index

    def window_available(self) -> bool:
        """True when a collected window is already queued (or collection
        failed) — lets the trainer stop opportunistic extra train cycles the
        moment fresh data is waiting."""
        return self._error is not None or not self._buffer.empty()

    def seconds_until_window_ready(self) -> float | None:
        """Conservative ETA for whether another whole train cycle fits."""
        if self.window_available():
            return 0.0
        with self._lock:
            ready_at = self._estimated_window_ready_at
        if ready_at is None:
            return None
        return max(0.0, ready_at - time.monotonic())

    def raise_if_failed(self) -> None:
        """A background failure is fatal even if a good window is queued."""
        if self._error is not None:
            raise RuntimeError(
                f"FATAL: background rollout collection failed: {self._error}"
            ) from self._error
        raise_if_run_failed(getattr(self, "run_root", None))

    def next_window(
        self,
        *,
        timeout: float | None = None,
        check_health: Callable[[], None] | None = None,
    ) -> CollectedWindow:
        deadline = None if timeout is None else time.monotonic() + timeout
        poll_interval = 1.0
        while True:
            self.raise_if_failed()
            if check_health is not None:
                check_health()
            remaining = (
                None if deadline is None else max(0.1, deadline - time.monotonic())
            )
            queue_timeout = (
                poll_interval if remaining is None else min(poll_interval, remaining)
            )
            try:
                window = self._buffer.get(timeout=queue_timeout)
                self.raise_if_failed()
                if check_health is not None:
                    check_health()
                if self._synchronous_collection:
                    with self._policy_ready:
                        if (
                            window.policy_iteration
                            != self._last_collected_policy_iteration
                            or window.policy_iteration
                            == self._last_consumed_policy_iteration
                        ):
                            raise RuntimeError(
                                "Synchronous rollout window does not match its "
                                "single collected policy version."
                            )
                        self._last_consumed_policy_iteration = window.policy_iteration
                return window
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the next rollout window.")
                continue

    def _desired_sync_target(self) -> tuple[Path | None, int]:
        with self._lock:
            return self._pending_policy_path, self._pending_policy_iteration

    def _sync_pending_policy_if_needed(self) -> None:
        desired_policy_path, desired_policy_iteration = self._desired_sync_target()
        if self._should_sync(desired_policy_path, desired_policy_iteration):
            self._sync_policy(desired_policy_path, desired_policy_iteration)

    def _should_sync(self, policy_path: Path | None, policy_iteration: int) -> bool:
        return policy_path is not None and (
            policy_iteration > self._active_policy_iteration
            or (
                policy_iteration == self._active_policy_iteration == -1
                and self._active_policy_path is None
            )
        )

    def _sync_runtime_adapter(self, policy_path: Path, policy_iteration: int) -> None:
        sync_started_at = time.perf_counter()
        model_id = _STATIC_LORA_ALIAS
        log.info(
            "hot-loading inference adapter from trainer iteration %s as %s on %d server(s): %s",
            policy_iteration,
            model_id,
            len(self.collectors),
            policy_path,
        )
        _wait_for_adapter_ready(policy_path)
        self._run_on_all_collectors(
            "adapter-sync",
            lambda collector: self._sync_collector_runtime_adapter(
                collector,
                policy_path=policy_path,
                policy_iteration=policy_iteration,
            ),
        )
        self._active_policy_path = policy_path
        self._active_policy_iteration = policy_iteration
        self._active_model_id = model_id
        log.info(
            "policy sync complete for trainer iteration %s in %.2fs",
            policy_iteration,
            time.perf_counter() - sync_started_at,
        )

    def _sync_collector_runtime_adapter(
        self,
        collector: RolloutCollector,
        *,
        policy_path: Path,
        policy_iteration: int,
    ) -> None:
        sync_collector_runtime_adapter(
            collector,
            policy_path=policy_path,
            policy_iteration=policy_iteration,
        )

    def _sync_policy(self, policy_path: Path, policy_iteration: int) -> None:
        self._sync_runtime_adapter(policy_path, policy_iteration)

    def _ensure_server_ready(self) -> None:
        if self._server_started:
            return
        log.info(
            "ensuring %d inference server(s) at %s",
            len(self.collectors),
            ", ".join(collector._health_url() for collector in self.collectors),
        )
        managed_model_id = self._base_model_id()
        self._run_on_all_collectors(
            "server-start",
            lambda collector: collector.ensure_server(),
        )
        self._active_policy_path = None
        self._published_policy_ready = False
        self._active_policy_iteration = -1
        self._active_model_id = managed_model_id
        self._server_started = True
        log.info(
            "inference server(s) ready at %s",
            ", ".join(collector._health_url() for collector in self.collectors),
        )

    def prefix_cache_hit_rates(self) -> dict[str, float]:
        """Cumulative prefix-cache hit rate per server (0..1), for metrics."""
        rates: dict[str, float] = {}
        for server_index, collector in enumerate(self.collectors):
            counters = collector.prefix_cache_counters()
            if counters is None:
                continue
            hits, queries = counters
            rates[str(server_index)] = hits / queries
        return rates

    def windowed_prefix_cache_hit_rates(self) -> dict[str, float]:
        """Hit rate per server over the counter delta since the previous
        call (0..1). The cumulative ratio buries recent behavior under
        lifetime history; this is the rate for the window just collected.
        A counter reset (server restart) skips one reading."""
        prev: dict[str, tuple[float, float]] = getattr(self, "_prefix_cache_prev", {})
        current: dict[str, tuple[float, float]] = {}
        rates: dict[str, float] = {}
        for server_index, collector in enumerate(self.collectors):
            counters = collector.prefix_cache_counters()
            if counters is None:
                continue
            key = str(server_index)
            current[key] = counters
            prev_hits, prev_queries = prev.get(key, (0.0, 0.0))
            delta_queries = counters[1] - prev_queries
            delta_hits = counters[0] - prev_hits
            if delta_queries > 0 and delta_hits >= 0:
                rates[key] = delta_hits / delta_queries
        self._prefix_cache_prev = current
        return rates

    def _collect_window_across_servers(
        self,
        *,
        output_root: Path,
        schedule_index: int,
        scheduled_games: list[str],
        scheduled_rollout_seeds: list[str | int | None],
        window_kind: str,
    ) -> CollectedWindow:
        """Collect one logical window, splitting the scheduled rollouts
        into complete balanced passes per managed server and merging the
        results back into a single CollectedWindow in the original slot
        order."""
        if len(self.collectors) == 1:
            return self.collector.collect_window(
                output_root=output_root,
                model_id=self._active_model_id,
                policy_iteration=self._active_policy_iteration,
                schedule_index=schedule_index,
                scheduled_games=scheduled_games,
                scheduled_rollout_seeds=scheduled_rollout_seeds,
                window_kind=window_kind,
            )

        window_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        window_dir = output_root / f"window_{window_id}"
        server_count = len(self.collectors)
        if len(scheduled_rollout_seeds) != len(scheduled_games):
            raise ValueError(
                "scheduled_rollout_seeds must have the same length as scheduled_games."
            )
        # Preserve original slot indices through balanced server assignment,
        # so repeated games keep their individual seeds and result positions.
        chunks: list[list[tuple[int, str, str | int | None]]] = [
            [
                (
                    slot_index,
                    scheduled_games[slot_index],
                    scheduled_rollout_seeds[slot_index],
                )
                for slot_index in chunk_slot_indices
            ]
            for chunk_slot_indices in split_slots_into_balanced_chunks(
                scheduled_games, server_count
            )
        ]

        child_windows: dict[int, CollectedWindow] = {}

        def _collect_chunk(server_index: int) -> None:
            chunk = chunks[server_index]
            if not chunk:
                return
            child_windows[server_index] = self.collectors[server_index].collect_window(
                output_root=window_dir / f"server_{server_index}",
                model_id=self._active_model_id,
                policy_iteration=self._active_policy_iteration,
                schedule_index=schedule_index,
                scheduled_games=[game_id for _, game_id, _ in chunk],
                scheduled_rollout_seeds=[seed for _, _, seed in chunk],
                window_kind=window_kind,
            )

        with ThreadPoolExecutor(
            max_workers=server_count, thread_name_prefix="collector-window"
        ) as executor:
            futures = [
                executor.submit(_collect_chunk, server_index)
                for server_index in range(server_count)
            ]
            errors: list[BaseException] = []
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
        if errors:
            if self._stop_event.is_set() and all(
                isinstance(error, InterruptedError) for error in errors
            ):
                raise InterruptedError(
                    "rollout collection was deliberately stopped"
                ) from errors[0]
            raise RuntimeError(
                f"{window_kind} window collection failed on "
                f"{len(errors)}/{server_count} inference servers."
            ) from next(
                (error for error in errors if not isinstance(error, InterruptedError)),
                errors[0],
            )

        # Reassemble rollouts in the original slot order. Child windows order
        # rollouts by their local slot index, which maps back through chunk
        # order to the original index.
        merged: list[tuple[int, RolloutTrajectory]] = []
        for server_index, child_window in child_windows.items():
            chunk = chunks[server_index]
            if len(child_window.rollouts) != len(chunk):
                if self._stop_event.is_set():
                    # Shutdown mid-window: a child returned a partial chunk whose
                    # slot alignment is no longer reliable; drop it.
                    continue
                raise RuntimeError(
                    f"server {server_index} returned {len(child_window.rollouts)} "
                    f"rollouts for {len(chunk)} scheduled games in window {window_id}."
                )
            for (original_index, _, _), rollout in zip(
                chunk, child_window.rollouts, strict=True
            ):
                merged.append((original_index, rollout))
        merged.sort(key=lambda item: item[0])

        children = list(child_windows.values())
        return CollectedWindow(
            window_id=window_id,
            window_dir=window_dir,
            rollouts=[rollout for _, rollout in merged],
            policy_iteration=self._active_policy_iteration,
            schedule_index=max(0, int(schedule_index)),
            collection_duration_seconds=max(
                (child.collection_duration_seconds for child in children),
                default=0.0,
            ),
            launched_rollouts=sum(child.launched_rollouts for child in children),
            zero_action_rollouts=sum(child.zero_action_rollouts for child in children),
            failed_processes=sum(child.failed_processes for child in children),
            artifact_rollouts=sum(child.artifact_rollouts for child in children),
            crashed_rollouts=sum(child.crashed_rollouts for child in children),
        )

    def _sample_training_games(self, *, schedule_index: int) -> list[str]:
        games = self.config.inference.games or [self.config.inference.game]
        return sample_training_games(
            games,
            slot_count=self.config.parallel_rollouts,
            seed=self.config.training.seed,
            schedule_index=schedule_index,
        )

    def _collect_training_window(self) -> CollectedWindow:
        schedule_index = self._next_schedule_index
        with self._collection_lock:
            self._sync_pending_policy_if_needed()
            if self._active_policy_path is None and not self._published_policy_ready:
                raise RuntimeError(
                    "Rollout collection requires a published policy before "
                    "the first rollout."
                )
            policy_identity = self._active_policy_path, self._active_policy_iteration
            scheduled_games = self._sample_training_games(schedule_index=schedule_index)
            iteration_seed = self._rolling_seed.iteration("train", schedule_index)
            # Distinct, reproducible decode streams for every slot and window,
            # including repeated attempts at the same game's fixed layout.
            scheduled_rollout_seeds: list[str | int | None] = [
                iteration_seed.child(game_id, "slot", slot_index)
                for slot_index, game_id in enumerate(scheduled_games)
            ]
            runtime_minutes = self.collector._effective_max_runtime_minutes("train")
            with self._lock:
                self._estimated_window_ready_at = (
                    None
                    if runtime_minutes is None
                    else time.monotonic() + max(0.0, float(runtime_minutes) * 60.0)
                )
            try:
                window = self._collect_window_across_servers(
                    output_root=self.output_root,
                    schedule_index=schedule_index,
                    scheduled_games=scheduled_games,
                    scheduled_rollout_seeds=scheduled_rollout_seeds,
                    window_kind="train",
                )
            finally:
                with self._lock:
                    self._estimated_window_ready_at = time.monotonic()
            if (
                window.policy_iteration != policy_identity[1]
                or (self._active_policy_path, self._active_policy_iteration)
                != policy_identity
            ):
                raise RuntimeError("Rollout collection changed policy within a window.")
        self._next_schedule_index += 1
        return window

    def _run(self) -> None:
        try:
            self._ensure_server_ready()
            while not self._stop_event.is_set():
                if self._window_limit_reached():
                    return
                if self._synchronous_collection and not self._wait_for_next_policy():
                    return
                if self._buffer.full():
                    time.sleep(0.5)
                    continue
                window = self._collect_training_window()
                if self._synchronous_collection:
                    with self._policy_ready:
                        self._last_collected_policy_iteration = window.policy_iteration
                while not self._stop_event.is_set():
                    try:
                        self._buffer.put(window, timeout=0.5)
                        with self._lock:
                            self._windows_produced += 1
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, InterruptedError) and self._stop_event.is_set():
                return
            self._error = exc
            log.exception("FATAL: background rollout collection failed")
            record_run_failure(self.run_root, exc)
