"""Trainer-owned capture and admission for the native vLLM engine service.

Every generation carries exact input token IDs and a run_id#nonce stamp.
Sampling and append-only history are checked before forwarding /generate;
the returned IDs, sampled logprobs, and adapter identity are captured at this
boundary. The harness supplies only the full-game outcomes and step linkage.

No chat, OpenAI compatibility, generation probes, or alternate generation
routes are exposed. A transport or contract failure stops the training run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

import aiohttp
from aiohttp import web

from training.policy_identity import (
    PUBLISHED_IDENTITY_FIELD_SETS,
    full_weight_identity,
    policy_weights_identity,
    published_identity,
)
from training.engine_protocol import (
    EngineProtocolError,
    normalize_logit_bias,
    unpack_generation,
    unpack_output,
)
from training.gateway_lineage import LineageError, LineageLedger
from training.gateway_progress import ProgressError, ProgressLedger
from training.harness import get_trainer_support
from training.config import DEFAULT_MALFORMED_CALL_PENALTY
from training.context_contract import ContextContract

logger = logging.getLogger(__name__)

# Endpoint the harness worker POSTs game outcomes to; also the path the
# worker receives inside its spec as ``results_url``.
RESULTS_PATH = "/rl/results"

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "content-encoding",
}


def token_ids_crc32(token_ids: Sequence[int]) -> int:
    """Canonical checksum both sides of the results contract compute."""
    return zlib.crc32(",".join(str(int(t)) for t in token_ids).encode("ascii"))


@dataclass
class StepCapture:
    """One native generation, as witnessed at the serving edge."""

    run_id: str
    nonce: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    finish_reason: str
    order: int  # completion order at the gateway (diagnostics only)
    # Human-facing copies for the viewer/debug surface only — the PPO data
    # path runs entirely on the token ids above.
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)
    # Set only after comparing the served IDs with the integer wire prompt.
    native_token_prompt: bool = False
    lineage: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    adapter: dict[str, Any] = field(default_factory=dict)
    media: dict[str, Any] | None = None
    media_evidence: dict[str, Any] | None = None
    logit_bias: dict[int, float] = field(default_factory=dict)
    summary_request: bool = False
    # Only generated rows are stored; collation scatters them to causal targets.
    completion_topk_ids: np.ndarray | None = None
    completion_topk_logprobs: np.ndarray | None = None


@dataclass
class RunResult:
    """The harness's end-of-run report for one game run."""

    run_id: str
    game_id: str
    state: str
    levels_completed: int
    number_of_levels: int
    actions_per_level: list[int]
    base_actions_per_level: list[int]
    final_wallclock_seconds: float | None
    action_wallclock_seconds: list[float]
    steps: list[
        dict[str, Any]
    ]  # {nonce, action_index, output_len, output_crc32, continued}

    episode_return: float | None = None
    terminal_progress: dict[str, Any] | None = None
    progress_reconciliation: dict[str, Any] = field(default_factory=dict, compare=False)
    environment: str = "arc3"
    terminal_reward: float = 0.0
    arc2_outcome: dict[str, Any] | None = None
    rollout_slot_id: str | None = None
    rollout_seed: int | None = None
    behavior_policy_snapshot: dict[str, Any] | None = None
    serving_adapter: dict[str, Any] | None = None
    # The augmentation the slot was built with (harness_support/augmentation.py);
    # the collector checks it against its own derivation for the window.
    augmentation: dict[str, Any] | None = None
    # Settings the worker read off the agent it built (the harness's resolved
    # values, not the spec's); the harness adapter says which it requires.
    agent_settings: dict[str, Any] | None = None
    # The harness's note on how the run ended (TAAF ``GameRun.solver_note``):
    # a crashed run's traceback, a give-up's reason. Diagnostics only.
    solver_note: str | None = None
    # Trusted trainer configuration, never read from the worker's payload; no
    # cap by default.
    malformed_call_penalty: float = DEFAULT_MALFORMED_CALL_PENALTY
    max_episode_penalty: float | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        malformed_call_penalty: float = DEFAULT_MALFORMED_CALL_PENALTY,
        max_episode_penalty: float | None = None,
    ) -> "RunResult":
        # Validate the wire types before any conversion can turn True, 1.5,
        # or "1" into apparently valid action/count evidence.
        result = cls(
            run_id=payload["run_id"],
            game_id=payload["game_id"],
            state=payload["state"],
            levels_completed=payload["levels_completed"],
            number_of_levels=payload["number_of_levels"],
            actions_per_level=payload["actions_per_level"],
            base_actions_per_level=payload["base_actions_per_level"],
            final_wallclock_seconds=payload["final_wallclock_seconds"],
            action_wallclock_seconds=payload["action_wallclock_seconds"],
            steps=payload["steps"],
            episode_return=payload.get("episode_return"),
            terminal_progress=payload.get("terminal_progress"),
            progress_reconciliation=payload.get("progress_reconciliation", {}),
            environment=payload.get("environment", "arc3"),
            terminal_reward=payload.get("terminal_reward", 0.0),
            arc2_outcome=payload.get("arc2_outcome"),
            rollout_slot_id=payload.get("rollout_slot_id"),
            rollout_seed=payload.get("rollout_seed"),
            behavior_policy_snapshot=payload.get("behavior_policy_snapshot"),
            serving_adapter=payload.get("serving_adapter"),
            augmentation=payload.get("augmentation"),
            agent_settings=payload.get("agent_settings"),
            solver_note=payload.get("solver_note"),
            malformed_call_penalty=malformed_call_penalty,
            max_episode_penalty=max_episode_penalty,
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Validate outcome and action linkage before calculating task rewards."""
        label = f"run {self.run_id}"
        try:
            support = get_trainer_support(self.environment)
        except ValueError as error:
            raise GatewayError(f"{label}: unsupported environment") from error
        if type(self.terminal_reward) not in (int, float) or not math.isfinite(
            self.terminal_reward
        ):
            raise GatewayError(f"{label}: terminal reward must be finite")
        if self.augmentation is not None and not isinstance(self.augmentation, dict):
            raise GatewayError(f"{label}: augmentation must be an object or null")
        if self.agent_settings is not None and not isinstance(self.agent_settings, dict):
            raise GatewayError(f"{label}: agent_settings must be an object or null")
        if self.solver_note is not None and not isinstance(self.solver_note, str):
            raise GatewayError(f"{label}: solver_note must be a string or null")

        def count(value, name):
            if type(value) is not int or value < 0:
                raise GatewayError(f"{label}: {name} must be a nonnegative integer")

        def elapsed(value, name, *, optional=False):
            if optional and value is None:
                return
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise GatewayError(
                    f"{label}: {name} must be a finite nonnegative number"
                )

        for name in ("run_id", "game_id", "state"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise GatewayError(f"{label}: {name} must be a nonempty string")
        for name in ("levels_completed", "number_of_levels"):
            count(getattr(self, name), name)
        if self.levels_completed > self.number_of_levels:
            raise GatewayError(f"{label}: inconsistent level counts")
        for name in ("actions_per_level", "base_actions_per_level"):
            values = getattr(self, name)
            if not isinstance(values, list):
                raise GatewayError(f"{label}: {name} must be a list")
            for value in values:
                count(value, name)
        if not isinstance(self.action_wallclock_seconds, list):
            raise GatewayError(f"{label}: action_wallclock_seconds must be a list")
        total_actions = len(self.action_wallclock_seconds)
        if sum(self.actions_per_level) != total_actions:
            raise GatewayError(
                f"{label}: actions_per_level must sum to the action history length"
            )
        elapsed(self.final_wallclock_seconds, "final_wallclock_seconds", optional=True)
        previous_time = 0.0
        for value in self.action_wallclock_seconds:
            elapsed(value, "action_wallclock_seconds")
            if value < previous_time:
                raise GatewayError(f"{label}: action wallclocks must be chronological")
            if (
                self.final_wallclock_seconds is not None
                and value > self.final_wallclock_seconds
            ):
                raise GatewayError(f"{label}: action wallclock exceeds final duration")
            previous_time = value

        if not isinstance(self.steps, list) or any(
            not isinstance(step, dict) for step in self.steps
        ):
            raise GatewayError(f"{label}: steps must be a list of objects")
        if self.episode_return is not None:
            if type(self.episode_return) not in (int, float) or not math.isfinite(
                self.episode_return
            ):
                raise GatewayError(f"{label}: episode_return must be finite")
            for step in self.steps:
                reward = step.get("reward")
                if type(reward) not in (int, float) or not math.isfinite(reward):
                    raise GatewayError(f"{label}: step reward must be finite")
            if not math.isclose(
                sum(step["reward"] for step in self.steps) + self.terminal_reward,
                self.episode_return,
                rel_tol=1e-9,
                abs_tol=1e-7,
            ):
                raise GatewayError(
                    f"{label}: episode_return does not equal reported step rewards"
                )
        # The harness's outcome semantics, applied here by the trainer rather
        # than trusted from the worker's report.
        try:
            support.validate_run_result(self)
        except ValueError as error:
            raise GatewayError(f"{label}: {error}") from error
        nonces = set()
        action_starts = []
        for step in self.steps:
            nonce = step.get("nonce")
            if not isinstance(nonce, str) or not nonce or nonce in nonces:
                raise GatewayError(f"{label}: every reported step needs a unique nonce")
            nonces.add(nonce)
            count(step.get("output_len"), f"step {nonce} output_len")
            count(step.get("output_crc32"), f"step {nonce} output_crc32")
            if "action_index" not in step:
                raise GatewayError(f"{label}: step {nonce} lacks action_index")
            action = step["action_index"]
            if action is not None:
                count(action, f"step {nonce} action_index")
                if action >= total_actions or (
                    action_starts and action <= action_starts[-1]
                ):
                    raise GatewayError(
                        f"{label}: action_index must increase within executed actions"
                    )
                action_starts.append(action)
        if total_actions and (not action_starts or action_starts[0] != 0):
            raise GatewayError(f"{label}: executed actions lack reported step linkage")


@dataclass
class WindowCapture:
    """Everything the gateway collected for one rollout window."""

    captures: dict[tuple[str, str], StepCapture] = field(default_factory=dict)
    results: dict[str, RunResult] = field(default_factory=dict)


class GatewayError(RuntimeError):
    """A capture-contract violation that must stop the training run."""


@dataclass(frozen=True)
class SamplingPolicy:
    """Trainer-owned request contract, independent of harness presets.

    Training uses T=1, top_p=1 and top_k=-1; evaluation can deliberately
    choose another distribution. Penalties and hidden processors are never
    permitted. Missing sampling fields on real draws are errors, since vLLM
    can otherwise inherit checkpoint generation defaults.
    """

    model: str
    temperature: float
    top_p: float
    top_k: int
    context: ContextContract
    max_runtime_seconds: float | None = None
    expected_adapter: dict[str, Any] | None = None
    require_lora: bool = False
    require_full_weight: bool = False
    # Flash: the base name serving merged policy weights (published over /weights/*).
    require_policy_weights: bool = False
    require_progress: bool = False
    malformed_call_penalty: float = DEFAULT_MALFORMED_CALL_PENALTY
    max_episode_penalty: float | None = None
    score_centering_top_k: int = 0
    vocab_size: int = 0

    def __post_init__(self) -> None:
        if self.expected_adapter is not None:
            object.__setattr__(
                self,
                "expected_adapter",
                _adapter_identity(self.expected_adapter, self.model),
            )
        if self.max_runtime_seconds is not None and (
            not math.isfinite(self.max_runtime_seconds) or self.max_runtime_seconds <= 0
        ):
            raise ValueError("rollout runtime must be finite and positive")

    def validate(self, body: dict[str, Any]) -> None:
        sampling = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else -1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
            "min_tokens": 0,
        }
        neutral = {
            "n": 1,
            "ignore_eos": False,
            "skip_special_tokens": False,
            "logprobs": 0,
        }
        allowed = (
            set(sampling)
            | set(neutral)
            | {
                "model",
                "user",
                "arc3_lineage",
                "arc3_progress",
                "prompt",
                "media",
                "max_tokens",
                "seed",
                "logit_bias",
            }
        )
        if not isinstance(body, dict):
            raise GatewayError("native generation must be a JSON object")
        unknown = set(body) - allowed
        if unknown:
            raise GatewayError(f"unapproved vLLM request fields: {sorted(unknown)}")
        if body.get("model") != self.model:
            raise GatewayError(
                f"wrong policy model: expected {self.model!r}, got {body.get('model')!r}"
            )
        stamp = body.get("user")
        if (
            not isinstance(stamp, str)
            or "#" not in stamp
            or not all(stamp.rpartition("#")[::2])
        ):
            raise GatewayError("native generation requires a run_id#nonce stamp")
        for key, expected in sampling.items():
            value = body.get(key)
            if key == "top_k" and expected == -1 and type(value) is int and value == 0:
                value = body[key] = -1
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value != expected
            ):
                raise GatewayError(
                    f"vLLM {key} must be explicitly {expected!r}; got {value!r}"
                )
        for key, expected in neutral.items():
            if key in body and (
                type(body[key]) is not type(expected) or body[key] != expected
            ):
                raise GatewayError(
                    f"vLLM {key} must be {expected!r}; got {body[key]!r}"
                )
        if "seed" in body and type(body["seed"]) is not int:
            raise GatewayError("vLLM seed must be an integer")
        prompt = body.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(t) is not int or t < 0 for t in prompt)
        ):
            raise GatewayError(
                "native generation requires one exact integer token prompt"
            )
        limit = body.get("max_tokens")
        if type(limit) is not int or limit <= 0:
            raise GatewayError(
                "an explicit completion limit must be a positive integer"
            )
        over_trigger = len(prompt) > self.context.compaction_trigger_tokens
        if "logit_bias" in body and not over_trigger:
            raise GatewayError("logit_bias is permitted only on over-trigger requests")
        summary_request = self.is_summary_request(body)
        if summary_request:
            try:
                bias = normalize_logit_bias(body.get("logit_bias", {}))
            except EngineProtocolError as error:
                raise GatewayError(str(error)) from error
            expected_bias = {
                token: -100.0 for token in self.context.summary_forbidden_token_ids
            }
            if bias != expected_bias:
                raise GatewayError(
                    "logit_bias must exactly match the over-trigger contract"
                )
            if "logit_bias" in body:
                body["logit_bias"] = bias
        cap = (
            self.context.max_summary_completion_tokens
            if summary_request
            else self.context.max_completion_tokens
        )
        if summary_request and cap == 0:
            raise GatewayError(
                "over-trigger requests require a positive summary completion cap"
            )
        if limit > cap:
            raise GatewayError(
                f"an explicit completion limit in [1, {cap}] is required"
            )
        if len(prompt) + limit > self.context.max_served_sequence_tokens:
            raise GatewayError(
                "prompt plus completion budget exceeds trainer context limit"
            )

    def is_summary_request(self, body: dict[str, Any]) -> bool:
        """Distinguish biased/overflow draws from ordinary budget-limited replies."""
        return len(body["prompt"]) > self.context.compaction_trigger_tokens and (
            "logit_bias" in body
            or len(body["prompt"])
            + body["max_tokens"]
            + self.context.safety_margin_tokens
            > self.context.context_window_tokens
        )


class CaptureGateway:
    """Validated serving boundary and window store on a background thread."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        port: int,
        host: str = "127.0.0.1",
        on_contract_violation: Callable[[GatewayError], None] | None = None,
    ):
        # The native service exposes routes directly at its root URL.
        self.upstream = upstream_base_url.rstrip("/")
        if self.upstream.endswith("/v1"):
            raise ValueError("native vLLM requires a root URL without /v1")
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._pending_generations = 0
        self._lineage = LineageLedger()
        self._progress = ProgressLedger()
        self._window = WindowCapture()
        self._order = 0
        self._runner: web.AppRunner | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._policy: SamplingPolicy | None = None
        self._contract_error: str | None = None
        self._on_contract_violation = on_contract_violation

    # ------------------------------------------------------------------ store

    def drain_window(
        self,
        *,
        timeout_seconds: float = 300.0,
        check_health: Callable[[], None] | None = None,
        require_results: bool = False,
    ) -> WindowCapture:
        """Close a window only after its outstanding generations are captured.

        A client deadline can end the harness before its upstream generation
        finishes. The result callback alone therefore does not close the token
        stream. Keep those responses in this window, including disconnected
        requests, before changing the store or the serving policy.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("drain timeout_seconds must be finite and nonnegative")
        deadline = time.monotonic() + timeout_seconds
        logged_wait = False
        failure = None
        while True:
            self.raise_if_invalid()
            if check_health is not None:
                try:
                    check_health()
                except InterruptedError:
                    # A deliberate stop is not a serving contract violation.
                    raise
                except Exception as error:
                    raise GatewayError(self._invalidate(error)) from error
            with self._idle:
                if self._contract_error is not None:
                    raise GatewayError(self._contract_error)
                if not self._pending_generations:
                    orphaned = {run_id for run_id, _ in self._window.captures} - set(
                        self._window.results
                    )
                    if require_results and orphaned:
                        failure = GatewayError(
                            f"captured generations lack result callbacks for runs {sorted(orphaned)}"
                        )
                        break
                    self._lineage.drain()
                    for run in self._window.results.values():
                        if run.terminal_progress is not None:
                            run.progress_reconciliation = self._progress.close(
                                run.run_id, [step["nonce"] for step in run.steps]
                            )
                    window = self._window
                    self._window = WindowCapture()
                    self._order = 0
                    return window
                pending = self._pending_generations
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not logged_wait:
                    logger.info(
                        "Waiting for %d outstanding gateway generations before closing window",
                        pending,
                    )
                    logged_wait = True
                self._idle.wait(timeout=min(0.25, remaining))
        error = failure or GatewayError(
            f"timed out after {timeout_seconds:g}s waiting for {pending} "
            "outstanding gateway generations; window was not drained"
        )
        raise GatewayError(self._invalidate(error)) from error

    def set_policy(self, policy: SamplingPolicy) -> None:
        """Called before dispatch, never inferred from a harness request."""
        self.raise_if_invalid()
        if policy.require_lora and (
            policy.expected_adapter is None or policy.expected_adapter["id"] == 0
        ):
            raise GatewayError(
                self._invalidate(
                    GatewayError("production rollouts require a loaded LoRA adapter")
                )
            )
        if policy.require_full_weight and (
            policy.require_lora
            or policy.expected_adapter is None
            or full_weight_identity(policy.expected_adapter) is None
        ):
            raise GatewayError(
                self._invalidate(
                    GatewayError(
                        "production rollouts require a published full-weight policy"
                    )
                )
            )
        if policy.require_policy_weights and (
            policy.require_lora
            or policy.require_full_weight
            or policy.expected_adapter is None
            or policy_weights_identity(policy.expected_adapter) is None
        ):
            raise GatewayError(
                self._invalidate(
                    GatewayError(
                        "production rollouts require applied merged policy weights"
                    )
                )
            )
        with self._lock:
            active = (
                self._pending_generations
                or self._lineage.runs
                or self._window.captures
                or self._window.results
            )
            if not active:
                self._policy = policy
        if active:
            raise GatewayError(
                self._invalidate(
                    GatewayError(
                        "cannot change serving policy before the previous window is drained"
                    )
                )
            )

    def raise_if_invalid(self) -> None:
        with self._lock:
            error = self._contract_error
        if error is not None:
            raise GatewayError(error)

    def _invalidate(self, error: Exception) -> str:
        # An HTTP error alone can be swallowed by the harness. Poison the
        # trainer's window too, permanently for this gateway's lifetime.
        with self._idle:
            first_error = self._contract_error is None
            if self._contract_error is None:
                self._contract_error = f"rollout serving contract violated: {error}"
            message = self._contract_error
            self._idle.notify_all()
        logger.error("FATAL: %s", message)
        if first_error and self._on_contract_violation is not None:
            failure = GatewayError(message)
            failure.__cause__ = error
            self._on_contract_violation(failure)
        return message

    def _reject(self, error: Exception) -> web.Response:
        message = self._invalidate(error)
        return web.json_response({"error": message}, status=400)

    def _validate_request(self, body: dict[str, Any]) -> None:
        self.raise_if_invalid()
        with self._lock:
            policy = self._policy
        if policy is None:
            raise GatewayError("trainer has not configured the window sampling policy")
        policy.validate(body)

    def _admit_generation(self, body: dict[str, Any]) -> None:
        metadata = body.get("arc3_lineage")
        run_id, _, nonce = body["user"].rpartition("#")
        with self._lock:
            if self._policy is None:
                raise GatewayError("trainer has not configured the history policy")
            try:
                snapshot = None
                if self._policy.require_progress or "arc3_progress" in body:
                    chain = self._lineage.runs.get(run_id)
                    snapshot = self._progress.validate_admission(
                        run_id,
                        nonce,
                        body.get("arc3_progress"),
                        previous_nonce=chain.nonces[-1]
                        if chain and chain.nonces
                        else None,
                        require_decision_clock=self._policy.require_progress,
                    )
                self._lineage.admit(
                    run_id,
                    nonce,
                    body["prompt"],
                    metadata,
                    self._policy.context.lineage_policy(
                        max_completion_tokens=body["max_tokens"]
                    ),
                    native_token_prompt=True,
                    media=body.get("media"),
                    summary_request=self._policy.is_summary_request(body),
                )
                if snapshot is not None:
                    self._progress.record_admission(run_id, nonce, snapshot)
            except (LineageError, ProgressError) as error:
                raise GatewayError(str(error)) from error

    def _record_capture(self, capture: StepCapture) -> None:
        with self._lock:
            key = (capture.run_id, capture.nonce)
            previous = self._window.captures.get(key)
            if previous is not None:
                raise GatewayError(
                    f"run {capture.run_id} nonce {capture.nonce}: duplicate generation capture"
                )
            try:
                capture.lineage = self._lineage.complete(
                    capture.run_id,
                    capture.nonce,
                    capture.prompt_token_ids,
                    capture.completion_token_ids,
                )
            except LineageError as error:
                raise GatewayError(str(error)) from error
            capture.progress = self._progress.snapshots.get(capture.run_id, {}).get(
                capture.nonce, {}
            )
            capture.order = self._order
            self._order += 1
            self._window.captures[key] = capture

    def _record_results(self, runs: list[RunResult]) -> None:
        with self._lock:
            pending: dict[str, RunResult] = {}
            for run in runs:
                previous = pending.get(run.run_id, self._window.results.get(run.run_id))
                if previous is not None:
                    if previous != run:
                        raise GatewayError(
                            f"run {run.run_id}: conflicting result evidence for a repeated run_id"
                        )
                else:
                    pending[run.run_id] = run
            for run in pending.values():
                # A crashed run is a scored terminal: its agent raised and the
                # harness contained it (on_agent_error="crash_game"), scoring
                # what it had reached. A serving failure never reaches here as
                # one: the generation handler poisons the window first.
                for step in run.steps:
                    capture = self._window.captures.get((run.run_id, step["nonce"]))
                    if capture is None:
                        raise GatewayError(
                            f"run {run.run_id}: reported step lacks a completed capture"
                        )
                    if (
                        step.get("exact_history") is not True
                        or type(step.get("continued")) is not bool
                        or step["continued"] != (capture.lineage["kind"] == "continue")
                        or type(step.get("history_epoch")) is not int
                        or step["history_epoch"] != capture.lineage["epoch"]
                    ):
                        raise GatewayError(
                            f"run {run.run_id}: callback disagrees with admitted history lineage"
                        )
                    if step["output_len"] != len(capture.completion_token_ids) or step[
                        "output_crc32"
                    ] != token_ids_crc32(capture.completion_token_ids):
                        raise GatewayError(
                            f"run {run.run_id}: callback output differs from the captured draw"
                        )
                try:
                    if (
                        (self._policy and self._policy.require_progress)
                        or run.run_id in self._progress.snapshots
                        or run.terminal_progress is not None
                    ):
                        if run.episode_return is None:
                            raise GatewayError(
                                f"run {run.run_id}: progress requires callback rewards"
                            )
                        for step in run.steps:
                            snapshot = self._progress.snapshots.get(run.run_id, {}).get(
                                step["nonce"], {}
                            )
                            clock = snapshot.get("data", {}).get(
                                "decision_elapsed_seconds"
                            )
                            if (
                                clock is not None
                                and step.get("decision_elapsed_seconds") != clock
                            ):
                                raise GatewayError(
                                    f"run {run.run_id}: callback decision timestamp differs from capture"
                                )
                        self._progress.reconcile(
                            run.run_id,
                            run.terminal_progress,
                            run.steps,
                            run.episode_return,
                            terminal_reward=run.terminal_reward,
                        )
                    runtime = self._policy.max_runtime_seconds if self._policy else None
                    self._lineage.report(
                        run.run_id,
                        [str(s["nonce"]) for s in run.steps],
                        # One terminal draw may go unreported where the run
                        # was cut at it: by the deadline, or by its agent
                        # crashing before the draw was processed.
                        deadline_reached=run.state == "crashed"
                        or (
                            runtime is not None
                            and run.final_wallclock_seconds is not None
                            and (
                                run.state
                                in get_trainer_support(
                                    run.environment
                                ).DEADLINE_TERMINAL_STATES
                            )
                            and run.final_wallclock_seconds >= runtime
                        ),
                    )
                except (LineageError, ProgressError) as error:
                    raise GatewayError(str(error)) from error
            # Validate the whole batch before publishing any new reports.
            self._window.results.update(pending)

    # ------------------------------------------------------------- lifecycle

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def results_url(self) -> str:
        return f"http://{self.host}:{self.port}{RESULTS_PATH}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="rollout-gateway", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            raise GatewayError("rollout gateway failed to start within 30s")
        if self._startup_error is not None:
            raise GatewayError(
                f"rollout gateway startup failed: {self._startup_error!r}"
            )

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._thread = None
        self._loop = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._start_app())
            self._ready.set()
            loop.run_forever()
        except BaseException as error:  # noqa: BLE001 - surfaced to start()
            self._startup_error = error
            self._ready.set()
        finally:
            try:
                if self._runner is not None:
                    loop.run_until_complete(self._runner.cleanup())
                loop.run_until_complete(self._session.close())
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            loop.close()

    async def _start_app(self) -> None:
        # No total timeout: a completion under heavy batch load can take
        # minutes; connect failures still surface promptly.
        # Rollout slots bound concurrency. The default 100-connection pool
        # queues part of ARC-2's 120-puzzle evaluation until connect times out.
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None, connect=30.0),
        )
        app = web.Application(
            client_max_size=256 * 1024 * 1024,
            middlewares=[self._track_generation_requests],
        )
        app.router.add_post("/generate", self._handle_generate)
        app.router.add_post(RESULTS_PATH, self._handle_results)
        app.router.add_route("*", "/{tail:.*}", self._handle_passthrough)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        self._runner = runner
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.port = int(site._server.sockets[0].getsockname()[1])

    # -------------------------------------------------------------- handlers

    @web.middleware
    async def _track_generation_requests(self, request: web.Request, handler):
        if request.method != "POST" or request.path != "/generate":
            return await handler(request)
        with self._idle:
            self._pending_generations += 1
        try:
            return await handler(request)
        except Exception as error:
            failure = GatewayError(
                f"generation handler failed: {type(error).__name__}: {error}"
            )
            failure.__cause__ = error
            return self._reject(failure)
        finally:
            with self._idle:
                self._pending_generations -= 1
                self._idle.notify_all()

    async def _handle_results(self, request: web.Request) -> web.Response:
        try:
            self.raise_if_invalid()
            payload = await request.json()
            runs_payload = payload.get("runs")
            if not isinstance(runs_payload, list):
                raise GatewayError("results payload must carry a 'runs' list")
            if self._policy is None:
                raise GatewayError(
                    "trainer has not configured the window sampling policy"
                )
            runs = [
                RunResult.from_payload(
                    entry,
                    malformed_call_penalty=self._policy.malformed_call_penalty,
                    max_episode_penalty=self._policy.max_episode_penalty,
                )
                for entry in runs_payload
            ]
            self._record_results(runs)
        except Exception as error:
            return self._reject(GatewayError(f"malformed run result: {error}"))
        return web.json_response({"accepted": len(runs)})

    async def _handle_generate(self, request: web.Request) -> web.Response:
        try:
            wire = await request.json()
            if self._policy is None:
                raise GatewayError(
                    "trainer has not configured the window sampling policy"
                )
            body = unpack_generation(
                wire, model=self._policy.model, adapter=self._policy.expected_adapter
            )
            self._validate_request(body)
            self._admit_generation(body)
            # Task metadata terminates at the trainer gateway.
            wire = {**wire, "request_context": {}}
            wire["sampling_params"] = {
                **{
                    key: value
                    for key, value in wire["sampling_params"].items()
                    if key != "extra_args"
                },
                "logprobs": self._policy.score_centering_top_k,
            }
            if "logit_bias" in body:
                wire["sampling_params"]["logit_bias"] = body["logit_bias"]
            status, raw, content_type = await self._upstream(
                "POST",
                f"{self.upstream}/generate",
                json=wire,
                headers=_forward_headers(request.headers),
            )
            if status != 200:
                _log_upstream_error("POST /generate", status, raw, body)
                raise GatewayError(
                    f"native vLLM generation failed: HTTP {status}: {raw[:600].decode('utf-8', 'replace')}"
                )
            envelope = json.loads(raw)
            head = envelope.pop("sampler_topk", None)
            response = unpack_output(envelope)
            if head is not None:
                raw = json.dumps(envelope).encode()
            if response.get("request_id") != body["user"]:
                raise GatewayError(
                    "vLLM response request_id differs from the admitted request"
                )
            if response.get("model") != body["model"]:
                raise GatewayError(
                    "vLLM response model differs from the requested policy"
                )
            adapter = _adapter_identity(response.get("adapter"), body["model"])
            if (
                self._policy.expected_adapter is not None
                and adapter != self._policy.expected_adapter
            ):
                raise GatewayError(
                    "vLLM response adapter identity differs from the trainer's pinned policy"
                )
            run_id, _, nonce = body["user"].rpartition("#")
            capture = _capture_from_native_response(
                run_id=run_id, nonce=nonce, response=response
            )
            if self._policy.score_centering_top_k:
                from training.score_centering import unpack_head

                capture.completion_topk_ids, capture.completion_topk_logprobs = (
                    unpack_head(
                        head,
                        tokens=len(capture.completion_token_ids),
                        k=self._policy.score_centering_top_k,
                        vocab_size=self._policy.vocab_size,
                    )
                )
            elif head is not None:
                raise GatewayError("unexpected sampler top-k evidence")
            capture.logit_bias = dict(body.get("logit_bias", {}))
            capture.summary_request = self._policy.is_summary_request(body)
            capture.media = body.get("media")
            capture.media_evidence = response.get("media_evidence")
            expected_media = (
                {
                    key: value
                    for key, value in capture.media.items()
                    if key not in {"version", "images"}
                }
                if capture.media is not None
                else None
            )
            if capture.media_evidence != expected_media:
                raise GatewayError(
                    "server media processing differs from the admitted image evidence"
                )
            if capture.prompt_token_ids != body["prompt"]:
                raise GatewayError(
                    "vLLM served prompt differs from the exact wire token prompt"
                )
            self._validate_capture(capture, body)
            self._record_capture(capture)
        except (
            GatewayError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            OverflowError,
        ) as error:
            return self._reject(error)
        return web.Response(body=raw, status=status, content_type=content_type)

    def _validate_capture(self, capture: StepCapture, body: dict[str, Any]) -> None:
        if any(not math.isfinite(p) or p > 0 for p in capture.completion_logprobs):
            raise GatewayError("sampled token logprobs must be finite and nonpositive")
        if len(capture.completion_token_ids) > body["max_tokens"]:
            raise GatewayError("vLLM returned more tokens than requested")
        if (
            self._policy is None
            or len(capture.prompt_token_ids) + len(capture.completion_token_ids)
            > self._policy.context.max_served_sequence_tokens
        ):
            raise GatewayError("served sequence exceeds trainer context limit")

    async def _upstream(
        self, method: str, url: str, **kwargs: Any
    ) -> tuple[int, bytes, str]:
        """Send generation once; only read-only utility requests may retry."""
        last_error: BaseException | None = None
        generating = url.endswith("/generate")
        attempts = 1 if generating else 2
        for attempt in range(attempts):
            try:
                async with self._session.request(method, url, **kwargs) as upstream:
                    raw = await upstream.read()
                    return upstream.status, raw, upstream.content_type
            except (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
            ) as error:
                last_error = error
                logger.warning(
                    "rollout gateway: upstream connection dropped on %s %s (attempt %d): %s",
                    method,
                    url,
                    attempt + 1,
                    error,
                )
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.5)
        body = json.dumps(
            {
                "error": f"rollout gateway: upstream unreachable after {attempts} attempt(s); "
                f"generation retries are forbidden: {last_error}"
            }
        ).encode("utf-8")
        return 502, body, "application/json"

    async def _handle_passthrough(self, request: web.Request) -> web.Response:
        # Discovery is read-only; adapter controls belong to the trainer.
        if (request.method, request.path) not in {
            ("GET", "/health"),
            ("GET", "/metadata"),
            ("GET", "/runtime"),
            ("GET", "/models"),
            ("GET", "/metrics"),
        }:
            return self._reject(
                GatewayError(
                    f"unapproved serving route: {request.method} {request.path}"
                )
            )
        data = await request.read()
        status, raw, content_type = await self._upstream(
            request.method,
            f"{self.upstream}/{request.match_info['tail']}",
            data=data if data else None,
            params=request.rel_url.query,
            headers=_forward_headers(request.headers),
        )
        if status >= 400:
            _log_upstream_error(
                f"{request.method} /{request.match_info['tail']}", status, raw, data
            )
        return web.Response(body=raw, status=status, content_type=content_type)


def _log_upstream_error(route: str, status: int, raw: bytes, request_body: Any) -> None:
    """Upstream rejections are otherwise invisible: the harness is a black
    box and its children only see a closed host connection. Log enough of
    both sides to diagnose without a reproduction."""
    try:
        request_text = (
            request_body.decode("utf-8", "replace")
            if isinstance(request_body, (bytes, bytearray))
            else json.dumps(request_body, default=str)
        )
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        request_text = repr(request_body)
    logger.error(
        "rollout gateway: upstream %s -> %d; response=%s; request=%s",
        route,
        status,
        raw[:600].decode("utf-8", "replace"),
        request_text[:400],
    )


def _forward_headers(headers: Any) -> dict[str, str]:
    return {
        key: value for key, value in headers.items() if key.lower() not in _HOP_BY_HOP
    }


def _adapter_identity(value: Any, model: str) -> dict[str, Any]:
    fields = {"name", "id", "path", "server_instance_id"}
    if not isinstance(value, dict) or set(value) not in (
        fields,
        *(fields | published for published in PUBLISHED_IDENTITY_FIELD_SETS),
    ):
        raise GatewayError("native vLLM lacks exact adapter provenance")
    if value["name"] != model or type(value["id"]) is not int or value["id"] < 0:
        raise GatewayError("native vLLM adapter differs from the requested policy")
    if (
        not isinstance(value["server_instance_id"], str)
        or not value["server_instance_id"]
    ):
        raise GatewayError("native vLLM adapter lacks its server instance ID")
    if (value["id"] == 0 and value["path"] is not None) or (
        value["id"] > 0 and (not isinstance(value["path"], str) or not value["path"])
    ):
        raise GatewayError(
            "native vLLM adapter path does not match its base/LoRA identity"
        )
    try:
        published_identity(value)
    except ValueError as error:
        raise GatewayError(str(error)) from error
    return dict(value)


def _response_token_ids(value: Any, *, prompt: bool = False) -> list[int]:
    if (
        not isinstance(value, list)
        or (prompt and not value)
        or any(type(t) is not int or t < 0 for t in value)
    ):
        raise GatewayError("vLLM response token IDs must be exact nonnegative integers")
    return list(value)


def _capture_from_native_response(
    *, run_id: str, nonce: str, response: dict[str, Any]
) -> StepCapture:
    prompt_ids = _response_token_ids(response.get("prompt_token_ids"), prompt=True)
    output_ids = _response_token_ids(response.get("output_token_ids"))
    logprobs = response.get("output_logprobs")
    if (
        not isinstance(logprobs, list)
        or len(logprobs) != len(output_ids)
        or any(type(p) not in (int, float) for p in logprobs)
    ):
        raise GatewayError(
            "sampled output logprobs must align exactly with output token IDs"
        )
    if response.get("finish_reason") not in {"stop", "length", "abort"}:
        raise GatewayError("native generation lacks a valid finish_reason")
    if response["finish_reason"] == "abort":
        raise GatewayError("native generation aborted before a complete capture")
    if not isinstance(response.get("text"), str):
        raise GatewayError("native generation text must be a string")
    cached = response.get("num_cached_tokens")
    if type(cached) is not int or not 0 <= cached <= len(prompt_ids):
        raise GatewayError("native generation cached token count is invalid")
    return StepCapture(
        run_id=run_id,
        nonce=nonce,
        prompt_token_ids=prompt_ids,
        completion_token_ids=output_ids,
        completion_logprobs=list(logprobs),
        finish_reason=response["finish_reason"],
        order=0,
        native_token_prompt=True,
        assistant_message={"role": "assistant", "content": response["text"]},
        model=response["model"],
        adapter=dict(response["adapter"]),
    )
