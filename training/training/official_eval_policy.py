from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from training.policy_identity import full_weight_identity


_REQUEST_FILENAME = "official_eval_policy_request.json"
_READY_FILENAME = "official_eval_policy_ready.json"
_BEST_FILENAME = "best_eval.json"
_PROTOCOL_VERSION = 2


@dataclass(frozen=True, slots=True)
class EvalPolicy:
    iteration: int
    model_id: str
    identities: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class FullWeightEvalRequest:
    request_id: str
    cycle_index: int
    server_urls: tuple[str, ...]


def clear_full_weight_eval_handshake(run_root: Path) -> None:
    """Discard request state from a previous trainer/eval process generation."""
    (run_root / _REQUEST_FILENAME).unlink(missing_ok=True)
    (run_root / _READY_FILENAME).unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return payload


def record_full_weight_eval_result(run_root: Path, record: dict[str, Any]) -> None:
    """Retain the best completed evaluation's model iteration, not cycle index."""
    score = float(record["avg_arc_score"])
    if not math.isfinite(score):
        raise ValueError(f"Invalid evaluation score: {score}")
    best = _read_json(run_root / _BEST_FILENAME)
    if best is None or score > float(best["avg_arc_score"]):
        # Publish before the next request replaces the active evaluation pin.
        _write_json_atomic(
            run_root / _BEST_FILENAME,
            {
                key: record[key]
                for key in ("iteration", "cycle_index", "avg_arc_score", "completed_at")
            },
        )


def protected_full_weight_eval_iterations(run_root: Path) -> set[int]:
    # Read active first, then best: during a concurrent evaluation transition
    # this sees either the old active policy or its completed best record.
    active = _read_json(run_root / _READY_FILENAME)
    best = _read_json(run_root / _BEST_FILENAME)
    protected = set()
    if active is not None:
        protected.add(int(active["policy_iteration"]))
    if best is not None:
        protected.add(int(best["iteration"]))
    return protected


class FullWeightEvalPolicyProvider:
    """Request a trainer-side hot publication before starting an eval cycle."""

    def __init__(
        self,
        *,
        run_root: Path,
        base_model_id: str,
        server_urls: Sequence[str],
        poll_seconds: float = 1.0,
        request_timeout_seconds: float | None = 30.0 * 60.0,
        initial_request_timeout_seconds: float | None = None,
        liveness_check: Callable[[], None] | None = None,
    ) -> None:
        urls = tuple(str(url).strip() for url in server_urls if str(url).strip())
        if not urls:
            raise ValueError("Full-weight eval requires at least one server URL.")
        self._run_root = run_root
        self._base_model_id = base_model_id
        self._server_urls = urls
        self._poll_seconds = float(poll_seconds)
        self._request_timeout_seconds = (
            None if request_timeout_seconds is None else float(request_timeout_seconds)
        )
        if (
            self._request_timeout_seconds is not None
            and self._request_timeout_seconds <= 0.0
        ):
            raise ValueError("Full-weight eval request timeout must be positive.")
        if self._request_timeout_seconds is None and liveness_check is None:
            raise ValueError(
                "Full-weight eval waiting requires a liveness check or timeout"
            )
        self._initial_request_timeout_seconds = float(
            (self._request_timeout_seconds or 1800.0)
            if initial_request_timeout_seconds is None
            else initial_request_timeout_seconds
        )
        if self._initial_request_timeout_seconds <= 0.0:
            raise ValueError(
                "Full-weight initial eval request timeout must be positive."
            )
        self._has_acquired_policy = False
        self._liveness_check = liveness_check

    def acquire(self, *, cycle_index: int) -> EvalPolicy:
        request_id = uuid.uuid4().hex
        _write_json_atomic(
            self._run_root / _REQUEST_FILENAME,
            {
                "version": _PROTOCOL_VERSION,
                "request_id": request_id,
                "cycle_index": int(cycle_index),
                "server_urls": list(self._server_urls),
            },
        )
        ready_path = self._run_root / _READY_FILENAME
        timeout_seconds = (
            self._request_timeout_seconds
            if self._has_acquired_policy
            else self._initial_request_timeout_seconds
        )
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        while True:
            if self._liveness_check is not None:
                self._liveness_check()
            payload = _read_json(ready_path)
            if payload is not None and payload.get("request_id") == request_id:
                if int(payload.get("version", 0)) != _PROTOCOL_VERSION:
                    raise RuntimeError("Unsupported full-weight eval policy protocol.")
                from training.inference import _validated_adapter_identity

                identities = payload.get("publication_identities")
                if not isinstance(identities, list) or len(identities) != len(
                    self._server_urls
                ):
                    raise RuntimeError(
                        "Full-weight eval acknowledgement lacks publication identities"
                    )
                checked = tuple(
                    _validated_adapter_identity(identity, self._base_model_id)
                    for identity in identities
                )
                publications = [full_weight_identity(identity) for identity in checked]
                if any(
                    publication is None or publication != publications[0]
                    for publication in publications
                ):
                    raise RuntimeError(
                        "Full-weight eval servers acknowledged different policies"
                    )
                if payload.get("cycle_index") != cycle_index:
                    raise RuntimeError(
                        "Full-weight eval acknowledgement has a different cycle"
                    )
                policy = EvalPolicy(
                    iteration=int(payload["policy_iteration"]),
                    model_id=self._base_model_id,
                    identities=checked,
                )
                self._has_acquired_policy = True
                return policy
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the trainer to publish a full-weight "
                    "policy to the official-eval servers."
                )
            time.sleep(self._poll_seconds)


class FullWeightEvalRequestInbox:
    """Trainer-side half of the cycle-boundary request/ack handshake."""

    def __init__(self, *, run_root: Path, poll_seconds: float = 1.0) -> None:
        self._run_root = run_root
        self._poll_seconds = float(poll_seconds)
        self._last_acknowledged_request_id: str | None = None

    def pending(self) -> FullWeightEvalRequest | None:
        payload = _read_json(self._run_root / _REQUEST_FILENAME)
        if payload is None:
            return None
        if int(payload.get("version", 0)) != _PROTOCOL_VERSION:
            raise RuntimeError("Unsupported full-weight eval policy protocol.")
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            raise RuntimeError("Full-weight eval policy request has no request_id.")
        if request_id == self._last_acknowledged_request_id:
            return None
        raw_urls = payload.get("server_urls")
        if not isinstance(raw_urls, list) or not raw_urls:
            raise RuntimeError("Full-weight eval policy request has no server URLs.")
        return FullWeightEvalRequest(
            request_id=request_id,
            cycle_index=int(payload["cycle_index"]),
            server_urls=tuple(str(url) for url in raw_urls),
        )

    def wait(
        self,
        *,
        timeout_seconds: float,
        liveness_check: Callable[[], None] | None = None,
    ) -> FullWeightEvalRequest:
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            if liveness_check is not None:
                liveness_check()
            request = self.pending()
            if request is not None:
                return request
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the full-weight official-eval worker "
                    "to make its initial policy request."
                )
            time.sleep(self._poll_seconds)

    def acknowledge(
        self,
        request: FullWeightEvalRequest,
        *,
        policy_iteration: int,
        publication_identities: list[dict[str, Any]],
    ) -> None:
        _write_json_atomic(
            self._run_root / _READY_FILENAME,
            {
                "version": _PROTOCOL_VERSION,
                "request_id": request.request_id,
                "cycle_index": request.cycle_index,
                "policy_iteration": int(policy_iteration),
                "publication_identities": publication_identities,
            },
        )
        self._last_acknowledged_request_id = request.request_id
