"""Share the first fatal error across training, rollout and evaluation processes."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

from training.shutdown import shutdown_requested

log = logging.getLogger(__name__)

RUN_FAILURE_FILENAME = "run_failure.json"

# The gateway reports a stopped upstream server as a 502 body and the collector
# wraps that in a GatewayError; aiohttp's own connection errors sit underneath.
_TEARDOWN_MESSAGE_MARKERS = (
    "upstream unreachable",
    "Cannot connect to host",
    "Connect call failed",
    "Connection refused",
    "Server disconnected",
    "Connection reset",
)
_TEARDOWN_ERROR_TYPES = frozenset(
    {
        "ClientConnectorError",
        "ClientConnectionError",
        "ClientOSError",
        "ServerDisconnectedError",
    }
)


def is_teardown_error(error: BaseException) -> bool:
    """Whether an error is an upstream-connectivity failure a server stop explains.

    Follows the cause chain so the wrapped gateway report and the underlying
    connection error both count. Nothing else does: a real 502 from a server
    that died mid-run looks the same, so callers must also know a shutdown
    was requested before treating it as benign.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConnectionError):
            return True
        if type(current).__name__ in _TEARDOWN_ERROR_TYPES:
            return True
        message = str(current)
        if any(marker in message for marker in _TEARDOWN_MESSAGE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


class FatalTrainingError(RuntimeError):
    """The run has failed and must not consume more data or resume automatically."""


def record_run_failure(run_root: Path | None, error: BaseException) -> None:
    """Atomically publish the first error and prevent scheduler continuation.

    The temporary file is complete before its hard link becomes visible.
    Creating that link without replacing an existing destination lets many
    processes race to report the same failure without overwriting its cause.
    """
    if run_root is None:
        return
    if shutdown_requested() and is_teardown_error(error):
        # A scheduled segment end stops the serving hosts while a generation
        # can still be in flight. That connectivity failure is the shutdown
        # itself, not a run failure; latching it would block the continuation
        # the shutdown exists to allow.
        log.info(
            "scheduled shutdown: upstream teardown error is not a run failure: %s",
            error,
        )
        return
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    # Set the launcher's existing stop latch first: even a process killed
    # while publishing its diagnostic must not schedule a continuation.
    (root / "STOP_AUTO_RELAUNCH").touch(exist_ok=True)
    destination = root / RUN_FAILURE_FILENAME
    if destination.exists():
        return
    payload = {
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "pid": os.getpid(),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    fd, temporary_name = tempfile.mkstemp(prefix=".run_failure.", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass  # Another process published the first failure.
    finally:
        temporary.unlink(missing_ok=True)


def raise_if_run_failed(run_root: Path | None) -> None:
    """Fail closed on a recorded error, even when the originating worker lives."""
    if run_root is None:
        return
    path = Path(run_root) / RUN_FAILURE_FILENAME
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except (OSError, UnicodeError) as error:
        raise FatalTrainingError(
            f"Cannot read the fatal run record {path}: {error}"
        ) from error
    try:
        payload = json.loads(contents)
        if not isinstance(payload, dict) or any(
            not isinstance(payload.get(key), str) for key in ("error_type", "message")
        ):
            raise ValueError("missing error_type or message")
    except (ValueError, TypeError) as error:
        raise FatalTrainingError(f"Invalid fatal run record {path}: {error}") from error
    raise FatalTrainingError(
        f"Fatal training failure: {payload['error_type']}: {payload['message']} "
        f"(recorded in {path}; automatic relaunch is disabled)"
    )
