"""ARC-2 adapters for exact-history training and processed-draw accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from training_backend import TrainingLLMAgentSolver


@dataclass
class ProcessedResponseObserver:
    """Journal processed ARC-2 draws before tool execution or cancellation."""

    def __call__(self, response: Any) -> None:
        """Journal penalty evidence after parsing and before tool execution.

        ``empty_reply`` is the reply the Duck ends a game on under
        ``terminate_on_empty_reply``: kept as malformed with no text and no
        tool call, by the Duck's own test (``_last_turn_was_empty_reply``) and
        ullm's, which counts it one malformed call in the turn manager. It is
        charged as that malformed call; the flag only counts empty replies
        (harness_support/arc2/results.py).
        """
        from ullm.backend import _is_empty_content

        stamp = response.generation_request_id
        if not isinstance(stamp, str) or "#" not in stamp:
            raise RuntimeError(
                "processed ARC-2 response lacks its exact generation identity"
            )
        run_id, _, nonce = stamp.rpartition("#")
        if not run_id.isalnum() or not nonce.isalnum():
            raise RuntimeError("invalid ARC-2 response identity")
        record = {
            "nonce": nonce,
            "malformed_call_count": response.malformed_call_count,
            "truncation_cap_source": response.truncation_cap_source,
            "stop_reason": response.stop_reason,
            "has_malformed_error": response.malformed_error is not None,
            "has_tool_calls": bool(response.final_content.tool_calls()),
            "empty_reply": response.stop_reason == "malformed"
            and _is_empty_content(response.final_content),
        }
        root = Path(os.environ["ARC2_TRAINING_RESPONSE_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"{run_id}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


@dataclass
class ContractARC2LLMAgentSolver(TrainingLLMAgentSolver):
    """Require terminal infrastructure failures to reach the training worker."""

    environment: str = "arc2"
    strict_teardown: bool = True
    _trace_recorder: Any = field(default=None, init=False, repr=False)

    def configure_game_backend(self, game: Any, host_backend: Any, client: Any) -> None:
        """Journal host draws after the harness registers their trace destination."""
        super().configure_game_backend(game, host_backend, client)
        if host_backend.trace_recorder is None or self.job_dir is None:
            raise RuntimeError(
                "ARC-2 draw journaling requires traces and a job directory"
            )
        # Deliberate exception to the hook's solver-owned tracing boundary:
        # training needs durable evidence of every host draw, including failures.
        # Keep the solver's recorder authoritative and call it first; migrate to
        # an additional public recorder hook when Platypus provides one.
        # Each game shares the host. Capture the collector once, before wrapping it.
        if host_backend.trace_recorder != self._record_native_draw:
            self._trace_recorder = host_backend.trace_recorder
        host_backend.trace_recorder = self._record_native_draw

    def _record_native_draw(self, record: Any) -> None:
        """Keep the authoritative collector and separately persist failure evidence."""
        from harness_support.arc2.draw_journal import append_native_draw

        assert self._trace_recorder is not None and self.job_dir is not None
        self._trace_recorder(record)
        append_native_draw(self.job_dir, record)
