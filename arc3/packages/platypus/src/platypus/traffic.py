"""A recorded run's traces, as the calls ``ullm.traffic`` builds a workload from.

The recording is an ordinary platypus run with traces on
(``LLMConfig.traces``, ideally ``trace_content="ids"``): every step carries the
engine's own input and output ids, its images, and -- where the host granted
the run a slot -- the ``after`` link ``TraceCollector`` stamps. This module
reads those folders back; ``ullm.traffic`` owns everything after.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import numpy as np
import taaf.trace
import ullm.traffic


def recorded_runs(trace_root: str | Path) -> tuple[list[ullm.traffic.RecordedRun], dict[str, bytes]]:
    """Every completed trace under ``trace_root`` as a ``RecordedRun``, and
    the images they cite by content key (the sha256 of their bytes).

    Either trace content serves: both carry the ids and the images.
    """
    runs: list[ullm.traffic.RecordedRun] = []
    media: dict[str, bytes] = {}
    for folder in sorted(Path(trace_root).iterdir()):
        if not (folder / taaf.trace.SUCCESS_MARKER).exists():
            continue
        trace = taaf.trace.load_trace(folder)
        calls: list[ullm.traffic.RecordedCall] = []
        for step in trace.steps:
            keys: list[str] = []
            for ref in step.media:
                data = (folder / ref.path).read_bytes()
                key = hashlib.sha256(data).hexdigest()
                media.setdefault(key, data)
                keys.append(key)
            link = step.extra.get("after")
            calls.append(
                ullm.traffic.RecordedCall(
                    input_ids=np.asarray(step.input_token_ids, dtype=np.int32),
                    output_ids=np.asarray(step.output_token_ids, dtype=np.int32),
                    images=tuple(keys),
                    finished_at=datetime.datetime.fromisoformat(step.created_at).timestamp(),
                    continued=bool(step.extra.get("continued")),
                    opens="after" in step.extra,
                    after=(link["run_id"], int(link["step_index"])) if link else None,
                )
            )
        runs.append(ullm.traffic.RecordedRun(run_id=trace.run_id, calls=tuple(calls)))
    return runs, media
