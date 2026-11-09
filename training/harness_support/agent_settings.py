"""Agent settings a rollout worker reports with every run of its window.

The worker reads them off the agent it built, after the harness preset and
the training configuration have both applied, so they are the harness's
resolved values rather than the spec's. The trainer receives them as
``RunResult.agent_settings``; each harness adapter says which it requires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AGENT_SETTINGS_FILE = "agent_settings.json"


def record_agent_settings(job_dir: Path, settings: dict[str, Any]) -> None:
    """Keep the built agent's settings beside the window's harness output."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / AGENT_SETTINGS_FILE).write_text(
        json.dumps(settings, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def attach_agent_settings(job_dir: Path, runs: list[dict[str, Any]]) -> None:
    """Report the window's agent settings with each of its runs."""
    settings = json.loads((job_dir / AGENT_SETTINGS_FILE).read_text(encoding="utf-8"))
    for run in runs:
        run["agent_settings"] = dict(settings)
