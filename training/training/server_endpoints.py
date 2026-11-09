"""Shared launcher/collector endpoint identity; credentials stay in the environment."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def server_auth_headers() -> dict[str, str]:
    """Use the native service key without storing it in a run artifact."""
    key = os.environ.get("SERVER_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def server_ready(url: str, *, authenticate: bool = True) -> bool:
    """Bounded readiness check; staging health deliberately has separate auth."""
    import requests

    try:
        return (
            requests.get(
                url, headers=server_auth_headers() if authenticate else {}, timeout=2
            ).status_code
            == 200
        )
    except requests.RequestException:
        return False


def configure_server_log_paths(payload: dict, run_root: Path) -> None:
    """Record the exact single-node launch paths, including evaluation servers."""
    inference = payload["inference"]
    evaluation = payload.get("official_eval", {})
    train_count = max(1, len(inference.get("server_gpus", [])))
    eval_count = (
        len(evaluation.get("server_gpus", [])) if evaluation.get("enabled") else 0
    )
    total = train_count + eval_count
    logs = [
        str(
            run_root.resolve()
            / (
                "inference_server.log"
                if total == 1
                else f"inference_server_{index}.log"
            )
        )
        for index in range(total)
    ]
    inference["server_log_paths"] = logs[:train_count]
    if evaluation:
        evaluation["server_log_paths"] = logs[train_count:]


def collector_server_log_paths(
    config, run_root: Path, *, evaluation: bool = False
) -> list[Path]:
    """Pair collectors with explicit logs or the single-node launch convention."""
    section = config.official_eval if evaluation else config.inference
    count = len(section.server_gpus) if evaluation else max(1, len(section.server_gpus))
    explicit = section.server_log_paths
    if explicit:
        paths = [Path(value) for value in explicit]
        if len(paths) != count or any(not path.is_absolute() for path in paths):
            name = "official_eval" if evaluation else "inference"
            raise ValueError(
                f"{name}.server_log_paths requires one absolute path per server"
            )
        if len(set(paths)) != len(paths):
            raise ValueError("Each server must have a distinct log path")
        return paths
    payload = {
        "inference": {"server_gpus": config.inference.server_gpus},
        "official_eval": {
            "server_gpus": config.official_eval.server_gpus,
            "enabled": config.official_eval.enabled,
        },
    }
    configure_server_log_paths(payload, run_root)
    return [
        Path(value)
        for value in payload["official_eval" if evaluation else "inference"][
            "server_log_paths"
        ]
    ]


if __name__ == "__main__":
    # Only the URL appears in argv or shell tracing; never print response headers.
    raise SystemExit(0 if server_ready(sys.argv[1]) else 1)
