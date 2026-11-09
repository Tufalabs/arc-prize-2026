from __future__ import annotations

from pathlib import Path

from .commons import JSONValue, read_json, repo_root, write_json


def build_game_catalog(*, root: Path | None = None) -> list[dict[str, JSONValue]]:
    root_dir = root or repo_root()
    environment_root = root_dir / "re_arc" / "environment_files"
    ideas: list[dict[str, JSONValue]] = []

    for metadata_path in sorted(environment_root.glob("*/**/metadata.json")):
        payload = read_json(metadata_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object in {metadata_path}")
        ideas.append({"idea": str(payload.get("idea") or "")})

    return ideas


def write_game_catalog(*, output_path: Path, root: Path | None = None) -> list[dict[str, JSONValue]]:
    payload = build_game_catalog(root=root)
    write_json(output_path, payload)
    return payload
