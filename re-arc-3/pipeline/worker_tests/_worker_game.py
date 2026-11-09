from __future__ import annotations

import json
from pathlib import Path

REFERENCE_GAME_SLUGS = frozenset(json.loads("""__REFERENCE_GAME_SLUGS_JSON__"""))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_generated_game() -> dict[str, object]:
    root = repo_root()
    environment_root = root / "re_arc" / "environment_files"
    agents_root = root / "re_arc" / "dsl" / "agents"
    environment_slugs = (
        {path.name for path in environment_root.iterdir() if path.is_dir()} if environment_root.exists() else set()
    )
    agent_slugs = (
        {path.stem for path in agents_root.glob("*.py") if path.is_file() and path.stem != "__init__"}
        if agents_root.exists()
        else set()
    )
    candidate_slugs = sorted((environment_slugs & agent_slugs) - REFERENCE_GAME_SLUGS)
    assert len(candidate_slugs) == 1, (
        "Expected exactly one generated game slug, "
        f"found {candidate_slugs!r} with reference slugs {sorted(REFERENCE_GAME_SLUGS)!r}."
    )
    slug = candidate_slugs[0]
    metadata_paths = sorted((environment_root / slug).rglob("metadata.json"))
    assert len(metadata_paths) == 1, f"Expected one metadata.json for {slug!r}, found {len(metadata_paths)}."
    payload = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    game_id = str(payload.get("game_id") or "").strip()
    assert game_id, f"metadata.json for {slug!r} did not define game_id."
    env_files = sorted(path for path in (environment_root / slug).rglob("*.py") if path.is_file())
    assert env_files, f"No environment source files found for generated slug {slug!r}."
    return {"slug": slug, "game_id": game_id, "metadata_path": metadata_paths[0], "env_files": env_files}
