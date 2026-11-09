from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from re_arc.cli import _run_replay_ui

from .commons import JSONValue, read_json, repo_root, write_json

VALID_REVIEW_DECISIONS = {"pending", "accepted", "rejected"}


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("._") or "item"


def _run_dir(*, run_id: str, root: Path) -> Path:
    return root / "pipeline" / "runs" / run_id


def _worker_packages(*, run_id: str, root: Path) -> list[dict[str, Any]]:
    return _worker_packages_from_run_dir(_run_dir(run_id=run_id, root=root))


def _worker_packages_from_run_json(run_json_path: Path) -> list[dict[str, Any]]:
    if not run_json_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {run_json_path}")
    payload = read_json(run_json_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {run_json_path}")
    packages = payload.get("worker_packages")
    if not isinstance(packages, list):
        raise ValueError(f"Expected worker_packages list in {run_json_path}")
    return [pkg for pkg in packages if isinstance(pkg, dict) and str(pkg.get("status") or "") != "failed"]


def _fallback_worker_spec_path(*, worker_root: Path) -> str:
    spec_path = worker_root.parent.parent / "specs" / f"{worker_root.name}.md"
    return str(spec_path) if spec_path.exists() else ""


def _fallback_worker_idea_details(*, worker_root: Path) -> dict[str, str]:
    idea_path = worker_root / "idea.json"
    if not idea_path.exists():
        return {}
    payload = read_json(idea_path)
    if not isinstance(payload, dict):
        return {}
    return {
        "idea_title": str(payload.get("title") or "").strip(),
        "idea_description": str(payload.get("description") or "").strip(),
    }


def _worker_packages_from_workers_dir(workers_dir: Path) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    if not workers_dir.exists():
        return packages

    for worker_root in sorted(path for path in workers_dir.iterdir() if path.is_dir()):
        idea_details = _fallback_worker_idea_details(worker_root=worker_root)
        replay_paths = sorted((worker_root / "repo" / "artifacts" / "replays").glob("*.replay.json"))
        for replay_path in replay_paths:
            game_id = replay_path.name.removesuffix(".replay.json")
            try:
                payload = json.loads(replay_path.read_text(encoding="utf-8"))
                metadata = payload.get("metadata") if isinstance(payload, dict) else None
                if isinstance(metadata, dict):
                    discovered_game_id = str(metadata.get("game_id") or "").strip()
                    if discovered_game_id:
                        game_id = discovered_game_id
            except Exception:
                pass
            gif_path = replay_path.parent / f"{replay_path.name.removesuffix('.replay.json')}.gif"
            package: dict[str, Any] = {
                "idea_id": worker_root.name,
                "idea_title": str(idea_details.get("idea_title") or ""),
                "idea_description": str(idea_details.get("idea_description") or ""),
                "game_id": game_id,
                "worker_root": str(worker_root),
                "worker_spec_path": _fallback_worker_spec_path(worker_root=worker_root),
                "worker_idea_path": str(worker_root / "idea.json") if (worker_root / "idea.json").exists() else "",
                "dsl_replay_path": str(replay_path),
                "dsl_gif_path": str(gif_path) if gif_path.exists() else "",
            }
            packages.append(package)
    return packages


def _worker_packages_from_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    run_json_path = run_dir / "run.json"
    if run_json_path.exists():
        packages = _worker_packages_from_run_json(run_json_path)
        if packages:
            return packages
    return _worker_packages_from_workers_dir(run_dir / "workers")


def _copy_review_entry(*, replay_dir: Path, package: dict[str, Any], prefix: str) -> dict[str, str] | None:
    replay_path_raw = package.get("dsl_replay_path")
    if not replay_path_raw:
        return None
    replay_path = Path(str(replay_path_raw))
    if not replay_path.exists():
        return None

    game_id = str(package.get("game_id") or "").strip()
    idea_id = str(package.get("idea_id") or "").strip()
    target_replay_path = replay_dir / f"{prefix}.replay.json"
    shutil.copy2(replay_path, target_replay_path)

    gif_target_path = None
    gif_path_raw = package.get("dsl_gif_path")
    if gif_path_raw:
        gif_path = Path(str(gif_path_raw))
        if gif_path.exists():
            gif_target_path = replay_dir / f"{prefix}.gif"
            shutil.copy2(gif_path, gif_target_path)

    return {
        "idea_id": idea_id,
        "idea_title": str(package.get("idea_title") or ""),
        "idea_description": str(package.get("idea_description") or ""),
        "game_id": game_id,
        "generated_slug": str(package.get("generated_slug") or ""),
        "source_replay_path": str(replay_path),
        "review_replay_path": str(target_replay_path),
        "source_gif_path": str(gif_path_raw) if gif_path_raw else "",
        "review_gif_path": str(gif_target_path) if gif_target_path is not None else "",
        "worker_root": str(package.get("worker_root") or ""),
        "worker_repo_path": str(package.get("worker_repo_path") or ""),
        "worker_spec_path": str(package.get("worker_spec_path") or ""),
    }


def _review_manifest_path(review_dir: Path) -> Path:
    return review_dir / "manifest.json"


def _review_decisions_path(review_dir: Path) -> Path:
    return review_dir / "decisions.json"


def _write_initial_review_decisions(*, review_dir: Path, entries: list[dict[str, str]]) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "review_dir": str(review_dir),
        "submitted": False,
        "entries": [
            {
                "review_replay_name": Path(entry["review_replay_path"]).name,
                "idea_id": entry.get("idea_id", ""),
                "game_id": entry.get("game_id", ""),
                "decision": "pending",
            }
            for entry in entries
        ],
    }
    write_json(_review_decisions_path(review_dir), payload)
    return payload


def _load_review_manifest(review_dir: Path) -> dict[str, object]:
    payload = read_json(_review_manifest_path(review_dir))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {_review_manifest_path(review_dir)}")
    return {str(key): value for key, value in payload.items()}


def load_review_decisions(*, review_dir: Path) -> dict[str, JSONValue]:
    decisions_path = _review_decisions_path(review_dir)
    if not decisions_path.exists():
        manifest = _load_review_manifest(review_dir)
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Expected entries list in {_review_manifest_path(review_dir)}")
        typed_entries = [entry for entry in entries if isinstance(entry, dict)]
        return _write_initial_review_decisions(review_dir=review_dir, entries=typed_entries)
    payload = read_json(decisions_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {decisions_path}")
    return {str(key): value for key, value in payload.items()}


def record_review_decision(*, review_dir: Path, review_replay_name: str, decision: str) -> dict[str, JSONValue]:
    normalized = decision.strip().lower()
    if normalized not in VALID_REVIEW_DECISIONS - {"pending"}:
        raise ValueError(f"Invalid review decision: {decision!r}")
    payload = load_review_decisions(review_dir=review_dir)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Expected entries list in {_review_decisions_path(review_dir)}")
    updated = False
    typed_entries: list[dict[str, JSONValue]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = {str(key): value for key, value in raw_entry.items()}
        if str(entry.get("review_replay_name") or "") == review_replay_name:
            entry["decision"] = normalized
            updated = True
        typed_entries.append(entry)
    if not updated:
        raise KeyError(f"Unknown review replay: {review_replay_name}")
    payload["entries"] = typed_entries
    write_json(_review_decisions_path(review_dir), payload)
    return payload


def _generated_slug_from_worker_root(worker_root: Path) -> str:
    worker_repo = worker_root / "repo"
    environment_root = worker_repo / "re_arc" / "environment_files"
    agents_root = worker_repo / "re_arc" / "dsl" / "agents"
    environment_slugs = (
        {path.name for path in environment_root.iterdir() if path.is_dir()} if environment_root.exists() else set()
    )
    agent_slugs = (
        {path.stem for path in agents_root.glob("*.py") if path.is_file() and path.stem != "__init__"}
        if agents_root.exists()
        else set()
    )
    reference_slugs = {"ft09_close", "ls20_close"}
    candidate_slugs = sorted((environment_slugs & agent_slugs) - reference_slugs)
    if len(candidate_slugs) != 1:
        raise RuntimeError(f"Expected exactly one generated slug under {worker_root}, found {candidate_slugs!r}.")
    return candidate_slugs[0]


def _generated_slug_from_game_id(game_id: str) -> str:
    value = str(game_id or "").strip()
    if "-" not in value:
        return ""
    slug, _, suffix = value.rpartition("-")
    return slug if slug and suffix else ""


def _promoted_game_id_for_slug(*, source_slug: str, promoted_slug: str, source_game_id: str) -> str:
    if source_game_id.startswith(f"{source_slug}-"):
        return f"{promoted_slug}{source_game_id[len(source_slug) :]}"
    if source_game_id == source_slug:
        return promoted_slug
    if source_slug in source_game_id:
        return source_game_id.replace(source_slug, promoted_slug, 1)
    return source_game_id


def _unique_promoted_slug(*, root_dir: Path, base_slug: str, source_game_id: str) -> str:
    candidate = base_slug
    version = 2
    while True:
        env_dir = root_dir / "re_arc" / "environment_files" / candidate
        agent_path = root_dir / "re_arc" / "dsl" / "agents" / f"{candidate}.py"
        promoted_game_id = _promoted_game_id_for_slug(
            source_slug=base_slug, promoted_slug=candidate, source_game_id=source_game_id
        )
        replay_path, gif_path = _promoted_replay_paths(root_dir=root_dir, promoted_game_id=promoted_game_id)
        if not env_dir.exists() and not agent_path.exists() and not replay_path.exists() and not gif_path.exists():
            return candidate
        candidate = f"{base_slug}_v{version}"
        version += 1


def _promoted_env_game_ids(target_env_dir: Path) -> set[str]:
    if not target_env_dir.exists():
        return set()
    game_ids: set[str] = set()
    for metadata_path in sorted(target_env_dir.rglob("metadata.json")):
        payload = read_json(metadata_path)
        if not isinstance(payload, dict):
            continue
        game_id = str(payload.get("game_id") or "").strip()
        if game_id:
            game_ids.add(game_id)
    return game_ids


def _promoted_slug_for_submission(*, root_dir: Path, generated_slug: str, source_game_id: str) -> str:
    promoted_game_id = _promoted_game_id_for_slug(
        source_slug=generated_slug, promoted_slug=generated_slug, source_game_id=source_game_id
    )
    target_env_dir = root_dir / "re_arc" / "environment_files" / generated_slug
    if target_env_dir.exists() and promoted_game_id in _promoted_env_game_ids(target_env_dir):
        return generated_slug
    return _unique_promoted_slug(root_dir=root_dir, base_slug=generated_slug, source_game_id=source_game_id)


def _rewrite_promoted_metadata_game_id(
    *, target_env_dir: Path, source_slug: str, promoted_slug: str, source_game_id: str, idea_description: str = ""
) -> str:
    metadata_paths = sorted(target_env_dir.rglob("metadata.json"))
    promoted_game_id = _promoted_game_id_for_slug(
        source_slug=source_slug, promoted_slug=promoted_slug, source_game_id=source_game_id
    )
    if not metadata_paths:
        return promoted_game_id
    for metadata_path in metadata_paths:
        payload = read_json(metadata_path)
        if not isinstance(payload, dict):
            continue
        updated_payload = {str(key): value for key, value in payload.items()}
        updated_payload["game_id"] = promoted_game_id
        if idea_description.strip():
            updated_payload["idea"] = idea_description.strip()
        raw_tags = updated_payload.get("tags")
        tags = [str(tag).strip() for tag in raw_tags] if isinstance(raw_tags, list) else []
        tags = [tag for tag in tags if tag]
        if promoted_slug != source_slug and "further-version" not in tags:
            tags.append("further-version")
        if tags:
            updated_payload["tags"] = tags
        write_json(metadata_path, updated_payload)
    return promoted_game_id


def _promoted_replay_paths(*, root_dir: Path, promoted_game_id: str) -> tuple[Path, Path]:
    docs_replays_dir = root_dir / "docs" / "replays"
    docs_replays_dir.mkdir(parents=True, exist_ok=True)
    replay_path = docs_replays_dir / f"{promoted_game_id}.replay.json"
    gif_path = docs_replays_dir / f"{promoted_game_id}.gif"
    return replay_path, gif_path


def _copy_promoted_replay_artifacts(
    *, root_dir: Path, source_replay_path: str, source_gif_path: str, promoted_game_id: str
) -> tuple[str, str]:
    replay_path = Path(source_replay_path)
    if not source_replay_path or not replay_path.exists():
        return "", ""

    target_replay_path, target_gif_path = _promoted_replay_paths(root_dir=root_dir, promoted_game_id=promoted_game_id)

    payload = read_json(replay_path)
    if isinstance(payload, dict):
        updated_payload = {str(key): value for key, value in payload.items()}
        metadata = updated_payload.get("metadata")
        if isinstance(metadata, dict):
            updated_metadata = {str(key): value for key, value in metadata.items()}
            updated_metadata["game_id"] = promoted_game_id
            updated_metadata["gif_path"] = (
                str(target_gif_path) if source_gif_path and Path(source_gif_path).exists() else ""
            )
            updated_payload["metadata"] = updated_metadata
        write_json(target_replay_path, updated_payload)
    else:
        shutil.copy2(replay_path, target_replay_path)

    copied_gif_path = ""
    if source_gif_path:
        gif_path = Path(source_gif_path)
        if gif_path.exists():
            shutil.copy2(gif_path, target_gif_path)
            copied_gif_path = str(target_gif_path)
    elif target_gif_path.exists():
        target_gif_path.unlink()

    return str(target_replay_path), copied_gif_path


def _sync_promoted_precomputed_actions(*, root_dir: Path, game_ids: list[str]) -> list[str]:
    if not game_ids:
        return []

    from re_arc.cli import _build_env_sampler, _load_config
    from re_arc.dsl.precomputed_actions import (
        _generate_record_for_game,
        sync_baseline_actions,
        write_precomputed_actions,
    )

    config_path = root_dir / "config.env"
    config = _load_config(str(config_path)) if config_path.exists() else {}
    config = dict(config)
    config["ENVIRONMENTS_DIR"] = str(root_dir / "re_arc" / "environment_files")
    sampler = _build_env_sampler(config, seed=0, augment=False)
    precomputed_dir = root_dir / "re_arc" / "dsl" / "precomputed_actions"

    for game_id in game_ids:
        record = _generate_record_for_game(sampler=sampler, game_id=game_id, seed=0, max_actions=12000)
        write_precomputed_actions(record, directory=precomputed_dir)

    _results, failures = sync_baseline_actions(
        config=config, sampler=sampler, game_ids=game_ids, seed=0, check_only=False
    )
    return failures


def submit_review(*, review_dir: Path, root: Path | None = None) -> dict[str, JSONValue]:
    root_dir = root or repo_root()
    manifest = _load_review_manifest(review_dir)
    manifest_entries = manifest.get("entries")
    if not isinstance(manifest_entries, list):
        raise ValueError(f"Expected entries list in {_review_manifest_path(review_dir)}")
    decisions = load_review_decisions(review_dir=review_dir)
    decision_entries = decisions.get("entries")
    if not isinstance(decision_entries, list):
        raise ValueError(f"Expected entries list in {_review_decisions_path(review_dir)}")

    decisions_by_name: dict[str, dict[str, JSONValue]] = {}
    for raw_entry in decision_entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = {str(key): value for key, value in raw_entry.items()}
        replay_name = str(entry.get("review_replay_name") or "")
        if replay_name:
            decisions_by_name[replay_name] = entry

    for raw_entry in manifest_entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = {str(key): value for key, value in raw_entry.items()}
        replay_name = Path(str(entry.get("review_replay_path") or "")).name
        decision = str((decisions_by_name.get(replay_name) or {}).get("decision") or "pending")
        if decision == "pending":
            raise RuntimeError("Cannot submit review until every replay has an accept/reject decision.")

    promoted: list[dict[str, str]] = []
    for raw_entry in manifest_entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = {str(key): value for key, value in raw_entry.items()}
        replay_name = Path(str(entry.get("review_replay_path") or "")).name
        decision = str((decisions_by_name.get(replay_name) or {}).get("decision") or "pending")
        if decision != "accepted":
            continue
        worker_root = Path(str(entry.get("worker_root") or ""))
        if not worker_root.exists():
            raise FileNotFoundError(f"Worker root not found for accepted replay {replay_name}: {worker_root}")
        generated_slug = (
            str(entry.get("generated_slug") or "").strip()
            or _generated_slug_from_game_id(str(entry.get("game_id") or ""))
            or _generated_slug_from_worker_root(worker_root)
        )
        worker_repo_path = (
            Path(str(entry.get("worker_repo_path") or "")).resolve()
            if entry.get("worker_repo_path")
            else worker_root / "repo"
        )
        source_env_dir = worker_repo_path / "re_arc" / "environment_files" / generated_slug
        source_agent_path = worker_repo_path / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py"
        if not source_env_dir.exists():
            raise FileNotFoundError(f"Generated environment directory not found: {source_env_dir}")
        if not source_agent_path.exists():
            raise FileNotFoundError(f"Generated agent file not found: {source_agent_path}")

        source_game_id = str(entry.get("game_id") or "")
        promoted_slug = _promoted_slug_for_submission(
            root_dir=root_dir, generated_slug=generated_slug, source_game_id=source_game_id
        )
        target_env_dir = root_dir / "re_arc" / "environment_files" / promoted_slug
        target_agent_path = root_dir / "re_arc" / "dsl" / "agents" / f"{promoted_slug}.py"
        if target_env_dir.exists():
            shutil.rmtree(target_env_dir)
        shutil.copytree(source_env_dir, target_env_dir)
        target_agent_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_agent_path, target_agent_path)
        promoted_game_id = _rewrite_promoted_metadata_game_id(
            target_env_dir=target_env_dir,
            source_slug=generated_slug,
            promoted_slug=promoted_slug,
            source_game_id=source_game_id,
            idea_description=str(entry.get("idea_description") or ""),
        )
        docs_replay_path, docs_gif_path = _copy_promoted_replay_artifacts(
            root_dir=root_dir,
            source_replay_path=str(entry.get("source_replay_path") or ""),
            source_gif_path=str(entry.get("source_gif_path") or ""),
            promoted_game_id=promoted_game_id,
        )
        promoted.append(
            {
                "idea_id": str(entry.get("idea_id") or ""),
                "game_id": promoted_game_id,
                "generated_slug": generated_slug,
                "promoted_slug": promoted_slug,
                "target_env_dir": str(target_env_dir),
                "target_agent_path": str(target_agent_path),
                "target_spec_path": "",
                "target_replay_path": docs_replay_path,
                "target_gif_path": docs_gif_path,
            }
        )

    decisions["submitted"] = True
    decisions["promoted"] = promoted
    write_json(_review_decisions_path(review_dir), decisions)
    sync_failures = _sync_promoted_precomputed_actions(
        root_dir=root_dir, game_ids=[str(item["game_id"]) for item in promoted if str(item.get("game_id") or "")]
    )
    submission_payload: dict[str, JSONValue] = {
        "review_dir": str(review_dir),
        "promoted_count": len(promoted),
        "promoted": promoted,
        "precompute_failures": sync_failures,
    }
    return submission_payload


def prepare_review_bundle(*, run_id: str, root: Path | None = None) -> dict[str, JSONValue]:
    root_dir = root or repo_root()
    run_dir = _run_dir(run_id=run_id, root=root_dir)
    review_dir = run_dir / "review"
    replay_dir = review_dir / "replays"
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    for index, package in enumerate(_worker_packages(run_id=run_id, root=root_dir), start=1):
        replay_path_raw = package.get("dsl_replay_path")
        game_id = str(package.get("game_id") or "").strip()
        idea_id = str(package.get("idea_id") or "").strip()
        replay_path = Path(str(replay_path_raw)) if replay_path_raw else Path("item")
        entry = _copy_review_entry(
            replay_dir=replay_dir,
            package=package,
            prefix=f"{index:03d}-{_safe_name(idea_id or game_id or replay_path.stem)}",
        )
        if entry is not None:
            entries.append(entry)

    if not entries:
        raise FileNotFoundError(f"No generated replay artifacts found for run {run_id!r} under {run_dir}.")

    manifest: dict[str, JSONValue] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "review_dir": str(review_dir),
        "replay_dir": str(replay_dir),
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest_path = review_dir / "manifest.json"
    write_json(manifest_path, manifest)
    _write_initial_review_decisions(review_dir=review_dir, entries=entries)
    return manifest


def review_run(
    *,
    run_id: str,
    root: Path | None = None,
    host: str | None = None,
    port: int | None = None,
    launch_ui: bool = True,
    replay_ui_runner: Any = _run_replay_ui,
) -> dict[str, JSONValue]:
    root_dir = root or repo_root()
    manifest = prepare_review_bundle(run_id=run_id, root=root_dir)
    if launch_ui:
        args = SimpleNamespace(
            config=str(root_dir / "config.env"),
            replay=None,
            replay_dir=str(manifest["replay_dir"]),
            host=host,
            port=port,
        )
        replay_ui_runner(args)
    return manifest
