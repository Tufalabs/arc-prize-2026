from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.review import (
    load_review_decisions,
    prepare_review_bundle,
    record_review_decision,
    review_run,
    submit_review,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture(autouse=True)
def _stub_precompute_sync(monkeypatch):
    monkeypatch.setattr("pipeline.review._sync_promoted_precomputed_actions", lambda **_: [])


def test_prepare_review_bundle_collects_worker_replays(tmp_path: Path):
    root = tmp_path
    run_id = "20260324-180000"
    run_dir = root / "pipeline" / "runs" / run_id

    replay_a = run_dir / "workers" / "idea-a" / "repo" / "artifacts" / "replays" / "game-a-0001.replay.json"
    gif_a = replay_a.with_suffix(".gif")
    replay_b = run_dir / "workers" / "idea-b" / "repo" / "artifacts" / "replays" / "game-b-0001.replay.json"
    _write_text(replay_a, json.dumps({"version": 2, "metadata": {"game_id": "game-a-0001"}}, indent=2))
    _write_text(gif_a, "gif-a\n")
    _write_text(replay_b, json.dumps({"version": 2, "metadata": {"game_id": "game-b-0001"}}, indent=2))

    run_json = {
        "worker_packages": [
            {
                "idea_id": "idea-a",
                "idea_title": "Idea A",
                "idea_description": "First generated idea.",
                "game_id": "game-a-0001",
                "worker_root": str(run_dir / "workers" / "idea-a"),
                "dsl_replay_path": str(replay_a),
                "dsl_gif_path": str(gif_a),
            },
            {
                "idea_id": "idea-b",
                "idea_title": "Idea B",
                "idea_description": "Second generated idea.",
                "game_id": "game-b-0001",
                "worker_root": str(run_dir / "workers" / "idea-b"),
                "dsl_replay_path": str(replay_b),
                "dsl_gif_path": str(replay_b.with_suffix(".gif")),
            },
        ]
    }
    _write_text(run_dir / "run.json", json.dumps(run_json, indent=2))

    manifest = prepare_review_bundle(run_id=run_id, root=root)

    replay_dir = Path(str(manifest["replay_dir"]))
    copied_replays = sorted(path.name for path in replay_dir.glob("*.replay.json"))
    assert copied_replays == ["001-idea-a.replay.json", "002-idea-b.replay.json"]
    assert (replay_dir / "001-idea-a.gif").read_text(encoding="utf-8") == "gif-a\n"
    assert not (replay_dir / "002-idea-b.gif").exists()
    assert manifest["entry_count"] == 2
    assert (Path(str(manifest["review_dir"])) / "manifest.json").exists()
    assert manifest["entries"][0]["idea_title"] == "Idea A"
    assert manifest["entries"][0]["idea_description"] == "First generated idea."
    assert manifest["entries"][1]["idea_title"] == "Idea B"
    decisions = load_review_decisions(review_dir=Path(str(manifest["review_dir"])))
    assert decisions["submitted"] is False
    assert [entry["decision"] for entry in decisions["entries"]] == ["pending", "pending"]


def test_prepare_review_bundle_removes_stale_replays(tmp_path: Path):
    root = tmp_path
    run_id = "20260324-180003"
    run_dir = root / "pipeline" / "runs" / run_id
    replay_path = run_dir / "workers" / "idea-a" / "repo" / "artifacts" / "replays" / "game-a-0001.replay.json"
    stale_replay = run_dir / "review" / "replays" / "999-stale.replay.json"
    stale_gif = run_dir / "review" / "replays" / "999-stale.gif"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "game-a-0001"}}, indent=2))
    _write_text(stale_replay, "{}\n")
    _write_text(stale_gif, "stale\n")
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "game_id": "game-a-0001",
                        "worker_root": str(run_dir / "workers" / "idea-a"),
                        "dsl_replay_path": str(replay_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)

    replay_dir = Path(str(manifest["replay_dir"]))
    assert sorted(path.name for path in replay_dir.iterdir()) == ["001-idea-a.replay.json"]


def test_prepare_review_bundle_skips_failed_worker_packages(tmp_path: Path):
    root = tmp_path
    run_id = "20260324-180002"
    run_dir = root / "pipeline" / "runs" / run_id
    replay_path = run_dir / "workers" / "idea-a" / "repo" / "artifacts" / "replays" / "game-a-0001.replay.json"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "game-a-0001"}}, indent=2))
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "game_id": "game-a-0001",
                        "worker_root": str(run_dir / "workers" / "idea-a"),
                        "dsl_replay_path": str(replay_path),
                    },
                    {
                        "idea_id": "idea-b",
                        "status": "failed",
                        "error": "transient model failure",
                        "worker_root": str(run_dir / "workers" / "idea-b"),
                    },
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)

    assert manifest["entry_count"] == 1
    assert manifest["entries"][0]["idea_id"] == "idea-a"


def test_prepare_review_bundle_falls_back_to_workers_when_run_json_has_no_worker_packages(tmp_path: Path):
    root = tmp_path
    run_id = "20260331-170346"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    replay_path = run_dir / "workers" / "idea-a" / "repo" / "artifacts" / "replays" / "game-a-0001.replay.json"
    gif_path = replay_path.with_suffix(".gif")
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "game-a-0001"}}, indent=2))
    _write_text(gif_path, "gif-a\n")
    _write_text(
        worker_root / "idea.json",
        json.dumps({"idea_id": "idea-a", "title": "Idea A", "description": "Fallback idea."}, indent=2),
    )
    _write_text(run_dir / "run.json", json.dumps({"worker_packages": []}, indent=2))

    manifest = prepare_review_bundle(run_id=run_id, root=root)

    assert manifest["entry_count"] == 1
    replay_dir = Path(str(manifest["replay_dir"]))
    assert (replay_dir / "001-idea-a.replay.json").exists()
    assert not (replay_dir / "001-idea-a.gif").exists()
    assert manifest["entries"][0]["idea_title"] == "Idea A"
    assert manifest["entries"][0]["idea_description"] == "Fallback idea."


def test_review_run_prepares_bundle_without_launching_ui(tmp_path: Path):
    root = tmp_path
    run_id = "20260324-180001"
    run_dir = root / "pipeline" / "runs" / run_id
    replay_path = run_dir / "workers" / "idea-a" / "repo" / "artifacts" / "replays" / "game-a-0001.replay.json"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "game-a-0001"}}, indent=2))
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "game_id": "game-a-0001",
                        "worker_root": str(run_dir / "workers" / "idea-a"),
                        "dsl_replay_path": str(replay_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    captured: dict[str, object] = {}

    def fake_replay_ui_runner(args):
        captured["args"] = args

    manifest = review_run(
        run_id=run_id, root=root, host="127.0.0.1", port=8123, launch_ui=False, replay_ui_runner=fake_replay_ui_runner
    )

    assert manifest["entry_count"] == 1
    assert captured == {}


def test_submit_review_promotes_accepted_games(tmp_path: Path):
    root = tmp_path
    run_id = "20260331-150000"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    worker_repo = worker_root / "repo"
    replay_path = worker_repo / "artifacts" / "replays" / "game-a-0001.replay.json"
    gif_path = worker_repo / "artifacts" / "replays" / "game-a-0001.gif"
    spec_path = run_dir / "specs" / "idea-a.md"
    generated_slug = "fresh_game"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "game-a-0001"}}, indent=2))
    _write_text(gif_path, "gif-a\n")
    _write_text(spec_path, "# Fresh Game\n\nSpec text.\n")
    _write_text(worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json", "{}\n")
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py", "# env\n"
    )
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# agent\n")
    _write_text(root / "re_arc" / "environment_files" / "ft09_close" / "0001" / "metadata.json", "{}\n")
    _write_text(root / "re_arc" / "dsl" / "agents" / "ft09_close.py", "# ref agent\n")
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "game_id": "game-a-0001",
                        "generated_slug": generated_slug,
                        "worker_root": str(worker_root),
                        "worker_repo_path": str(worker_repo),
                        "worker_spec_path": str(spec_path),
                        "dsl_replay_path": str(replay_path),
                        "dsl_gif_path": str(gif_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)
    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    submission = submit_review(review_dir=review_dir, root=root)

    assert submission["promoted_count"] == 1
    assert (root / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py").exists()
    assert (root / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py").read_text(encoding="utf-8") == "# agent\n"
    assert not (root / "re_arc" / "environment_files" / generated_slug / "source_spec.md").exists()
    promoted = submission["promoted"][0]
    assert promoted["target_spec_path"] == ""
    assert Path(str(promoted["target_replay_path"])).read_text(encoding="utf-8")
    replay_payload = json.loads(Path(str(promoted["target_replay_path"])).read_text(encoding="utf-8"))
    assert replay_payload["metadata"]["game_id"] == "game-a-0001"
    assert replay_payload["metadata"]["gif_path"] == str(root / "docs" / "replays" / "game-a-0001.gif")
    assert (root / "docs" / "replays" / "game-a-0001.gif").read_text(encoding="utf-8") == "gif-a\n"
    decisions = load_review_decisions(review_dir=review_dir)
    assert decisions["submitted"] is True


def test_submit_review_runs_precompute_for_promoted_games(tmp_path: Path, monkeypatch):
    root = tmp_path
    run_id = "20260331-150006"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    worker_repo = worker_root / "repo"
    replay_path = worker_repo / "artifacts" / "replays" / "fresh_game-0001.replay.json"
    generated_slug = "fresh_game"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "fresh_game-0001"}}, indent=2))
    _write_text(worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json", "{}\n")
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py", "# env\n"
    )
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# agent\n")
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "idea_description": "Fresh generated idea.",
                        "game_id": "fresh_game-0001",
                        "generated_slug": generated_slug,
                        "worker_root": str(worker_root),
                        "worker_repo_path": str(worker_repo),
                        "dsl_replay_path": str(replay_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    captured: dict[str, object] = {}

    def fake_sync(*, root_dir: Path, game_ids: list[str]) -> list[str]:
        captured["root_dir"] = root_dir
        captured["game_ids"] = list(game_ids)
        return []

    monkeypatch.setattr("pipeline.review._sync_promoted_precomputed_actions", fake_sync)

    manifest = prepare_review_bundle(run_id=run_id, root=root)
    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    submission = submit_review(review_dir=review_dir, root=root)

    assert captured["root_dir"] == root
    assert captured["game_ids"] == ["fresh_game-0001"]
    assert submission["precompute_failures"] == []


def test_submit_review_overwrites_existing_slug_when_target_exists(tmp_path: Path):
    root = tmp_path
    run_id = "20260331-150002"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    worker_repo = worker_root / "repo"
    replay_path = worker_repo / "artifacts" / "replays" / "fresh_game-0001.replay.json"
    spec_path = run_dir / "specs" / "idea-a.md"
    generated_slug = "fresh_game"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "fresh_game-0001"}}, indent=2))
    _write_text(spec_path, "# Fresh Game\n\nVariant spec text.\n")
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json",
        json.dumps({"game_id": "fresh_game-0001"}, indent=2),
    )
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py", "# env\n"
    )
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# agent\n")
    _write_text(
        root / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json",
        json.dumps({"game_id": "fresh_game-0001"}, indent=2),
    )
    _write_text(root / "re_arc" / "environment_files" / generated_slug / "old.txt", "old\n")
    _write_text(root / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# existing agent\n")
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "idea_description": "Fresh generated idea.",
                        "game_id": "fresh_game-0001",
                        "generated_slug": generated_slug,
                        "worker_root": str(worker_root),
                        "worker_repo_path": str(worker_repo),
                        "worker_spec_path": str(spec_path),
                        "dsl_replay_path": str(replay_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)
    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    submission = submit_review(review_dir=review_dir, root=root)

    assert submission["promoted_count"] == 1
    promoted = submission["promoted"][0]
    assert promoted["generated_slug"] == "fresh_game"
    assert promoted["promoted_slug"] == "fresh_game"
    assert (root / "re_arc" / "dsl" / "agents" / "fresh_game.py").read_text(encoding="utf-8") == "# agent\n"
    assert not (root / "re_arc" / "environment_files" / "fresh_game" / "old.txt").exists()
    assert not (root / "re_arc" / "environment_files" / "fresh_game" / "source_spec.md").exists()
    assert promoted["target_spec_path"] == ""
    assert Path(str(promoted["target_replay_path"])).name == "fresh_game-0001.replay.json"
    metadata = json.loads(
        (root / "re_arc" / "environment_files" / "fresh_game" / "0001" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["game_id"] == "fresh_game-0001"
    assert metadata["idea"] == "Fresh generated idea."
    assert "further-version" not in metadata.get("tags", [])
    replay_payload = json.loads(Path(str(promoted["target_replay_path"])).read_text(encoding="utf-8"))
    assert replay_payload["metadata"]["game_id"] == "fresh_game-0001"


def test_submit_review_uses_unique_slug_when_existing_slug_is_different_game(tmp_path: Path):
    root = tmp_path
    run_id = "20260331-150007"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    worker_repo = worker_root / "repo"
    replay_path = worker_repo / "artifacts" / "replays" / "fresh_game-0002.replay.json"
    generated_slug = "fresh_game"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "fresh_game-0002"}}, indent=2))
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json",
        json.dumps({"game_id": "fresh_game-0002"}, indent=2),
    )
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py", "# new env\n"
    )
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# new agent\n")
    _write_text(
        root / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json",
        json.dumps({"game_id": "fresh_game-0001"}, indent=2),
    )
    _write_text(root / "re_arc" / "environment_files" / generated_slug / "old.txt", "old\n")
    _write_text(root / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# existing agent\n")
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "idea_description": "Second generated idea.",
                        "game_id": "fresh_game-0002",
                        "generated_slug": generated_slug,
                        "worker_root": str(worker_root),
                        "worker_repo_path": str(worker_repo),
                        "dsl_replay_path": str(replay_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)
    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    submission = submit_review(review_dir=review_dir, root=root)

    assert submission["promoted_count"] == 1
    promoted = submission["promoted"][0]
    assert promoted["generated_slug"] == "fresh_game"
    assert promoted["promoted_slug"] == "fresh_game_v2"
    assert (root / "re_arc" / "environment_files" / "fresh_game" / "old.txt").read_text(encoding="utf-8") == "old\n"
    assert (root / "re_arc" / "dsl" / "agents" / "fresh_game.py").read_text(encoding="utf-8") == "# existing agent\n"
    assert (root / "re_arc" / "dsl" / "agents" / "fresh_game_v2.py").read_text(encoding="utf-8") == "# new agent\n"
    metadata = json.loads(
        (root / "re_arc" / "environment_files" / "fresh_game_v2" / "0001" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["game_id"] == "fresh_game_v2-0002"
    assert "further-version" in metadata["tags"]
    assert Path(str(promoted["target_replay_path"])).name == "fresh_game_v2-0002.replay.json"


def test_submit_review_overwrites_replay_artifacts_with_same_names(tmp_path: Path):
    root = tmp_path
    run_id = "20260331-150005"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    worker_repo = worker_root / "repo"
    replay_path = worker_repo / "artifacts" / "replays" / "fresh_game-0001.replay.json"
    gif_path = worker_repo / "artifacts" / "replays" / "fresh_game-0001.gif"
    generated_slug = "fresh_game"

    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "fresh_game-0001"}}, indent=2))
    _write_text(gif_path, "gif-new\n")
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json",
        json.dumps({"game_id": "fresh_game-0001"}, indent=2),
    )
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py", "# env\n"
    )
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# agent\n")
    _write_text(
        root / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json",
        json.dumps({"game_id": "fresh_game-0001"}, indent=2),
    )
    _write_text(root / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# existing agent\n")
    _write_text(root / "docs" / "replays" / "fresh_game-0001.replay.json", json.dumps({"version": 2}, indent=2))
    _write_text(root / "docs" / "replays" / "fresh_game-0001.gif", "gif-old\n")
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "game_id": "fresh_game-0001",
                        "generated_slug": generated_slug,
                        "worker_root": str(worker_root),
                        "worker_repo_path": str(worker_repo),
                        "dsl_replay_path": str(replay_path),
                        "dsl_gif_path": str(gif_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)
    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    submission = submit_review(review_dir=review_dir, root=root)

    promoted = submission["promoted"][0]
    assert promoted["promoted_slug"] == "fresh_game"
    assert Path(str(promoted["target_replay_path"])).name == "fresh_game-0001.replay.json"
    assert Path(str(promoted["target_gif_path"])).name == "fresh_game-0001.gif"
    replay_payload = json.loads(Path(str(promoted["target_replay_path"])).read_text(encoding="utf-8"))
    assert replay_payload["metadata"]["game_id"] == "fresh_game-0001"
    assert replay_payload["metadata"]["gif_path"] == str(root / "docs" / "replays" / "fresh_game-0001.gif")
    assert Path(str(promoted["target_gif_path"])).read_text(encoding="utf-8") == "gif-new\n"


def test_submit_review_does_not_copy_spec_when_bundle_falls_back_to_workers_dir(tmp_path: Path):
    root = tmp_path
    run_id = "20260331-150004"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    worker_repo = worker_root / "repo"
    replay_path = worker_repo / "artifacts" / "replays" / "fresh_game-0001.replay.json"
    spec_path = run_dir / "specs" / "idea-a.md"
    generated_slug = "fresh_game"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "fresh_game-0001"}}, indent=2))
    _write_text(spec_path, "# Fresh Game\n\nFallback spec text.\n")
    _write_text(worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json", "{}\n")
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py", "# env\n"
    )
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# agent\n")
    _write_text(root / "re_arc" / "environment_files" / "ft09_close" / "0001" / "metadata.json", "{}\n")
    _write_text(root / "re_arc" / "dsl" / "agents" / "ft09_close.py", "# ref agent\n")
    _write_text(run_dir / "run.json", json.dumps({"worker_packages": []}, indent=2))

    manifest = prepare_review_bundle(run_id=run_id, root=root)

    assert manifest["entries"][0]["worker_spec_path"] == str(spec_path)

    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    submission = submit_review(review_dir=review_dir, root=root)

    assert submission["promoted_count"] == 1
    assert submission["promoted"][0]["target_spec_path"] == ""
    assert not (root / "re_arc" / "environment_files" / generated_slug / "source_spec.md").exists()


def test_submit_review_derives_generated_slug_from_game_id_when_worker_repo_has_reference_games(tmp_path: Path):
    root = tmp_path
    run_id = "20260403-112046"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    worker_repo = worker_root / "repo"
    generated_slug = "conveyor_sort_gpt"
    replay_path = worker_repo / "artifacts" / "replays" / f"{generated_slug}-0001.replay.json"
    spec_path = run_dir / "specs" / "idea-a.md"

    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": f"{generated_slug}-0001"}}, indent=2))
    _write_text(spec_path, "# Conveyor Sort\n\nSpec text.\n")
    _write_text(worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json", "{}\n")
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py", "# env\n"
    )
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# agent\n")
    _write_text(worker_repo / "re_arc" / "environment_files" / "ft09" / "0d8bbf25" / "metadata.json", "{}\n")
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / "ft09.py", "# ref agent\n")
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "game_id": f"{generated_slug}-0001",
                        "generated_slug": "",
                        "worker_root": str(worker_root),
                        "worker_repo_path": str(worker_repo),
                        "worker_spec_path": str(spec_path),
                        "dsl_replay_path": str(replay_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)
    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    submission = submit_review(review_dir=review_dir, root=root)

    assert submission["promoted_count"] == 1
    assert (root / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py").exists()


def test_submit_review_skips_spec_copy_when_worker_spec_path_missing(tmp_path: Path):
    root = tmp_path
    run_id = "20260331-150003"
    run_dir = root / "pipeline" / "runs" / run_id
    worker_root = run_dir / "workers" / "idea-a"
    worker_repo = worker_root / "repo"
    replay_path = worker_repo / "artifacts" / "replays" / "fresh_game-0001.replay.json"
    generated_slug = "fresh_game"
    _write_text(replay_path, json.dumps({"version": 2, "metadata": {"game_id": "fresh_game-0001"}}, indent=2))
    _write_text(worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / "metadata.json", "{}\n")
    _write_text(
        worker_repo / "re_arc" / "environment_files" / generated_slug / "0001" / f"{generated_slug}.py", "# env\n"
    )
    _write_text(worker_repo / "re_arc" / "dsl" / "agents" / f"{generated_slug}.py", "# agent\n")
    _write_text(root / "re_arc" / "environment_files" / "ft09_close" / "0001" / "metadata.json", "{}\n")
    _write_text(root / "re_arc" / "dsl" / "agents" / "ft09_close.py", "# ref agent\n")
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "game_id": "fresh_game-0001",
                        "generated_slug": generated_slug,
                        "worker_root": str(worker_root),
                        "worker_repo_path": str(worker_repo),
                        "dsl_replay_path": str(replay_path),
                    }
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)
    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    submission = submit_review(review_dir=review_dir, root=root)

    assert submission["promoted_count"] == 1
    assert submission["promoted"][0]["target_spec_path"] == ""
    assert not (root / "re_arc" / "environment_files" / generated_slug / "source_spec.md").exists()


def test_submit_review_requires_decision_for_every_entry(tmp_path: Path):
    root = tmp_path
    run_id = "20260331-150001"
    run_dir = root / "pipeline" / "runs" / run_id
    replay_a = run_dir / "workers" / "idea-a" / "repo" / "artifacts" / "replays" / "game-a-0001.replay.json"
    replay_b = run_dir / "workers" / "idea-b" / "repo" / "artifacts" / "replays" / "game-b-0001.replay.json"
    _write_text(replay_a, json.dumps({"version": 2, "metadata": {"game_id": "game-a-0001"}}, indent=2))
    _write_text(replay_b, json.dumps({"version": 2, "metadata": {"game_id": "game-b-0001"}}, indent=2))
    _write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "worker_packages": [
                    {
                        "idea_id": "idea-a",
                        "game_id": "game-a-0001",
                        "worker_root": str(run_dir / "workers" / "idea-a"),
                        "dsl_replay_path": str(replay_a),
                    },
                    {
                        "idea_id": "idea-b",
                        "game_id": "game-b-0001",
                        "worker_root": str(run_dir / "workers" / "idea-b"),
                        "dsl_replay_path": str(replay_b),
                    },
                ]
            },
            indent=2,
        ),
    )

    manifest = prepare_review_bundle(run_id=run_id, root=root)
    review_dir = Path(str(manifest["review_dir"]))
    replay_name = Path(str(manifest["entries"][0]["review_replay_path"])).name
    record_review_decision(review_dir=review_dir, review_replay_name=replay_name, decision="accepted")

    with pytest.raises(RuntimeError, match="every replay has an accept/reject decision"):
        submit_review(review_dir=review_dir, root=root)
