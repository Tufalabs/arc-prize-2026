"""Repeated full games keep independent model RNGs and original task identities."""

import importlib.util
import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_support.agent_settings import record_agent_settings


def _worker():
    """Load the standalone worker without requiring scripts as an installed package."""
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "platypus_rollout_worker.py"
    )
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("full_game_slots_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec(tmp_path: Path) -> dict:
    """Describe uneven repeated games with an explicit seed for every slot."""
    return {
        "games": ["ar25", "ar25", "ls20"],
        "slot_seeds": [11, 22, 33],
        "rng_seed": 100,
        "max_concurrent": 3,
        "job_dir": str(tmp_path),
        "base_url": "http://unused.invalid",
        "served_model": "test-model",
        "context_window_tokens": 32768,
        "max_completion_tokens": 4096,
        "compact_to_tokens": 16384,
        "safety_margin_tokens": 512,
        "malformed_retries": 0,
        "accept_malformed_output": True,
        "compaction_trigger_tokens": 28160,
        "max_summary_tokens": 0,
        "summary_forbidden_token_ids": [],
        "max_served_sequence_tokens": 32768,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
    }


def test_repeated_slots_start_fresh_full_games_with_independent_sampling(
    tmp_path: Path,
) -> None:
    """Exercise real GameAPI startup and the harness's per-call seed resolution."""
    pytest.importorskip("duck.agent")
    pytest.importorskip("re_arc")
    from taaf.game import RunSession
    from taaf.game_api import GameAPI

    benchmark = _worker()._build_benchmark(_spec(tmp_path))
    assert benchmark.n_passes == 1
    assert [type(game) for game in benchmark.games] == [GameAPI] * 3
    assert [game.env_name for game in benchmark.games] == ["ar25", "ar25", "ls20"]
    identities = ["ar25#s0", "ar25#s1", "ls20#s2"]
    assert [game.external_game_id for game in benchmark.games] == identities
    agent = benchmark.solver.agent
    assert agent.run_config.rng_seed_per_game == dict(zip(identities, [11, 22, 33]))
    assert agent.run_config.on_agent_error == "crash_game"
    assert agent.malformed_retries == 0
    assert agent.llm_config.backend_factory.engine._http is None

    session = RunSession()
    streams = []
    try:
        for index, game in enumerate(benchmark.games):
            initial = game.start_game(session)
            assert initial.levels_completed == 0
            assert game.game_id == identities[index]
            assert game.game_run is not None
            assert game.game_run.game_id == identities[index]
            assert game.game_run.levels_completed == 0
            assert game.game_run.history == []
            assert sum(game.game_run.actions_per_level) == 0
            seed = benchmark.solver._game_seed(game, index)
            assert seed == [11, 22, 33][index]
            ctx = SimpleNamespace(rng=random.Random(seed))
            profile = next(iter(agent.llm_config.sampling_profiles))
            streams.append(
                tuple(agent.resolve_sampling(ctx, profile).seed for _ in range(3))
            )
        assert len(set(streams)) == 3
    finally:
        for game in benchmark.games:
            if game.game_run is not None:
                game.finish_game()
        session.close()


def test_results_map_repeated_slots_back_to_original_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both independent runs retain the task ID expected by trainer assembly."""
    worker = _worker()
    spec = _spec(tmp_path)
    spec.update(
        games=["ar25", "ar25"],
        results_url="http://unused.invalid/results",
        max_runtime_s_per_game=60,
    )
    (tmp_path / "benchmark.json").write_text(
        json.dumps(
            {
                "game_runs": [
                    {"run_id": "first", "game_id": "ar25#s0", "state": "gave_up"},
                    {"run_id": "second", "game_id": "ar25#s1", "state": "gave_up"},
                ]
            }
        ),
        encoding="utf-8",
    )
    captured = []

    def capture(request, timeout):
        """Capture the callback body without contacting a server."""
        captured.append(json.loads(request.data))
        return io.BytesIO(b'{"accepted": 2}')

    monkeypatch.setattr(worker.urllib.request, "urlopen", capture)
    record_agent_settings(tmp_path, {"max_tokens_per_level": [90_000]})
    worker._post_results(spec)
    assert len(captured) == 1
    assert [(run["run_id"], run["game_id"]) for run in captured[0]["runs"]] == [
        ("first", "ar25"),
        ("second", "ar25"),
    ]
    assert [run["agent_settings"] for run in captured[0]["runs"]] == [
        {"max_tokens_per_level": [90_000]}
    ] * 2


def test_results_reject_unknown_external_slot(tmp_path: Path) -> None:
    """An artifact from another window cannot be silently assigned to a task."""
    (tmp_path / "benchmark.json").write_text(
        json.dumps({"game_runs": [{"run_id": "run", "game_id": "ar25#s99"}]}),
        encoding="utf-8",
    )
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="unknown rollout slot"):
        _worker()._post_results(spec)
