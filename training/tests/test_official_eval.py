import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.config import ExperimentConfig
from training.official_eval import (
    _collect_eval_chunk,
    _collector_config,
    _eval_server_base_urls,
    _latest_published_policy,
    _merge_eval_child_windows,
    _rollout_seed,
)


def test_collect_eval_chunk_passes_explicit_cycle_state(tmp_path: Path) -> None:
    calls: list[dict] = []

    class _Collector:
        def collect_window(self, **kwargs):
            calls.append(kwargs)
            return "child-window"

    result = _collect_eval_chunk(
        collector=_Collector(),
        chunk=[(4, "game-a", 101), (9, "game-b", 202)],
        output_root=tmp_path / "cycle_00003" / "server_1",
        model_id="policy",
        policy_iteration=17,
        cycle_index=3,
    )

    assert result == "child-window"
    assert calls == [
        {
            "output_root": tmp_path / "cycle_00003" / "server_1",
            "model_id": "policy",
            "policy_iteration": 17,
            "schedule_index": 3,
            "scheduled_games": ["game-a", "game-b"],
            "scheduled_rollout_seeds": [101, 202],
            "window_kind": "eval",
        }
    ]


def test_collect_eval_chunk_skips_empty_server_share(tmp_path: Path) -> None:
    class _Collector:
        def collect_window(self, **_kwargs):
            raise AssertionError("empty shares must not launch a rollout window")

    assert (
        _collect_eval_chunk(
            collector=_Collector(),
            chunk=[],
            output_root=tmp_path,
            model_id="base",
            policy_iteration=-1,
            cycle_index=0,
        )
        is None
    )


def test_merge_eval_child_windows_restores_original_slot_order() -> None:
    chunks = [
        [(0, "game-a", 1), (2, "game-c", 3)],
        [(1, "game-b", 2)],
    ]
    child_windows = {
        0: SimpleNamespace(rollouts=["rollout-a", "rollout-c"]),
        1: SimpleNamespace(rollouts=["rollout-b"]),
    }

    assert _merge_eval_child_windows(
        chunks=chunks,
        child_windows=child_windows,
    ) == ["rollout-a", "rollout-b", "rollout-c"]


@pytest.mark.parametrize(
    ("chunks", "child_windows", "message"),
    [
        (
            [[(0, "game-a", 1), (1, "game-b", 2)]],
            {0: SimpleNamespace(rollouts=["rollout-a"])},
            "returned 1 rollouts for 2 scheduled",
        ),
        (
            [[(0, "game-a", 1)]],
            {},
            "returned no window for 1 scheduled",
        ),
    ],
)
def test_merge_eval_child_windows_rejects_incomplete_results(
    chunks: list[list[tuple[int, str, int]]],
    child_windows: dict[int, SimpleNamespace],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _merge_eval_child_windows(chunks=chunks, child_windows=child_windows)


def _base_payload() -> dict:
    return {
        "num_iterations": 2,
        "parallel_rollouts": 4,
        "rollout_buffer_size": 1,
        "inference": {
            "agent_max_completion_tokens": 4096,
            "base_url": "http://127.0.0.1:1234/v1",
            "model_id": "local-qwen",
            "games": ["train_game-0001", "train_game-0002"],
            "max_runtime_minutes": 5.0,
            "platypus_venv": "/tmp/platypus-venv",
        },
        "training": {
            "model_name": "Qwen/Qwen3.6-27B",
            "output_dir": "runs",
            "device": "cpu",
            "seed": 42,
            "max_sequence_tokens": 512,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "epochs_per_window": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "max_grad_norm": 1.0,
            "bf16": False,
            "gradient_checkpointing": False,
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj"],
            },
        },
    }


def test_every_tracked_ppo_config_loads() -> None:
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    config_paths = sorted(configs_dir.glob("ppo*.json"))
    assert config_paths

    for config_path in config_paths:
        ExperimentConfig.load(config_path)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_official_eval_config_defaults_to_disabled(tmp_path: Path) -> None:
    config = ExperimentConfig.load(_write_config(tmp_path, _base_payload()))
    assert config.official_eval.enabled is False
    assert config.official_eval.games == []


def test_config_allows_policy_adapter_init_without_trainer_state(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["training"]["init_adapter_dir"] = "/tmp/policy-adapter"

    config = ExperimentConfig.load(_write_config(tmp_path, payload))

    assert config.training.init_adapter_dir == "/tmp/policy-adapter"
    assert config.training.init_trainer_state_path is None


def test_config_rejects_trainer_state_init_without_policy_adapter(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["training"]["init_trainer_state_path"] = "/tmp/trainer-state.pt"

    with pytest.raises(ValueError, match="requires training.init_adapter_dir"):
        ExperimentConfig.load(_write_config(tmp_path, payload))


def test_official_eval_requires_games_and_gpu(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["official_eval"] = {"enabled": True, "games": [], "server_gpus": ["7"]}
    with pytest.raises(ValueError, match="official_eval.games"):
        ExperimentConfig.load(_write_config(tmp_path, payload))

    payload["official_eval"] = {"enabled": True, "games": ["g-1"], "server_gpus": []}
    with pytest.raises(ValueError, match="official_eval.server_gpus"):
        ExperimentConfig.load(_write_config(tmp_path, payload))


@pytest.mark.parametrize("field", ["eval_games", "eval_every_iterations"])
def test_config_rejects_removed_trainer_eval_fields(tmp_path: Path, field: str) -> None:
    payload = _base_payload()
    payload["inference"][field] = [] if field == "eval_games" else 5
    with pytest.raises(ValueError, match=field):
        ExperimentConfig.load(_write_config(tmp_path, payload))


def test_official_eval_gpu_must_be_disjoint(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["inference"]["server_gpus"] = ["0", "1"]
    payload["training"]["device"] = "cuda:2"
    payload["training"]["device_map_gpus"] = [2, 3]
    payload["official_eval"] = {
        "enabled": True,
        "games": ["g-1"],
        "server_gpus": ["3"],
    }
    with pytest.raises(ValueError, match="disjoint GPUs"):
        ExperimentConfig.load(_write_config(tmp_path, payload))


def test_next_cycle_index_resumes_across_relaunches(tmp_path: Path) -> None:
    from training.official_eval import _next_cycle_index

    # Fresh run: no ledger, no artifacts.
    assert _next_cycle_index(tmp_path) == 0

    # Ledger rows from a prior leg (plus junk that must be skipped).
    (tmp_path / "official_eval_metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"cycle_index": 0, "avg_arc_score": 0.4}),
                json.dumps({"cycle_index": 3, "avg_arc_score": 0.5}),
                "not-json",
                json.dumps({"no_cycle_key": True}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert _next_cycle_index(tmp_path) == 4

    # A cycle that crashed after creating its artifact dir but before its
    # ledger row still claims its index.
    rollouts = tmp_path / "official_eval_rollouts"
    (rollouts / "cycle_00007").mkdir(parents=True)
    (rollouts / "cycle_junk").mkdir()
    assert _next_cycle_index(tmp_path) == 8


def test_next_cycle_index_scans_node_local_rollout_root(tmp_path: Path) -> None:
    from training.official_eval import _next_cycle_index

    run_root = tmp_path / "runs" / "20260723_120000"
    scratch_root = (
        tmp_path
        / "scratch"
        / "rollout_windows"
        / run_root.name
        / "official_eval_rollouts"
    )
    (scratch_root / "cycle_00004").mkdir(parents=True)
    (run_root / "official_eval_rollouts" / "cycle_00006").mkdir(parents=True)

    assert _next_cycle_index(run_root, rollouts_root=scratch_root) == 7


def test_latest_published_policy_reads_training_state(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError, match="initial LoRA policy has not been published"
    ):
        _latest_published_policy(tmp_path)
    (tmp_path / "initial_policy.json").write_text(
        json.dumps(
            {
                "last_completed_iteration": -1,
                "runtime_adapter_dir": "checkpoints/adapter_initial/runtime_vllm",
            }
        )
    )
    assert _latest_published_policy(tmp_path) == (
        -1,
        tmp_path / "checkpoints/adapter_initial/runtime_vllm",
    )

    (tmp_path / "training_state.json").write_text(
        json.dumps(
            {
                "last_completed_iteration": 7,
                "runtime_adapter_dir": "checkpoints/adapter_iter_000007/runtime_vllm",
            }
        ),
        encoding="utf-8",
    )
    iteration, adapter_dir = _latest_published_policy(tmp_path)
    assert iteration == 7
    assert adapter_dir == tmp_path / "checkpoints/adapter_iter_000007/runtime_vllm"


def test_collector_config_applies_eval_budget_and_context(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["official_eval"] = {
        "enabled": True,
        "games": ["official-1", "official-2", "official-3"],
        "server_gpus": ["7"],
        "cycle_minutes": 45,
        "context_window_tokens": 65536,
    }
    config = ExperimentConfig.load(_write_config(tmp_path, payload))
    eval_config = _collector_config(config, base_url="http://127.0.0.1:9123/v1")

    assert eval_config.parallel_rollouts == 3
    assert eval_config.inference.games == ["official-1", "official-2", "official-3"]
    assert eval_config.inference.server_gpus == []
    assert eval_config.inference.max_runtime_minutes == 45.0
    # Eval-only context window overrides the training 32k constraint.
    assert eval_config.inference.agent_context_window == 65536
    assert eval_config.inference.base_url.endswith(":9123/v1")
    assert eval_config.training.device == "cpu"
    # The original config is untouched.
    assert config.inference.max_runtime_minutes == 5.0


def test_collector_config_sizes_slots_for_multi_server_rollouts_per_game(
    tmp_path: Path,
) -> None:
    # Decision 13: 3 games x 2 rollouts/game = 6 slots, one complete pass
    # per eval server -> each collector holds ceil(6/2)=3 slots, pinned to
    # its own GPU and pre-launched server URL.
    payload = _base_payload()
    payload["official_eval"] = {
        "enabled": True,
        "games": ["official-1", "official-2", "official-3"],
        "server_gpus": ["6", "7"],
        "rollouts_per_game": 2,
        "cycle_minutes": 30,
    }
    config = ExperimentConfig.load(_write_config(tmp_path, payload))
    assert config.official_eval.rollouts_per_game == 2
    first = _collector_config(config, base_url="http://127.0.0.1:9200/v1")
    second = _collector_config(config, base_url="http://127.0.0.1:9201/v1")
    assert first.parallel_rollouts == 3
    assert second.parallel_rollouts == 3
    assert first.inference.base_url.endswith(":9200/v1")
    assert second.inference.base_url.endswith(":9201/v1")
    assert first.inference.server_gpus == []


def test_shipped_smaller_eval_context_has_its_own_compaction_target() -> None:
    from training.config import full_game_context_contract

    config = ExperimentConfig.load(
        Path(__file__).resolve().parents[1] / "configs/ppo_5gpu_arc3_officials_platypus.json"
    )
    evaluation = _collector_config(config, base_url="http://unused.invalid/v1")

    assert evaluation.inference.agent_context_window == 32768
    assert full_game_context_contract(evaluation.inference) == (28160, 16384, 512)
    assert config.inference.agent_context_window == 65536
    assert config.inference.agent_compact_to_tokens == 32768
    # The target survives freezing/serialization as an explicit eval setting.
    assert config.to_dict()["official_eval"]["compact_to_tokens"] == 16384


@pytest.mark.parametrize("target", [0, -1, 32768])
def test_invalid_eval_compaction_target_is_rejected_without_silent_clamping(
    target: int,
) -> None:
    from training.config import full_game_context_contract

    config = ExperimentConfig.load(
        Path(__file__).resolve().parents[1] / "configs/ppo_5gpu_arc3_officials_platypus.json"
    )
    config.official_eval.compact_to_tokens = target
    evaluation = _collector_config(config, base_url="http://unused.invalid/v1")
    with pytest.raises(ValueError, match="agent_compact_to_tokens"):
        full_game_context_contract(evaluation.inference)


def test_eval_server_base_urls_continue_the_training_port_contract(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["inference"]["base_url"] = "http://127.0.0.1:50000/v1"
    payload["inference"]["server_gpus"] = ["0", "1"]
    payload["official_eval"] = {
        "enabled": True,
        "games": ["official-1"],
        "server_gpus": ["6", "7"],
    }
    config = ExperimentConfig.load(_write_config(tmp_path, payload))

    # Training servers own indices 0..1 (ports 50000/50100); the eval
    # servers continue the deterministic sequence — never discovered free
    # ports, which no launcher could have been given.
    assert _eval_server_base_urls(config) == [
        "http://127.0.0.1:50200/v1",
        "http://127.0.0.1:50300/v1",
    ]

    payload["inference"].pop("server_gpus")
    single_server = ExperimentConfig.load(_write_config(tmp_path, payload))
    assert _eval_server_base_urls(single_server) == [
        "http://127.0.0.1:50100/v1",
        "http://127.0.0.1:50200/v1",
    ]


def test_eval_server_base_urls_prefer_explicit_config(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["official_eval"] = {
        "enabled": True,
        "games": ["official-1"],
        "server_gpus": ["6", "7"],
        "base_urls": [
            "http://10.0.0.5:8100/v1",
            "http://10.0.0.6:8100/v1",
        ],
    }
    config = ExperimentConfig.load(_write_config(tmp_path, payload))
    assert _eval_server_base_urls(config) == [
        "http://10.0.0.5:8100/v1",
        "http://10.0.0.6:8100/v1",
    ]

    config.official_eval.base_urls = ["http://10.0.0.5:8100/v1"]
    with pytest.raises(ValueError, match="official_eval.base_urls lists 1 URLs"):
        _eval_server_base_urls(config)


def test_rollout_seeds_are_stable_per_game_copy_and_cycle() -> None:
    seed_a = _rollout_seed(training_seed=42, game_id="bp35-0a0ad940")
    seed_b = _rollout_seed(training_seed=42, game_id="bp35-0a0ad940")
    seed_c = _rollout_seed(training_seed=42, game_id="ls20-9607627b")
    assert seed_a == seed_b  # deterministic: a resumed run replays its seeds
    assert seed_a != seed_c
    # Copies of one game in a cycle, and the same game across cycles, draw
    # different decode seeds (they sampled identically when they shared one).
    copies = {
        _rollout_seed(training_seed=42, game_id="bp35-0a0ad940", copy_index=i)
        for i in range(25)
    }
    assert len(copies) == 25
    cycles = {
        _rollout_seed(training_seed=42, game_id="bp35-0a0ad940", cycle_index=c)
        for c in range(10)
    }
    assert len(cycles) == 10


def test_stats_at_minutes_drops_late_level_completions() -> None:
    from training.contracts import RolloutTrajectory
    from training.official_eval import _stats_at_minutes

    rollout = RolloutTrajectory(
        trajectory_id="eval-a",
        task_id="bp35-0a0ad940",
        steps=[],
        trajectory_reward=0.0,
        metadata={
            "game_total_levels": 6.0,
            "level_scores": {1: 100.0, 2: 100.0},
            "level_completion_elapsed_seconds": {1: 300.0, 2: 1500.0},
        },
    )
    no_timing = RolloutTrajectory(
        trajectory_id="eval-b",
        task_id="ls20-9607627b",
        steps=[],
        trajectory_reward=0.0,
        metadata={
            "game_total_levels": 4.0,
            "level_scores": {},
            "level_completion_elapsed_seconds": {},
        },
    )

    stats = _stats_at_minutes([rollout, no_timing], minutes=10.0)

    # Level 2 completed at minute 25 is dropped; level 1 of 6 at full
    # efficiency contributes its level-index weight (1 of 1+..+6 = 21),
    # averaged with the scoreless rollout.
    assert stats["rollouts"] == 2.0
    assert stats["avg_levels_completed"] == pytest.approx(0.5)
    assert stats["avg_arc_score"] == pytest.approx((100.0 * 1 / 21) / 2.0)

    late_stats = _stats_at_minutes([rollout], minutes=45.0)
    assert late_stats["avg_levels_completed"] == pytest.approx(2.0)


def test_official_eval_stats_minutes_parses_and_sorts(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["official_eval"] = {
        "enabled": True,
        "games": ["g-1"],
        "server_gpus": ["7"],
        "stats_minutes": [30, 10.0, 10],
    }
    config = ExperimentConfig.load(_write_config(tmp_path, payload))
    assert config.official_eval.stats_minutes == [10.0, 30.0]

    payload["official_eval"]["stats_minutes"] = []
    config = ExperimentConfig.load(_write_config(tmp_path, payload))
    assert config.official_eval.stats_minutes == []

    del payload["official_eval"]["stats_minutes"]
    config = ExperimentConfig.load(_write_config(tmp_path, payload))
    # Default is now EMPTY: mark-stats logging is opt-in (owner 2026-07-18).
    assert config.official_eval.stats_minutes == []


def test_tokens_per_generation_falls_back_to_usage_counts() -> None:
    from training.contracts import RolloutStep, RolloutTrajectory
    from training.metrics import summarize_rollouts

    def _step(index: int, *, token_ids: list[int], usage: dict | None) -> RolloutStep:
        metadata = {"usage": usage} if usage is not None else {}
        return RolloutStep(
            step_index=index,
            prompt_messages=[],
            tool_schemas=[],
            assistant_message={"role": "assistant"},
            reward=0.0,
            done=False,
            completion_token_ids=token_ids,
            metadata=metadata,
        )

    rollout = RolloutTrajectory(
        trajectory_id="eval-usage",
        task_id="bp35-0a0ad940",
        steps=[
            # Captured ids win when present.
            _step(1, token_ids=[1, 2, 3], usage={"completion_tokens": 999}),
            # Capture off (official eval): server usage counts fill in.
            _step(2, token_ids=[], usage={"completion_tokens": 400}),
            _step(3, token_ids=[], usage=None),
        ],
        trajectory_reward=0.0,
        metadata={},
    )

    metrics = summarize_rollouts([rollout])

    assert metrics.avg_tokens_per_generation == pytest.approx((3 + 400 + 0) / 3)


def test_collector_config_applies_eval_sampling_params(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["inference"]["agent_temperature"] = 1.0
    payload["official_eval"] = {
        "enabled": True,
        "games": ["g-1"],
        "server_gpus": ["7"],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    config = ExperimentConfig.load(_write_config(tmp_path, payload))
    eval_config = _collector_config(config, base_url="http://127.0.0.1:9000/v1")

    assert eval_config.inference.agent_temperature == 0.6
    assert eval_config.inference.agent_top_p == 0.95
    assert eval_config.inference.agent_top_k == 20
    # Training sampling untouched.
    assert config.inference.agent_temperature == 1.0


def test_latest_published_policy_requires_explicit_reexport_publication(
    tmp_path: Path,
) -> None:
    from training.run_artifacts import write_json_atomic

    state_path = tmp_path / "training_state.json"
    old_export = tmp_path / "previous-job" / "runtime_vllm"
    new_export = tmp_path / "this-job" / "runtime_vllm"
    state = {
        "last_completed_iteration": 4,
        "runtime_adapter_dir": str(old_export),
    }
    write_json_atomic(state_path, state)
    assert _latest_published_policy(tmp_path) == (4, old_export)
    new_export.mkdir(parents=True)
    # Creating an export does not publish it: readers use the atomic state.
    assert _latest_published_policy(tmp_path) == (4, old_export)
    write_json_atomic(state_path, {**state, "runtime_adapter_dir": str(new_export)})
    assert _latest_published_policy(tmp_path) == (4, new_export)


def test_eval_future_failure_is_latched_before_aborting_a_blocked_sibling(
    tmp_path: Path,
) -> None:
    import training.official_eval as official_eval
    from training.run_failure import RUN_FAILURE_FILENAME

    started = threading.Event()
    aborted = threading.Event()
    error = RuntimeError("invalid serving distribution")

    class _Collector:
        def abort_window(self):
            assert (tmp_path / RUN_FAILURE_FILENAME).exists()
            aborted.set()

    def slow_first_server():
        started.set()
        assert aborted.wait(timeout=2), (
            "failed server waited behind its blocked sibling"
        )
        return "unused window"

    def failed_second_server():
        assert started.wait(timeout=2)
        raise error

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(slow_first_server): 0,
            executor.submit(failed_second_server): 1,
        }
        with pytest.raises(
            RuntimeError, match="invalid serving distribution"
        ) as raised:
            official_eval._resolve_eval_futures(
                futures, collectors=[_Collector()], run_root=tmp_path
            )
    assert raised.value is error
    assert aborted.is_set()
    failure = json.loads((tmp_path / RUN_FAILURE_FILENAME).read_text())
    assert failure["message"] == str(error)


@pytest.mark.parametrize(
    ("failure_source", "message"),
    [
        ("ensure", "invalid serving distribution"),
        ("collect", "broken token continuation"),
        ("adapter", "still missing"),
    ],
)
def test_eval_errors_stop_the_run_before_teardown_without_base_fallback(
    tmp_path: Path, monkeypatch, failure_source: str, message: str
) -> None:
    import training.official_eval as official_eval
    from training.run_failure import RUN_FAILURE_FILENAME

    payload = _base_payload()
    payload["official_eval"] = {
        "enabled": True,
        "games": ["tu93-0768757b"],
        "server_gpus": ["7"],
    }
    config_path = _write_config(tmp_path, payload)
    run_root = tmp_path / "run"
    run_root.mkdir()
    initial_adapter = run_root / "initial-lora"
    initial_adapter.mkdir()
    (run_root / "initial_policy.json").write_text(
        json.dumps(
            {
                "last_completed_iteration": -1,
                "runtime_adapter_dir": str(initial_adapter),
            }
        )
    )
    if failure_source == "adapter":
        (run_root / "training_state.json").write_text(
            json.dumps(
                {"last_completed_iteration": 7, "runtime_adapter_dir": "missing-export"}
            )
        )
        monkeypatch.setattr(official_eval, "_ADAPTER_WAIT_SECONDS", 0)
    calls = []
    teardown = []

    def assert_failure_recorded():
        failure = json.loads((run_root / RUN_FAILURE_FILENAME).read_text())
        assert message in failure["message"]

    class _Collector:
        def __init__(self, *_args, **_kwargs):
            assert _kwargs["failure_root"] == run_root
            assert _kwargs["run_root"] == run_root / "official_eval_worker"

        def ensure_server(self):
            if failure_source == "ensure":
                raise RuntimeError(message)

        def load_runtime_adapter(self, **kwargs):
            assert failure_source == "collect"
            assert kwargs == {"adapter_name": "policy", "adapter_path": initial_adapter}

        def collect_window(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("broken token continuation")

        def abort_window(self):
            assert_failure_recorded()

    class _Writer:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            assert_failure_recorded()
            teardown.append("writer")

    def stage_out(**_kwargs):
        assert_failure_recorded()
        teardown.append("artifacts")

    monkeypatch.setattr(official_eval, "_configure_hf_cache_defaults", lambda: None)
    monkeypatch.setattr(
        official_eval, "setup_logging_for_experiment", lambda **kwargs: None
    )
    monkeypatch.setattr(
        official_eval, "resolve_rollout_windows_root", lambda path: path
    )
    monkeypatch.setattr(official_eval, "RolloutCollector", _Collector)
    monkeypatch.setattr(official_eval, "SummaryWriter", _Writer)
    monkeypatch.setattr(official_eval, "stage_out_rollout_scratch", stage_out)

    with pytest.raises(RuntimeError, match=message):
        official_eval.run_official_eval(config_path=str(config_path), run_root=run_root)
    assert teardown == ["writer", "artifacts"]
    assert (run_root / "STOP_AUTO_RELAUNCH").exists()
    assert not (run_root / "official_eval_metrics.jsonl").exists()
    if failure_source == "collect":
        assert len(calls) == 1
        assert calls[0]["model_id"] == "policy"
        assert calls[0]["policy_iteration"] == -1
    else:
        assert calls == []


@pytest.mark.parametrize("iteration", [-1, 7])
def test_published_iteration_requires_an_adapter_path(
    tmp_path: Path, iteration
) -> None:
    import training.official_eval as official_eval

    with pytest.raises(
        RuntimeError, match="initial LoRA policy has not been published"
    ):
        official_eval._await_published_policy(tmp_path)
    (tmp_path / "training_state.json").write_text(
        json.dumps({"last_completed_iteration": iteration, "runtime_adapter_dir": None})
    )
    with pytest.raises(
        RuntimeError,
        match=f"published iteration {iteration} has no runtime adapter path",
    ):
        official_eval._await_published_policy(tmp_path)


@pytest.mark.parametrize("weight_mode", ["lora", "full_weight"])
@pytest.mark.parametrize("resumed", [False, True])
def test_eval_uses_policy_steps_and_waits_for_new_policies(
    tmp_path: Path, monkeypatch, weight_mode: str, resumed: bool
) -> None:
    import training.official_eval as official_eval
    import training.official_eval_policy as eval_policy
    from training.contracts import RolloutStep, RolloutTrajectory

    payload = _base_payload()
    payload["official_eval"] = {
        "enabled": True,
        "games": ["tu93-0768757b"],
        "server_gpus": ["7"],
    }
    # The loop only needs weight_mode; bypass unrelated full-weight GPU guards.
    config = ExperimentConfig.load(_write_config(tmp_path, payload))
    config.training.weight_mode = weight_mode
    monkeypatch.setattr(ExperimentConfig, "load", lambda _: config)
    ledger = tmp_path / "official_eval_metrics.jsonl"
    if resumed:
        ledger.write_text(json.dumps({"cycle_index": 3, "iteration": 7}) + "\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    policies = iter([7, 7, 10, 10, 14] if resumed else [-1, -1, 7, 7, 10, 10, 14])
    acquired_cycles = []
    waits = []
    collected = []
    activated = []
    scalars = []

    class _Done(BaseException):
        pass

    def next_policy():
        try:
            return next(policies)
        except StopIteration:
            raise _Done

    class _Provider:
        def __init__(self, **_kwargs):
            pass

        def acquire(self, *, cycle_index):
            acquired_cycles.append(cycle_index)
            iteration = next_policy()
            return eval_policy.EvalPolicy(
                iteration, "policy", ({"iteration": iteration},)
            )

    class _Collector:
        def __init__(self, *_args, **_kwargs):
            pass

        def ensure_server(self):
            pass

        def load_runtime_adapter(self, **kwargs):
            activated.append(kwargs)

        def pin_full_weight_identity(self, identity):
            activated.append(identity)

        def collect_window(self, **kwargs):
            collected.append(kwargs)
            rollout = RolloutTrajectory(
                trajectory_id="eval-a",
                task_id="tu93-0768757b",
                steps=[
                    RolloutStep(
                        i,
                        [],
                        [],
                        {},
                        0.0,
                        False,
                        metadata={
                            "segment_start": True,
                            "usage": {"completion_tokens": 3},
                        },
                    )
                    for i in range(2)
                ],
            )
            return SimpleNamespace(rollouts=[rollout], collection_duration_seconds=2.0)

    class _Writer:
        def __init__(self, **_kwargs):
            pass

        def add_scalar(self, tag, value, step):
            scalars.append((tag, value, step))

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(official_eval, "_configure_hf_cache_defaults", lambda: None)
    monkeypatch.setattr(official_eval, "setup_logging_for_experiment", lambda **_: None)
    monkeypatch.setattr(official_eval, "resolve_rollout_windows_root", lambda path: path)
    monkeypatch.setattr(official_eval, "stage_out_rollout_scratch", lambda **_: None)
    monkeypatch.setattr(official_eval, "RolloutCollector", _Collector)
    monkeypatch.setattr(official_eval, "SummaryWriter", _Writer)
    monkeypatch.setattr(official_eval.time, "sleep", waits.append)
    monkeypatch.setattr(
        official_eval, "_await_published_policy", lambda _: (next_policy(), adapter)
    )
    monkeypatch.setattr(eval_policy, "FullWeightEvalPolicyProvider", _Provider)

    with pytest.raises(_Done):
        official_eval.run_official_eval(config_path="unused", run_root=tmp_path)

    iterations = [10, 14] if resumed else [-1, 7, 10, 14]
    cycles = [4, 5] if resumed else [0, 1, 2, 3]
    assert [call["policy_iteration"] for call in collected] == iterations
    assert [call["schedule_index"] for call in collected] == cycles
    assert [call["output_root"].parent.name for call in collected] == [
        f"cycle_{cycle:05d}" for cycle in cycles
    ]
    assert len(activated) == len(iterations)
    assert len(waits) == 3
    assert [
        (value, step) for tag, value, step in scalars if tag == "eval/cycle_index"
    ] == list(zip(cycles, iterations))
    for metric, expected in (
        ("eval/tokens_per_s", 3.0),
        ("eval/compactions_per_rollout", 1.0),
    ):
        assert [(value, step) for tag, value, step in scalars if tag == metric] == [
            (expected, iteration) for iteration in iterations
        ]
    assert {step for _, _, step in scalars} == set(iterations)
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["iteration"] for row in records] == (
        ([7] if resumed else []) + iterations
    )
    if weight_mode == "full_weight":
        # Even an already evaluated resumed policy completes the startup handshake.
        assert acquired_cycles[:2] == ([4, 4] if resumed else [0, 1])



def test_last_evaluated_iteration_ignores_incomplete_ledger_records(tmp_path: Path) -> None:
    from training.official_eval import _last_evaluated_iteration

    assert _last_evaluated_iteration(tmp_path) is None
    ledger = tmp_path / "official_eval_metrics.jsonl"
    ledger.write_text('{"iteration": -1}\n')
    assert _last_evaluated_iteration(tmp_path) == -1
    with ledger.open("a") as handle:
        handle.write('{"iteration": 7}\n{"cycle_index": 3}\n{"iteration": 4}\n{partial')
    assert _last_evaluated_iteration(tmp_path) == 7


def test_long_horizon_eval_preserves_all_puzzles_and_admission_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import harness_support.arc2.trainer as arc2_support

    # Exercise scheduling without requiring the private harness checkout.
    # The real packed datasets are validated in test_arc2_integration.py.
    puzzle_ids = [f"eval-puzzle-{index:03d}" for index in range(120)]

    def dataset_manifest(name: str) -> dict:
        assert name == "arc2-eval"
        return {
            "name": name,
            "sha256": "test-manifest",
            "puzzle_ids": puzzle_ids,
            "test_outputs": dict.fromkeys(puzzle_ids, 1),
        }

    monkeypatch.setattr(arc2_support, "dataset_manifest", dataset_manifest)

    config = ExperimentConfig.load(
        "configs/grpo_8gpu_arc2_full_weight_overfit_60k.json"
    )
    arc2_support.resolve_datasets(config)
    collector = _collector_config(config, base_url="http://localhost:1234")
    assert collector.inference.games == puzzle_ids
    assert collector.parallel_rollouts == 120
    assert collector.inference.max_concurrent_episodes == 16
    assert collector.inference.agent_max_episode_tokens == 60000
    assert collector.inference.max_runtime_minutes == 60
    assert config.parallel_rollouts == 32
    assert config.inference.server_gpus == ["0", "1"]
    assert config.training.full_weight.device_ids == [2, 3, 4, 5, 6]
