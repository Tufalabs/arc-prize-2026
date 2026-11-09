"""Construct the pinned ARC-3 worker without a server or local weights."""

from types import SimpleNamespace

import pytest

duck = pytest.importorskip("duck.agent")
if not hasattr(duck.DuckAgent, "multimodal_context"):
    pytest.skip("requires the ARC-3 Duck environment", allow_module_level=True)


def worker_spec(tmp_path, *, summarizing=False):
    return dict(
        games=["fixture"],
        rng_seed=7,
        max_concurrent=1,
        max_runtime_s_per_game=900,
        base_url="http://127.0.0.1:1",
        served_model="policy",
        tokenizer_id="Qwen/Qwen3.8-27B",
        max_completion_tokens=4096,
        context_window_tokens=32768,
        max_summary_tokens=4096 if summarizing else 0,
        max_served_sequence_tokens=49152 if summarizing else 32768,
        summary_forbidden_token_ids=[248058] if summarizing else [],
        compact_to_tokens=28160 if summarizing else 16384,
        safety_margin_tokens=512,
        compaction_trigger_tokens=28160,
        max_episode_tokens=12345,
        malformed_retries=0,
        accept_malformed_output=True,
        sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0},
        job_dir=str(tmp_path),
    )


@pytest.mark.parametrize("tokenizer_revision", ["pinned-revision", None])
@pytest.mark.parametrize("summarizing", [False, True])
@pytest.mark.parametrize("image_mode", ["", "current_grid", "level_start"])
def test_worker_uses_the_supplied_engine(
    monkeypatch, tmp_path, summarizing, image_mode, tokenizer_revision
):
    pytest.importorskip("vllm")
    from ullm.backends import deployment
    from ullm.backends.engine_proxy import EngineProxy, EngineProxyFactory
    from harness_support.arc3.worker import build_benchmark

    def forbidden(*args, **kwargs):
        raise AssertionError("supplied engine must not load local weights")

    monkeypatch.setattr(deployment, "build", forbidden)
    monkeypatch.setattr(
        "taaf.game_api.GameAPI", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        "taaf.benchmark.Benchmark", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    spec = worker_spec(tmp_path, summarizing=summarizing)
    spec.update(multimodal_context=image_mode, model_revision="pinned-revision")
    agent = build_benchmark(spec).solver.agent
    factory = agent.llm_config.backend_factory
    assert isinstance(factory, EngineProxyFactory)
    assert isinstance(factory.engine, EngineProxy)
    assert factory.engine._http is None
    assert not getattr(agent, "derive_max_pixels", False)
    assert agent.compaction_strategy == ("summarize" if summarizing else "prune")
    assert agent.journal is not summarizing
    assert agent.preserve_failed_draws is summarizing
    assert agent.malformed_retries == 0
    assert agent.summary_thinking is False
    assert agent.multimodal_context == image_mode
    assert agent._turn_tools(), "image input must preserve the ordinary Duck tools"
    assert agent.max_tokens_per_game == 12345
    assert agent.engine_ceiling_tokens == (49152 if summarizing else 32768)
    assert agent.context_budget_tokens() == 28160
    assert agent.compact_target_tokens() == (28160 if summarizing else 16384)
    assert agent.llm_config.exact_history
    assert factory.backend_kwargs["accept_malformed_output"]
    assert not factory.backend_kwargs["replay_on_link_loss"]

    # Hosted models report the pinned revision; local merged checkpoints
    # report None even though the training recipe retains its HF revision.
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    from ullm.backends.engine_proxy import tokenizer_fingerprint

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    metadata = {
        name: getattr(tokenizer, name)
        for name in (
            "chat_template", "special_tokens_map", "bos_token_id",
            "eos_token_id", "pad_token_id",
        )
    }
    metadata.update(
        tokenizer_id=(
            spec["tokenizer_id"] if tokenizer_revision
            else str(tmp_path / "merged-checkpoint")
        ),
        tokenizer_revision=tokenizer_revision,
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
    )
    loads = []
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda path, **kwargs: loads.append((path, kwargs)) or tokenizer,
    )
    assert factory.engine._load_tokenizer(metadata) is tokenizer
    assert loads[0][0] == spec["tokenizer_id"]
    assert loads[0][1]["revision"] == tokenizer_revision
    metadata["tokenizer_fingerprint"] = {"sha256": "wrong"}
    with pytest.raises(ValueError, match="tokenizer fingerprint differs"):
        factory.engine._load_tokenizer(metadata)


def test_summary_worker_requires_retained_draw_support(monkeypatch, tmp_path):
    from harness_support.arc3.worker import build_benchmark

    monkeypatch.delattr(duck.DuckAgent, "preserve_failed_draws", raising=False)
    with pytest.raises(ValueError, match="preserve_failed_draws"):
        build_benchmark(worker_spec(tmp_path, summarizing=True))


INJECTION = (
    "inject_text_on_first_prompt",
    "inject_text_on_later_prompts",
    "prompt_injection_schedule",
)


@pytest.mark.parametrize("engine", ["qwen3.8-27b", "qwen3.8-flash-next"])
def test_worker_builds_the_engine_preset_the_spec_names(monkeypatch, tmp_path, engine):
    """The agent serves the prompt injection the engine's preset resolves,
    all three fields, and every run reports it."""
    pytest.importorskip("vllm")
    import json
    from duck import presets
    from harness_support.agent_settings import AGENT_SETTINGS_FILE
    from harness_support.arc3.worker import build_benchmark

    monkeypatch.setattr("taaf.game_api.GameAPI", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("taaf.benchmark.Benchmark", lambda **kwargs: SimpleNamespace(**kwargs))
    spec = worker_spec(tmp_path)
    spec.update(tokenizer_id=str(tmp_path / "checkpoint"), engine=engine)
    agent = build_benchmark(spec).solver.agent
    assert agent.llm_config.backend_factory.engine.tokenizer_path == str(tmp_path / "checkpoint")
    preset = duck.DuckAgent()
    presets.configure(
        preset,
        engine=engine,
        environment="slurm-b200",
        backend_factory=agent.llm_config.backend_factory,
    )
    expected = {name: getattr(preset, name) for name in INJECTION}
    assert {name: getattr(agent, name) for name in INJECTION} == expected
    recorded = json.loads((tmp_path / AGENT_SETTINGS_FILE).read_text())
    assert {name: recorded[name] for name in INJECTION} == expected


@pytest.mark.parametrize("engine", ["qwen3.8-27b", "qwen3.8-flash-next"])
def test_worker_renders_with_the_engines_own_backend_kwargs(monkeypatch, tmp_path, engine):
    """The engine's parsers and template options are ullm's for that engine,
    plus the training flags, never another engine's."""
    pytest.importorskip("vllm")
    from ullm.backends import deployment
    from harness_support.arc3.worker import build_benchmark

    monkeypatch.setattr("taaf.game_api.GameAPI", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("taaf.benchmark.Benchmark", lambda **kwargs: SimpleNamespace(**kwargs))
    spec = worker_spec(tmp_path)
    spec.update(engine=engine)
    agent = build_benchmark(spec).solver.agent
    assert agent.llm_config.backend_factory.backend_kwargs == {
        **deployment.backend_kwargs(engine),
        "accept_malformed_output": True,
        "require_explicit_sampling": True,
        "verify_engine_output": True,
        "token_count_evidence": True,
        "replay_on_link_loss": False,
    }


@pytest.mark.parametrize("engine", ["qwen3.8-27b", "qwen3.8-flash-next"])
def test_worker_keeps_and_reports_the_presets_per_level_budget(monkeypatch, tmp_path, engine):
    """Training never sets the Duck's per-level token budget: the engine's
    preset decides, and every run reports the value its agent played under."""
    pytest.importorskip("vllm")
    import json
    from harness_support.agent_settings import AGENT_SETTINGS_FILE
    from harness_support.arc3.worker import build_benchmark, enrich_results

    monkeypatch.setattr("taaf.game_api.GameAPI", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("taaf.benchmark.Benchmark", lambda **kwargs: SimpleNamespace(**kwargs))
    spec = worker_spec(tmp_path)
    spec.update(engine=engine)
    agent = build_benchmark(spec).solver.agent
    preset = duck.DuckAgent.build(
        engine=engine,
        environment="slurm-b200",
        backend_factory=agent.llm_config.backend_factory,
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
    )
    assert agent.max_tokens_per_level == preset.max_tokens_per_level
    expected = (
        None if preset.max_tokens_per_level is None else list(preset.max_tokens_per_level)
    )
    recorded = json.loads((tmp_path / AGENT_SETTINGS_FILE).read_text())
    assert recorded["max_tokens_per_level"] == expected
    runs = [
        dict(
            run_id="run", game_id="fixture", state="gave_up", levels_completed=0,
            number_of_levels=1, actions_per_level=[], base_actions_per_level=[1],
            action_wallclock_seconds=[], final_wallclock_seconds=1.0, steps=[],
        )
    ]
    enrich_results({**spec, "max_runtime_s_per_game": 900}, runs, [])
    assert runs[0]["agent_settings"]["max_tokens_per_level"] == expected
