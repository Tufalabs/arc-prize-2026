"""Worker overrides preserve run settings composed by the deployment column."""

from dataclasses import dataclass, field
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "environment,summarizing", [("arc2", True), ("arc3", False), ("arc3", True)]
)
@pytest.mark.parametrize("context_budget", [28160, 28000])
def test_worker_preserves_column_run_settings(
    monkeypatch, environment, summarizing, context_budget
):
    @dataclass
    class RunConfig:
        start_gap_s: float = 0.5
        process_mode: str = "inprocess"
        min_request_gap_s: float = 5.0
        max_concurrent: int = 32
        max_runtime_s_per_game: float | None = None
        on_agent_error: str = "continue"
        rng_seed: int | None = None
        rng_seed_per_game: dict = field(default_factory=dict)

    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    inherited = RunConfig(start_gap_s=7.5)
    agent = SimpleNamespace(
        run_config=inherited,
        prepare_for_backend=lambda: None,
        context_budget_tokens=lambda: context_budget,
        compact_to_tokens=16384,
        context_window_tokens=32768,
        _uses_context_summary=lambda: summarizing,
        _require_summary_fits=lambda ceiling: None,
        _summary_target_tokens=lambda: 2048,
        compact_target_tokens=lambda: context_budget if summarizing else 16384,
        engine_ceiling_tokens=49152 if summarizing else 32768,
        _derived_engine_window=lambda: 49152 if summarizing else 32768,
        _reply_reserve_tokens=lambda: 4096,
        summary_thinking=False,
        compaction_forbidden_token_ids=(248058,) if summarizing else (),
        context_observer=SimpleNamespace(reply_cap_tokens=4096),
        malformed_retries=0,
        inject_text_on_first_prompt="We need solve",
        inject_text_on_later_prompts="We need",
        prompt_injection_schedule="game",
        max_tokens_per_level=(90_000,),
    )

    def build(**kwargs):
        if environment == "arc2":
            assert kwargs["engine_ceiling_tokens"] == 49152
            assert kwargs["summary_thinking"] is False
            assert "compaction_strategy" not in kwargs
        else:
            assert kwargs["compaction_strategy"] == (
                "summarize" if summarizing else "prune"
            )
            assert kwargs["journal"] is not summarizing
            assert kwargs.get("preserve_failed_draws", False) is summarizing
            assert kwargs["engine_ceiling_tokens"] == (49152 if summarizing else 32768)
        agent.llm_config = SimpleNamespace(backend_factory=kwargs["backend_factory"])
        return agent

    class Configured(Exception):
        pass

    def configured_agent(configured, **kwargs):
        assert configured is agent

    def validated_agent(validated):
        assert validated is agent
        raise Configured

    stub("platypus", RunConfig=RunConfig)
    stub(
        "duck.agent",
        DuckAgent=SimpleNamespace(build=build, preserve_failed_draws=False),
    )
    stub("ullm.backends.engine_proxy", EngineProxy=SimpleNamespace)
    stub(
        "serving_contract",
        ContractLLMAgentSolver=object,
        install_contract_backend=validated_agent,
    )
    stub("taaf.benchmark", Benchmark=object)
    stub("taaf.game_api", GameAPI=object)
    stub("taaf.game", ArcGame=object)
    stub("taaf.datasets", load=lambda _: [SimpleNamespace(game_id="puzzle")])
    stub(
        "harness_support.arc2.contract",
        ProcessedResponseObserver=object,
        ContractARC2LLMAgentSolver=object,
    )
    stub(
        "training_backend",
        configure_training_agent=configured_agent,
        qwen_engine_factory=lambda proxy, *, engine: SimpleNamespace(backend_kwargs={}),
        training_backend_factory=lambda source_factory: source_factory,
        training_sampling=lambda sampling: sampling,
        prompt_injection_settings=lambda agent: {},
    )
    worker = importlib.import_module(f"harness_support.{environment}.worker")
    if environment == "arc2":
        # What the Duck's own summary request carries, and a server whose
        # image ceiling fits its renders (both need the real harness).
        monkeypatch.setattr(
            worker,
            "duck_summary_request_caps",
            lambda agent: {"cap": 4096, "reasoning_cap": 12288},
        )
        monkeypatch.setattr(worker, "check_served_image_ceiling", lambda agent, spec: None)
    spec = dict(
        dataset="fixture",
        games=["puzzle", "puzzle"],
        slot_seeds=[11, 22],
        rng_seed=7,
        max_concurrent=2,
        max_runtime_s_per_game=900,
        base_url="http://unused.invalid",
        served_model="policy",
        tokenizer_id="fixture",
        model_revision="fixture",
        max_completion_tokens=4096,
        context_window_tokens=32768,
        max_summary_tokens=4096 if summarizing else 0,
        summary_forbidden_token_ids=[248058] if summarizing else [],
        max_served_sequence_tokens=49152 if summarizing else 32768,
        compact_to_tokens=28160 if summarizing else 16384,
        safety_margin_tokens=512,
        reasoning_effort="medium",
        sampling={},
        compaction_trigger_tokens=28160,
        accept_malformed_output=True,
    )
    # Exercise the public budget check before stopping at backend validation.
    if context_budget == 28160:
        with pytest.raises(Configured):
            worker.build_benchmark(spec)
    else:
        with pytest.raises(ValueError, match="compaction trigger differs"):
            worker.build_benchmark(spec)
    assert agent.run_config.start_gap_s == 7.5
    assert agent.run_config is not inherited
    assert inherited == RunConfig(start_gap_s=7.5)
    assert agent.run_config.process_mode == "subprocess"
    assert agent.run_config.min_request_gap_s == 0
    assert agent.run_config.max_concurrent == 2
    assert agent.run_config.max_runtime_s_per_game == 900
    # The deployment's policy: an agent error crashes its own game only.
    assert agent.run_config.on_agent_error == "crash_game"
    assert agent.run_config.rng_seed == 7
    assert agent.run_config.rng_seed_per_game == {"puzzle#s0": 11, "puzzle#s1": 22}
