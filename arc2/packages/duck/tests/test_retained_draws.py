"""Retained draws preserve unsent history and its original failure."""

import random
from dataclasses import dataclass

import pytest
import ullm.backend
import ullm.backends.engine_proxy
import ullm.backends.stub
import ullm.conv
import ullm.manager
from duck import presets
from duck.agent import DuckAgent

import taaf.game
from platypus.context import Produced, SubmitAnswer
from platypus.llm import LLMConfig, build_llm_context


def _collect_into(submissions: list[taaf.game.Submission]) -> SubmitAnswer:
    """A submit channel that keeps every answer, in order."""

    def submit(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        del produced
        submissions.append(submission)

    return submit


@dataclass
class VerifiedBackend(ullm.backends.stub.StubBackend):
    classify_tool_calls: bool = False
    require_explicit_sampling: bool = True
    verify_engine_output: bool = True
    token_count_evidence: bool = True
    accept_malformed_output: bool = True
    replay_on_link_loss: bool = False


def _agent() -> DuckAgent:
    return DuckAgent(
        reply_cap_tokens=None,
        llm_config=LLMConfig(sampling_profiles=presets.analyzer_sampling_profiles()),
        context_window_tokens=32768,
        compact_to_tokens=16384,
        max_tokens_per_game=None,
        preserve_failed_draws=True,
        perturb_retry_prompt=False,
        malformed_retries=0,
        malformed_nudges=0,
        classify_tool_calls=True,
    )


def _factory() -> ullm.backends.engine_proxy.EngineProxyFactory:
    return ullm.backends.engine_proxy.EngineProxyFactory(
        engine=ullm.backends.engine_proxy.EngineProxy(base_url="http://127.0.0.1:1", model="policy")
    )


@pytest.mark.parametrize("retries", [None, 1, 5])
def test_retention_rejects_retries_before_backend_start(retries: int | None) -> None:
    agent = (
        DuckAgent(preserve_failed_draws=True)
        if retries is None
        else DuckAgent(preserve_failed_draws=True, malformed_retries=retries)
    )
    with pytest.raises(ValueError, match="preserve_failed_draws=True requires malformed_retries=0"):
        agent.prepare_for_backend()


def test_retention_revalidates_retries_assigned_after_construction() -> None:
    agent = _agent()
    agent.prepare_for_backend()
    agent.malformed_retries = 1
    with pytest.raises(ValueError, match="preserve_failed_draws=True requires malformed_retries=0"):
        agent.prepare_for_backend()


def test_supplied_engine_build_validates_the_requested_reply_reserve() -> None:
    agent = DuckAgent.build(
        backend_factory=_factory(),
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        compaction_strategy="prune",
        context_window_tokens=32768,
        compact_to_tokens=16384,
        reply_cap_tokens=4096,
        safety_margin_tokens=512,
    )
    assert agent._context_budget_tokens() == 28160
    agent.reply_cap_tokens = 20480
    with pytest.raises(ValueError, match="compact_to_tokens"):
        agent.prepare_for_backend()


@pytest.mark.parametrize("training", [False, True])
async def test_unsent_failure_preserves_original_exception_and_training_history(
    monkeypatch: pytest.MonkeyPatch, training: bool
) -> None:
    agent = _agent()
    agent.preserve_failed_draws = training
    prior = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="prior task"))
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="next request"))
    agent.messages.append(prior)
    failure = RuntimeError("native engine connection failed")
    rewrites: list[DuckAgent] = []
    mark_history_rewritten = DuckAgent._mark_history_rewritten

    async def fail_before_generation(self: DuckAgent) -> None:
        raise failure

    def record_rewrite(self: DuckAgent) -> None:
        rewrites.append(self)
        if self.preserve_failed_draws:
            raise RuntimeError("history may change only during a verified context compaction")
        mark_history_rewritten(self)

    monkeypatch.setattr(DuckAgent, "_trim_live_messages", fail_before_generation)
    monkeypatch.setattr(DuckAgent, "_mark_history_rewritten", record_rewrite)
    submissions: list[taaf.game.Submission] = []
    ctx, _meter = build_llm_context(
        game_id="unsent-failure",
        submissions=submissions,
        rng=random.Random(7),
        backend=VerifiedBackend.from_script("unused"),
        submit=_collect_into(submissions),
    )
    with pytest.raises(RuntimeError) as caught:
        await agent._run_turn_block(ctx, [prompt], remaining_calls=None)
    assert caught.value is failure
    assert agent.messages[0] is prior
    if training:
        assert rewrites == []
        assert agent.messages[1] is prompt
        assert len(agent.messages) == 3
        assert isinstance(agent.messages[-1], ullm.conv.ResponseTurn)
        assert agent.messages[-1].elements == []
        assert agent.history_epoch == 0
    else:
        assert rewrites == [agent]
        assert agent.messages == [prior]
        assert agent.history_epoch == 1


@pytest.mark.parametrize("retries", [0, 1])
def test_retained_summary_build_and_backend_preparation(retries: int) -> None:
    def build() -> DuckAgent:
        return DuckAgent.build(
            backend_factory=_factory(),
            derive_max_model_len=False,
            derive_max_num_seqs=False,
            derive_max_pixels=False,
            compaction_strategy="summarize",
            engine_ceiling_tokens=65536,
            context_window_tokens=32768,
            reply_cap_tokens=4096,
            preserve_failed_draws=True,
            malformed_retries=retries,
            malformed_nudges=0,
        )

    if retries:
        with pytest.raises(ValueError, match="preserve_failed_draws=True requires malformed_retries=0"):
            build().prepare_for_backend()
    else:
        build().prepare_for_backend()
