"""Retained draws preserve unsent history and its original failure.

Forked from arc2-platypus `packages/duck/tests/test_retained_draws.py`, bar what
differs underneath it: ARC3's duck discards an unsent turn after a failure only
when it summarizes, so the unsent-failure test drives the summarize arm and
fails it where the request is prepared. Not carried: ARC2's reply-reserve test
for a supplied engine, which checks the prune budget rather than retention,
over a factory argument (`derive_max_pixels`) ARC3 does not have.
"""

from __future__ import annotations

import random

import pytest
import ullm.backend
import ullm.backends.engine_proxy
import ullm.backends.stub
import ullm.conv

from duck import DuckAgent
from platypus.contract import Action, GameInfo, Move, Observation
from platypus.llm import LLMConfig, LLMMoveContext


def _agent() -> DuckAgent:
    return DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        context_window_tokens=32768,
        compact_to_tokens=16384,
        max_tokens_per_game=None,
        preserve_failed_draws=True,
        malformed_retries=0,
    )


def _factory() -> ullm.backends.engine_proxy.EngineProxyFactory:
    return ullm.backends.engine_proxy.EngineProxyFactory(
        engine=ullm.backends.engine_proxy.EngineProxy(base_url="http://127.0.0.1:1", model="policy")
    )


def _context(backend: ullm.backends.stub.StubBackend) -> LLMMoveContext:
    import arcengine
    import numpy as np
    import taaf.game

    raw = arcengine.FrameDataRaw(
        game_id="unsent-failure",
        state=arcengine.GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=[1, 2],
    )
    raw.frame = [np.zeros((8, 8), dtype=np.int8)]
    observation = Observation.from_state(
        taaf.game.GameState(raw=raw), outcome="playing", level=0, index=1, caused_by=Action("ACTION1")
    )
    return LLMMoveContext(
        history=[Move(action=Action("ACTION1"), observations=(observation,))],
        game=GameInfo(game_id="unsent-failure", hint=None, action_space=("ACTION1", "ACTION2"), number_of_levels=1),
        rng=random.Random(7),
        shared={},
        backend=backend,
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


@pytest.mark.parametrize("training", [False, True])
async def test_unsent_failure_preserves_original_exception_and_training_history(
    monkeypatch: pytest.MonkeyPatch, training: bool
) -> None:
    agent = DuckAgent(
        reply_cap_tokens=2048,
        compaction_strategy="summarize",
        file_tools="none",
        multimodal_context="",
        max_tokens_per_game=None,
        preserve_failed_draws=training,
        malformed_retries=0,
        llm_config=LLMConfig(
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)}
        ),
    )
    prior = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="prior task"))
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="next request"))
    agent.messages.append(prior)
    failure = RuntimeError("native engine connection failed")
    rewrites: list[DuckAgent] = []
    mark_history_rewritten = DuckAgent._mark_history_rewritten

    async def fail_before_generation(self: DuckAgent, *args: object, **kwargs: object) -> tuple[int, int]:
        raise failure

    def record_rewrite(self: DuckAgent) -> None:
        rewrites.append(self)
        if self.preserve_failed_draws:
            raise RuntimeError("history may change only during a verified context compaction")
        mark_history_rewritten(self)

    monkeypatch.setattr(DuckAgent, "_prepare_summarized_request", fail_before_generation)
    monkeypatch.setattr(DuckAgent, "_mark_history_rewritten", record_rewrite)
    ctx = _context(ullm.backends.stub.StubBackend.from_script("unused"))
    agent._turn_prompt = prompt  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError) as caught:
        await agent._run_turn_block(ctx, prompt, remaining_calls=None)  # pyright: ignore[reportPrivateUsage]
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
    pytest.importorskip("duck.presets", reason="needs the vllm extra (linux)")

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
        )

    if retries:
        with pytest.raises(ValueError, match="preserve_failed_draws=True requires malformed_retries=0"):
            build().prepare_for_backend()
    else:
        build().prepare_for_backend()
