"""Reply limits and context reserves share the agent's per-call authority."""

import random
from types import SimpleNamespace
from typing import cast

import pytest
from ullm.backend import GenericSamplingParameters

from platypus import LLMConfig, LLMMoveContext
from seal.agent import SealAgent


def test_call_caps_follow_agent_changes_without_mutating_the_profile() -> None:
    profile = GenericSamplingParameters(max_tokens=None, temperature=0.6)
    agent = SealAgent(llm_config=LLMConfig(sampling_profiles={"play": profile}), compaction_strategy="prune")
    ctx = cast(LLMMoveContext, SimpleNamespace(rng=random.Random(3)))
    first = agent._sampling(ctx)
    assert first.max_tokens == 32768
    for cap in (4096, None, 777):
        agent.reply_cap_tokens = cap
        agent.prepare_for_backend()
        current = agent._sampling(ctx)
        assert current.max_tokens == cap
        assert agent._reply_reserve_tokens() == (512 if cap is None else cap)
        assert profile.max_tokens is None and profile.seed is None
    assert first.max_tokens == 32768


@pytest.mark.parametrize("cap", [0, -1, True, "profile"])
def test_invalid_caps_are_refused(cap: object) -> None:
    with pytest.raises(ValueError, match="reply_cap_tokens"):
        SealAgent(reply_cap_tokens=cap)  # type: ignore[arg-type]


@pytest.mark.parametrize("cap", [123, 32768])
@pytest.mark.parametrize("stage", ["construction", "preparation", "request"])
def test_profile_caps_are_rejected_even_after_preparation(cap: int, stage: str) -> None:
    profile = GenericSamplingParameters(max_tokens=None)
    config = LLMConfig(sampling_profiles={"play": profile})
    if stage == "construction":
        profile.max_tokens = cap
        with pytest.raises(ValueError, match="reply_cap_tokens"):
            SealAgent(llm_config=config)
        return
    agent = SealAgent(llm_config=config)
    agent.prepare_for_backend()
    profile.max_tokens = cap
    with pytest.raises(ValueError, match="reply_cap_tokens"):
        if stage == "preparation":
            agent.prepare_for_backend()
        else:
            agent._sampling(cast(LLMMoveContext, SimpleNamespace(rng=random.Random(3))))
    assert profile.max_tokens == cap
