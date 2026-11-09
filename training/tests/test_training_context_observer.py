"""Ordinary Duck preserves exact training history through public observations."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

pytest.importorskip("duck.agent")
import ullm.backend
import ullm.conv
import ullm.continuation
from duck.state import MemoryEntry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serving_contract import CompactionTokenCount
from training_agent_fixture import context_agent


def _user(text):
    return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))


def _fixture():
    agent = context_agent(
        compaction_strategy="prune",
        journal=True,
        multimodal_context="",
        token_native_context=True,
        context_window_tokens=4096,
        compact_to_tokens=1500,
        safety_margin_tokens=0,
    )
    agent.messages = [
        _user("play"),
        ullm.conv.ResponseTurn(elements=[], status="intermediate"),
    ]
    conversation = ullm.conv.Conversation(
        elements=agent.messages,
        status="intermediate",
        system_prompt=ullm.conv.SystemPrompt(
            content=ullm.conv.StringContent(text="system")
        ),
    )

    async def count(params, conversation):
        size = 300
        for element in conversation.elements:
            if isinstance(element, ullm.conv.UserPrompt):
                size += len(element.content.to_string())
            elif isinstance(element, ullm.conv.ResponseTurn):
                for child in element.elements:
                    if isinstance(child, ullm.conv.LLMResponse):
                        size += len(child.final_content.to_string()) + len(
                            child.reasoning_content.to_string()
                        )
                    elif isinstance(child, ullm.conv.ToolResult):
                        size += len(child.content.to_string())
        if size > agent._context_budget_tokens():
            return CompactionTokenCount(size, list(range(size)))
        return size

    agent._live_backend = SimpleNamespace(token_count=count)
    agent._live_conversation = conversation
    agent._live_params = ullm.backend.GenericSamplingParameters(max_tokens=512)
    return agent, conversation


async def _prepare_request(agent):
    await agent._trim_live_messages()
    await agent._observe_request_attempt(0)


def test_measurement_does_not_stamp_or_advance_generation():
    async def scenario():
        agent, conversation = _fixture()
        for _ in range(3):
            assert await agent._measured_prompt_tokens(agent.messages) == 304
        assert agent.context_observer.last_request_epoch is None
        assert "arc3_lineage" not in agent._live_params.request_context
        await agent._trim_live_messages()
        assert "arc3_lineage" not in agent._live_params.request_context
        await agent._observe_request_attempt(0)
        assert agent._live_params.request_context["arc3_lineage"] == {
            "version": 1,
            "epoch": 0,
            "kind": "start",
            "compaction_prompt_token_ids": None,
        }
        before = list(agent.messages)
        await _prepare_request(agent)
        assert agent._live_params.request_context["arc3_lineage"]["kind"] == "continue"
        assert agent.messages == before
        assert conversation.history_epoch == 0

    asyncio.run(scenario())


def test_real_eviction_compacts_once_with_proof_and_keeps_journal_and_outputs():
    async def scenario():
        agent, conversation = _fixture()
        await _prepare_request(agent)
        agent.memory_entries = [
            MemoryEntry(
                level=0,
                step=1,
                index=1,
                response=1,
                last_action=None,
                text="Keep the useful observation.",
            )
        ]
        agent.memory_entry_count = 1
        original_notes = list(agent.memory_entries)
        response = ullm.conv.LLMResponse(
            reasoning_content=ullm.conv.StringContent(text="unfinished" * 550)
        )
        response.stop_reason = "max_tokens"
        agent.messages[-1].elements.extend(
            [
                response,
                ullm.conv.ToolResult(
                    call_id="bad-output",
                    tool_name="python",
                    content=ullm.conv.StringContent(text="invalid model tool format"),
                    is_error=True,
                ),
            ]
        )
        # Post-turn retention defers the complete, bounded trim to pre-round.
        before = list(agent.messages[-1].elements)
        await agent._persist_history()
        assert agent.messages[-1].elements == before
        assert conversation.history_epoch == 0
        recorded = []
        await agent._trim_live_messages(
            record_pruned_round=lambda removed: recorded.extend(removed)
        )
        await agent._observe_request_attempt(0)
        metadata = agent._live_params.request_context["arc3_lineage"]
        assert metadata["kind"] == "reset" and metadata["epoch"] == 1
        assert (
            len(metadata["compaction_prompt_token_ids"])
            > agent._context_budget_tokens()
        )
        assert (
            await agent._measured_prompt_tokens(agent.messages)
            <= agent.compact_to_tokens
        )
        assert response in recorded
        assert response.reasoning_content.to_string() == "unfinished" * 550
        assert response.stop_reason == "max_tokens"
        assert agent.memory_entries == original_notes and agent.memory_entry_count == 1
        assert (
            "Keep the useful observation."
            in conversation.system_prompt.content.to_string()
        )
        await _prepare_request(agent)
        assert agent._live_params.request_context["arc3_lineage"] == {
            "version": 1,
            "epoch": 1,
            "kind": "continue",
            "compaction_prompt_token_ids": None,
        }

    asyncio.run(scenario())


def test_undeclared_or_below_threshold_history_rewrite_is_fatal():
    async def scenario():
        agent, conversation = _fixture()
        await _prepare_request(agent)
        with pytest.raises(RuntimeError, match="only during a verified"):
            agent._mark_history_rewritten()
        conversation.history_epoch = 1
        with pytest.raises(RuntimeError, match="undeclared history rewrite"):
            await _prepare_request(agent)
        agent, _ = _fixture()
        await _prepare_request(agent)
        agent.messages.insert(0, ullm.conv.ResponseTurn(elements=[]))
        with pytest.raises(RuntimeError, match="only during a verified"):
            await _prepare_request(agent)

    asyncio.run(scenario())


def test_count_history_drift_is_not_silently_downgraded():
    async def scenario():
        agent, _ = _fixture()

        async def drift(*args):
            raise ullm.continuation.HistoryDriftError("served prefix changed")

        agent._live_backend = SimpleNamespace(token_count=drift)
        with pytest.raises(
            ullm.continuation.HistoryDriftError, match="served prefix changed"
        ):
            await _prepare_request(agent)

    asyncio.run(scenario())


def test_shared_recipe_keeps_training_values_out_of_the_harness():
    from duck.agent import DuckAgent
    from training_backend import configure_training_agent, training_sampling

    agent = DuckAgent(
        reply_cap_tokens=4096, derive_max_model_len=False, derive_max_num_seqs=False
    )
    agent.run_config.min_request_gap_s = 5.0
    factory = object()
    sampling = training_sampling({"temperature": 0.9, "top_p": 0.8, "top_k": 12})
    original = sampling.max_tokens
    original_config = agent.llm_config
    original_run_config = agent.run_config
    auxiliary = ullm.backend.GenericSamplingParameters(temperature=0.2)
    original_config.sampling_profiles["auxiliary"] = auxiliary
    injection = {
        name: getattr(agent, name)
        for name in (
            "inject_text_on_first_prompt",
            "inject_text_on_later_prompts",
            "prompt_injection_schedule",
        )
    }
    configured = configure_training_agent(
        agent,
        backend_factory=factory,
        sampling=sampling,
        max_concurrent=7,
    )
    assert type(configured) is DuckAgent
    assert configured.llm_config.backend_factory is factory
    assert original_config.backend_factory is not factory
    assert original_run_config.min_request_gap_s == 5.0
    assert configured.llm_config.sampling_profiles["auxiliary"] is auxiliary
    assert configured.run_config.min_request_gap_s == 0.0
    assert configured.run_config.max_concurrent == 7
    profile = configured.llm_config.sampling_profiles[configured.profile]
    assert (profile.temperature, profile.top_p, profile.max_tokens) == (
        0.9,
        0.8,
        original,
    )
    assert profile.send_back_reasoning == "all" and sampling.max_tokens == original
    assert not configured.derive_max_model_len
    # The harness's prompt injection is served as it resolved it.
    assert {name: getattr(configured, name) for name in injection} == injection
    assert configured.context_observer is not None


def test_compaction_metadata_keeps_exact_multimodal_evidence():
    from serving_contract import TrainingContextObserver

    observer = TrainingContextObserver()
    params = ullm.backend.GenericSamplingParameters()
    observer.prepare_request(params)
    observer.request_prepared(params, epoch=0, prompt_tokens=2, target=4)
    observer.prepare_request(params)
    media = {"images": [{"sha256": "image-identity"}]}
    proof = CompactionTokenCount(5, [1, 2, 3, 4, 5], media=media)
    observer.prompt_measured(proof, epoch=0, trigger=4, compacting=True)
    observer.history_rewritten(epoch=0)
    observer.request_prepared(params, epoch=1, prompt_tokens=3, target=4)
    assert params.request_context["arc3_lineage"]["compaction_media"] == media
    assert params.request_context["arc3_lineage"]["compaction_prompt_token_ids"] == [
        1,
        2,
        3,
        4,
        5,
    ]


@pytest.mark.parametrize("served", [False, True])
def test_compaction_hook_sequence_with_backend_token_count(served):
    from context_observer_sequences import compaction_stamps, expected_stamps
    from serving_contract import TrainingContextObserver

    assert compaction_stamps(
        TrainingContextObserver(), CompactionTokenCount, served=served
    ) == expected_stamps(served=served)
