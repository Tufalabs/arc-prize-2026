"""Both harnesses expose the same public compaction lifecycle to the trainer."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "arc2-platypus/packages/duck/src/duck/agent.py").exists(),
    reason="Requires the pinned ARC-2 submodule",
)
def test_pinned_arc2_observer_preserves_compaction_and_continuation():
    pytest.importorskip("ullm.backend")
    program = """
import asyncio
from types import SimpleNamespace
from ullm.backend import GenericSamplingParameters, TokenCount
from ullm.conv import Conversation, UserPrompt, StringContent, ResponseTurn, LLMResponse
from duck.agent import DuckAgent
from serving_contract import TrainingContextObserver

async def check():
    agent = DuckAgent(token_native_context=True, context_window_tokens=4096, reply_cap_tokens=512,
                                 require_exact_context=True, compact_before_request_only=True,
                                 context_observer=TrainingContextObserver(),
                                 compact_to_tokens=1500, safety_margin_tokens=0)
    agent.messages = [UserPrompt(content=StringContent(text='play')),
                      ResponseTurn(elements=[], status='intermediate')]
    agent._live_conversation = Conversation(elements=agent.messages, status='intermediate')
    agent._live_params = GenericSamplingParameters(max_tokens=512)
    async def count(*args):
        size = 300 + sum(len(element.to_string()) for element in agent.messages)
        return TokenCount(size, list(range(size)))
    agent._live_backend = SimpleNamespace(token_count=count)
    agent._prepare_context_observer()
    await agent._trim_live_messages()
    await agent._observe_request_attempt(0)
    assert agent._live_params.request_context['arc3_lineage']['kind'] == 'start'
    agent.messages[-1].elements.append(LLMResponse(reasoning_content=StringContent(text='thought '*800)))
    await agent._persist_history()
    assert agent._live_conversation.history_epoch == 0
    agent._prepare_context_observer()
    await agent._trim_live_messages()
    await agent._observe_request_attempt(0)
    metadata = agent._live_params.request_context['arc3_lineage']
    assert metadata['kind'] == 'reset' and metadata['epoch'] == 1
    assert len(metadata['compaction_prompt_token_ids']) > agent._context_budget_tokens()
    assert await agent._measured_prompt_tokens(agent.messages) <= 1500
    agent._prepare_context_observer()
    await agent._trim_live_messages()
    await agent._observe_request_attempt(0)
    assert agent._live_params.request_context['arc3_lineage']['kind'] == 'continue'
asyncio.run(check())
"""
    paths = [
        ROOT / "arc2-platypus/packages" / name / "src"
        for name in ("duck", "platypus", "taaf")
    ]
    paths += [ROOT / "platypus/packages/ullm/src", ROOT / "scripts", ROOT]
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
