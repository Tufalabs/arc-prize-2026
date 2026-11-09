"""Pinned Duck compaction and budget decisions pass trainer admission on CPU."""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

duck = pytest.importorskip("duck.agent")
if not hasattr(duck.DuckAgent, "_uses_context_summary") or not hasattr(
    duck.DuckAgent, "images"
):
    pytest.skip("requires the ARC-2 Duck environment", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "arc2-platypus/packages/duck/tests"))
from test_context_summarization import (  # noqa: E402 - requires the ARC-2 environment
    LATEST,
    SUMMARY,
    _CountingScript,
    _history,
    _response,
    _run_block,
    _summarizing_agent,
)
from serving_contract import CompactionTokenCount, TrainingContextObserver  # noqa: E402
from training.context_contract import resolve_context_contract  # noqa: E402
from training.gateway_lineage import LineageError, LineageLedger  # noqa: E402
from training.rollout_gateway import SamplingPolicy  # noqa: E402


def _policy():
    return SamplingPolicy(
        "model",
        1.0,
        1.0,
        -1,
        context=resolve_context_contract(32768, 4096, 28672, 0, 4096, 49152, (248058,)),
    )


def _draw(request, size):
    return dict(
        model="model",
        user="run#draw",
        prompt=list(range(size)),
        max_tokens=request.max_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        min_p=0.0,
        min_tokens=0,
        **request.extra_params,
    )


@pytest.mark.parametrize("reset_size,remaining", [(20000, None), (30000, 1001)])
def test_duck_retains_a_long_tail_and_the_gateway_accepts_its_reset(
    monkeypatch, reset_size, remaining
):
    async def scenario():
        original = _CountingScript._token_count

        async def exact_count(self, params, conversation):
            count = await original(self, params, conversation)
            return CompactionTokenCount(count, list(range(count)))

        monkeypatch.setattr(_CountingScript, "_token_count", exact_count)
        agent = _summarizing_agent(
            window=32768,
            profile_max_tokens=4096,
            max_tokens_per_game=remaining,
            compaction_forbidden_token_ids=(248058,),
        )
        agent.compact_to_tokens = 28672
        agent.safety_margin_tokens = 0
        agent.context_observer = TrainingContextObserver(reply_cap_tokens=4096)
        agent.context_observer.request_prepared(
            SimpleNamespace(request_context={}),
            epoch=0,
            prompt_tokens=CompactionTokenCount(2000, list(range(2000))),
            target=28672,
        )
        _history(agent)
        backend = _CountingScript(
            responses=[_response(SUMMARY), _response("DONE")],
            ordinary_count=32700,
            summary_count=33000,
            compacted_count=5000,
            compacted_with_tail_count=reset_size,
        )
        await _run_block(agent, backend)
        assert [r.summary for r in backend.seen_requests] == [True, False]
        assert any(LATEST in text for text in backend.seen_requests[-1].element_texts)
        assert backend.counted[-1] == reset_size
        policy = _policy()
        ledger = LineageLedger()
        first = list(range(2000))
        ledger.admit(
            "run",
            "first",
            first,
            dict(version=1, epoch=0, kind="start", compaction_prompt_token_ids=None),
            policy.context.lineage_policy(),
            native_token_prompt=True,
        )
        ledger.complete("run", "first", first, [2000])
        for index, (request, size) in enumerate(
            zip(backend.seen_requests, [33000, reset_size])
        ):
            draw = _draw(request, size)
            policy.validate(draw)
            lineage = policy.context.lineage_policy(
                max_completion_tokens=draw["max_tokens"]
            )
            ledger.admit(
                "run",
                str(index),
                draw["prompt"],
                request.request_context["arc3_lineage"],
                lineage,
                native_token_prompt=True,
                summary_request=policy.is_summary_request(draw),
            )
            ledger.complete("run", str(index), draw["prompt"], [size])
            if index == 0:
                with pytest.raises(LineageError, match="continuation after a summary"):
                    ledger.admit(
                        "run",
                        "retry",
                        list(range(size + 2)),
                        dict(
                            version=1,
                            epoch=0,
                            kind="continue",
                            compaction_prompt_token_ids=None,
                        ),
                        lineage,
                        native_token_prompt=True,
                        summary_request=True,
                    )
        assert ledger.runs["run"].epoch == 2

    asyncio.run(scenario())


def test_duck_final_budget_reply_is_admitted_without_summary_bias():
    async def scenario():
        agent = _summarizing_agent(
            window=32768,
            profile_max_tokens=4096,
            max_tokens_per_game=1000,
            compaction_forbidden_token_ids=(248058,),
        )
        agent.safety_margin_tokens = 0
        _history(agent)
        backend = _CountingScript(responses=[_response("DONE")], ordinary_count=30000)
        await _run_block(agent, backend)
        (request,) = backend.seen_requests
        assert not request.summary and request.max_tokens == 1000
        draw = _draw(request, 30000)
        assert "logit_bias" not in draw
        _policy().validate(draw)
        assert not _policy().is_summary_request(draw)

    asyncio.run(scenario())
