"""Real ARC-3 Duck summary requests satisfy observer and gateway contracts."""

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

duck = pytest.importorskip("duck.agent")
if not hasattr(duck.DuckAgent, "_uses_context_summary") or not hasattr(
    duck.DuckAgent, "multimodal_context"
):
    pytest.skip(
        "requires the ARC-3 summarize Duck environment", allow_module_level=True
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "arc3_summary_fixtures",
    ROOT / "platypus/packages/duck/tests/test_context_summarization.py",
)
fixtures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixtures
spec.loader.exec_module(fixtures)

from serving_contract import CompactionTokenCount, TrainingContextObserver  # noqa: E402
from training.context_contract import resolve_context_contract  # noqa: E402
from training.gateway_lineage import LineageLedger  # noqa: E402
from training.rollout_gateway import SamplingPolicy  # noqa: E402


def policy():
    return SamplingPolicy(
        "model",
        1.0,
        1.0,
        -1,
        context=resolve_context_contract(32768, 4096, 28672, 0, 4096, 49152, (248058,)),
    )


def draw(request, size):
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


@pytest.mark.parametrize("tail,remaining", [(20000, None), (30000, 1001)])
def test_summary_and_retained_tail_pass_gateway_admission(monkeypatch, tail, remaining):
    async def scenario():
        original = fixtures._CountingScript._token_count

        async def exact_count(self, params, conversation):
            count = await original(self, params, conversation)
            return CompactionTokenCount(count, list(range(count)))

        monkeypatch.setattr(fixtures._CountingScript, "_token_count", exact_count)
        agent = fixtures._summarizing_agent(
            window=32768,
            profile_max_tokens=4096,
            max_tokens_per_game=remaining,
            compaction_forbidden_token_ids=(248058,),
        )
        agent.engine_ceiling_tokens = 49152
        agent.preserve_failed_draws = True
        agent.malformed_retries = 0
        agent.safety_margin_tokens = 0
        agent.context_observer = TrainingContextObserver(reply_cap_tokens=4096)
        agent.context_observer.request_prepared(
            SimpleNamespace(request_context={}),
            epoch=0,
            prompt_tokens=CompactionTokenCount(2000, list(range(2000))),
            target=28672,
        )
        fixtures._history(agent)
        backend = fixtures._CountingScript(
            responses=[
                fixtures._response(fixtures.SUMMARY),
                fixtures._response("DONE"),
            ],
            ordinary_count=32700,
            summary_count=33000,
            compacted_count=5000,
            compacted_with_tail_count=tail,
        )
        await fixtures._run_block(agent, backend)
        assert [r.summary for r in backend.seen_requests] == [True, False]
        assert fixtures.LATEST in "\n".join(backend.seen_requests[-1].element_texts)
        gateway = policy()
        ledger = LineageLedger()
        first = list(range(2000))
        ledger.admit(
            "run",
            "first",
            first,
            dict(version=1, epoch=0, kind="start", compaction_prompt_token_ids=None),
            gateway.context.lineage_policy(),
            native_token_prompt=True,
        )
        ledger.complete("run", "first", first, [2000])
        for index, (request, size) in enumerate(
            zip(backend.seen_requests, [33000, tail])
        ):
            body = draw(request, size)
            gateway.validate(body)
            ledger.admit(
                "run",
                str(index),
                body["prompt"],
                request.request_context["arc3_lineage"],
                gateway.context.lineage_policy(
                    max_completion_tokens=body["max_tokens"]
                ),
                native_token_prompt=True,
                summary_request=gateway.is_summary_request(body),
            )
            ledger.complete("run", str(index), body["prompt"], [size])
        assert ledger.runs["run"].epoch == 2
        assert agent.context_summaries == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["empty", "tool-markup", "tool-call", "malformed"])
def test_failed_summary_keeps_one_draw_without_reset_or_retry(monkeypatch, failure):
    async def scenario():
        original = fixtures._CountingScript._token_count

        async def exact_count(self, params, conversation):
            count = await original(self, params, conversation)
            return CompactionTokenCount(count, list(range(count)))

        monkeypatch.setattr(fixtures._CountingScript, "_token_count", exact_count)
        agent = fixtures._summarizing_agent(
            window=32768,
            profile_max_tokens=4096,
            compaction_forbidden_token_ids=(248058,),
        )
        agent.engine_ceiling_tokens = 49152
        agent.safety_margin_tokens = 0
        agent.preserve_failed_draws = True
        agent.malformed_retries = 0
        agent.context_observer = TrainingContextObserver(reply_cap_tokens=4096)
        agent.context_observer.request_prepared(
            SimpleNamespace(request_context={}),
            epoch=0,
            prompt_tokens=CompactionTokenCount(2000, list(range(2000))),
            target=28672,
        )
        fixtures._history(agent)
        responses = {
            "empty": fixtures._response(" ", completion_tokens=7),
            "tool-markup": fixtures._response(
                "<tool_call>{}</tool_call>", completion_tokens=7
            ),
            "tool-call": fixtures._tool_response(completion_tokens=7),
            "malformed": fixtures._response(fixtures.SUMMARY, completion_tokens=7),
        }
        backend_type = (
            fixtures._MalformedSummaryOnceScript
            if failure == "malformed"
            else fixtures._CountingScript
        )
        backend = backend_type(
            responses=[responses[failure]],
            ordinary_count=32700,
            summary_count=33000,
        )
        with pytest.raises(fixtures.GiveUpError, match="retained draw, no retry"):
            await fixtures._run_block(agent, backend)
        (request,) = backend.seen_requests
        assert request.summary
        assert len(request.turn.elements) == 1
        assert isinstance(request.turn.elements[0], fixtures.ullm.conv.LLMResponse)
        assert agent.history_epoch == 0 and agent.context_summaries == 0
        assert agent.context_observer.reset_proof is None
        gateway = policy()
        ledger = LineageLedger()
        first = list(range(2000))
        ledger.admit(
            "run",
            "first",
            first,
            dict(version=1, epoch=0, kind="start", compaction_prompt_token_ids=None),
            gateway.context.lineage_policy(),
            native_token_prompt=True,
        )
        ledger.complete("run", "first", first, [2000])
        body = draw(request, 33000)
        gateway.validate(body)
        ledger.admit(
            "run",
            "summary",
            body["prompt"],
            request.request_context["arc3_lineage"],
            gateway.context.lineage_policy(max_completion_tokens=body["max_tokens"]),
            native_token_prompt=True,
            summary_request=gateway.is_summary_request(body),
        )
        output = list(range(33000, 33007))
        ledger.complete("run", "summary", body["prompt"], output)
        assert ledger.runs["run"].last_output == output
        assert ledger.runs["run"].nonces == ["first", "summary"]
        assert ledger.runs["run"].pending_nonce is None

    asyncio.run(scenario())


def test_final_short_reply_stays_an_ordinary_request():
    async def scenario():
        agent = fixtures._summarizing_agent(
            window=32768,
            profile_max_tokens=4096,
            max_tokens_per_game=1000,
            compaction_forbidden_token_ids=(248058,),
        )
        agent.safety_margin_tokens = 0
        fixtures._history(agent)
        backend = fixtures._CountingScript(
            responses=[fixtures._response("DONE")],
            ordinary_count=30000,
        )
        await fixtures._run_block(agent, backend)
        (request,) = backend.seen_requests
        body = draw(request, 30000)
        policy().validate(body)
        assert not policy().is_summary_request(body)
        assert request.max_tokens == 1000

    asyncio.run(scenario())
