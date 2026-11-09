"""Processed draw journals supply the malformed-call reward evidence, an empty
reply one malformed call, and count the empty replies for observation.

The observer asks ullm whether a reply's content is empty. Where ullm imports
(the harness environments) the responses carry ullm's own content; CI has no
ullm, so there a module stands in for it (the Duck's own draws run in
tests/test_arc2_empty_reply_duck.py, in the ARC-2 worker environment)."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from harness_support.arc2.results import enrich_arc2_result
from harness_support.arc2.worker import add_response_reports
from test_arc2_integration import result_payload
from training.rollout_gateway import RunResult

try:
    import ullm.backend  # noqa: F401 - the observer's predicate
    import ullm.conv as conv
except ImportError:
    conv = None


@pytest.fixture
def observer(monkeypatch, tmp_path):
    module = ModuleType("training_backend")
    module.TrainingLLMAgentSolver = object
    monkeypatch.setitem(sys.modules, "training_backend", module)
    if conv is None:
        package = ModuleType("ullm")
        backend = ModuleType("ullm.backend")
        # The doubles below carry their own emptiness (_content).
        backend._is_empty_content = lambda content: content.empty
        package.backend = backend
        monkeypatch.setitem(sys.modules, "ullm", package)
        monkeypatch.setitem(sys.modules, "ullm.backend", backend)
    spec = importlib.util.spec_from_file_location(
        "_arc2_contract_under_test",
        Path(__file__).resolve().parents[1] / "harness_support/arc2/contract.py",
    )
    contract = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, contract)
    spec.loader.exec_module(contract)
    monkeypatch.setenv(
        "ARC2_TRAINING_RESPONSE_DIR", str(tmp_path / "processed_responses")
    )
    return contract.ProcessedResponseObserver()


def _content(*, tools, text):
    """A reply's final content: ullm's own where it imports."""
    if conv is None:
        calls = [object()] if tools else []
        return SimpleNamespace(tool_calls=lambda: calls, empty=not (tools or text))
    elements = [conv.StringContent(text="<tool_call>raw text")] if text else []
    if tools:
        elements.append(conv.ToolCallContent(tool_name="python", arguments={"code": "1"}))
    if not elements:
        return conv.EmptyContent()
    return elements[0] if len(elements) == 1 else conv.ListContent(elements=elements)


def _response(reason, count, *, tools=False, text=False):
    """A processed response as the observer receives it."""
    return SimpleNamespace(
        generation_request_id="abc#first",
        malformed_call_count=count,
        truncation_cap_source=None,
        stop_reason=reason,
        malformed_error="kept as malformed" if reason == "malformed" else None,
        final_content=_content(tools=tools, text=text),
    )


def _scored(tmp_path, *, malformed_call_penalty=0.05):
    run = result_payload()
    add_response_reports(run, job_dir=tmp_path, max_runtime_seconds=None)
    enrich_arc2_result(
        run,
        benchmark_run=dict(
            finished=True,
            history=[{}],
            result={"pairs": [{"correct": True}, {"correct": False}]},
            final_score=50,
        ),
        test_output_count=2,
        malformed_call_penalty=malformed_call_penalty,
        max_episode_penalty=None,
    )
    return run


@pytest.mark.parametrize(
    "reason,count,tools,text,empty,malformed",
    [
        # An ordinary empty reply: ullm's turn manager counted it one call.
        ("malformed", 1, False, False, True, 1),
        # An empty summary: its manager does not classify, so the malformed
        # stop reason supplies the call (no double counting either way).
        ("malformed", 0, False, False, True, 1),
        ("malformed", 2, False, False, True, 2),
        # Raw text the backend could not parse, with and without tool markup.
        ("malformed", 1, False, True, False, 1),
        ("malformed", 0, False, True, False, 1),
        ("tool_use", 2, True, False, False, 2),
        ("tool_use", 0, True, False, False, 0),
        ("stop", 0, False, True, False, 0),
        # A budget stop with nothing in it is not an empty reply.
        ("max_tokens", 0, False, False, False, 0),
    ],
)
def test_journal_fields_and_uniform_penalty(
    observer, tmp_path, reason, count, tools, text, empty, malformed
):
    observer(_response(reason, count, tools=tools, text=text))
    record = json.loads((tmp_path / "processed_responses/abc.jsonl").read_text())
    assert record == dict(
        nonce="first",
        malformed_call_count=count,
        truncation_cap_source=None,
        stop_reason=reason,
        has_malformed_error=reason == "malformed",
        has_tool_calls=tools,
        empty_reply=empty,
    )
    run = _scored(tmp_path)
    assert run["steps"][0]["malformed_call_count"] == malformed
    assert run["steps"][0]["empty_reply"] is empty
    outcome = run["arc2_outcome"]
    assert outcome["malformed_call_count"] == malformed
    assert outcome["empty_reply_count"] == int(empty)
    assert outcome["malformed_penalty"] == pytest.approx(0.05 * malformed)
    assert run["terminal_reward"] == pytest.approx(0.5 - 0.05 * malformed)
    RunResult.from_payload(run)


def test_a_journal_without_the_empty_reply_indicator_is_refused(tmp_path):
    directory = tmp_path / "processed_responses"
    directory.mkdir()
    (directory / "abc.jsonl").write_text(
        json.dumps({"nonce": "first", "malformed_call_count": 1, "stop_reason": "malformed"})
        + "\n"
    )
    run = result_payload()
    del run["steps"][0]["empty_reply"]
    add_response_reports(run, job_dir=tmp_path, max_runtime_seconds=None)
    with pytest.raises(ValueError, match="explicit empty-reply indicator"):
        enrich_arc2_result(
            run,
            benchmark_run=dict(
                finished=True,
                history=[{}],
                result={"pairs": [{"correct": True}, {"correct": False}]},
                final_score=50,
            ),
            test_output_count=2,
            malformed_call_penalty=0.05,
            max_episode_penalty=None,
        )
