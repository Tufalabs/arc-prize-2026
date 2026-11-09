"""U5 trace storage: the R5 trace sink end to end.

Hermetic by design: the trace machinery is exercised through a
trace-capable STUB backend riding the real topology (host backend behind
the Unix-socket wrapper, per-game ProxyBackend clients stamping run_id),
and the taaf boundary is the real ``register_trace`` (validate + write +
``_SUCCESS``). The engine-adjacent halves the cheap tier cannot reach —
real vllmx and OAI-server backends — run in the gpu tier below.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import arcengine
import pytest
import taaf.benchmark
import taaf.game
import taaf.game_examples
import taaf.trace
import ullm.backend
import ullm.backends.oai
import ullm.backends.stub
import ullm.conv
import ullm.factory
import ullm.tracing
import ullm.traffic

import platypus.llm_solver
import platypus.tracing
import platypus.traffic
from platypus import LLMAgentSolver, LLMConfig, RunConfig
from reference_agents import PromptedRandomAgent

_CAPABILITY = ullm.tracing.TraceCapability(
    provider_name="stub",
    provider_version="0",
    api_kind="test_stub",
    provider_model_id="stub-model",
    validation_method="test.stub_roundtrip.v1",
    model_id="stub-model",
    tokenizer_id="stub-tokenizer",
    contents=("full", "ids"),
)


@dataclass
class TraceStubBackend(ullm.backends.stub.StubBackend):
    """Scripted stub that also produces internally-consistent trace
    records — witness ids are fabricated deterministically, which is fine
    because the input roundtrip is backend-attested by contract."""

    async def trace_capability(self) -> ullm.tracing.TraceCapability | None:
        return _CAPABILITY

    async def _build_trace_step(
        self,
        *,
        run_id: str,
        sampling_params: ullm.backend.SamplingParameters,
        llm_input: Any,
        element: ullm.conv.LLMResponse,
        conversation: ullm.conv.Conversation,
    ) -> ullm.tracing.TraceStepRecord:
        messages = [{"role": "system", "content": conversation.system_prompt.content.to_string()}]
        for conv_element in conversation.elements:
            role = "user" if isinstance(conv_element, ullm.conv.UserPrompt) else "assistant"
            messages.append({"role": role, "content": conv_element.to_string()})
        output_text = element.final_content.to_string() or "empty"
        # An ids-only run (taaf R5.39) reports the exact side alone, as vllmx does.
        ids_only = self.trace_ids_only
        return ullm.tracing.TraceStepRecord(
            run_id=run_id,
            messages=[] if ids_only else messages,
            input_token_ids=[len(str(message.get("content"))) + 1 for message in messages],
            output_token_ids=list(range(1, len(output_text) + 1)),
            output_text="" if ids_only else output_text,
            output_message=None if ids_only else {"role": "assistant", "content": output_text},
            sampling_params=ullm.tracing.sampling_to_dict(sampling_params),
            request_payload={"model": "stub-model", "messages": [] if ids_only else messages},
            output_token_reconstruction_status="not_applicable" if ids_only else "ok",
            uncached_input_tokens=3,
        )


def _llm_config(*, traces: bool, capable: bool = True, **config_kwargs: Any) -> LLMConfig:
    cls = TraceStubBackend if capable else ullm.backends.stub.StubBackend
    backend = cls.from_script("8", on_exhausted="loop")
    return LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
        traces=traces,
        **config_kwargs,
    )


def _agent(config: LLMConfig, *, mode: str = "subprocess") -> PromptedRandomAgent:
    return PromptedRandomAgent(
        max_moves=4,
        profile="play",
        randomize_sum=False,
        llm_config=config,
        run_config=RunConfig(process_mode=mode, rng_seed=7),  # type: ignore[arg-type]
    )


# --- the fail-loud gate (U5: never a silent skip) ---------------------------


@pytest.mark.asyncio
async def test_traces_on_noncapable_backend_is_fatal_at_setup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The capability gate fires after the backend starts but before any
    game plays. TAAF's R2.15 surface for a solver-fatal: every game ends
    crashed, nothing is traced, and the error is printed loudly —
    the same surface as the token-accounting hard error."""
    agent = _agent(_llm_config(traces=True, capable=False))
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame()], solver=LLMAgentSolver(agent=agent), job_dir=tmp_path / "job"
    )
    await benchmark.run()
    assert all(run.state == "crashed" for run in benchmark.game_runs)
    assert "cannot produce R5-grade traces" in capsys.readouterr().err
    trace_root = tmp_path / "job" / "traces"
    assert not trace_root.exists() or not any(trace_root.iterdir())


@pytest.mark.asyncio
async def test_ids_only_traces_on_a_full_only_backend_are_fatal_at_setup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A served backend (OpenAIServer) writes full traces only; asked for
    ids-only ones it must refuse before a game plays, not fail the first
    step's payload reconstruction."""
    monkeypatch.setattr(TraceStubBackend, "trace_capability", _full_only_capability)
    agent = _agent(_llm_config(traces=True, trace_content="ids"))
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame()], solver=LLMAgentSolver(agent=agent), job_dir=tmp_path / "job"
    )
    await benchmark.run()
    assert all(run.state == "crashed" for run in benchmark.game_runs)
    assert "produces only ['full'] traces" in capsys.readouterr().err


async def _full_only_capability(self: TraceStubBackend) -> ullm.tracing.TraceCapability:
    return replace(_CAPABILITY, contents=("full",))


@pytest.mark.asyncio
async def test_exact_history_with_replay_dir_rejected(tmp_path: Path) -> None:
    """Replay serves recorded answers, never ids: the promise is refused at
    setup, before the replay path's early return could bypass the gate."""
    agent = _agent(_llm_config(traces=False, replay_dir=str(tmp_path), exact_history=True))
    benchmark = taaf.benchmark.Benchmark(games=[taaf.game_examples.ExampleGame()], solver=LLMAgentSolver(agent=agent))
    with pytest.raises(ValueError, match="exact_history=True is incompatible with replay_dir"):
        await benchmark.run()


@pytest.mark.asyncio
async def test_traces_with_replay_dir_rejected(tmp_path: Path) -> None:
    agent = _agent(_llm_config(traces=True, replay_dir=str(tmp_path)))
    benchmark = taaf.benchmark.Benchmark(games=[taaf.game_examples.ExampleGame()], solver=LLMAgentSolver(agent=agent))
    with pytest.raises(ValueError, match="incompatible with replay_dir"):
        await benchmark.run()


@pytest.mark.asyncio
async def test_respond_without_run_id_is_fatal_when_tracing() -> None:
    backend = TraceStubBackend.from_script("8", on_exhausted="loop")

    def _drop(record: ullm.tracing.TraceStepRecord) -> None:
        """Accept and discard: this row only needs the capability gate."""

    backend.trace_recorder = _drop
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="s"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="hi")))
    conversation.elements.append(ullm.conv.ResponseTurn(status="intermediate"))
    conversation.status = "intermediate"
    params = ullm.backend.GenericSamplingParameters(max_tokens=10)
    with pytest.raises(RuntimeError, match="without run_id"):
        async for _partial in backend.respond(params, conversation):
            pass


# --- LoRA identity (U7) ------------------------------------------------------


def test_hf_snapshot_commit_pin() -> None:
    import ullm.backends.vllmx as vllmx

    sha = "a" * 40
    assert vllmx._hf_snapshot_commit(f"/data/hub/models--org--m/snapshots/{sha}/x") == sha
    assert vllmx._hf_snapshot_commit("/data/weights/qwen") == ""


@pytest.mark.asyncio
async def test_vllmx_lora_capability_names_the_base() -> None:
    """The capability names the base; adapter identity belongs to each step."""
    import ullm.backends.vllmx as vllmx

    engine = SimpleNamespace(
        model_config=SimpleNamespace(model="m", tokenizer="m", revision=None, tokenizer_revision=None),
        vllm_config=SimpleNamespace(lora_config=object()),
    )
    backend = vllmx.VLLM(engine=engine)  # type: ignore[arg-type]
    # vllmx is outside the pyright include (see pyrightconfig.json), so what
    # its backend returns is unknown here.
    capability = await backend.trace_capability()  # pyright: ignore[reportUnknownVariableType]
    assert capability is not None and capability.model_id == "m"


@pytest.mark.asyncio
async def test_oai_identity_drift_stops_the_run() -> None:
    """Adapter hot-swap mid-run: the per-step re-resolve must refuse to
    keep tracing under a changed identity."""
    backend = ullm.backends.oai.OpenAIServer(base_url="http://localhost:1/v1", model="policy", streaming=False)
    identities = [
        {"model_revision": "/ckpt/a", "model_weight_hash": "h1", "parent_model_revision": "base"},
        {"model_revision": "/ckpt/b", "model_weight_hash": "h2", "parent_model_revision": "base"},
    ]

    async def fake_fetch() -> dict[str, str]:
        return identities.pop(0)

    backend._trace_fetch_identity = fake_fetch  # type: ignore[method-assign]
    backend._trace_identity = await backend._trace_fetch_identity()
    with pytest.raises(ullm.tracing.TraceReconstructionError, match="identity changed mid-run"):
        await backend._trace_check_identity_drift()
    await backend.close()


@pytest.mark.asyncio
async def test_oai_streaming_with_tracing_is_rejected_before_any_http() -> None:
    backend = ullm.backends.oai.OpenAIServer(base_url="http://localhost:1/v1", model="m", streaming=True)
    try:
        with pytest.raises(ValueError, match="requires streaming=False"):
            await backend.trace_capability()
    finally:
        await backend.close()


# --- snapshot linkage + the real taaf boundary ------------------------------


def _act(action_id: int) -> arcengine.ActionInput:
    return arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id), data={})


def _stub_record(run_id: str, index: int) -> ullm.tracing.TraceStepRecord:
    messages = [{"role": "user", "content": f"turn {index}"}]
    return ullm.tracing.TraceStepRecord(
        run_id=run_id,
        messages=messages,
        input_token_ids=[1, 2, 3],
        output_token_ids=[4, 5],
        output_text="ok",
        output_message={"role": "assistant", "content": "ok"},
        sampling_params={"temperature": 0.0},
        request_payload={"model": "stub-model", "messages": messages},
        output_token_reconstruction_status="ok",
        uncached_input_tokens=2,
    )


def _marker_record(run_id: str, n_images: int) -> ullm.tracing.TraceStepRecord:
    """A step shaped the way a marker-style backend (vllmx) produces one: the
    messages carry bare image markers and the bytes ride alongside, in marker
    order."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": "look"}]
    content += [{"type": "image"} for _ in range(n_images)]
    messages = [{"role": "user", "content": content}]
    return ullm.tracing.TraceStepRecord(
        run_id=run_id,
        messages=messages,
        input_token_ids=[1, 2, 3],
        output_token_ids=[4, 5],
        output_text="ok",
        output_message={"role": "assistant", "content": "ok"},
        sampling_params={"temperature": 0.0},
        request_payload={"model": "stub-model", "messages": messages},
        output_token_reconstruction_status="ok",
        uncached_input_tokens=2,
        images=[ullm.tracing.TraceImage(png=png) for _ in range(n_images)],
    )


def test_marker_images_name_the_file_they_were_written_to(tmp_path: Path) -> None:
    """A marker-style backend ships bare ``{"type": "image"}`` parts and its
    bytes separately. The written step must still say which file belongs in
    which slot: a consumer reads images out of the messages, and the declared
    message format has no bare image part at all. Without the linkage the
    media is on disk and unreachable, and the payload roundtrip cannot see it
    — the split leaves the messages untouched, so re-inlining them matches
    trivially."""
    game = taaf.game_examples.ExampleGame()
    game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    run = game.game_run
    assert run is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    collector.register_run(run)
    collector.record(_marker_record(run.run_id, n_images=2))
    game.execute_action(_act(1))
    game.finish_game()

    folder = collector.finalize(run)
    assert folder is not None
    trace = taaf.trace.load_trace(folder)
    (step,) = trace.steps
    parts = [p for p in step.messages[0]["content"] if p.get("type") != "text"]
    assert [p["type"] for p in parts] == ["image_url", "image_url"], parts
    referenced = [p["image_url"]["url"] for p in parts]
    assert referenced == [ref.path for ref in step.media], (referenced, [r.path for r in step.media])
    for path in referenced:
        assert (folder / path).exists(), path


def test_a_run_that_frees_a_slot_before_its_next_step_passes_its_link_on(tmp_path: Path) -> None:
    """Slot provenance: a run granted a slot that frees it again before its
    next step has no step to be followed by, so whoever takes the slot then
    follows what that run itself followed. Linking to the run's older step
    would give that tenure two successors and break the lane."""
    games = [taaf.game_examples.ExampleGame(label=label) for label in ("a", "b", "c")]
    for game in games:
        game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    a, b, c = (game.game_run for game in games)
    assert a is not None and b is not None and c is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    for run in (a, b, c):
        collector.register_run(run)
    collector.note_grant(a.run_id, None)
    collector.record(_stub_record(a.run_id, 0))
    collector.note_grant(b.run_id, a.run_id)  # a is done; b takes its slot
    collector.note_grant(c.run_id, b.run_id)  # b gives it up without a step
    collector.record(_stub_record(c.run_id, 0))
    links = {}
    for game, run in ((games[0], a), (games[2], c)):
        game.finish_game()
        folder = collector.finalize(run)
        assert folder is not None
        links[run.run_id] = taaf.trace.load_trace(folder).steps[0].extra["after"]
    assert links == {a.run_id: None, c.run_id: {"run_id": a.run_id, "step_index": 0}}


def test_snapshot_linkage_links_first_of_batch_and_skips_idle_calls(tmp_path: Path) -> None:
    """The in-snippet interleaving the mechanism exists for:
    respond -> execute a, b (batch) -> respond (inspect only) -> respond ->
    execute c. Steps link to their batch's FIRST action; the idle call
    links nothing."""
    game = taaf.game_examples.ExampleGame()
    game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    run = game.game_run
    assert run is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    collector.register_run(run)

    collector.record(_stub_record(run.run_id, 0))  # L=0
    game.execute_action(_act(1))  # index 0
    game.execute_action(_act(2))  # index 1 (same batch)
    collector.record(_stub_record(run.run_id, 1))  # L=2, no growth before next
    collector.record(_stub_record(run.run_id, 2))  # L=2
    game.execute_action(_act(0))  # index 2
    game.finish_game()

    folder = collector.finalize(run)
    assert folder is not None and (folder / "_SUCCESS").exists()
    trace = taaf.trace.load_trace(folder)
    linkage = [(step.action_index, step.action.id.value if step.action else None) for step in trace.steps]
    assert linkage == [(0, 1), (None, None), (2, 0)]
    # The dataframe bridge sees the same rows (R5.65).
    rows = list(taaf.trace.iter_trace_rows(tmp_path / "traces"))
    assert [row["step_index"] for row in rows] == [0, 1, 2]


def test_template_render_view_parses_tool_arguments() -> None:
    """Stored output_messages keep OpenAI-wire JSON-string arguments; HF
    templates iterate arguments as a mapping (jinja dictsort), so the
    render view parses them — without touching the stored form."""
    stored: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "python", "arguments": '{"code": "print(1)"}'}}
        ],
    }
    view = ullm.tracing.template_render_view(stored)
    assert view["tool_calls"][0]["function"]["arguments"] == {"code": "print(1)"}
    assert stored["tool_calls"][0]["function"]["arguments"] == '{"code": "print(1)"}'


def test_collector_rejects_unknown_run_id() -> None:
    collector = platypus.tracing.TraceCollector(_CAPABILITY)
    with pytest.raises(RuntimeError, match="unknown run_id"):
        collector.record(_stub_record("nope", 0))


def test_media_split_is_exactly_invertible(tmp_path: Path) -> None:
    """A data-URL image is split to media/, the message rewritten to the
    relative path, and the payload-hash pair proves re-inlining from the
    stored bytes reproduces the request exactly."""
    import base64

    game = taaf.game_examples.ExampleGame()
    game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    run = game.game_run
    assert run is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    collector.register_run(run)

    png = base64.b64encode(b"\x89PNG fake bytes").decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
            ],
        }
    ]
    record = _stub_record(run.run_id, 0)
    record.messages = messages
    record.request_payload = {"model": "stub-model", "messages": messages}
    collector.record(record)
    game.execute_action(_act(1))
    game.finish_game()

    folder = collector.finalize(run)
    assert folder is not None
    trace = taaf.trace.load_trace(folder)
    step = trace.steps[0]
    # The filename is TAAF's to choose (it addresses media by content), so
    # what is pinned here is the join: one media entry, the message pointing
    # at it, and the original bytes behind it.
    assert len(step.media) == 1
    stored_path = step.media[0].path
    assert stored_path.startswith("media/")
    assert (folder / stored_path).read_bytes() == b"\x89PNG fake bytes"
    stored_part = step.messages[0]["content"][1]
    assert stored_part["image_url"]["url"] == stored_path
    validation = step.provider_validation
    assert validation["original_payload_sha256"] == validation["reconstructed_payload_sha256"]


@pytest.mark.asyncio
async def test_oai_identity_resolution(tmp_path: Path) -> None:
    """Hermetic _trace_fetch_identity: /v1/models parsing + the
    runtime-export summary contract (adapter_sha256 / base_model_name),
    with the fail-loud cases."""
    backend = ullm.backends.oai.OpenAIServer(base_url="http://localhost:1/v1", model="policy", streaming=False)

    payload: dict[str, object] = {"data": [{"id": "policy", "root": str(tmp_path), "parent": "org/base"}]}

    class _Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> object:
            return payload

    class _Client:
        async def request(self, method: str, url: str, *, json: object = None) -> "_Response":
            assert method == "GET"
            assert json is None
            return _Response()

        async def aclose(self) -> None:
            pass

    backend._server_http = _Client()
    try:
        # Adapter without the runtime export: unusable provenance, loud.
        with pytest.raises(FileNotFoundError, match="runtime_adapter_summary"):
            await backend._trace_fetch_identity()
        # Proper runtime export: sha + base lineage resolved.
        (tmp_path / "runtime_adapter_summary.json").write_text(
            json.dumps({"adapter_sha256": "cafe" * 16, "base_model_name": "org/base-v2"})
        )
        identity = await backend._trace_fetch_identity()
        assert identity == {
            "model_revision": str(tmp_path),
            "model_weight_hash": "cafe" * 16,
            "parent_model_revision": "org/base-v2",
        }
        # Base model (no parent): empty hash is the correct posture.
        payload = {"data": [{"id": "policy", "root": "/weights/base", "parent": ""}]}
        identity = await backend._trace_fetch_identity()
        assert identity == {"model_revision": "/weights/base", "model_weight_hash": "", "parent_model_revision": ""}
        # Alias not served: loud.
        payload = {"data": [{"id": "other", "root": "/x", "parent": ""}]}
        with pytest.raises(RuntimeError, match="not in /v1/models"):
            await backend._trace_fetch_identity()
    finally:
        await backend.close()


def test_output_message_from_element_shapes() -> None:
    """Reasoning + text + tool call element -> the R5.37 structured form
    (reasoning keyed reasoning_content, arguments as JSON strings)."""
    element = ullm.conv.LLMResponse()
    element.reasoning_content = ullm.conv.StringContent(text="think hard")
    element.final_content = ullm.conv.ListContent(
        elements=[
            ullm.conv.StringContent(text="doing it"),
            ullm.conv.ToolCallContent(call_id="c0", tool_name="python", arguments={"code": "print(1)"}),
        ]
    )
    message = ullm.tracing.output_message_from_element(element)
    assert message == {
        "role": "assistant",
        "reasoning_content": "think hard",
        "content": "doing it",
        "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "python", "arguments": '{"code": "print(1)"}'}}
        ],
    }


def test_output_render_against_real_tokenizer_fixture() -> None:
    """Regression-lock the output-status render (agp prefix, framing-newline
    strip, dictsort argument adaptation) against real captured steps from a
    validated GPU run, classified by the REAL serving tokenizer. Statuses
    were computed in production by this same helper and verified by the
    100%-explained bar — a logic change that reclassifies any of them fails
    here without needing an engine."""
    from transformers import AutoTokenizer

    # transformers ships no stubs, so this seam is untyped at its source.
    tokenizer: Any = AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownVariableType]
        "vrfai/Qwen3.6-27B-FP8"
    )
    fixture = Path(__file__).parent / "references" / "trace_output_render_steps.jsonl"
    rows = [json.loads(line) for line in fixture.read_text().splitlines()]
    assert {row["expected_status"] for row in rows} == {"ok", "whitespace_mismatch", "mismatch"}
    for row in rows:
        status = ullm.tracing.classify_output_render_hf(tokenizer, {}, row["output_message"], row["output_token_ids"])
        assert status == row["expected_status"], row["note"]


# --- gpu tier: the engine-adjacent halves the cheap tier cannot reach --------


def _gpu_guard(request: pytest.FixtureRequest) -> None:
    """Mirrors test_llm_solver's guard: skip quietly when gpu wasn't
    selected, fail loudly when it was but no GPU is visible."""
    pytest.importorskip("vllm")
    import torch

    if not torch.cuda.is_available():
        if "gpu" in (request.config.option.markexpr or ""):
            pytest.fail("gpu tests selected but no GPU is visible; run under `make test-gpu-srun`")
        pytest.skip("no GPU available")


_GPU_TEST_MODEL = "Qwen/Qwen3-VL-2B-Instruct"  # the small instruct VL model; the ullm matrix runs Qwen3.5-9B


def _gpu_engine_args() -> "object":
    import vllm

    return vllm.AsyncEngineArgs(
        model=_GPU_TEST_MODEL, max_model_len=8192, gpu_memory_utilization=0.30, enforce_eager=True
    )


async def _traced_gpu_run(
    factory: ullm.factory.BackendFactory, tmp_path: Path, *, expect_continued: bool, injection: str | None = None
) -> taaf.trace.Trace:
    """One traced PromptedRandom game over ONE growing conversation
    (multimodal: the frame rides along as an image every move, exercising
    the image-token roundtrip), returning the loaded Trace after asserting
    the R5 folder + per-step invariants. ``expect_continued``: every call
    after the first extended the served ids — the in-process engine
    continues image deltas through its processor; a server cannot carry
    them and renders afresh (still traced, never claimed as continued).
    ``injection``: text served after every call's generation prompt — the
    trace must stay token-faithful (the roundtrip passes) and say so."""
    agent = PromptedRandomAgent(
        max_moves=3,
        attach_frame_image=True,
        image_scale=4,
        keep_history=True,
        llm_config=LLMConfig(
            backend_factory=factory,  # type: ignore[arg-type]
            # Past reasoning goes back whole: exact history has nothing to
            # continue when it is stripped.
            sampling_profiles={
                "play": ullm.backend.GenericSamplingParameters(
                    temperature=0.0, max_tokens=50, send_back_reasoning="all", response_injection=injection
                )
            },
            traces=True,
            exact_history=expect_continued,
        ),
        run_config=RunConfig(process_mode="subprocess", rng_seed=1),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame()], solver=LLMAgentSolver(agent=agent), job_dir=tmp_path / "job"
    )
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "gave_up", f"expected a clean gave-up run, got {run.state} ({run.solver_note})"
    folders = [p for p in (tmp_path / "job" / "traces").iterdir() if (p / "_SUCCESS").exists()]
    assert len(folders) == 1
    trace = taaf.trace.load_trace(folders[0])
    assert trace.model_id
    assert trace.n_steps == 3  # one seam-check call per decision move
    for step in trace.steps:
        validation = step.provider_validation
        assert validation["input_token_reconstruction_status"] == "ok"
        assert validation["output_token_reconstruction_status"] != "unavailable"
        assert validation["original_payload_sha256"] == validation["reconstructed_payload_sha256"]
        assert step.action_index is not None
        assert step.media, "the frame image must land as a media ref"
        assert step.input_token_ids and step.output_token_ids and step.output_text
    assert trace.extra["response_injected"] is (injection is not None)
    assert [step.extra.get("response_injection") for step in trace.steps] == [injection] * trace.n_steps
    continued = [bool(step.extra.get("continued")) for step in trace.steps]
    if expect_continued:
        assert continued == [False, True, True], continued
        assert trace.extra["nesting_breaks"] == 0
        # A continued step pays for its delta, never for the whole prompt:
        # the served prefix is a cache hit. (The delta itself — a new image
        # is ~64 tokens here — can outweigh this tiny history, so the claim
        # is against the full prompt, not the previous one.)
        for step in trace.steps[1:]:
            assert step.uncached_input_tokens < len(step.input_token_ids), (
                step.uncached_input_tokens,
                len(step.input_token_ids),
            )
    else:
        assert not any(continued), continued
    return trace


@pytest.mark.gpu
async def test_gpu_traced_vllmx_engine(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """The vllmx trace path against a real engine: witness ids off the
    RequestOutput and the in-process input roundtrip — INCLUDING the
    multimodal processor expansion — plus the local output-status render.
    This is the automated stand-in for the massive validation runs.

    Run WITH an injection, which is how the shipped duck runs: the roundtrip
    then has to reproduce the served ids on the fresh first call and the echo
    on the continued ones. One run, not two — injection is a per-request
    sampling parameter, so a second case would buy a second engine in this
    column for a difference the engine never sees, and this column wedges on
    a second engine (see the naming note below). The uninjected trace path is
    the OAI column's, and the unmarked trace is pinned in the cheap tier.

    Named WITHOUT 'direct'/'server'/'proxy': each traced test is its own
    GPU_MATRIX_COLUMNS column (own srun/GPU). Stacking a third engine into
    the shared columns wedged engine startup (session fixture + per-test
    engines); isolated, this test runs in ~2 min."""
    _gpu_guard(request)
    import ullm.backends.vllm_factory
    import ullm.backends.vllmx

    factory = ullm.backends.vllm_factory.EngineFactory(
        engine_args=_gpu_engine_args(),  # type: ignore[arg-type]
        backend_cls=ullm.backends.vllmx.VLLM,
        backend_kwargs={"tool_call_parser": "hermes"},
        mode="direct",
    )
    await _traced_gpu_run(factory, tmp_path, expect_continued=True, injection="We need")


@pytest.mark.gpu
async def test_gpu_traffic_records_ids_and_replays_them(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """The traffic benchmark end to end on a real engine, images included.
    Two games through one slot record ids-only traces; the engine is kept;
    the traces become a one-lane workload; the replay serves it through
    ``VLLM.serve_ids`` on that same engine. The replayed replies are exactly
    the recorded lengths, the continued calls hit the prefix cache on the
    lane's own history, and the second game follows the first on the lane.

    Its own column (``gpu_traffic``): a second engine in a shared column
    wedges startup (see ``test_gpu_traced_vllmx_engine``)."""
    _gpu_guard(request)
    # Aliased: a bare `import ullm.x` here would make `ullm` local to the test.
    import ullm.backends.vllm_factory as vllm_factory
    import ullm.backends.vllmx as vllmx

    factory = vllm_factory.EngineFactory(
        engine_args=_gpu_engine_args(),  # type: ignore[arg-type]
        backend_cls=vllmx.VLLM,
        backend_kwargs={"tool_call_parser": "hermes"},
        mode="direct",
    )
    play = ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50, send_back_reasoning="all")
    agent = PromptedRandomAgent(
        max_moves=3,
        attach_frame_image=True,
        image_scale=4,
        keep_history=True,
        llm_config=LLMConfig(
            backend_factory=factory,  # type: ignore[arg-type]
            sampling_profiles={"play": play},
            traces=True,
            trace_content="ids",
            exact_history=True,
        ),
        run_config=RunConfig(process_mode="subprocess", rng_seed=1, max_concurrent=1),
    )
    kept = platypus.llm_solver.KeptBackend()
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame(label=f"game{i}") for i in range(2)],
        solver=LLMAgentSolver(agent=agent, keep_backend=kept),
        job_dir=tmp_path / "job",
    )
    try:
        await benchmark.run()
        trace_root = tmp_path / "job" / "traces"
        for folder in (p for p in trace_root.iterdir() if (p / "_SUCCESS").exists()):
            trace = taaf.trace.load_trace(folder)
            assert trace.content == taaf.trace.CONTENT_IDS
            statuses = [step.provider_validation["input_token_reconstruction_status"] for step in trace.steps]
            assert statuses == ["not_applicable", "ok", "ok"], "the fresh render has nothing to check; the echo does"
            assert all(step.media and not step.messages and not step.output_text for step in trace.steps)

        runs, media = platypus.traffic.recorded_runs(trace_root)
        workload = ullm.traffic.build_workload(runs, media, horizon_tokens=10**9, snapshot=0.0)
        assert workload.meta["recorded_lanes"] == 1
        (lane,) = workload.lanes
        assert [call.sequence for call in lane.calls] == [0, 0, 0, 1, 1, 1]

        assert kept.backend is not None
        # The window outlasts the recording, so the lane runs dry: loudly.
        with pytest.raises(RuntimeError, match="ran out of workload"):
            await ullm.traffic.replay(
                kept.backend,  # type: ignore[arg-type]
                workload,
                lanes=1,
                window_s=600.0,
                gap_s=0.0,
                sampling_params=ullm.backend.GenericSamplingParameters(temperature=0.0, send_back_reasoning="all"),
                out_dir=tmp_path / "replay",
                restart=False,
            )
        summary = json.loads((tmp_path / "replay" / "traffic_result.json").read_text())
        assert summary["generated_tokens"] == lane.output_tokens
        rows = [json.loads(line) for line in (tmp_path / "replay" / "traffic_calls.jsonl").read_text().splitlines()]
        assert [row["generated"] for row in rows] == [call.output_len for call in lane.calls]
        for row in rows:
            if row["shared"]:
                assert row["cached"] > 0, row
    finally:
        await kept.close()


@pytest.mark.gpu
async def test_gpu_traced_oai_backend(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """The OAI trace path against a real vLLM server: the capability probe
    (/tokenize present, return_token_ids honored, /v1/models identity),
    non-streaming SDK extras survival, and the server-side /tokenize input
    roundtrip."""
    _gpu_guard(request)
    import ullm.backends.vllm_factory

    factory = ullm.backends.vllm_factory.InprocessServerFactory(
        engine_args=_gpu_engine_args(),  # type: ignore[arg-type]
        backend_kwargs={"streaming": False, "send_back_reasoning": "none"},
    )
    trace = await _traced_gpu_run(factory, tmp_path, expect_continued=False)
    assert trace.model_revision, "server identity resolution must fill the weights path"


# --- end to end through the real topology -----------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["subprocess", "inprocess"])
async def test_e2e_stub_run_writes_validated_traces(tmp_path: Path, mode: str) -> None:
    """Full stack: solver gate passes, run_id rides the RUN frame from the
    per-game client to the host stub, the collector assembles, and TAAF
    validates + writes one _SUCCESS trace per game."""
    agent = _agent(_llm_config(traces=True), mode=mode)
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame(label=f"game{i}") for i in range(2)],
        solver=LLMAgentSolver(agent=agent),
        job_dir=tmp_path / "job",
    )
    await benchmark.run()
    trace_root = tmp_path / "job" / "traces"
    folders = sorted(path for path in trace_root.iterdir() if path.is_dir())
    assert len(folders) == 2
    run_ids = {run.run_id for run in benchmark.game_runs}
    assert {folder.name for folder in folders} == run_ids
    for folder in folders:
        trace = taaf.trace.load_trace(folder)
        assert trace.n_steps > 0
        assert trace.model_id == "stub-model"
        assert trace.source_kind == "platypus"
        # PromptedRandomAgent asks the LLM once per decision move, then
        # returns the action: every step must link to a real action.
        run = next(run for run in benchmark.game_runs if run.run_id == folder.name)
        for step in trace.steps:
            assert step.action_index is not None
            assert step.action is not None
            recorded = json.loads(json.dumps(taaf.trace._action_to_json(run.history[step.action_index].action)))
            assert taaf.trace._action_to_json(step.action) == recorded


@pytest.mark.asyncio
async def test_ids_only_traces_keep_the_exact_side_alone(tmp_path: Path) -> None:
    """trace_content="ids" end to end: the solver tells the backend, the
    collector keeps the ids, and TAAF writes traces marked ids-only with no
    portable side on any step (a backend that ignored the flag would hand the
    collector output text, and TAAF would refuse the trace)."""
    agent = _agent(_llm_config(traces=True, trace_content="ids"), mode="inprocess")
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame(label=f"game{i}") for i in range(2)],
        solver=LLMAgentSolver(agent=agent),
        job_dir=tmp_path / "job",
    )
    await benchmark.run()
    folders = sorted(path for path in (tmp_path / "job" / "traces").iterdir() if path.is_dir())
    assert len(folders) == 2
    for folder in folders:
        trace = taaf.trace.load_trace(folder)
        assert trace.content == taaf.trace.CONTENT_IDS
        assert trace.steps
        for step in trace.steps:
            assert step.messages == [] and step.output_text == "" and step.output_message is None
            assert step.input_token_ids and step.output_token_ids


@pytest.mark.asyncio
async def test_slot_links_chain_every_call_into_one_lane_per_slot(tmp_path: Path) -> None:
    """Five games through two slots: the traces' slot links chain the runs
    into exactly two lanes, and every recorded call lands in one of them --
    what the traffic replay plays back in place of the games."""
    agent = _agent(_llm_config(traces=True, trace_content="ids"), mode="inprocess")
    agent.run_config.max_concurrent = 2
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame(label=f"game{i}") for i in range(5)],
        solver=LLMAgentSolver(agent=agent),
        job_dir=tmp_path / "job",
    )
    await benchmark.run()
    runs, media = platypus.traffic.recorded_runs(tmp_path / "job" / "traces")
    assert len(runs) == 5
    assert sum(1 for run in runs if run.calls[0].opens and run.calls[0].after is None) == 2
    # From before the first call and with no horizon to stop at, the lanes
    # hold the whole recording.
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=10**9, snapshot=0.0)
    assert workload.meta["recorded_lanes"] == 2
    assert sum(len(lane.calls) for lane in workload.lanes) == sum(len(run.calls) for run in runs)


@pytest.mark.asyncio
async def test_traces_off_leaves_backend_untouched(tmp_path: Path) -> None:
    """traces=False must not set a recorder or write anything."""
    agent = _agent(_llm_config(traces=False))
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame()],
        solver=LLMAgentSolver(agent=agent),
        job_dir=tmp_path / "job",
    )
    await benchmark.run()
    trace_root = tmp_path / "job" / "traces"
    assert not trace_root.exists() or not any(trace_root.iterdir())


def test_steps_are_stamped_with_whether_they_nest(tmp_path: Path) -> None:
    """The U5 nesting invariant is measured by the collector, for every
    backend: a step whose prompt ids extend the previous prompt + sampled
    ids is stamped as nesting, one that re-tokenized the sampled span (or
    lost it) as not, and the trace carries the break count."""
    game = taaf.game_examples.ExampleGame()
    game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    run = game.game_run
    assert run is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    collector.register_run(run)

    served: list[tuple[list[int], list[int]]] = [
        ([1, 2, 3], [4, 5]),  # first step: nothing to nest under
        ([1, 2, 3, 4, 5, 6, 7], [8]),  # extends prompt + output
        ([1, 2, 3, 4, 5, 9, 7, 8, 10], [11]),  # sampled span re-tokenized (5,6 -> 5,9)
        ([1, 2, 3, 4, 5, 9, 7, 8, 10, 11, 12], []),  # extends again
        ([1, 2, 3], [13]),  # history rewritten: shorter than what was served
    ]
    for index, (prompt, output) in enumerate(served):
        record = _stub_record(run.run_id, index)
        record.input_token_ids = prompt
        record.output_token_ids = output
        collector.record(record)
    game.execute_action(_act(1))
    game.finish_game()

    folder = collector.finalize(run)
    assert folder is not None
    trace = taaf.trace.load_trace(folder)
    assert [step.extra.get("nests_previous") for step in trace.steps] == [None, True, False, True, False]
    assert trace.extra["nesting_breaks"] == 2


def test_logprobs_reach_the_stored_trace_aligned_to_the_sampled_ids(tmp_path: Path) -> None:
    """The collector carries what the backend measured; taaf's own validator
    is the one that insists on the alignment, so this goes through
    ``load_trace`` rather than reading the record back."""
    game = taaf.game_examples.ExampleGame()
    game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    run = game.game_run
    assert run is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    collector.register_run(run)

    scored = _stub_record(run.run_id, 0)
    scored.logprobs = [
        {"token_id": 4, "logprob": -0.25, "rank": 1, "top": [[4, -0.25, 1], [7, -1.5, 2]]},
        {"token_id": 5, "logprob": -0.75, "rank": 2, "top": [[6, -0.5, 1], [5, -0.75, 2]]},
    ]
    collector.record(scored)
    collector.record(_stub_record(run.run_id, 1))  # unscored: None is spec-valid
    game.execute_action(_act(1))
    game.finish_game()

    folder = collector.finalize(run)
    assert folder is not None
    steps = taaf.trace.load_trace(folder).steps
    assert steps[1].logprobs is None
    assert steps[0].logprobs is not None
    assert [entry["token_id"] for entry in steps[0].logprobs] == steps[0].output_token_ids
    assert steps[0].logprobs[0]["logprob"] == -0.25


def test_an_injected_step_marks_the_trace(tmp_path: Path) -> None:
    """A step that served injected text is faithful in ids only: the trace
    says so at the top (``response_injected``) so a text-retokenising
    consumer can refuse it without reading steps, and the step names the
    text. A trace with no injected step says False, not nothing."""
    game = taaf.game_examples.ExampleGame()
    game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    run = game.game_run
    assert run is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    collector.register_run(run)
    injected = _stub_record(run.run_id, 0)
    injected.extra = {"response_injection": "Okay,"}
    collector.record(injected)
    collector.record(_stub_record(run.run_id, 1))
    game.execute_action(_act(1))
    game.finish_game()
    folder = collector.finalize(run)
    assert folder is not None
    trace = taaf.trace.load_trace(folder)
    assert trace.extra["response_injected"] is True
    assert [step.extra.get("response_injection") for step in trace.steps] == ["Okay,", None]

    plain_game = taaf.game_examples.ExampleGame()
    plain_game.start_game(taaf.game.RunSession(trace_root=tmp_path / "plain"))
    plain = plain_game.game_run
    assert plain is not None
    collector.register_run(plain)
    collector.record(_stub_record(plain.run_id, 0))
    plain_game.execute_action(_act(1))
    plain_game.finish_game()
    plain_folder = collector.finalize(plain)
    assert plain_folder is not None
    assert taaf.trace.load_trace(plain_folder).extra["response_injected"] is False


def test_logprobs_from_vllm_shapes_one_entry_per_sampled_token() -> None:
    """vLLM hands back a dict per position keyed by token id; the trace wants
    a list in sampled order, ids kept as ints (JSON object keys could not be)."""

    @dataclass
    class _Logprob:
        logprob: float
        rank: int | None = None

    completion = SimpleNamespace(
        logprobs=[
            {11: _Logprob(-0.25, 1), 12: _Logprob(-2.0, 2)},
            {13: _Logprob(-3.0, 4), 12: _Logprob(-0.5, 1)},  # sampled token ranked below the top
        ]
    )
    out = ullm.tracing.logprobs_from_vllm(completion, [11, 13])
    assert out is not None
    assert [entry["token_id"] for entry in out] == [11, 13]
    assert [entry["logprob"] for entry in out] == [-0.25, -3.0]
    assert [entry["rank"] for entry in out] == [1, 4]
    assert out[1]["top"] == [[13, -3.0, 4], [12, -0.5, 1]]
    # Not asked for: absent, not empty — the two mean different things.
    assert ullm.tracing.logprobs_from_vllm(SimpleNamespace(logprobs=None), [11]) is None


def test_logprobs_length_mismatch_fails_at_the_backend() -> None:
    """taaf rejects a misaligned list for the WHOLE trace at write time — a
    run that played every game and then stored nothing. Catching it where it
    is produced names the call instead."""
    completion = SimpleNamespace(logprobs=[{1: SimpleNamespace(logprob=-1.0, rank=1)}])
    with pytest.raises(ullm.tracing.TraceReconstructionError, match="align one-to-one"):
        ullm.tracing.logprobs_from_vllm(completion, [1, 2])


def test_exact_history_refuses_a_non_nesting_step_within_an_epoch(tmp_path: Path) -> None:
    """Under exact history (LLMConfig.exact_history) the collector is the
    second line behind the backend: a step that does not extend the previous
    prompt + sampled ids, with the same history epoch, stops the run; a
    declared rewrite (a new epoch) is a boundary, not a violation."""
    game = taaf.game_examples.ExampleGame()
    game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    run = game.game_run
    assert run is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    collector.register_run(run)

    def _step(index: int, prompt: list[int], output: list[int], epoch: int) -> ullm.tracing.TraceStepRecord:
        record = _stub_record(run.run_id, index)
        record.input_token_ids = prompt
        record.output_token_ids = output
        record.extra = {"exact_history": True, "history_epoch": epoch, "continued": index > 0}
        return record

    collector.record(_step(0, [1, 2, 3], [4, 5], epoch=0))
    collector.record(_step(1, [1, 2, 3, 4, 5, 6], [7], epoch=0))  # nests
    collector.record(_step(2, [9, 9], [7], epoch=1))  # declared rewrite: allowed
    with pytest.raises(ullm.tracing.TraceReconstructionError, match="exact history violated") as refused:
        collector.record(_step(3, [9, 9, 8], [1], epoch=1))  # re-tokenised span, same epoch
    # The error says where: the reply's first id was served as 7 and sent as 8.
    assert "first differs at id 2, id 0 of the previous reply; served [7], prompt [8]" in str(refused.value)


@pytest.mark.parametrize(
    "prompt, says",
    [
        # A retry that resends the request a refused reply answered.
        ([1, 2, 3], "the previous request sent again, without its reply"),
        ([1, 2, 3, 4], "the prompt stops at id 4, inside the previous reply (served history rolled back)"),
        (
            [1, 2, 0, 4, 5, 6],
            "first differs at id 2, id 2 of the previous prompt; served [3, 4, 5], prompt [0, 4, 5, 6]",
        ),
    ],
)
def test_a_non_nesting_step_says_where_it_left_the_served_ids(prompt: list[int], says: str) -> None:
    """The exact-history error names the break, so a trace that died with its
    job still says what the agent sent."""
    assert says in ullm.tracing.describe_nesting_break([1, 2, 3], [4, 5], prompt)


@pytest.mark.asyncio
async def test_exact_history_on_noncapable_backend_is_fatal_at_setup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A backend that cannot serve an id prompt from its own ids (a stub, a
    hosted API) fails the continuation gate at setup — never a silent
    re-rendering run under a promise of exact history."""
    agent = _agent(_llm_config(traces=True, exact_history=True))
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame()], solver=LLMAgentSolver(agent=agent), job_dir=tmp_path / "job"
    )
    await benchmark.run()
    assert all(run.state == "crashed" for run in benchmark.game_runs)
    assert "cannot continue the served history by tokens" in capsys.readouterr().err


def test_nesting_is_not_claimed_without_a_token_witness(tmp_path: Path) -> None:
    """A hosted-API record carries no ids (status ``unavailable``): its steps
    are neither stamped as nesting nor counted as breaks, and a trace whose
    steps never carried a witness reports ``nesting_breaks=None`` — not
    measurable is not the same as not broken."""
    game = taaf.game_examples.ExampleGame()
    game.start_game(taaf.game.RunSession(trace_root=tmp_path / "traces"))
    run = game.game_run
    assert run is not None
    collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
    collector.register_run(run)
    for index in range(2):
        record = _stub_record(run.run_id, index)
        record.input_token_ids = []
        record.output_token_ids = []
        record.input_token_reconstruction_status = "unavailable"
        record.extra = {"exact_history": True, "history_epoch": 0}
        collector.record(record)  # exact history cannot be violated where nothing is measurable
    game.execute_action(_act(1))
    game.finish_game()
    folder = collector.finalize(run)
    assert folder is not None
    trace = taaf.trace.load_trace(folder)
    assert all("nests_previous" not in step.extra for step in trace.steps)
    assert trace.extra["nesting_breaks"] is None


# --- one engine across runs (LLMAgentSolver.keep_backend) -------------------


async def _run_on_kept(config: LLMConfig, kept: platypus.llm_solver.KeptBackend, job_dir: Path) -> taaf.game.GameRun:
    """One game under a solver sharing ``kept``; returns its run."""
    benchmark = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame()],
        solver=LLMAgentSolver(agent=_agent(config), keep_backend=kept),
        job_dir=job_dir,
    )
    await benchmark.run()
    (run,) = benchmark.game_runs
    return run


@pytest.mark.asyncio
async def test_an_untraced_run_on_a_kept_backend_plays_after_a_traced_one(tmp_path: Path) -> None:
    """The traced run leaves its collector on the shared backend; the next
    run, untraced, must play rather than die on its first step reporting to
    a collector that never registered it. The backend meets the new run
    with the old run's per-run wiring taken off."""
    kept = platypus.llm_solver.KeptBackend()
    try:
        traced = await _run_on_kept(_llm_config(traces=True), kept, tmp_path / "traced")
        assert traced.state == "gave_up" and traced.history, traced.solver_note
        assert kept.backend is not None and kept.backend.trace_recorder is not None, (
            "the stale hook this guards against"
        )
        untraced = await _run_on_kept(_llm_config(traces=False), kept, tmp_path / "untraced")
        assert untraced.state == "gave_up" and untraced.history, untraced.solver_note
        assert kept.backend.trace_recorder is None and kept.backend.exact_history is False
    finally:
        await kept.close()


@pytest.mark.asyncio
async def test_a_run_on_a_kept_backend_reconciles_its_own_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The backend's counters carry every earlier run; the reconciliation
    (I1) must read this run's share, or a second run reports a mismatch
    that is not one -- printed, or raised and swallowed by TAAF behind
    completed games, depending on the teardown's exception context."""
    raised: list[BaseException] = []
    original = LLMAgentSolver._verify_token_accounting

    def spying(self: LLMAgentSolver, games: Any, *, hard: bool = True) -> None:
        """The real check, with what it raises kept where the test can see it."""
        try:
            original(self, games, hard=hard)
        except BaseException as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(LLMAgentSolver, "_verify_token_accounting", spying)
    kept = platypus.llm_solver.KeptBackend()
    try:
        first = await _run_on_kept(_llm_config(traces=False), kept, tmp_path / "first")
        assert kept.backend is not None and kept.backend.total_usage.completion_tokens > 0, "a run that spent tokens"
        second = await _run_on_kept(_llm_config(traces=False), kept, tmp_path / "second")
        assert first.state == second.state == "gave_up"
        assert raised == [], [str(exc) for exc in raised]
        assert "token accounting mismatch" not in capsys.readouterr().out
    finally:
        await kept.close()


def test_packed_extra_values_are_written_as_they_were(tmp_path: Path) -> None:
    """A backend may pack a big ``extra`` value while the collector holds its
    step (``ullm.tracing.PackedJson``: an API teacher's per-token
    alternatives); the written step carries the value itself, byte for byte
    what writing it unpacked gives."""
    alternatives = [
        {"token": "391", "bytes": [51, 57, 49], "logprob": -0.012345678901234567, "top_logprobs": []},
        {
            "token": " é",
            "bytes": [32, 195, 169],
            "logprob": -3.3e-07,
            "top_logprobs": [{"token": "x", "logprob": -9.5}],
        },
    ]
    lines = {}
    for name, value in (("packed", ullm.tracing.PackedJson.pack(alternatives)), ("plain", alternatives)):
        game = taaf.game_examples.ExampleGame()
        game.start_game(taaf.game.RunSession(trace_root=tmp_path / name))
        run = game.game_run
        assert run is not None
        collector = platypus.tracing.TraceCollector(_CAPABILITY, source_kind="test")
        collector.register_run(run)
        record = _stub_record(run.run_id, 0)
        record.extra["teacher_logprobs"] = value
        collector.record(record)
        game.execute_action(_act(1))
        game.finish_game()
        folder = collector.finalize(run)
        assert folder is not None
        (line,) = (folder / "steps.jsonl").read_text().splitlines()
        lines[name] = json.loads(line)["extra"]["teacher_logprobs"]
    assert lines["packed"] == lines["plain"] == alternatives
    assert json.dumps(lines["packed"]) == json.dumps(alternatives)
