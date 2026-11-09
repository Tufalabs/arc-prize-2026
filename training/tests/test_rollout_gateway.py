"""Live-loopback tests for the rollout gateway: capture of stamped calls,
sampled-token evidence, single admission, exact continuation, neutral sampling
and the results callback."""

import json
import socket
import threading
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from aiohttp import web

from engine_wire import generation, output
from training.engine_protocol import unpack_generation

from training.rollout_gateway import (
    CaptureGateway,
    GatewayError,
    SamplingPolicy,
    token_ids_crc32,
)
from training.context_contract import resolve_context_contract


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _adapter(model="stub-model"):
    return {"name": model, "id": 0, "path": None, "server_instance_id": "test-instance"}


class _StubVllm(BaseHTTPRequestHandler):
    """Simulate the native engine service while exercising the real HTTP edge."""

    seen: list[dict] = []
    seen_wires: list[dict] = []
    completion_ids = [3, 4]
    wrong_prompt = False
    invalid_logprobs = False
    malformed_response = False
    response_status = 200
    drop_next = 0
    response_changes: dict = {}

    def _respond(self, body, status=200):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _drop(self):
        if type(self).drop_next:
            type(self).drop_next -= 1
            self.close_connection = True
            self.connection.close()
            return True
        return False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen_wires.append(body)
        selected = body["lora_request"]
        adapter = (
            None
            if selected is None
            else {
                "name": selected["lora_name"],
                "id": selected["lora_int_id"],
                "path": selected["lora_path"],
            }
        )
        body = unpack_generation(body, model=body["model"], adapter=adapter)
        type(self).seen.append(body)
        assert self.path == "/generate"
        if self._drop():
            return
        if self.response_status != 200:
            return self._respond({"error": "generation rejected"}, self.response_status)
        if self.malformed_response:
            return self._respond({"output_token_ids": None})
        self._respond(
            output(
                {
                    "request_id": body["user"],
                    "model": body["model"],
                    "adapter": _adapter(body["model"]),
                    "prompt_token_ids": [999]
                    if self.wrong_prompt
                    else list(body["prompt"]),
                    "output_token_ids": self.completion_ids,
                    "output_logprobs": [
                        float("nan") if self.invalid_logprobs else -0.1,
                        -0.2,
                    ][: len(self.completion_ids)],
                    "text": "ok",
                    "finish_reason": "stop",
                    "stop_reason": None,
                    "num_cached_tokens": 0,
                    **self.response_changes,
                }
            )
        )

    def do_GET(self):
        type(self).seen.append({"get": self.path})
        if not self._drop():
            self._respond({"models": [_adapter()], "adapter": _adapter()})

    def log_message(self, *args):
        pass


@pytest.fixture()
def gateway():
    upstream_port = _free_port()
    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), _StubVllm)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    _StubVllm.seen = []
    _StubVllm.seen_wires = []
    _StubVllm.drop_next = 0
    _StubVllm.wrong_prompt = False
    _StubVllm.response_changes = {}
    _StubVllm.invalid_logprobs = False
    _StubVllm.malformed_response = False
    _StubVllm.response_status = 200
    gw = CaptureGateway(
        upstream_base_url=f"http://127.0.0.1:{upstream_port}", port=_free_port()
    )
    gw.set_policy(
        SamplingPolicy(
            model="stub-model",
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            context=resolve_context_contract(32768, 4096, 16384, 0),
            expected_adapter=_adapter(),
        )
    )
    gw.start()
    yield gw
    gw.stop()
    upstream.shutdown()


def _lineage(kind="start", epoch=0, proof=None):
    return {
        "version": 1,
        "epoch": epoch,
        "kind": kind,
        "compaction_prompt_token_ids": proof,
    }


def _start(gateway, user=None, **changes):
    return _generate(gateway, prompt=[1, 2], user=user, **changes)


def _sampling():
    return dict(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        min_p=0.0,
        min_tokens=0,
    )


def _generate(gateway, **changes):
    payload = dict(
        model="stub-model",
        prompt=[11, 12, 13],
        max_tokens=4,
        user="run-9#tok",
        arc3_lineage=_lineage(),
        skip_special_tokens=False,
        **_sampling(),
    )
    payload.update(changes)
    return requests.post(
        f"{gateway.base_url}/generate",
        data=json.dumps(
            generation(
                payload, gateway._policy.expected_adapter or _adapter(payload["model"])
            )
        ),
        headers={"Content-Type": "application/json"},
        timeout=5,
    )


def test_stamped_call_is_captured_with_injected_evidence(gateway) -> None:
    response = _start(gateway, user="run-1#abc")
    assert response.status_code == 200
    # The gateway, not the client, asks for RL evidence.
    assert _StubVllm.seen[-1]["logprobs"] == 0
    assert _StubVllm.seen[-1]["skip_special_tokens"] is False
    window = gateway.drain_window()
    capture = window.captures[("run-1", "abc")]
    assert capture.prompt_token_ids == [1, 2]
    assert capture.completion_token_ids == [3, 4]
    assert capture.completion_logprobs == pytest.approx([-0.1, -0.2])
    assert capture.finish_reason == "stop"
    assert capture.request_messages == []
    assert capture.model == "stub-model" and capture.adapter == _adapter()
    # drain cleared the store.
    assert not gateway.drain_window().captures


def test_pinned_engine_proxy_prefix_cache_hints_do_not_reach_native_sampling(
    gateway, monkeypatch
):
    from unittest.mock import AsyncMock

    proxy_module = pytest.importorskip("ullm.backends.engine_proxy")
    proxy = proxy_module.EngineProxy(base_url=gateway.base_url, model="stub-model")
    # Exercise the pinned serializer and HTTP client without downloading a tokenizer.
    monkeypatch.setattr(proxy, "connect", AsyncMock())
    proxy._client()

    async def scenario():
        try:
            outputs = [
                item
                async for item in proxy.generate(
                    {"prompt_token_ids": [1, 2]},
                    {
                        **_sampling(),
                        "max_tokens": 4,
                        "logprobs": 0,
                        "skip_special_tokens": False,
                        "extra_args": {
                            "prefix_cache_context": "run-1/thread",
                            "prefix_cache_epoch": 0,
                        },
                    },
                    "run-1#cache",
                    request_context={"arc3_lineage": _lineage()},
                )
            ]
            assert outputs[0].outputs[0].token_ids == [3, 4]
        finally:
            await proxy.aclose()

    asyncio.run(scenario())
    assert "extra_args" not in _StubVllm.seen_wires[-1]["sampling_params"]
    assert _StubVllm.seen_wires[-1]["request_context"] == {}
    assert gateway.drain_window().captures[("run-1", "cache")].prompt_token_ids == [
        1, 2,
    ]


def test_unknown_sampling_extras_are_rejected_before_forwarding(gateway):
    response = _start(
        gateway, extra_args={"prefix_cache_epoch": 0, "hidden_processor": True}
    )
    assert response.status_code == 400
    assert not _StubVllm.seen_wires


@pytest.mark.parametrize("expected_adapter", [None, _adapter()])
def test_required_lora_rejects_missing_or_base_identity_before_dispatch(
    gateway, expected_adapter
):
    with pytest.raises(GatewayError, match="require a loaded LoRA"):
        gateway.set_policy(
            replace(
                gateway._policy, require_lora=True, expected_adapter=expected_adapter
            )
        )
    assert not _StubVllm.seen
    with pytest.raises(GatewayError, match="require a loaded LoRA"):
        gateway.raise_if_invalid()


@pytest.mark.parametrize("served_id", [0, 1, 2])
def test_required_lora_keeps_per_response_identity_validation(gateway, served_id):
    expected = {**_adapter(), "id": 1, "path": "/tmp/initialized-lora"}
    gateway.set_policy(
        replace(gateway._policy, require_lora=True, expected_adapter=expected)
    )
    observed = {**expected, "id": served_id}
    if served_id == 0:
        observed["path"] = None
    _StubVllm.response_changes = {"adapter": observed}
    response = _start(gateway, user="run-1#initial-lora")
    if served_id == 1:
        assert response.status_code == 200
        assert gateway.drain_window().captures
    else:
        assert response.status_code == 400
        with pytest.raises(GatewayError, match="adapter"):
            gateway.drain_window()


def test_generation_failure_record_retains_original_handler_traceback(tmp_path):
    from types import SimpleNamespace

    from training.run_failure import record_run_failure

    gateway = CaptureGateway(
        upstream_base_url="http://unused.invalid",
        port=0,
        on_contract_violation=lambda error: record_run_failure(tmp_path, error),
    )

    async def original_transport_failure(_request):
        raise ConnectionResetError("diagnostic socket reset")

    response = asyncio.run(
        gateway._track_generation_requests(
            SimpleNamespace(method="POST", path="/generate"), original_transport_failure
        )
    )
    assert response.status == 400
    failure = json.loads((tmp_path / "run_failure.json").read_text())
    assert "ConnectionResetError: diagnostic socket reset" in failure["traceback"]
    assert "original_transport_failure" in failure["traceback"]
    assert "generation handler failed" in failure["traceback"]
    assert "direct cause" in failure["traceback"]
    with pytest.raises(GatewayError, match="diagnostic socket reset"):
        gateway.raise_if_invalid()
    assert gateway._pending_generations == 0


def test_retry_reusing_nonce_is_rejected_before_another_generation(gateway) -> None:
    assert _start(gateway, user="run-1#abc").status_code == 200
    assert _start(gateway, user="run-1#abc").status_code == 400
    assert len([b for b in _StubVllm.seen if "max_tokens" in b]) == 1
    with pytest.raises(GatewayError, match="retries"):
        gateway.drain_window()


def test_exact_native_continuation_and_verified_native_reset(gateway):
    gateway.set_policy(
        replace(
            gateway._policy,
            context=replace(
                gateway._policy.context,
                compaction_trigger_tokens=8,
                reset_prompt_ceiling_tokens=3,
            ),
        )
    )
    assert _start(gateway, user="r#first").status_code == 200
    assert (
        _generate(
            gateway,
            user="r#second",
            prompt=[1, 2, 3, 4, 5],
            arc3_lineage=_lineage("continue"),
        ).status_code
        == 200
    )
    proof = [1, 2, 3, 4, 5, 3, 4, 8, 9]
    assert (
        _start(
            gateway, user="r#reset", arc3_lineage=_lineage("reset", 1, proof)
        ).status_code
        == 200
    )
    assert (
        _generate(
            gateway,
            user="r#after",
            prompt=[1, 2, 3, 4, 10],
            arc3_lineage=_lineage("continue", 1),
        ).status_code
        == 200
    )
    assert len(gateway.drain_window().captures) == 4


@pytest.mark.parametrize(
    "change", ["missing_output", "fresh_nonce_retry", "epoch_jump", "unverified_reset"]
)
def test_lineage_violations_fail_before_another_generation(gateway, change):
    assert _generate(gateway, user="r#first", prompt=[1, 2]).status_code == 200
    payload = {
        "user": "r#second",
        "prompt": [1, 2, 3, 4, 5],
        "arc3_lineage": _lineage("continue"),
    }
    if change == "missing_output":
        payload["prompt"] = [1, 2, 5]
    if change == "fresh_nonce_retry":
        payload.update(prompt=[1, 2], arc3_lineage=_lineage())
    if change == "epoch_jump":
        payload["arc3_lineage"] = _lineage("continue", 1)
    if change == "unverified_reset":
        payload["arc3_lineage"] = _lineage("reset", 1)
    assert _generate(gateway, **payload).status_code == 400
    assert len(_StubVllm.seen) == 1


def test_generation_transport_failure_is_not_retried(gateway):
    _StubVllm.drop_next = 1
    assert _generate(gateway).status_code == 400
    assert len(_StubVllm.seen) == 1
    with pytest.raises(GatewayError, match="generation failed"):
        gateway.raise_if_invalid()


def _result(*, steps, state="gave_up", elapsed=10.0):
    return dict(
        run_id="r",
        game_id="ar25",
        state=state,
        levels_completed=0,
        number_of_levels=2,
        actions_per_level=[],
        base_actions_per_level=[2, 4],
        final_wallclock_seconds=elapsed,
        action_wallclock_seconds=[],
        steps=steps,
    )


def _report(gateway, result):
    return requests.post(gateway.results_url, json={"runs": [result]}, timeout=5)


def _step():
    return dict(
        nonce="first",
        action_index=None,
        output_len=2,
        output_crc32=token_ids_crc32([3, 4]),
        continued=False,
        history_epoch=0,
        exact_history=True,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"continued": True},
        {"history_epoch": 1},
        {"exact_history": False},
        {"output_len": 1},
        {"output_crc32": 0},
    ],
)
def test_callback_lineage_and_output_must_match_captured_draw(gateway, change):
    assert _generate(gateway, user="r#first", prompt=[1, 2]).status_code == 200
    assert _report(gateway, _result(steps=[{**_step(), **change}])).status_code == 400
    with pytest.raises(GatewayError):
        gateway.raise_if_invalid()


@pytest.mark.parametrize(
    "state,elapsed,accepted",
    [
        ("gave_up", 10.0, True),
        ("gave_up", 9.0, False),
        ("won", 10.0, False),
        # A crash can cut a run at a draw its agent never processed, at any
        # time (on_agent_error="crash_game").
        ("crashed", 10.0, True),
        ("crashed", 3.0, True),
    ],
)
def test_only_a_terminal_deadline_or_crash_can_omit_one_draw(
    gateway, state, elapsed, accepted
):
    gateway.set_policy(replace(gateway._policy, max_runtime_seconds=10.0))
    assert _generate(gateway, user="r#first", prompt=[1, 2]).status_code == 200
    response = _report(gateway, _result(steps=[], state=state, elapsed=elapsed))
    assert response.status_code == (200 if accepted else 400)
    if accepted:
        window = gateway.drain_window()
        assert ("r", "first") in window.captures


def test_a_crashed_run_still_cannot_omit_two_draws(gateway):
    assert _generate(gateway, user="r#first", prompt=[1, 2]).status_code == 200
    assert (
        _generate(
            gateway,
            user="r#second",
            prompt=[1, 2, 3, 4, 5],
            arc3_lineage=_lineage("continue"),
        ).status_code
        == 200
    )
    response = _report(gateway, _result(steps=[], state="crashed", elapsed=3.0))
    assert response.status_code == 400
    with pytest.raises(GatewayError, match="unreported generations"):
        gateway.raise_if_invalid()


def test_drain_waits_for_a_generation_after_the_deadline_callback(gateway, monkeypatch):
    gateway.set_policy(replace(gateway._policy, max_runtime_seconds=10.0))
    entered, release = threading.Event(), threading.Event()
    original = gateway._upstream

    async def delayed(method, url, **kwargs):
        if url.endswith("/generate"):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        return await original(method, url, **kwargs)

    monkeypatch.setattr(gateway, "_upstream", delayed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        response = executor.submit(_generate, gateway, user="r#first", prompt=[1, 2])
        try:
            assert entered.wait(timeout=5)
            assert _report(gateway, _result(steps=[])).status_code == 200
            drained = executor.submit(gateway.drain_window, timeout_seconds=5)
            assert not drained.done()
        finally:
            release.set()
        assert response.result(timeout=5).status_code == 200
        window = drained.result(timeout=5)
    assert ("r", "first") in window.captures
    assert not gateway.drain_window().captures
    assert _generate(gateway, user="r#reuse", prompt=[1]).status_code == 400


def test_pending_drain_timeout_is_fatal_without_discarding_window():
    gateway = CaptureGateway(upstream_base_url="http://unused", port=1)
    gateway._pending_generations = 1
    window = gateway._window
    with pytest.raises(GatewayError, match="timed out"):
        gateway.drain_window(timeout_seconds=0)
    assert gateway._window is window
    with pytest.raises(GatewayError):
        gateway.raise_if_invalid()


def test_final_drain_rejects_generation_hidden_under_another_run_id(gateway):
    assert _generate(gateway, user="r#first", prompt=[1, 2]).status_code == 200
    assert _report(gateway, _result(steps=[_step()])).status_code == 200
    assert (
        _generate(gateway, user="unreported-run#first", prompt=[1, 2]).status_code
        == 200
    )
    window = gateway._window
    with pytest.raises(GatewayError, match="lack result callbacks"):
        gateway.drain_window(require_results=True)
    assert gateway._window is window
    with pytest.raises(GatewayError):
        gateway.raise_if_invalid()


def test_unstamped_generation_is_rejected(gateway) -> None:
    response = _start(gateway)
    assert response.status_code == 400
    assert not _StubVllm.seen
    with pytest.raises(GatewayError, match="stamp"):
        gateway.drain_window()


def test_streaming_is_refused(gateway) -> None:
    response = _start(gateway, user="run-1#abc", stream=True)
    assert response.status_code == 400


def test_generic_passthrough(gateway) -> None:
    response = requests.get(f"{gateway.base_url}/models", timeout=10)
    assert response.status_code == 200
    assert response.json()["models"][0]["name"] == "stub-model"


def test_harness_status_monitor_can_read_metrics(gateway) -> None:
    response = requests.get(gateway.base_url + "/metrics", timeout=5)
    assert response.status_code == 200
    gateway.raise_if_invalid()


def test_results_callback_lands_in_the_window(gateway) -> None:
    assert _start(gateway, user="run-1#abc").status_code == 200
    payload = {
        "runs": [
            {
                "run_id": "run-1",
                "game_id": "pathfinder-0001",
                "state": "won",
                "levels_completed": 1,
                "number_of_levels": 2,
                "actions_per_level": [2],
                "base_actions_per_level": [2, 4],
                "final_wallclock_seconds": 12.5,
                "action_wallclock_seconds": [5.0, 12.5],
                "steps": [
                    {
                        "nonce": "abc",
                        "action_index": 0,
                        "output_len": 2,
                        "output_crc32": token_ids_crc32([3, 4]),
                        "continued": False,
                        "history_epoch": 0,
                        "exact_history": True,
                    }
                ],
            }
        ]
    }
    response = requests.post(gateway.results_url, json=payload, timeout=10)
    assert response.status_code == 200
    assert response.json() == {"accepted": 1}
    window = gateway.drain_window()
    run = window.results["run-1"]
    assert run.game_id == "pathfinder-0001"
    assert run.action_wallclock_seconds == [5.0, 12.5]

    malformed = requests.post(gateway.results_url, json={"runs": [{}]}, timeout=10)
    assert malformed.status_code == 400


def test_completion_turn_is_captured_from_token_prompt(gateway) -> None:
    gw = gateway
    response = requests.post(
        f"{gw.base_url}/generate",
        json=generation(
            {
                **_sampling(),
                "skip_special_tokens": False,
                "model": "stub-model",
                "prompt": [11, 12, 13],
                "max_tokens": 4,
                "user": "run-9#tok",
                "arc3_lineage": _lineage(),
            }
        ),
        timeout=5,
    )
    assert response.status_code == 200
    assert _StubVllm.seen[-1]["logprobs"] == 0
    assert _StubVllm.seen[-1]["skip_special_tokens"] is False
    capture = gw.drain_window().captures[("run-9", "tok")]
    assert capture.prompt_token_ids == [11, 12, 13]
    assert capture.native_token_prompt is True
    assert capture.completion_token_ids == [3, 4]
    assert capture.completion_logprobs == pytest.approx([-0.1, -0.2])
    assert capture.assistant_message == {"role": "assistant", "content": "ok"}


@pytest.mark.parametrize(
    "changes",
    [
        {"presence_penalty": 1.5},
        {"frequency_penalty": 0.1},
        {"repetition_penalty": 1.1},
        {"min_p": 0.1},
        {"min_tokens": 1},
        {"temperature": 0.6},
        {"top_p": 0.95},
        {"top_k": 20},
        {"top_k": None},
        {"temperature": True},
        {"presence_penalty": float("nan")},
        {"model": "stale-policy"},
        {"n": 2},
        {"best_of": 1},
        {"return_token_ids": True},
        {"messages": [{"role": "user", "content": "ping"}]},
        {"use_beam_search": False},
        {"logprobs": 1},
        {"echo": False},
        {"stream": True},
        {"logit_bias": {"3": 1}},
        {"allowed_token_ids": [3]},
        {"bad_words": ["!"]},
        {"logits_processors": ["processor"]},
        {"structured_outputs": {"json": {}}},
        {"thinking_token_budget": 10},
        {"repetition_detection": {"max_pattern_size": 10}},
        {"vllm_xargs": {}},
        {"truncate_prompt_tokens": 2},
        {"prompt": "rerendered text"},
        {"prompt": [[1, 2], [3, 4]]},
        {"prompt": [True]},
        {"skip_special_tokens": True},
        {"echo": True},
        {"max_tokens": 4097},
        {"max_completion_tokens": 4},
    ],
)
def test_invalid_wire_settings_fail_before_serving_and_poison_window(gateway, changes):
    response = _generate(gateway, **changes)
    assert response.status_code == 400, response.text
    assert not _StubVllm.seen
    # The harness swallowing the HTTP error cannot turn this into usable data.
    with pytest.raises(GatewayError):
        gateway.raise_if_invalid()
    with pytest.raises(GatewayError):
        gateway.drain_window()
    assert _generate(gateway).status_code == 400


def test_violation_stops_the_run_even_if_harness_catches_error(gateway, tmp_path):
    from training.run_failure import (
        FatalTrainingError,
        record_run_failure,
        raise_if_run_failed,
    )

    gateway._on_contract_violation = lambda error: record_run_failure(tmp_path, error)
    assert _generate(gateway, presence_penalty=1.5).status_code == 400
    # The trainer reads its own shared failure channel, not the worker's
    # eventual exit status or interpretation of this HTTP response.
    with pytest.raises(FatalTrainingError, match="presence_penalty"):
        raise_if_run_failed(tmp_path)
    assert (tmp_path / "STOP_AUTO_RELAUNCH").exists()


def test_invalid_result_callback_is_also_a_fatal_contract_violation(gateway, tmp_path):
    from training.run_failure import (
        FatalTrainingError,
        record_run_failure,
        raise_if_run_failed,
    )

    gateway._on_contract_violation = lambda error: record_run_failure(tmp_path, error)
    response = requests.post(gateway.results_url, json={"runs": [{}]}, timeout=5)
    assert response.status_code == 400
    with pytest.raises(FatalTrainingError, match="malformed run result"):
        raise_if_run_failed(tmp_path)


def test_upstream_generation_failure_cannot_be_swallowed(gateway):
    _StubVllm.response_status = 500
    response = _generate(gateway)
    assert response.status_code == 400
    with pytest.raises(GatewayError, match="generation failed: HTTP 500"):
        gateway.raise_if_invalid()


@pytest.mark.parametrize("key", list(_sampling()))
def test_sampling_cannot_silently_inherit_server_defaults(gateway, key):
    payload = dict(
        model="stub-model",
        prompt=[1],
        max_tokens=4,
        user="r#n",
        skip_special_tokens=False,
        **_sampling(),
    )
    del payload[key]
    response = requests.post(
        f"{gateway.base_url}/generate",
        json=generation(
            payload, gateway._policy.expected_adapter or _adapter(payload["model"])
        ),
        timeout=5,
    )
    assert response.status_code == 400
    assert key in response.text
    assert not _StubVllm.seen


@pytest.mark.parametrize(
    "fault,match",
    [
        ("wrong_prompt", "served prompt differs"),
        ("invalid_logprobs", "finite"),
    ],
)
def test_invalid_server_evidence_poison_window(gateway, fault, match):
    setattr(_StubVllm, fault, True)
    response = _generate(gateway)
    assert response.status_code == 400
    with pytest.raises(GatewayError, match=match):
        gateway.drain_window()


def test_malformed_server_response_always_poison_window(gateway):
    _StubVllm.malformed_response = True
    response = _generate(gateway)
    assert response.status_code == 400
    with pytest.raises(GatewayError):
        gateway.drain_window()


@pytest.mark.parametrize("token", [1.5, True, "3", -1])
def test_response_token_ids_are_not_coerced(gateway, token):
    _StubVllm.completion_ids = [token, 4]
    try:
        response = _generate(gateway)
        assert response.status_code == 400
        with pytest.raises(GatewayError, match="exact nonnegative integers"):
            gateway.drain_window()
    finally:
        _StubVllm.completion_ids = [3, 4]


@pytest.mark.parametrize(
    "path",
    [
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/responses",
        "/adapters/load",
        "/adapters/unload",
        "/adapters/warm",
        "/tokenize",
        "/detokenize",
    ],
)
def test_harness_cannot_bypass_capture_or_change_policy(gateway, path):
    response = requests.post(gateway.base_url + path, json={}, timeout=5)
    assert response.status_code == 400
    assert not _StubVllm.seen
    with pytest.raises(GatewayError, match="route"):
        gateway.drain_window()


def test_explicit_eval_policy_and_memory_reply_limit(gateway):
    gateway.set_policy(
        SamplingPolicy(
            model="eval-model",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            context=resolve_context_contract(32768, 4096, 16384, 0),
        )
    )
    assert (
        _generate(
            gateway,
            model="eval-model",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=3072,
        ).status_code
        == 200
    )
    assert gateway.drain_window().captures


def test_dropped_upstream_connection_is_retried_once(gateway) -> None:
    gw = gateway
    _StubVllm.drop_next = 1
    response = requests.get(f"{gw.base_url}/metadata", timeout=10)
    assert response.status_code == 200
    assert len(_StubVllm.seen) == 2  # the dropped attempt and the retry


def test_persistently_unreachable_upstream_answers_json_502(gateway) -> None:
    gw = gateway
    _StubVllm.drop_next = 99
    response = requests.get(f"{gw.base_url}/metadata", timeout=10)
    assert response.status_code == 502
    assert "upstream unreachable" in response.json()["error"]


def test_full_arc2_eval_reaches_engine_before_any_generation_finishes():
    """The HTTP pool must admit all 120 puzzles, as the engine proxy does."""

    async def run():
        seen = Counter()
        admitted, release = asyncio.Event(), asyncio.Event()

        async def generate(request):
            seen[await request.text()] += 1
            if len(seen) == 120:
                admitted.set()
            await release.wait()
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_post("/generate", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        gateway = CaptureGateway(upstream_base_url=url, port=0)
        tasks = []
        try:
            await gateway._start_app()
            tasks = [
                asyncio.create_task(
                    gateway._upstream("POST", url + "/generate", data=str(i))
                )
                for i in range(120)
            ]
            try:
                await asyncio.wait_for(admitted.wait(), timeout=5)
            except TimeoutError:
                pytest.fail(f"only {len(seen)}/120 draws reached the engine")
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            assert all(status == 200 for status, _, _ in responses)
            assert seen == Counter({str(i): 1 for i in range(120)})
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if gateway._runner is not None:
                await gateway._runner.cleanup()
            await gateway._session.close()
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": "other#request"},
        {"adapter": None},
        {"adapter": {**_adapter(), "name": "wrong-model"}},
        {"adapter": {**_adapter(), "id": True}},
        {"adapter": {**_adapter(), "id": 1, "path": "/different/checkpoint"}},
        {"adapter": {**_adapter(), "server_instance_id": "restarted-server"}},
        {"adapter": {**_adapter(), "path": "/unexpected/base-path"}},
        {"output_logprobs": [-0.1]},
        {"finish_reason": None},
        {"finish_reason": "abort"},
        {"num_cached_tokens": True},
    ],
)
def test_native_engine_identity_and_evidence_must_match_admission(gateway, change):
    _StubVllm.response_changes = change
    assert _generate(gateway).status_code == 400
    with pytest.raises(GatewayError):
        gateway.drain_window()


@pytest.mark.parametrize("pending", [False, True])
def test_policy_cannot_change_before_the_window_is_drained(gateway, pending):
    if pending:
        gateway._pending_generations = 1
    else:
        assert _generate(gateway).status_code == 200
    with pytest.raises(GatewayError, match="previous window is drained"):
        gateway.set_policy(
            replace(gateway._policy, model="other", expected_adapter=_adapter("other"))
        )
    gateway._pending_generations = 0
    with pytest.raises(GatewayError):
        gateway.raise_if_invalid()


def test_policy_can_change_after_successful_final_drain(gateway):
    assert _generate(gateway, user="r#first", prompt=[1, 2]).status_code == 200
    assert _report(gateway, _result(steps=[_step()])).status_code == 200
    gateway.drain_window(require_results=True)
    gateway.set_policy(
        replace(gateway._policy, model="other", expected_adapter=_adapter("other"))
    )
    assert _generate(gateway, model="other").status_code == 200


def test_legacy_flat_generation_is_rejected_before_sampling(gateway):
    response = requests.post(
        gateway.base_url + "/generate",
        json={
            "model": "stub-model",
            "user": "r#n",
            "prompt": [1],
            "max_tokens": 4,
            "skip_special_tokens": False,
            "arc3_lineage": _lineage(),
            **_sampling(),
        },
        timeout=5,
    )
    assert response.status_code == 400
    assert not _StubVllm.seen
    with pytest.raises(GatewayError, match="shared engine envelope"):
        gateway.raise_if_invalid()
