"""The training preset against a bare vLLM-shaped server — no gateway (U5).

The external column is deployed both behind the trainer gateway and
straight against vLLM; a gateway can quietly normalize a request, so what
these tests pin is the DIRECT exchange: a real benchmark episode over real
sockets against a fake server that speaks exactly vLLM 0.25.1's surface
(``/tokenize``, ``/v1/models``, chat completions with ``return_token_ids``,
and id-prompt ``/v1/completions``) and nothing more. The episode must reach
the continued call — the incident signature is a run that never produces a
single ``/v1/completions`` line — and that call's wire fields are asserted
key by key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

pytest.importorskip("taaf.benchmark", reason="needs the arc framework")

import arcengine
import taaf.benchmark
import ullm.factory
from taaf.game_examples import ExampleGame

from duck import DuckAgent
from platypus import LLMAgentSolver, RunConfig

EOS = "<|im_end|>"


class _StubTokenizer:
    """The continuation-test tokenizer: a Qwen-shaped template over
    characters, so the local delta render is deterministic without HF."""

    specials = {EOS: 1000, "<|im_start|>": 1001, "<think>": 1002, "</think>": 1003}
    eos_token = EOS
    eos_token_id = 1000

    def apply_chat_template(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, **kwargs: Any) -> str:
        out = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                body = "<think>\n" + (message.get("reasoning_content") or "") + "\n</think>\n\n"
                body += message.get("content") or ""
                out += "<|im_start|>assistant\n" + body + EOS + "\n"
            else:
                out += "<|im_start|>" + role + "\n" + str(message.get("content")) + EOS + "\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n<think>\n"
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for special, sid in self.specials.items():
                if text.startswith(special, i):
                    ids.append(sid)
                    i += len(special)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        by_id = {sid: special for special, sid in self.specials.items()}
        return "".join(by_id.get(i, chr(i)) for i in ids)


_TOKENIZER = _StubTokenizer()

# The Qwen function-markup dialect the client-side salvage parses on
# continued calls (the completions route has no server-side tool parser).
_ACT = (
    "<tool_call>\n<function=python>\n<parameter=code>\nsubmit_action(['UP'])\n</parameter>\n</function>\n</tool_call>"
)


def _prompt_weight(messages: list[dict[str, Any]]) -> int:
    """One rule for every route: the engine's token count of a rendered
    conversation, deterministic in its JSON length."""
    return max(3, len(json.dumps(messages)) // 3)


class _FakeVllm(BaseHTTPRequestHandler):
    """vLLM 0.25.1's OpenAI surface, minimally: every request is logged as
    (path, payload) on the server class for the test to assert against."""

    requests: list[tuple[str, dict[str, Any]]] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — the stdlib signature
        """Quiet: the request ledger is the log."""

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length)) if length else {}

    def _reply(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
        """Serve ``/v1/models`` (the tokenizer-identity probe)."""
        type(self).requests.append((self.path, {}))
        self._reply({"data": [{"id": "served-model", "root": "stub-root"}]})

    def do_POST(self) -> None:  # noqa: N802 — same
        """Serve tokenize, chat, and id-prompt completions."""
        payload = self._body()
        type(self).requests.append((self.path, payload))
        if self.path.endswith("/tokenize"):
            count = _prompt_weight(payload.get("messages") or [])
            self._reply({"count": count, "tokens": list(range(count)), "max_model_len": 40960})
            return
        if self.path.endswith("/v1/chat/completions"):
            # Billed exactly what /tokenize says for the same messages: a
            # coherent engine, so both eras' reconciliations hold.
            prompt_ids = list(range(_prompt_weight(payload.get("messages") or [])))
            output_ids = _TOKENIZER.encode("\n</think>\n\nacting") + [_TOKENIZER.eos_token_id]
            self._reply(
                {
                    "id": "chat",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "served-model",
                    "prompt_token_ids": prompt_ids,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "token_ids": output_ids,
                            "message": {
                                "role": "assistant",
                                "content": "acting",
                                "reasoning": "",
                                "tool_calls": [
                                    {
                                        "id": "t1",
                                        "type": "function",
                                        "function": {
                                            "name": "python",
                                            "arguments": json.dumps({"code": "submit_action(['UP'])\n"}),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(output_ids),
                        "total_tokens": len(prompt_ids) + len(output_ids),
                    },
                }
            )
            return
        assert self.path.endswith("/v1/completions"), f"unexpected route {self.path}"
        sent = [int(t) for t in payload["prompt"]]
        text = "\n</think>\n\n" + _ACT
        output_ids = _TOKENIZER.encode(text) + [_TOKENIZER.eos_token_id]
        self._reply(
            {
                "id": "cmpl",
                "object": "text_completion",
                "created": 0,
                "model": "served-model",
                "choices": [
                    {
                        "index": 0,
                        "text": text + EOS,
                        "finish_reason": "stop",
                        "logprobs": None,
                        "token_ids": output_ids,
                        "prompt_token_ids": sent,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(sent),
                    "completion_tokens": len(output_ids),
                    "total_tokens": len(sent) + len(output_ids),
                },
            }
        )


async def test_direct_vllm_episode_reaches_the_continued_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole episode, gatewayless: probe, engine-rendered first call,
    then the id-prompt continuation — which must actually arrive at
    ``/v1/completions`` (the incident signature is that it never does),
    carrying exactly the wire fields vLLM 0.25.1 accepts."""
    from ullm.backends import oai

    _FakeVllm.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVllm)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def _stub_tokenizer(self: Any) -> _StubTokenizer:
        return _TOKENIZER

    monkeypatch.setattr(oai.OpenAIServer, "_trace_hf_tokenizer_get", _stub_tokenizer)
    try:
        # the external column's client wiring, verbatim (the preset module
        # itself imports the vllm extra, which the cheap tier does not have;
        # the preset's own field values are pinned by the preset tests).
        backend = oai.OpenAIServer(
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            api_key="unused",
            model="served-model",
            streaming=False,
            parse_think_tags=True,
            reasoning_history_field="reasoning",
            token_limit_param="max_completion_tokens",
            chat_template_kwargs={"preserve_thinking": True, "enable_thinking": True},
        )
        import ullm.backend as ullm_backend

        from platypus import LLMConfig

        agent = DuckAgent(
            derive_max_model_len=False,
            derive_max_num_seqs=False,
            derive_max_pixels=False,
            reply_cap_tokens=64,
            token_native_context=True,
            max_moves=6,
            multimodal_context="",
            # This server speaks the stock OpenAI-compatible route, which
            # rejects the fork's zero_recurrent_* sampling parameters. Any
            # deployment that is not the arc-prize-2026-vllm fork turns the blank
            # off, and the key-by-key assertion below is what holds that
            # line: the request carries what the route accepts and nothing
            # else.
            zero_recurrent=False,
            # Same line, drawn once more: this route cannot serve text after
            # the generation prompt either, and refuses rather than dropping
            # it, so a non-fork deployment opts out of the register too.
            inject_text_on_first_prompt="",
            inject_text_on_later_prompts="",
            llm_config=LLMConfig(
                backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
                sampling_profiles={
                    "play": ullm_backend.GenericSamplingParameters(
                        temperature=0.6,
                        top_p=0.89,
                        max_tokens=None,
                        extra_params={"top_k": 194, "presence_penalty": 1.5},
                    )
                },
                exact_history=True,
            ),
            run_config=RunConfig(rng_seed=99, on_agent_error="crash_game", process_mode="inprocess"),
        )
        benchmark = taaf.benchmark.Benchmark(
            games=[ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1] * 4], label="direct_game")],
            solver=LLMAgentSolver(agent=agent),
            n_passes=1,
        )
        await benchmark.run()
        (run,) = benchmark.game_runs
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert run.state == "won", f"the direct episode must play out: {run.state} / {run.solver_note}"
    generation_calls = [path for path, _ in _FakeVllm.requests if "completions" in path]
    assert len(generation_calls) <= 8, f"the episode churned: {len(generation_calls)} generation calls"
    continued = [payload for path, payload in _FakeVllm.requests if path.endswith("/v1/completions")]
    assert continued, (
        f"no /v1/completions request ever reached the server — the incident signature; "
        f"routes seen: {[path for path, _ in _FakeVllm.requests]}"
    )

    # The continued request, key by key: what vLLM 0.25.1's completions
    # route accepts, and nothing else — no chat-only fields, none of the
    # client's private bookkeeping.
    request = continued[0]
    prompt_field: list[Any] = request["prompt"]
    assert isinstance(prompt_field, list) and all(isinstance(t, int) for t in prompt_field)
    assert request["model"] == "served-model"
    extra_keys = set(request) - {"model", "prompt", "temperature", "top_p", "stop", "seed", "max_tokens"} - set()
    assert extra_keys <= {"top_k", "presence_penalty", "return_token_ids", "skip_special_tokens"}, extra_keys
    assert request["return_token_ids"] is True
    assert request["skip_special_tokens"] is False
    assert "messages" not in request and not any(k.startswith("_") for k in request)
    assert "chat_template_kwargs" not in request, "a template kwarg on an id prompt would re-render it"
    assert request["max_tokens"] <= 64

    # Append-only: every continued prompt extends the previous one.
    prompts = [payload["prompt"] for path, payload in _FakeVllm.requests if path.endswith("/v1/completions")]
    for earlier, later in zip(prompts, prompts[1:]):
        assert later[: len(earlier)] != [] and len(later) > len(earlier)


if __name__ == "__main__":
    pytest.main([__file__])
