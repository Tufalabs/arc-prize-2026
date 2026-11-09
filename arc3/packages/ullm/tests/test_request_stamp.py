"""The traced-request run stamp: an edge-capture gateway groups calls by
the OpenAI ``user`` field (``run_id#nonce``), so the oai backend must
stamp every prepared request while tracing, with a fresh nonce per
logical call, and leave the rest of the request untouched."""

from ullm.backend import Backend
from ullm.backends.oai import OpenAIServer


def _server() -> OpenAIServer:
    return OpenAIServer(base_url="http://127.0.0.1:1/v1", model="stub")


def test_oai_stamp_carries_run_and_fresh_nonce() -> None:
    server = _server()
    server.trace_recorder = [].append
    llm_input = {"model": "stub", "messages": [{"role": "user", "content": "go"}]}
    first = server._stamp_llm_input(llm_input, "run-abc")
    second = server._stamp_llm_input(llm_input, "run-abc")
    run_id, _, nonce = first["user"].rpartition("#")
    assert run_id == "run-abc" and nonce
    # A fresh nonce per logical call: retries re-send the SAME stamped
    # input, so distinct stamp() calls must never collide.
    assert first["user"] != second["user"]
    # The original request is not mutated, and nothing else changes.
    assert "user" not in llm_input
    assert {k: v for k, v in first.items() if k != "user"} == llm_input


def test_untraced_oai_attribution_does_not_change_the_request() -> None:
    server = _server()
    llm_input = {"model": "stub", "user": "caller-provided"}
    assert server._stamp_llm_input(llm_input, "run-abc") is llm_input
    assert llm_input["user"] == "caller-provided"


def test_base_backend_stamp_is_a_no_op() -> None:
    llm_input = object()
    assert Backend._stamp_llm_input(None, llm_input, "run-abc") is llm_input  # type: ignore[arg-type]


def test_render_template_kwargs_folds_native_reasoning_effort() -> None:
    """A /tokenize re-render must see the same reasoning effort the chat
    request rendered with: the API param has no /tokenize counterpart, so
    it is folded into chat_template_kwargs (explicit kwarg wins)."""
    from ullm.backends.oai import _render_template_kwargs

    assert _render_template_kwargs({"messages": []}) is None
    assert _render_template_kwargs({"reasoning_effort": "medium"}) == {"reasoning_effort": "medium"}
    merged = _render_template_kwargs(
        {"reasoning_effort": "low", "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    )
    assert merged == {"enable_thinking": True, "reasoning_effort": "low"}
    explicit = _render_template_kwargs(
        {"reasoning_effort": "low", "extra_body": {"chat_template_kwargs": {"reasoning_effort": "xhigh"}}}
    )
    assert explicit == {"reasoning_effort": "xhigh"}
