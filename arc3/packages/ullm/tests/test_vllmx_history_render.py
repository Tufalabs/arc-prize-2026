"""vllmx history rendering: past reasoning reaches the chat template as the
structured ``reasoning_content`` field with reasoning-free content, so a
think-style template (Qwen3.x) re-renders a turn the way it was sampled
(requirements U11, and the U5 nesting invariant on the in-process column).
No engine — ``VLLM._history_messages`` is pure over the parsers."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import ullm.backend
import ullm.conv

vllmx = pytest.importorskip("ullm.backends.vllmx")


class _ThinkParser:
    """The shape of vLLM's qwen3 reasoning parser this path relies on: the
    output text carries no opening tag (the generation prompt pre-filled it)
    and ``</think>`` closes the reasoning span. An output with no closing
    tag is read by construction-time ``enable_thinking``, as vLLM's is: all
    reasoning when the prompt opened a block the model never closed, all
    content when the prompt opened none."""

    def __init__(self, *, enable_thinking: bool = True) -> None:
        self.enable_thinking = enable_thinking

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
        del request
        if "</think>" not in text:
            return (text, "") if self.enable_thinking else (None, text)
        reasoning, content = text.split("</think>", 1)
        return reasoning, content


def _request_output(text: str) -> Any:
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def _backend(*, reasoning_parser: Any, native_reasoning_effort: bool = False) -> Any:
    backend = vllmx.VLLM.__new__(vllmx.VLLM)
    # Only the fields _history_messages reads; no engine, no tokenizer. The
    # no-thinking parser is what setup would build beside the configured one.
    backend.__dict__.update(
        reasoning_effort_prompts=None,
        chat_template_kwargs={},
        native_reasoning_effort=native_reasoning_effort,
        _reasoning_parser=reasoning_parser,
        _reasoning_lock=threading.Lock(),
        _reasoning_parser_no_thinking=_ThinkParser(enable_thinking=False) if reasoning_parser else None,
        _tool_parser=object(),
    )
    return backend


def _conversation(
    sampled: str, *, reasoning_effort: Any = "medium", thinking: bool = True
) -> tuple[ullm.conv.Conversation, ullm.conv.ResponseTurn]:
    """A past answer stored the way ``_process_output`` stores it: split by
    the parser that matched its render (``thinking``), the raw text kept."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(
            content=ullm.conv.StringContent(text="sys"), reasoning_effort=reasoning_effort
        )
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    reasoning, content = _ThinkParser(enable_thinking=thinking).extract_reasoning(sampled)
    response = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=reasoning.strip()) if reasoning else ullm.conv.EmptyContent(),
        final_content=ullm.conv.StringContent(text=(content or "").strip()),
    )
    response._model_contents = _request_output(sampled)
    past = ullm.conv.ResponseTurn()
    past.elements.append(response)
    conversation.elements.append(past)
    turn = ullm.conv.ResponseTurn()
    conversation.elements.append(turn)
    return conversation, turn


SAMPLED = "First, look at the board.\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n1+1\n</parameter>\n</function>\n</tool_call>"


def test_past_reasoning_rides_the_template_field_with_reasoning_free_content() -> None:
    conversation, turn = _conversation(SAMPLED)
    messages, images = _backend(reasoning_parser=_ThinkParser())._history_messages(turn, conversation, "all")
    assert images == []
    (assistant,) = [m for m in messages if m["role"] == "assistant"]
    assert assistant["reasoning_content"] == "First, look at the board."
    assert assistant["content"].startswith("<tool_call>"), assistant["content"]
    assert "</think>" not in assistant["content"]


def test_stripped_reasoning_leaves_no_field_and_no_think_markup() -> None:
    conversation, turn = _conversation(SAMPLED)
    messages, _ = _backend(reasoning_parser=_ThinkParser())._history_messages(turn, conversation, "none")
    (assistant,) = [m for m in messages if m["role"] == "assistant"]
    assert "reasoning_content" not in assistant
    assert "</think>" not in assistant["content"] and "First" not in assistant["content"]


def test_without_a_reasoning_parser_the_raw_text_is_the_history() -> None:
    """A model whose template has no think block gets its own text back verbatim."""
    conversation, turn = _conversation("plain answer")
    messages, _ = _backend(reasoning_parser=None)._history_messages(turn, conversation, "all")
    (assistant,) = [m for m in messages if m["role"] == "assistant"]
    assert assistant == {"role": "assistant", "content": "plain answer"}


NO_THINK_SAMPLED = "1. Hypotheses tested.\n2. Best candidate: def fn(g): return g[::-1]"


def test_a_request_rendered_without_thinking_is_split_by_a_parser_that_knows_it() -> None:
    """A no-thinking request on a thinking backend: the template pre-fills an
    empty think block, so the output carries no ``</think>`` — which the
    backend's own parser reads as reasoning that was never closed. Read that
    way, the whole answer lands in ``reasoning_content`` and re-renders
    inside a think block the prompt never opened, so the next prompt is not
    a suffix of what was served. The parser has to follow the render."""
    conversation, turn = _conversation(NO_THINK_SAMPLED, reasoning_effort=None, thinking=False)
    backend = _backend(reasoning_parser=_ThinkParser(), native_reasoning_effort=True)
    messages, _ = backend._history_messages(turn, conversation, "all")
    (assistant,) = [m for m in messages if m["role"] == "assistant"]
    assert not assistant.get("reasoning_content"), assistant
    # The content is re-extracted from the raw text at render time, so a
    # parser that reads the whole output as reasoning would leave it empty
    # even with the stored split right.
    assert assistant["content"] == NO_THINK_SAMPLED


def test_the_same_output_under_thinking_is_still_reasoning() -> None:
    """The other half of the disambiguation: with thinking on, an output with
    no closing tag IS an unfinished think block, and must stay one."""
    conversation, turn = _conversation(NO_THINK_SAMPLED, reasoning_effort="medium")
    backend = _backend(reasoning_parser=_ThinkParser(), native_reasoning_effort=True)
    messages, _ = backend._history_messages(turn, conversation, "all")
    (assistant,) = [m for m in messages if m["role"] == "assistant"]
    assert assistant["reasoning_content"] == NO_THINK_SAMPLED
    assert assistant["content"] == ""


def test_without_the_native_effort_path_the_backend_default_decides() -> None:
    """``reasoning_effort=None`` reaches the template only under
    ``native_reasoning_effort``; elsewhere it is inert, and the parser must
    stay the configured one rather than switching on a field nothing renders."""
    conversation, turn = _conversation(NO_THINK_SAMPLED, reasoning_effort=None)
    backend = _backend(reasoning_parser=_ThinkParser(), native_reasoning_effort=False)
    messages, _ = backend._history_messages(turn, conversation, "all")
    (assistant,) = [m for m in messages if m["role"] == "assistant"]
    assert assistant["reasoning_content"] == NO_THINK_SAMPLED


@pytest.mark.parametrize("partial", [False, True])
def test_history_and_output_share_serialized_reasoning_parser_access(partial: bool) -> None:
    """Counting must not race a stateful parser used by live response handling."""

    class StatefulParser(_ThinkParser):
        def __init__(self) -> None:
            super().__init__()
            self.current = ""
            self.active = 0
            self.peak = 0

        def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                self.current = text
                time.sleep(0.002)  # permit the other thread to enter this stateful parser
                return super().extract_reasoning(self.current, request)
            finally:
                self.active -= 1

        def count_reasoning_tokens(self, token_ids: Any) -> int:
            return 0

    parser = StatefulParser()
    backend = _backend(reasoning_parser=parser)
    conversation, turn = _conversation("history reasoning</think>history content")
    barrier = threading.Barrier(2)

    def histories() -> list[str]:
        barrier.wait(timeout=5)
        return [backend._history_messages(turn, conversation, "all")[0][-1]["content"] for _ in range(40)]

    def outputs() -> list[tuple[str, str]]:
        barrier.wait(timeout=5)
        results = []
        for _ in range(40):
            element = ullm.conv.LLMResponse(token_usage=ullm.backend.TokenUsage())
            text = "new reasoning</think>new content"
            if partial:
                backend._process_partial_output(
                    ullm.backend.GenericSamplingParameters(), _request_output(text), element
                )
            else:
                backend._parse_model_content(conversation, element, text, [])
            results.append((element.reasoning_content.to_string(), element.final_content.to_string()))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        history_future = executor.submit(histories)
        output_future = executor.submit(outputs)
        assert history_future.result(timeout=10) == ["history content"] * 40
        assert output_future.result(timeout=10) == [("new reasoning", "new content")] * 40
    assert parser.peak == 1
