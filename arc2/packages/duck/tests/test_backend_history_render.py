"""What the duck's own history looks like by the time the chat template sees it.

The gap this closes: the prompt golden pins the prompt objects the duck
builds, and a replay reference pins the request bytes one recorded run
produced — neither can see what the BACKEND does with a past answer on the
way back into the model. That is where a real defect lived: with a reasoning
parser the sampled text carries no opening ``<think>`` (the generation prompt
pre-filled it), so handing the raw text back as ``content`` made the Qwen
template render every prior round as an empty think block, the reasoning as
content, and a stray ``</think>``. Every round after the first, on every run,
and both instruments replayed green through it.

So this asserts the two halves that together decide the rendering: that our
engine config names the qwen3 reasoning parser at all, and that vllmx routes
past reasoning through the structured field the template renders inside the
think block. ``_history_messages`` is pure over the parsers — no engine, no
tokenizer, no GPU.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
import ullm.conv
from duck import DuckAgent

vllmx = pytest.importorskip("ullm.backends.vllmx")


class _ThinkParser:
    """The shape of vLLM's qwen3 reasoning parser this path relies on: the
    output text carries no opening tag, and ``</think>`` closes the span."""

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
        del request
        if "</think>" not in text:
            return text, ""
        reasoning, content = text.split("</think>", 1)
        return reasoning, content


#: One sampled answer in the duck's own shape: reasoning, then a tool call in
#: Qwen3's native XML — the markup that must survive back into the history.
SAMPLED = (
    "The colours look mirrored. Check the arrays.\n"
    "</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\nprint(train[0])\n"
    "</parameter>\n</function>\n</tool_call>"
)


def vllm_backend() -> Any:
    """A VLLM with only the fields ``_history_messages`` reads (also what
    ``test_context_summarization`` renders a rebuilt history with)."""
    backend = vllmx.VLLM.__new__(vllmx.VLLM)
    backend.__dict__.update(
        reasoning_effort_prompts=None,
        chat_template_kwargs={},
        native_reasoning_effort=False,
        _reasoning_parser=_ThinkParser(),
        _reasoning_lock=threading.Lock(),
        _reasoning_parser_no_thinking=None,
        _tool_parser=object(),
    )
    return backend


def _conversation() -> tuple[ullm.conv.Conversation, ullm.conv.ResponseTurn]:
    """A game's first round answered, and the second round about to render."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="the task")))
    reasoning, _, content = SAMPLED.partition("</think>")
    response = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=reasoning.strip()),
        final_content=ullm.conv.StringContent(text=content.strip()),
    )
    response._model_contents = SimpleNamespace(outputs=[SimpleNamespace(text=SAMPLED)])
    past = ullm.conv.ResponseTurn()
    past.elements.append(response)
    past.elements.append(
        ullm.conv.ToolResult(call_id="c0", tool_name="python", content=ullm.conv.StringContent(text="[[1, 2]]"))
    )
    conversation.elements.append(past)
    turn = ullm.conv.ResponseTurn()
    conversation.elements.append(turn)
    return conversation, turn


def _assistant(send_back_reasoning: str) -> dict[str, Any]:
    conversation, turn = _conversation()
    messages, images = vllm_backend()._history_messages(turn, conversation, send_back_reasoning)
    assert images == []
    (assistant,) = [m for m in messages if m["role"] == "assistant"]
    return assistant


def test_past_reasoning_rides_the_structured_field_and_leaves_the_content_clean() -> None:
    assistant = _assistant("all")
    assert assistant["reasoning_content"] == "The colours look mirrored. Check the arrays."
    assert assistant["content"].startswith("<tool_call>"), assistant["content"]
    assert "</think>" not in assistant["content"]


def test_the_tool_call_survives_in_the_models_native_markup() -> None:
    """Not the parsed call's Python repr, which is what the model would parrot."""
    assert "<function=python>" in _assistant("all")["content"]


@pytest.mark.parametrize("environment", ["slurm-b200", "kaggle-4xl4"])
def test_the_gpu_columns_name_the_reasoning_parser_the_rendering_assumes(environment: str) -> None:
    factory: Any = DuckAgent.build(environment=environment).llm_config.backend_factory
    assert factory.engine_args.reasoning_parser == "qwen3"
