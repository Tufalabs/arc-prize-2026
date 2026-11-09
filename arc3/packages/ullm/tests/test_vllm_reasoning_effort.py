"""Reasoning effort on the in-process vLLM path.

Until ``native_reasoning_effort`` existed, ``SystemPrompt.reasoning_effort``
never reached this backend at all — the value rode the conversation and did
nothing. Qwen3.8's chat template reads ``reasoning_effort`` as a kwarg, so
the flag makes the setting real; these pin both directions of that seam.

The byte-stability half matters most: every committed reference was recorded
with the flag off, so template kwargs must be provably identical whatever the
effort value says whenever it is off.
"""

from __future__ import annotations

from typing import Any

import pytest
import ullm.backend
import ullm.conv

vllmx = pytest.importorskip("ullm.backends.vllmx", reason="needs the vllm extra (linux)")


class _RecordingTokenizer:
    """Captures the kwargs the backend hands to ``apply_chat_template``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        """Record and return a token id list shaped like the real thing."""
        del messages
        self.calls.append(kwargs)
        return [1, 2, 3]


class _FakeEngine:
    """Only what ``_prepare_input`` touches: ``get_tokenizer``."""

    def __init__(self, tokenizer: _RecordingTokenizer) -> None:
        self._tokenizer = tokenizer

    def get_tokenizer(self) -> _RecordingTokenizer:
        """The recording tokenizer above."""
        return self._tokenizer


def _template_kwargs(*, native: bool, effort: Any) -> dict[str, Any]:
    """The kwargs one render produces under the given flag and effort."""
    tokenizer = _RecordingTokenizer()
    backend = vllmx.VLLM(
        engine=_FakeEngine(tokenizer),  # type: ignore[arg-type]
        chat_template_kwargs={"preserve_thinking": True},
        native_reasoning_effort=native,
    )
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), reasoning_effort=effort),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go"))],
    )
    turn = ullm.conv.ResponseTurn()
    backend._prepare_input(
        ullm.backend.GenericSamplingParameters(temperature=1.0, max_tokens=10),
        turn,
        conversation,
        send_back_reasoning="all",
    )
    (kwargs,) = tokenizer.calls
    kwargs.pop("add_generation_prompt", None)
    kwargs.pop("return_dict", None)
    return kwargs


def test_off_means_the_effort_value_cannot_move_a_byte() -> None:
    """With the flag off, every effort renders the same template kwargs.

    This is what keeps every existing recording valid: qwen3.6 configs never
    set the flag, so a conversation's effort value — whatever it says — is
    invisible to the token stream.
    """
    baseline = _template_kwargs(native=False, effort="high")
    for effort in ("low", "medium", "max", "xhigh", None):
        assert _template_kwargs(native=False, effort=effort) == baseline
    assert baseline == {"preserve_thinking": True}


def test_the_models_own_levels_pass_through_verbatim() -> None:
    """low/medium/xhigh are the model's vocabulary — no translation."""
    for effort in ("low", "medium", "xhigh"):
        kwargs = _template_kwargs(native=True, effort=effort)
        assert kwargs["reasoning_effort"] == effort
        assert "enable_thinking" not in kwargs


def test_none_disables_thinking_instead_of_naming_a_level() -> None:
    """None means what it means everywhere else: no thinking at all."""
    kwargs = _template_kwargs(native=True, effort=None)
    assert kwargs["enable_thinking"] is False
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("effort", ["high", "max"])
def test_levels_the_model_does_not_have_are_refused(effort: str) -> None:
    """ "high" and "max" are not on this model's scale, and reinterpreting
    them would silently run a different effort than the caller named."""
    with pytest.raises(ValueError, match="low.*medium.*xhigh"):
        _template_kwargs(native=True, effort=effort)


def test_harmony_refuses_levels_it_does_not_have() -> None:
    """No backend silently remaps a named level — raise, not clamp.

    Harmony's scale is low/medium/high; "max" and "xhigh" used to be
    approximated to HIGH, which ran a different effort than the caller
    named. The OAI client is verbatim for the same reason: it cannot know
    whether the far side is OpenAI (no xhigh) or a server hosting Qwen3.8
    (requires it), so the server's own rejection is the error surface.
    """
    assert set(vllmx._REASONING_EFFORT_MAP) == {"low", "medium", "high"}
