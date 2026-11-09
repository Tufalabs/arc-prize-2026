"""A token a request bans with logit bias -100 never reaches a completion
unnoticed: the vLLM backend checks each finished output and fails the call."""

from __future__ import annotations

import pytest
import ullm.backend
import vllm
from ullm.backends import vllmx


def test_only_a_bias_at_the_floor_is_a_ban() -> None:
    params = vllm.SamplingParams(logit_bias={5: -100.0, 6: -5.0, 7: 100.0})
    assert vllmx._banned_token_ids(params) == frozenset({5})
    assert vllmx._banned_token_ids(vllm.SamplingParams()) == frozenset()


def test_a_banned_token_in_the_completion_fails_the_call() -> None:
    with pytest.raises(ullm.backend.ForbiddenTokenError, match=r"\[5\]"):
        vllmx._check_banned_tokens(frozenset({5}), [1, 2, 5, 3])


def test_a_clean_completion_passes() -> None:
    vllmx._check_banned_tokens(frozenset({5}), [1, 2, 3])
    vllmx._check_banned_tokens(frozenset(), [5])


def test_the_failure_is_not_retried_as_malformed_output() -> None:
    assert not issubclass(ullm.backend.ForbiddenTokenError, ullm.backend.MalformedOutputError)
