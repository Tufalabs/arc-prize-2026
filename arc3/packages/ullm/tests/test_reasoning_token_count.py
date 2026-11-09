"""Reasoning-token counting when the chat template pre-fills ``<think>``.

vLLM's own counter walks from an OPENING think token and counts while
nested. Qwen3's template pre-fills that tag into the prompt, so the model
generates only the closing tag and the counter returns 0 for every response
— which silently zeroes reasoning accounting on the duck's primary backend.
"""

from typing import Any, Sequence

import pytest
from ullm.backends.vllmx import _count_reasoning_tokens

START, END = 100, 101


class _VllmLikeParser:
    """Mirrors vllm.reasoning.basic_parsers: count only while nested."""

    start_token_id = START
    end_token_id = END

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        count = depth = 0
        for t in token_ids:
            if t == START:
                depth += 1
            elif t == END:
                depth = max(0, depth - 1)
            elif depth > 0:
                count += 1
        return count


def test_prefilled_open_tag_still_counts() -> None:
    """The Qwen3 layout: no generated ``<think>``, reasoning runs up to the
    closing tag. vLLM's counter says 0; the real answer is 3."""
    parser: Any = _VllmLikeParser()
    token_ids = [7, 8, 9, END, 20, 21]
    assert parser.count_reasoning_tokens(token_ids) == 0
    assert _count_reasoning_tokens(parser, token_ids) == 3


def test_parser_count_wins_when_it_found_something() -> None:
    """With a generated opening tag the parser handles nesting and multiple
    spans, which the positional fallback cannot — so it must take priority."""
    parser: Any = _VllmLikeParser()
    token_ids = [START, 1, 2, END, 5, START, 6, END]
    assert _count_reasoning_tokens(parser, token_ids) == parser.count_reasoning_tokens(token_ids) == 3


def test_no_end_token_counts_nothing() -> None:
    """Nothing closed the reasoning span, so no span is countable — better
    than charging the whole response as reasoning."""
    parser: Any = _VllmLikeParser()
    assert _count_reasoning_tokens(parser, [1, 2, 3]) == 0


def test_parser_without_end_token_id_is_tolerated() -> None:
    class _NoTokens:
        def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
            return 0

    assert _count_reasoning_tokens(_NoTokens(), [1, 2, END]) == 0


if __name__ == "__main__":
    pytest.main([__file__])
