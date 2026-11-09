"""``thorough_summary`` asks for a thorough summary rather than a compact one,
states no size, and from the second summary on asks for what still holds of
the previous one (2026-09-23).

Off, the request is byte-identical whether or not a previous summary exists,
which is what the `compact_summary` golden renders. Built through the agent's own
``_prompt_modes()``, as every render is.
"""

from __future__ import annotations

from duck import DuckAgent
from duck.prompts import DuckPrompts

THOROUGH = (
    "Whatever you don't write in the summary is lost, so be thorough. Discuss everything you learned and "
    "the evidence behind it."
)
PREVIOUS = "Make sure to also take over anything that is still relevant from the previous summary."


def _agent(*, thorough: bool) -> DuckAgent:
    return DuckAgent(
        compaction_strategy="summarize", reply_cap_tokens=2000, prompts=DuckPrompts(thorough_summary=thorough)
    )


def _request(agent: DuckAgent, *, previous_summary: bool) -> str:
    return agent.prompts.compaction_prompt(
        agent._prompt_modes(),  # pyright: ignore[reportPrivateUsage]
        5632,
        current_level=1,
        previous_summary=previous_summary,
    )


def test_off_is_one_request_with_or_without_a_previous_summary() -> None:
    agent = _agent(thorough=False)
    first, later = _request(agent, previous_summary=False), _request(agent, previous_summary=True)
    assert first == later
    assert THOROUGH not in first and PREVIOUS not in first


def test_on_asks_for_thoroughness_and_from_the_second_summary_for_the_previous_one() -> None:
    agent = _agent(thorough=True)
    first, later = _request(agent, previous_summary=False), _request(agent, previous_summary=True)
    assert THOROUGH in first and PREVIOUS not in first
    assert f"{THOROUGH} {PREVIOUS} In this order:" in later
    assert "Write a thorough summary" in later and "compact summary" not in later
    assert "At most" not in later
    # The rest of the request is the compact one, those three changes apart.
    compact = _request(_agent(thorough=False), previous_summary=True)
    assert (
        later.replace(f" {THOROUGH} {PREVIOUS}", " At most ~5,632 tokens.").replace(
            "a thorough summary", "a compact summary"
        )
        == compact
    )
