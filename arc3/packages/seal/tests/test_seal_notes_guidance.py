"""``SealPrompts.notes_guidance``: what the solving guidance says files are for besides code.

Its wording is pinned by the ``notes_encourage`` and ``notes_discourage`` prompt
goldens. The journal refuses it: the journal's own lines already say where
findings go and what files are for, and a second say on it would contradict them.
"""

from __future__ import annotations

from typing import Literal

import pytest

from seal import SealAgent
from seal.prompts import SealPrompts


@pytest.mark.parametrize("guidance", ["encourage", "discourage"])
def test_the_journal_refuses_it(guidance: Literal["encourage", "discourage"]) -> None:
    agent = SealAgent(journal=True, level_reflection="none", prompts=SealPrompts(notes_guidance=guidance))
    with pytest.raises(ValueError, match="notes_guidance=.*requires journal=False"):
        agent.prepare_for_backend()


def test_without_the_journal_it_is_accepted() -> None:
    SealAgent(journal=False, prompts=SealPrompts(notes_guidance="encourage")).prepare_for_backend()
