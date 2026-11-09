"""The trace: what was observed on a level, and the model's predicted state
after each observation.

A value the search pickles into its subprocess alongside the model — and
pointedly *not* the conversation. The seal builds one from the level's
history and the state chain a passing ``validate_world_model`` produced
(:func:`seal.search.level_trace`); nothing here checks the model, because
that replay already did.

Steps hold plain frames rather than ``taaf.game.GameState``, which keeps the
trace picklable, diffable and free of the substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt

from platypus.contract import Action
from seal.model import Outcome


@dataclass(frozen=True)
class Step:
    """One observation: the action that caused it (``None`` for the
    level-initial observation) and what came back."""

    action: Action | None
    frame: npt.NDArray[np.int8]
    outcome: Outcome

    @property
    def is_reset(self) -> bool:
        """Whether this step's action was a RESET."""
        return self.action is not None and self.action.name == "RESET"


@dataclass
class LevelTrace:
    """Observations for one level, plus the model's predicted state at each.

    ``states`` is parallel to ``steps``: ``states[i]`` is the model's state
    after ``steps[i]``.
    """

    steps: list[Step] = field(default_factory=list["Step"])
    states: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def __len__(self) -> int:
        return len(self.steps)

    def current_state(self) -> dict[str, Any]:
        """The model's predicted state after the last observation."""
        if not self.states:
            raise RuntimeError("no predicted state yet — observe the level-initial step first")
        return self.states[-1]

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)
