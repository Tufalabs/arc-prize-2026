"""Agent contract and hosting machinery for ARC-AGI2.

Two class pairs: the LLM-free ``Agent`` / ``AgentSolver`` and the LLM-backed
``LLMAgent`` / ``LLMAgentSolver``. All configuration lives on the agent; the
solver classes hold none.

TAAF's value types are re-exported here so an agent has one import: it reads
a ``GameView`` of ``Grid``s and answers with a ``Submission``, and never
touches TAAF's hosting types.
"""

from platypus.agent import Agent
from platypus.context import MoveContext, SubmissionView
from platypus.contract import (
    ContractViolationError,
    GameOutcome,
    GiveUp,
    GiveUpError,
    RunConfig,
)
from platypus.llm import LLMAgent, LLMConfig, LLMMoveContext, grid_image
from platypus.llm_solver import LLMAgentSolver
from platypus.solver import AgentSolver
from taaf.game import N_ATTEMPTS, GameView, Pair, Submission
from taaf.grid import Grid

__all__ = [
    "N_ATTEMPTS",
    "Agent",
    "AgentSolver",
    "ContractViolationError",
    "GameOutcome",
    "GameView",
    "GiveUp",
    "GiveUpError",
    "Grid",
    "LLMAgent",
    "LLMAgentSolver",
    "LLMConfig",
    "LLMMoveContext",
    "MoveContext",
    "Pair",
    "RunConfig",
    "Submission",
    "SubmissionView",
    "grid_image",
]
