"""Platypus infrastructure layer: the agent contract and hosting
machinery (requirements.md). Two class pairs (§1):
the LLM-free ``Agent``/``AgentSolver`` and the LLM-backed
``LLMAgent``/``LLMAgentSolver`` (which pull in the ullm member's
CPU-only core; GPU serving stays behind ullm's ``vllm`` extra)."""

from platypus.agent import Agent
from platypus.context import HistoryView, MoveContext
from platypus.contract import (
    Action,
    ContractViolationError,
    GameInfo,
    GameResult,
    GiveUp,
    GiveUpError,
    Move,
    Observation,
    Outcome,
    RunConfig,
    action_names,
)
from platypus.llm import LLMAgent, LLMConfig, LLMMoveContext, frame_image
from platypus.llm_solver import LLMAgentSolver
from platypus.solver import AgentSolver

__all__ = [
    "Action",
    "Agent",
    "AgentSolver",
    "ContractViolationError",
    "GameInfo",
    "GameResult",
    "GiveUp",
    "GiveUpError",
    "HistoryView",
    "LLMAgent",
    "LLMAgentSolver",
    "LLMConfig",
    "LLMMoveContext",
    "Move",
    "MoveContext",
    "Observation",
    "Outcome",
    "RunConfig",
    "action_names",
    "frame_image",
]
