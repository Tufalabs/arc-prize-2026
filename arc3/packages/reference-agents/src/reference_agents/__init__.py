"""Reference/baseline agents: installable (deploy pickles agents by module
path) and reused by tests via small determinism knobs instead of parallel
scaffolding."""

from reference_agents.prompted_random_agent import PromptedRandomAgent
from reference_agents.random_agent import RandomAgent

__all__ = ["PromptedRandomAgent", "RandomAgent"]
