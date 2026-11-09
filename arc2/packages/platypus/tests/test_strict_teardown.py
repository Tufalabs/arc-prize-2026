"""Training teardown failures invalidate a run without hiding its primary error."""

from dataclasses import dataclass, field
from typing import Any, cast

import pytest
import ullm.backend

from platypus.llm_solver import LLMAgentSolver


@dataclass
class FailingTeardownSolver(LLMAgentSolver):
    closed: list[str] = field(default_factory=list[str])

    async def _close_server(self) -> None:
        self.closed.append("server")
        raise OSError("socket close failed")

    async def _close_factory(self) -> None:
        self.closed.append("factory")
        raise RuntimeError("factory close failed")

    async def _close_backend(self) -> None:
        self.closed.append("backend")


async def test_strict_teardown_attempts_all_resources_before_failing() -> None:
    solver = FailingTeardownSolver(agent=cast(Any, None), strict_teardown=True)
    solver._usage_before_games = ullm.backend.TokenUsage()
    with pytest.raises(RuntimeError, match="backend teardown failed"):
        await solver._teardown_backend()
    assert solver.closed == ["server", "factory", "backend"]
    assert len(solver.teardown_errors) == 2
    assert "socket close failed" in solver.teardown_errors[0]
    assert solver._usage_before_games is None


async def test_primary_failure_survives_strict_teardown() -> None:
    solver = FailingTeardownSolver(agent=cast(Any, None), strict_teardown=True)
    with pytest.raises(ValueError, match="primary failure"):
        try:
            raise ValueError("primary failure")
        finally:
            await solver._teardown_backend()
    assert len(solver.teardown_errors) == 2


async def test_standalone_teardown_keeps_best_effort_behavior() -> None:
    solver = FailingTeardownSolver(agent=cast(Any, None))
    await solver._teardown_backend()
    assert solver.closed == ["server", "factory", "backend"]
