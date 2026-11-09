"""Tufa ARC-AGI framework (TAAF) — the ARC-AGI2 orchestration layer.

No top-level re-exports — types live on their submodules
(``taaf.game.GameSpec``, not ``taaf.GameSpec``).

Submodules load on first use rather than at ``import taaf``. What forces
this is the cost of the ones nobody asked for: ``diagnostics`` pulls
matplotlib and scipy, and a process that only wants ``taaf.game`` pays
~80 MB of private heap for them. That heap is per-process, so a solver
hosting each game in a child process pays it again per child. ``taaf.X``
attribute access is unchanged; it just imports one module later.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-checkers only — the runtime path is ``__getattr__`` below.
    from taaf import (
        benchmark,
        datasets,
        deploy,
        deploy_inline,
        deploy_kaggle,
        deploy_slurm,
        diagnostics,
        game,
        game_examples,
        grid,
        solver,
        solver_examples,
        standard_benchmarks,
        submission,
        support,
        trace,
    )

__all__ = [
    "benchmark",
    "datasets",
    "deploy",
    "deploy_inline",
    "deploy_kaggle",
    "deploy_slurm",
    "diagnostics",
    "game",
    "game_examples",
    "grid",
    "solver",
    "solver_examples",
    "standard_benchmarks",
    "submission",
    "support",
    "trace",
]


def __getattr__(name: str) -> ModuleType:
    """Import a submodule on first attribute access (PEP 562).

    Only reached for a name not already bound, and ``import taaf.game``
    binds its own — so this runs at most once per submodule.
    """
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Every submodule, imported yet or not — ``dir(taaf)`` must not depend on load order."""
    return sorted({*__all__, *globals()})
