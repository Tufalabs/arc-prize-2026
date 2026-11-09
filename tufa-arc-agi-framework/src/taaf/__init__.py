"""ARC-AGI3 orchestration framework (TAAF) — Tufa Labs.

No top-level re-exports — types live on their submodules
(``taaf.game.GameState``, not ``taaf.GameState``).

Submodules load on first use rather than at ``import taaf``. What forces
this is the cost of the ones nobody asked for: ``diagnostics`` pulls
matplotlib and scipy, ``competition_arcade`` pulls ``arc_agi``, and a
process that only wants ``taaf.game`` pays ~80 MB of private heap for
them. That heap is per-process, so a solver hosting each game in a child
process pays it again per child. ``taaf.X`` attribute access is
unchanged; it just imports one module later.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-checkers only — the runtime path is ``__getattr__`` below.
    # ``as <name>`` aliases plus ``__all__`` mark these as intentional
    # re-exports for pyright strict (otherwise: reportUnusedImport on each line).
    import taaf.benchmark as benchmark
    import taaf.competition_arcade as competition_arcade
    import taaf.deploy as deploy
    import taaf.deploy_inline as deploy_inline
    import taaf.deploy_kaggle as deploy_kaggle
    import taaf.deploy_slurm as deploy_slurm
    import taaf.diagnostics as diagnostics
    import taaf.game as game
    import taaf.game_api as game_api
    import taaf.game_examples as game_examples
    import taaf.solver as solver
    import taaf.solver_examples as solver_examples
    import taaf.standard_benchmarks as standard_benchmarks
    import taaf.support as support
    import taaf.trace as trace

__all__ = [
    "benchmark",
    "competition_arcade",
    "deploy",
    "deploy_inline",
    "deploy_kaggle",
    "deploy_slurm",
    "diagnostics",
    "game",
    "game_api",
    "game_examples",
    "solver",
    "solver_examples",
    "standard_benchmarks",
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
