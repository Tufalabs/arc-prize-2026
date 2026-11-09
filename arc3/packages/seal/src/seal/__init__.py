"""The seal agent, on the platypus harness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from seal.harness import install_finder

# So ``runtime.game`` resolves wherever model state is unpickled, without
# importing the surface itself — see seal.harness._PresentedFinder.
install_finder()

if TYPE_CHECKING:
    from seal.agent import SealAgent

__all__ = ["SealAgent"]


def __getattr__(name: str) -> Any:
    """``from seal import SealAgent`` without importing the agent at package
    import: the kernel subprocess imports ``seal.harness.game`` for the
    model's surface, and an eager import here would drag ullm, platypus and
    the agent module into every kernel start."""
    if name == "SealAgent":
        from seal.agent import SealAgent

        return SealAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
