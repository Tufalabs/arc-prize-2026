"""The duck: an LLM solver for ARC-AGI-2, on the platypus agent contract.

Deliberately empty of imports at module level. ``duck.sandbox`` is imported
by the model's kernel — an isolated worker with a startup deadline — and
importing this package is what that costs, so pulling ``duck.agent`` (and
through it platypus, taaf and ullm) in here would put the whole agent stack
inside every kernel. The names below resolve on first attribute access
instead.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from duck.agent import DuckAgent as DuckAgent
    from duck.config import CheckMode as CheckMode
    from duck.multi_take import MultiTakeDuck as MultiTakeDuck
    from duck.multi_take import TieredTakeDuck as TieredTakeDuck
    from duck.prompts import DuckPrompts as DuckPrompts
    from duck.prompts import PromptsConfigurable as PromptsConfigurable
    from duck.python_tool import DuckPythonTool as DuckPythonTool
    from duck.state import AnswerReading as AnswerReading
    from duck.state import AnswerSlot as AnswerSlot

__all__ = [
    "AnswerReading",
    "AnswerSlot",
    "CheckMode",
    "DuckAgent",
    "DuckPrompts",
    "DuckPythonTool",
    "MultiTakeDuck",
    "PromptsConfigurable",
    "TieredTakeDuck",
]

_EXPORTS = {
    "AnswerReading": "duck.state",
    "AnswerSlot": "duck.state",
    "CheckMode": "duck.config",
    "DuckAgent": "duck.agent",
    "DuckPrompts": "duck.prompts",
    "DuckPythonTool": "duck.python_tool",
    "MultiTakeDuck": "duck.multi_take",
    "PromptsConfigurable": "duck.prompts",
    "TieredTakeDuck": "duck.multi_take",
}


def __getattr__(name: str) -> Any:
    """Resolve an export on first use, so importing ``duck`` costs nothing."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
