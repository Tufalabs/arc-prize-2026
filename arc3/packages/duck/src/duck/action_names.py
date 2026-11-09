"""Shared action-name mapping between model-facing labels and engine actions.

The engine names are the ones platypus and TAAF speak; the model never sees
them. Which labels the model is taught is the agent's business (contract §2,
ledger #8), and this is where the duck's choice of them lives.
"""

from __future__ import annotations

ENGINE_TO_MODEL_ACTION = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
    "ACTION7": "UNDO",
    "RESET": "RESET",
}

MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}


def to_model_action(name: str | None) -> str:
    """The label the model knows an engine action by; unknown names pass
    through unchanged so they stay recognizable in an error message."""
    raw = str(name or "").strip().upper()
    return ENGINE_TO_MODEL_ACTION.get(raw, raw)


def to_engine_action(name: str | None) -> str | None:
    """Engine name for a label the model wrote, or None if it is neither a
    known label nor a known engine name. Accepts engine names as-is, since
    the model does sometimes echo one back."""
    raw = str(name or "").strip().upper()
    if not raw:
        return None
    if raw in ENGINE_TO_MODEL_ACTION:
        return raw
    return MODEL_TO_ENGINE_ACTION.get(raw)
