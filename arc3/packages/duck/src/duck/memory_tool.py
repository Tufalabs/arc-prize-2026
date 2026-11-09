"""The Duck's host-backed persistent-memory append tool."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

import ullm.conv
import ullm.tool


def _memory_parameters() -> dict[str, Any]:
    """Return the native memory tool's append-only schema."""
    return {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "One new durable game finding to append.",
            }
        },
        "required": ["text"],
        "additionalProperties": False,
    }


async def _identity_append(text: str) -> str:
    """The default appender: echo the finding, owned journal absent."""
    return text


@dataclass
class DuckMemoryTool(ullm.tool.Tool):
    """Append to the bounded journal owned by one ``DuckAgent``."""

    name: str = "memory"
    description: str = (
        "Append one new durable game finding to persistent memory. Pass it in the required `text` argument."
    )
    parameters: dict[str, Any] = field(default_factory=_memory_parameters)
    clear_state_on: Literal["run", "turn", "never"] = "never"
    appender: Callable[[str], Awaitable[str] | str] = field(default=_identity_append, repr=False)

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Append the supplied finding and return a compact acknowledgement."""
        try:
            result = self.appender(str(arguments["text"]))
            rendered = await result if inspect.isawaitable(result) else result
        except ValueError as exc:
            self.is_error = True
            rendered = str(exc)
        return ullm.conv.StringContent(text=rendered)
