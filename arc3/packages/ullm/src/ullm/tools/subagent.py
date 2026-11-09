import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import ullm.conv
import ullm.manager
import ullm.tool


@dataclass
class Subagent(ullm.tool.Tool):
    """Tool that runs a subagent; this can be a different backend from the main agent."""

    name: str = "subagent"
    description: str = (
        "Prompt a subagent LLM with a given user prompt, and returns the final answer without reasoning traces."
    )
    clear_state_on: Literal["run", "turn", "never"] = "never"
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt for the subagent"},
            },
            "required": ["prompt"],
        }
    )

    turn_manager: ullm.manager.TurnManager = field(
        kw_only=True
    )  # turn manager to process the prompt; mandatory to provide on init

    _initial_turn_manager: Optional[ullm.manager.TurnManager] = field(default=None, init=False)  # for state clearing
    _logged_turn_managers: list[ullm.manager.TurnManager] = field(
        default_factory=lambda: list[ullm.manager.TurnManager](), init=False, repr=False
    )

    def _copy_turn_manager(self, tm: ullm.manager.TurnManager) -> ullm.manager.TurnManager:
        # Copy the conversation state but keep the SHARED live backend
        # (deepcopying a Backend yields a detached configuration clone,
        # which would split accounting and drop any capture sink).
        """A copy of the manager with its own conversation but the SAME live
        backend, so accounting and capture stay with the original.
        """
        backend = tm.response_manager.backend
        return copy.deepcopy(tm, {id(backend): backend})

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Run the sub-agent on the caller's prompt and return its reply."""
        if self._initial_turn_manager is None:
            self._initial_turn_manager = self._copy_turn_manager(self.turn_manager)
        self.turn_manager.conversation.elements.append(
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=arguments["prompt"]))
        )
        await self.turn_manager.run()  # process the prompt with the turn manager (including any tool calls)
        self.turn_manager.rearm()  # prepare for a new turn
        response_content = self.turn_manager.conversation.last_turn.last_response.final_content
        return ullm.conv.StringContent(text=copy.deepcopy(response_content.to_string()))

    async def _clear_state(self) -> None:
        # Reset the conversation to its initial state
        """Park the used conversation for diagnostics and restore the manager from
        its snapshot.
        """
        if self._initial_turn_manager is not None:
            self._logged_turn_managers.append(
                self.turn_manager
            )  # log the turn manager before resetting, for diagnostics
            self.turn_manager = self._copy_turn_manager(self._initial_turn_manager)
            self._initial_turn_manager = None
