from dataclasses import dataclass
from typing import Any, AsyncGenerator

import ullm.backend
import ullm.conv


@dataclass
class User(ullm.backend.Backend):
    """A backend where the user plays the LLM. This is useful for testing and debugging, and also to run the system with a human in the loop."""

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        """Render the conversation for a human to read."""
        return {
            "full_conversation": conversation.to_string(),
            "last_user_prompt": conversation.elements[-2].to_string(),
        }

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """Print the conversation and read the reply from stdin."""
        print("\n--- FULL CONVERSATION---")
        print(llm_input["full_conversation"])
        print("-----------------")
        print("(Note: in VS code, you need to enter your response at the top of the screen)")
        while True:
            response = input(llm_input["last_user_prompt"]).strip()
            if response:
                yield response
                return
            print("(empty response, try again)")

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Put what the human typed on the element."""
        element.final_content = ullm.conv.StringContent(text=llm_output)
        element.status = "processed"
