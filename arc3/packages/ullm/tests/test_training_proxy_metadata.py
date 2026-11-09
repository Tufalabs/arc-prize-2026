"""Native draw attribution and image-count evidence survive the child proxy."""

import tempfile
from dataclasses import dataclass
from typing import Optional

import ullm.backend
import ullm.backends.proxy
import ullm.backends.stub
import ullm.conv


@dataclass
class EvidenceBackend(ullm.backends.stub.StubBackend):
    """Supply native request identity and prompt media from the host."""

    def _fill_into(
        self,
        src: ullm.conv.LLMResponse,
        element: ullm.conv.LLMResponse,
        conversation: ullm.conv.Conversation,
        injection: Optional[str] = None,
    ) -> None:
        """Copy attribution that the native host parser attaches to its draw."""
        super()._fill_into(src, element, conversation, injection)
        element.generation_request_id = "puzzle#nonce"
        element.malformed_error = "invalid tool-call JSON"
        element.malformed_call_count = 1

    async def _token_count(
        self, sampling_params: ullm.backend.SamplingParameters, conversation: ullm.conv.Conversation
    ) -> int:
        """Return evidence as an integer-compatible count over the proxy wire."""
        return ullm.backend.TokenCount(3, [8, 9, 10], media={"image_hashes": ["first", "first", "last"]})


async def test_native_metadata_and_ordered_media_cross_the_proxy() -> None:
    backend = EvidenceBackend(
        responses=[ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="<tool_call>{"))]
    )
    with tempfile.TemporaryDirectory(prefix="ullm-", dir="/tmp") as folder:
        server = await ullm.backends.proxy.start_backend_server(backend, folder + "/backend.sock")
        proxy = ullm.backends.proxy.ProxyBackend(socket_path=folder + "/backend.sock", replay_on_link_loss=False)
        try:
            conversation = ullm.conv.Conversation(
                elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="puzzle"))],
                status="intermediate",
            )
            conversation.elements.append(ullm.conv.ResponseTurn(status="intermediate"))
            params = ullm.backend.GenericSamplingParameters(temperature=1, top_p=1)
            measured = await proxy.token_count(params, conversation)
            assert isinstance(measured, ullm.backend.TokenCount)
            assert measured.prompt_token_ids == (8, 9, 10)
            assert measured.media == {"image_hashes": ["first", "first", "last"]}
            async for _ in proxy.respond(params, conversation):
                pass
            response = conversation.last_turn.last_response
            assert response.generation_request_id == "puzzle#nonce"
            assert response.malformed_call_count == 1
            assert response.malformed_error == "invalid tool-call JSON"
            assert response.final_content.to_string() == "<tool_call>{"
        finally:
            await proxy.close()
            await server.close()
