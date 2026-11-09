"""ullm.capture unit level: canonical request stability, sink/stream
round-trip, and replay of defense-path (malformed retry) traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.capture
import ullm.conv
import ullm.manager
import ullm.utils


def _conversation(text: str) -> ullm.conv.Conversation:
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys prompt " * 30))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text)))
    return conversation


def test_canonical_request_is_stable_and_blocks_dedup() -> None:
    params = ullm.backend.GenericSamplingParameters(temperature=0.5, max_tokens=10, seed=7)
    blocks_a: dict[str, Any] = {}
    blocks_b: dict[str, Any] = {}
    conv_a, conv_b = _conversation("what is 1+1"), _conversation("what is 1+1")
    for conversation in (conv_a, conv_b):
        conversation.elements.append(ullm.conv.ResponseTurn(status="intermediate"))
    a = ullm.capture.canonicalize_request(params, conv_a, blocks_a)
    b = ullm.capture.canonicalize_request(params, conv_b, blocks_b)
    assert ullm.capture.canonical_dumps(a) == ullm.capture.canonical_dumps(b)
    # The long system prompt went to blocks (dedup identity), short text inlined.
    assert len(blocks_a) == 1 and blocks_a == blocks_b


async def test_capture_and_replay_roundtrip_through_manager(tmp_path: Path) -> None:
    # Record a two-call turn-manager run against the arithmetic stub, then
    # replay the identical run against the recorded stream.
    def run_manager(backend: ullm.backend.Backend) -> ullm.manager.TurnManager:
        manager = ullm.utils.make_turn_manager(backend, system_prompt="calc", temperature=0.0, seed=3)
        return manager

    recorded = ullm.backends.stub.ArithmeticStubBackend()
    recorded.capture_sink = ullm.capture.CaptureSink(directory=str(tmp_path))
    for prompt in ("2+2?", "compute 10+7 please"):
        manager = run_manager(recorded)
        manager.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=prompt)))
        await manager.run()

    replay = ullm.capture.ReplayBackend(directory=str(tmp_path))
    answers = []
    for prompt in ("2+2?", "compute 10+7 please"):
        manager = run_manager(replay)
        manager.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=prompt)))
        await manager.run()
        answers.append(manager.conversation.last_turn.last_response.final_content.to_string())
    assert answers == ["4", "17"]
    assert replay.unconsumed() == 0
    # Recorded usage reproduced exactly.
    assert replay.total_usage.completion_tokens == recorded.total_usage.completion_tokens


@dataclass
class _MalformedOnceBackend(ullm.backends.stub.ArithmeticStubBackend):
    """Arithmetic stub whose FIRST call fails output processing — captures
    a malformed record followed by the retry's success record."""

    _calls: int = field(default=0, init=False, repr=False)

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        self._calls += 1
        if self._calls == 1:
            raise ullm.backend.MalformedOutputError("deliberately malformed first attempt")
        super()._process_output(sampling_params, llm_output, conversation, turn, element)


async def test_malformed_retry_traffic_replays_deterministically(tmp_path: Path) -> None:
    # U4: defense paths are part of the reference. The recorded stream is
    # [malformed, response]; under replay the manager hits the same
    # malformed error, retries with the same deterministic seed bump and
    # newline perturbation, and the retry matches its own record.
    recorded = _MalformedOnceBackend()
    recorded.capture_sink = ullm.capture.CaptureSink(directory=str(tmp_path))
    manager = ullm.utils.make_turn_manager(recorded, system_prompt="calc", temperature=0.0, seed=5)
    manager.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="3+4?")))
    await manager.run()
    assert manager.conversation.last_turn.last_response.malformed_retries == 1

    records, _ = ullm.capture.load_stream(str(tmp_path))
    assert [r["kind"] for r in records] == ["malformed", "response"]

    replay = ullm.capture.ReplayBackend(directory=str(tmp_path))
    manager = ullm.utils.make_turn_manager(replay, system_prompt="calc", temperature=0.0, seed=5)
    manager.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="3+4?")))
    await manager.run()
    response = manager.conversation.last_turn.last_response
    assert response.final_content.to_string() == "7"
    assert response.malformed_retries == 1
    assert replay.unconsumed() == 0


def _image_prompt(text: str, image: "Any") -> ullm.conv.UserPrompt:
    return ullm.conv.UserPrompt(
        content=ullm.conv.ListContent(elements=[ullm.conv.StringContent(text=text), ullm.conv.ImageContent(data=image)])
    )


async def test_image_traffic_replays_and_verifies_pixels(tmp_path: Path) -> None:
    # Images always travel as content blocks (sha256 of the raw pixels).
    # Replay must accept the bit-identical image and reject a request
    # whose image differs by a single pixel — the text around it unchanged,
    # so pixels alone carry the match.
    import numpy as np

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    async def run(backend: ullm.backend.Backend, image: np.ndarray) -> str:
        manager = ullm.utils.make_turn_manager(backend, system_prompt="calc", temperature=0.0, seed=9)
        manager.conversation.elements.append(_image_prompt("what is 5+6?", image))
        await manager.run()
        return manager.conversation.last_turn.last_response.final_content.to_string()

    recorded = ullm.backends.stub.ArithmeticStubBackend()
    recorded.capture_sink = ullm.capture.CaptureSink(directory=str(tmp_path))
    assert await run(recorded, image) == "11"
    _, blocks = ullm.capture.load_stream(str(tmp_path))
    assert any("image_b64" in block for block in blocks.values())

    replay = ullm.capture.ReplayBackend(directory=str(tmp_path))
    assert await run(replay, image.copy()) == "11"
    assert replay.unconsumed() == 0

    tampered = image.copy()
    tampered[0, 0, 0] = 0
    replay = ullm.capture.ReplayBackend(directory=str(tmp_path))
    with pytest.raises(ullm.capture.ReplayMismatchError):
        await run(replay, tampered)


def test_identical_images_dedup_to_one_block() -> None:
    # The same frame sent twice (the common agent pattern) must produce one
    # block, referenced twice — the linear-storage property U3 promises.
    import numpy as np

    image = np.full((8, 8, 3), 7, dtype=np.uint8)
    params = ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=10, seed=1)
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
    )
    conversation.elements.append(_image_prompt("first", image))
    conversation.elements.append(_image_prompt("second", image.copy()))
    conversation.elements.append(ullm.conv.ResponseTurn(status="intermediate"))
    blocks: dict[str, Any] = {}
    canonical = ullm.capture.canonical_dumps(ullm.capture.canonicalize_request(params, conversation, blocks))
    image_blocks = [h for h, b in blocks.items() if "image_b64" in b]
    assert len(image_blocks) == 1
    assert canonical.count(image_blocks[0]) == 2


@dataclass
class _InjectingStub(ullm.backends.stub.ArithmeticStubBackend):
    """The arithmetic stub, declaring the injection support a vLLM host has."""

    def supports_response_injection(self) -> bool:
        return True


async def test_an_injected_response_replays_as_injected(tmp_path: Path) -> None:
    """The served prefix is part of what a recording has to reproduce.

    Replay rebuilds the element from the record rather than through the base
    ``respond``, so without restoring the marker a replayed injected call
    hands back the prefix as sampled text: the viewer and the diagnostics
    log then attribute to the model words it never chose.
    """
    params = ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=10, seed=3, response_injection="Okay,")
    recorded = _InjectingStub()
    recorded.capture_sink = ullm.capture.CaptureSink(directory=str(tmp_path))
    manager = ullm.utils.make_turn_manager(recorded, system_prompt="calc", temperature=0.0, seed=3)
    manager.response_manager.sampling_params = params
    manager.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="2+2?")))
    await manager.run()
    assert manager.conversation.last_turn.last_response.response_injection == "Okay,"

    replay = ullm.capture.ReplayBackend(directory=str(tmp_path))
    replayed = ullm.utils.make_turn_manager(replay, system_prompt="calc", temperature=0.0, seed=3)
    replayed.response_manager.sampling_params = params
    replayed.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="2+2?")))
    await replayed.run()
    assert replayed.conversation.last_turn.last_response.response_injection == "Okay,"
