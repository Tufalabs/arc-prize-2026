"""U5: an image's identity is taken once, where it is made, and every call
reuses it -- the continuation's served-image check, the witness and the
engine's caches -- and a reply's witness reuses the history its request was
built from, matching what a fresh render would witness."""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

import numpy as np
import pytest
import ullm.conv
from test_vllmx_engine_contract import _params, _respond
from test_vllmx_multimodal_routing import image_backend, image_conversation

vllmx = pytest.importorskip("ullm.backends.vllmx")


def _expected_digest(data: np.ndarray) -> str:
    """SHA-256 over the size and the row-order pixels, as the image is served."""
    return hashlib.sha256(f"RGB:{data.shape[1]}x{data.shape[0]}:".encode() + data.tobytes()).hexdigest()


def _image(value: int) -> ullm.conv.ImageContent:
    return ullm.conv.ImageContent(data=np.full((32, 32, 3), value, dtype=np.uint8))


def test_an_image_carries_its_digest_from_where_it_is_made(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _image(7)
    assert image.digest == _expected_digest(image.data)

    # Shipped to another process, it arrives with its digest: nothing hashes the pixels again.
    def hashed_again(data: np.ndarray) -> str:
        pytest.fail("pixels hashed again")

    monkeypatch.setattr(ullm.conv, "_pixel_digest", hashed_again)
    assert pickle.loads(pickle.dumps(image)).fingerprint() == image.digest
    monkeypatch.undo()
    # New pixels are a new image.
    image.data = np.full((32, 32, 3), 8, dtype=np.uint8)
    assert image.fingerprint() == _expected_digest(image.data) != _image(7).digest
    # One pickled before images carried a digest still has an identity.
    image.__dict__["digest"] = ""
    assert image.fingerprint() == _expected_digest(image.data)


@pytest.mark.parametrize("injection", [None, "prefix "])
async def test_the_engine_finds_each_image_by_its_digest(injection: str | None) -> None:
    backend, engine, _ = image_backend()
    conversation = image_conversation()
    params = _params(injection)
    try:
        await _respond(backend, params, conversation)
        prompt = conversation.elements[0]
        assert isinstance(prompt, ullm.conv.UserPrompt) and isinstance(prompt.content, ullm.conv.ImageContent)
        first = prompt.content
        assert engine.prompts[0]["multi_modal_uuids"] == {"image": [first.digest]}

        second = _image(9)
        conversation.elements.extend(
            [ullm.conv.UserPrompt(content=second), ullm.conv.ResponseTurn(status="intermediate")]
        )
        conversation.status = "intermediate"
        await _respond(backend, params, conversation)
        assert "prompt" not in engine.prompts[1]  # continued
        assert engine.prompts[1]["multi_modal_uuids"] == {"image": [first.digest, second.digest]}
    finally:
        await backend.close()


@pytest.mark.parametrize("remote_media", [False, True])
@pytest.mark.parametrize("injection", [None, "prefix "])
async def test_a_reply_renders_its_history_once_and_witnesses_as_a_fresh_render(
    monkeypatch: pytest.MonkeyPatch, injection: str | None, remote_media: bool
) -> None:
    """``remote_media``: an engine that prepares its media itself, as a game's
    engine proxy does, so the request is built on the counting worker and
    the render it makes must still reach the witness."""
    backend, engine, _ = image_backend()
    if remote_media:
        engine.prepare_media = lambda prompt: {"schema": "prepared-media"}  # type: ignore[attr-defined]
    conversation = image_conversation()
    params = _params(injection)
    renders = 0
    history_messages = vllmx.VLLM._history_messages

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal renders
        renders += 1
        return history_messages(self, *args, **kwargs)

    monkeypatch.setattr(vllmx.VLLM, "_history_messages", counted)

    # Keep the served prompt ids, which a witnessed reply sheds, to witness it again below.
    def keep_served_ids(self: Any, element: Any) -> None:
        pass

    monkeypatch.setattr(vllmx.VLLM, "_shed_served_ids", keep_served_ids)
    try:
        for call in range(2):
            if call:
                conversation.elements.extend(
                    [ullm.conv.UserPrompt(content=_image(5)), ullm.conv.ResponseTurn(status="intermediate")]
                )
                conversation.status = "intermediate"
            renders = 0
            answer = await _respond(backend, params, conversation)
            assert renders == 1
            witnessed = answer._served
            assert witnessed is not None and witnessed.continued is bool(call)
            # The same answer witnessed from a fresh render of its history.
            vllmx._PREPARED_RENDER.set(None)
            backend._leave_witness(conversation, conversation.last_turn, answer, "all", injection)
            assert answer._served == witnessed
    finally:
        await backend.close()
