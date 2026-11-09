"""Remote image preparation must not starve network I/O on a busy proxy host."""

import asyncio
import contextvars
import threading
import time
import types
from typing import Any

import pytest
import ullm.backend
import ullm.conv

vllmx = pytest.importorskip("ullm.backends.vllmx")
_caller = contextvars.ContextVar[int | None]("preparation_test_caller", default=None)


class _SlowBackend(vllmx.VLLM):
    """Exercise the real response/count scheduling with a slow CPU processor."""

    def _work(self, slot: int, kind: str) -> None:
        state = self.__dict__["test_state"]
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        state["work"].append((slot, kind, threading.get_ident(), _caller.get()))
        state["started"].set()
        try:
            # Only the first piece of work is held. Held work that times out
            # fails the test once, rather than every later item timing out too.
            if state["release"] is not None and len(state["work"]) == 1:
                assert state["release"].wait(5), "test did not release preparation"
            time.sleep(0.004)
        finally:
            state["active"] -= 1

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: Any) -> Any:
        slot = int(conversation.elements[0].content.text)
        self._work(slot, "prepare")
        return {"engine": {"type": "token", "prompt_token_ids": [slot, 248056], "media": {"image_hashes": [str(slot)]}}}

    async def _generate(self, sampling_params: Any, llm_input: Any) -> Any:
        self.__dict__["test_state"]["admitted"].append(llm_input)
        yield object()

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
        element.final_content = ullm.conv.StringContent(text="answer")
        element.token_usage = ullm.backend.TokenUsage(prompt_tokens=2, completion_tokens=1, num_requests=1)

    def _build_trace_step_sync(self, **kwargs: Any) -> Any:
        self._work(int(kwargs["run_id"]), "trace")
        return kwargs["llm_input"]


def _backend() -> Any:
    def prepare_media(prompt: Any) -> None:
        pass

    backend = _SlowBackend(engine=types.SimpleNamespace(prepare_media=prepare_media), token_count_evidence=True)
    backend.__dict__["test_state"] = {
        "active": 0,
        "peak": 0,
        "work": [],
        "admitted": [],
        "started": threading.Event(),
        "release": None,
    }
    return backend


def _conversation(slot: int) -> ullm.conv.Conversation:
    return ullm.conv.Conversation(
        elements=[
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=str(slot))),
            ullm.conv.ResponseTurn(status="intermediate"),
        ],
        status="intermediate",
    )


async def _respond(backend: Any, conversation: ullm.conv.Conversation, run_id: str) -> None:
    async for _ in backend.respond(ullm.backend.GenericSamplingParameters(), conversation, run_id=run_id):
        pass


def test_120_concurrent_counts_and_responses_keep_heartbeat_and_exact_identity() -> None:
    async def run() -> None:
        backend = _backend()
        state = backend.__dict__["test_state"]
        # The first piece of work is held in the worker until the loop has
        # ticked beside it.
        state["release"] = threading.Event()
        ticks = []
        finished = False

        async def heartbeat() -> None:
            while not finished:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.002)

        async def episode(slot: int) -> None:
            _caller.set(slot)
            conversation = _conversation(slot)
            count = await backend.token_count(ullm.backend.GenericSamplingParameters(), conversation)
            assert count.prompt_token_ids == (slot, 248056)
            assert count.media == {"image_hashes": [str(slot)]}
            await _respond(backend, conversation, f"slot-{slot}")
            usage = conversation.last_turn.last_response.token_usage
            assert usage is not None and usage.completion_tokens == 1

        async def release_once_the_loop_ticks() -> None:
            # Work run on the loop, or awaited by blocking it, would hold the
            # loop as long as the worker is held: no tick could land, and the
            # held work would time out instead. Counted, not clocked, so a
            # slow runner cannot fail it.
            while not state["started"].is_set():
                await asyncio.sleep(0.001)
            held_from = len(ticks)
            while len(ticks) < held_from + 3:
                await asyncio.sleep(0.001)
            assert state["active"] == 1, "the held work finished before it was released"
            state["release"].set()

        beat = asyncio.create_task(heartbeat())
        try:
            async with asyncio.timeout(10):
                await asyncio.gather(release_once_the_loop_ticks(), *(episode(slot) for slot in range(120)))
        finally:
            state["release"].set()
            finished = True
            await beat
            await backend.close()
        assert state["peak"] == 1
        loop_thread = threading.get_ident()
        assert all(thread != loop_thread for _, _, thread, _ in state["work"])
        assert all(slot == caller for slot, _, _, caller in state["work"])
        assert [slot for slot, _, _, _ in state["work"][:120]] == list(range(120))
        assert [draw["engine"]["prompt_token_ids"][0] for draw in state["admitted"]] == list(range(120))
        assert all(draw["request_id"].startswith(f"slot-{slot}#") for slot, draw in enumerate(state["admitted"]))
        assert len({draw["request_id"] for draw in state["admitted"]}) == 120

    with asyncio.Runner() as runner:
        runner.run(run())


def test_cancelled_preparation_drains_active_read_and_discards_queued_draw() -> None:
    async def run() -> None:
        backend = _backend()
        state = backend.__dict__["test_state"]
        state["release"] = threading.Event()
        first, queued = _conversation(0), _conversation(1)
        active = asyncio.create_task(_respond(backend, first, "active"))
        async with asyncio.timeout(2):
            while not state["started"].is_set():
                await asyncio.sleep(0.001)
        waiting = asyncio.create_task(_respond(backend, queued, "queued"))
        await asyncio.sleep(0.005)
        active.cancel()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await asyncio.sleep(0.005)
        assert not active.done()  # its thread still reads the original conversation
        active.cancel()  # repeated teardown cancellation must still drain safely
        state["release"].set()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert [slot for slot, _, _, _ in state["work"]] == [0]
        assert state["admitted"] == []
        assert first.last_turn.elements == queued.last_turn.elements == []
        await backend.close()

    with asyncio.Runner() as runner:
        runner.run(run())


def test_trace_rendering_uses_same_serial_worker_as_preparation() -> None:
    async def run() -> None:
        backend = _backend()
        state = backend.__dict__["test_state"]
        params = ullm.backend.GenericSamplingParameters()
        tasks = []
        for slot in range(12):
            conversation = _conversation(slot)
            tasks.append(backend.token_count(params, conversation))
            tasks.append(
                backend._build_trace_step(
                    run_id=str(slot),
                    sampling_params=params,
                    llm_input={"slot": slot},
                    element=ullm.conv.LLMResponse(),
                    conversation=conversation,
                )
            )
        results = await asyncio.gather(*tasks)
        assert results[1::2] == [{"slot": slot} for slot in range(12)]
        assert state["peak"] == 1
        assert len({thread for _, _, thread, _ in state["work"]}) == 1
        await backend.close()

    with asyncio.Runner() as runner:
        runner.run(run())
