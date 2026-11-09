"""On-demand live turn viewing over the job dir: the game writes
snapshots only while the viewer's request flag is fresh, and the viewer
reads them as plain files (so it works across hosts over NFS)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

import ullm.backend
import ullm.conv
import ullm.manager

import platypus.live
from platypus.diagnostics import BlockStore, SpanLogger
from platypus.llm import DiagnosticTurnManager

_CHUNKS = ["The answer", " is taking", " shape", " slowly", " but", " surely."]


@dataclass
class SlowStreamingBackend(ullm.backend.Backend):
    """Streams a fixed text in small chunks with real pauses — enough for
    an observer to catch a genuine mid-call partial."""

    chunk_sleep_s: float = 0.15

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        return None

    async def _generate(self, sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        text = ""
        for chunk in _CHUNKS:
            text += chunk
            yield text
            await asyncio.sleep(self.chunk_sleep_s)

    def _process_partial_output(self, sampling_params: Any, llm_output: Any, intermediate: Any) -> None:
        intermediate.final_content = ullm.conv.StringContent(text=llm_output)

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
        element.final_content = ullm.conv.StringContent(text=llm_output)
        element.stop_reason = "stop"
        element.token_usage = ullm.backend.TokenUsage(prompt_tokens=10, completion_tokens=6, num_requests=1)
        element.status = "processed"


def _manager(game_dir: Path, *, chunk_sleep_s: float = 0.15) -> DiagnosticTurnManager:
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    return DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(backend=SlowStreamingBackend(chunk_sleep_s=chunk_sleep_s)),
        conversation=conversation,
        diag=SpanLogger(game_dir / "spans.jsonl", BlockStore(game_dir.parent.parent / "diag_blocks")),
        label="analyzer",
    )


async def test_watched_turn_writes_midcall_snapshots(tmp_path: Path) -> None:
    game_dir = tmp_path / "games" / "live0001"
    game_dir.mkdir(parents=True)
    platypus.live.touch_request(game_dir)  # a viewer is watching

    manager = _manager(game_dir)
    task = asyncio.create_task(manager.run())
    try:
        partial = ""
        for _ in range(60):
            await asyncio.sleep(0.05)
            snapshot = platypus.live.read_snapshot(game_dir)
            if snapshot and snapshot.get("active") and snapshot.get("partial_final"):
                partial = snapshot["partial_final"]
                assert snapshot["label"] == "analyzer" and snapshot["state"] == "running"
                break
        assert partial.startswith("The answer"), f"no mid-call partial written: {partial!r}"
        assert partial != "".join(_CHUNKS), "should have caught the stream before completion"
    finally:
        await task
        # Closed here, so a failed assertion leaves no stream open for a
        # later test to trip over.
        manager.diag.close()

    # After the turn ends the writer stops; the last snapshot ages out.
    await asyncio.sleep(0)
    for _ in range(20):
        if platypus.live.read_snapshot(game_dir) is None:
            break
        await asyncio.sleep(0.5)
    # A game dir nobody wrote for: None, not an error.
    assert platypus.live.read_snapshot(tmp_path / "games" / "never_ran") is None


async def test_unwatched_turn_writes_nothing(tmp_path: Path) -> None:
    """No request flag → the writer never writes a snapshot (nothing
    happens unless someone asks)."""
    game_dir = tmp_path / "games" / "live0003"
    game_dir.mkdir(parents=True)
    manager = _manager(game_dir, chunk_sleep_s=0.5)  # a long turn, plenty of writer cycles
    task = asyncio.create_task(manager.run())
    try:
        await asyncio.sleep(1.0)  # several WRITE_INTERVAL_S windows
        assert not platypus.live.snapshot_path(game_dir).exists()
    finally:
        await task
        manager.diag.close()


async def test_live_page_renders_partial_and_idle(tmp_path: Path) -> None:
    from platypus.viewer import render_viewer_path

    game_dir = tmp_path / "games" / "live0002"
    # A multi-second turn so the writer's cadence catches a render-touched
    # request (real turns run seconds-to-minutes; only the test is brief).
    manager = _manager(game_dir, chunk_sleep_s=0.5)
    task = asyncio.create_task(manager.run())
    try:
        page = b""
        for _ in range(120):
            await asyncio.sleep(0.05)
            # Each render touches the request flag (a viewer is watching)
            # and then reads whatever snapshot the game has written. The
            # writer publishes the partial before the conversation, each in a
            # thread of its own, so a render can land between the two.
            rendered = render_viewer_path(tmp_path, "g_live0002_live.html")
            if rendered is not None and b"streaming response" in rendered[0] and b"conversation so far" in rendered[0]:
                page = rendered[0]
                break
        assert b"The answer" in page and b"live turn" in page
        assert b'http-equiv="refresh" content="5"' in page  # the 5 s update rate
        # The full conversation renders too (system prompt + the user
        # prompt), not just the streaming tail.
        assert b"conversation so far" in page and b"system prompt" in page and b"go" in page
    finally:
        await task
        manager.diag.close()

    # After the turn the snapshot ages out; the page drops the partial and
    # shows an idle reason. (This standalone manager has no game span, so
    # its root is the now-closed llm/turn — "the run has finished"; a real
    # run's open game span would instead read "no turn is running".)
    for _ in range(20):
        idle = render_viewer_path(tmp_path, "g_live0002_live.html")
        assert idle is not None
        if b"streaming response" not in idle[0] and (b"no turn is running" in idle[0] or b"has finished" in idle[0]):
            break
        await asyncio.sleep(0.5)
    else:
        raise AssertionError("live page never went idle after the turn ended")


async def test_live_page_links_to_in_flight_move_not_turn(tmp_path: Path) -> None:
    """The live link must target the in-flight infra/move span (which the
    move page is keyed by), not the deeper llm/turn span — else it points
    past the current move / 404s."""
    import re

    from platypus.viewer import render_viewer_path

    game_dir = tmp_path / "games" / "live0004"
    logger = SpanLogger(game_dir / "spans.jsonl", BlockStore(tmp_path / "diag_blocks"))
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    manager = DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(backend=SlowStreamingBackend(chunk_sleep_s=0.5)),
        conversation=conversation,
        diag=logger,
        label="analyzer",
    )

    async def run_under_spans() -> int:
        with logger.span("infra/game", "g"), logger.span("infra/move", "move 0") as move:
            await manager.run()  # llm/turn nests under the move via the ambient span
            return move.span_id

    task = asyncio.create_task(run_under_spans())
    try:
        page = b""
        for _ in range(120):
            await asyncio.sleep(0.05)
            rendered = render_viewer_path(tmp_path, "g_live0004_live.html")
            if rendered is not None and b"streaming response" in rendered[0]:
                page = rendered[0]
                break
        linked = re.search(rb"g_live0004_m(\d+)\.html", page)
        assert linked is not None, "live page has no move link"
        # The linked move page RESOLVES (it's a real move span, not the
        # turn span, which would 404).
        move_page = render_viewer_path(tmp_path, f"g_live0004_m{linked.group(1).decode()}.html")
        assert move_page is not None and b"move 0" in move_page[0]
    finally:
        move_id = await task
        logger.close()
    assert int(linked.group(1)) == move_id  # exactly the move span, not the turn


@dataclass
class BurstBackend(SlowStreamingBackend):
    """First stream event carries no content (a message_start analogue),
    then the full text arrives and the call ends quickly — the shape that
    must still ship its content through a single throttle window."""

    async def _generate(self, sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        yield ""
        await asyncio.sleep(0.05)
        yield "burst text"
        await asyncio.sleep(0.05)


async def test_proxy_relays_content_partial_for_short_bursts(tmp_path: Path) -> None:
    # Aliased: a bare `import ullm.backends.proxy` here would bind a LOCAL
    # `ullm` and shadow the module-level imports.
    from ullm.backends import proxy as _proxy

    server = await _proxy.start_backend_server(BurstBackend(), str(tmp_path / "b.sock"))
    client = _proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"), stream_interval=30.0)
    try:
        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
        )
        conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
        conversation.status = "intermediate"
        turn = ullm.conv.ResponseTurn(status="intermediate")
        conversation.elements.append(turn)
        partials: list[str] = []
        agen = client.respond(ullm.backend.GenericSamplingParameters(), conversation)
        async for partial in agen:
            partials.append(partial.final_content.to_string())
        # Even with a 30s throttle, the content-bearing partial gets through:
        # an empty snapshot does not consume the window.
        assert "burst text" in partials
    finally:
        await client.close()
        await server.close()
