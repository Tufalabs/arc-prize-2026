"""Serving one LoRA adapter through ``InprocessServerFactory``.

The serving itself is engine and server arguments; what these pin is the part
arguments cannot give. vLLM only addresses an adapter by its ``lora_modules``
name, and naming the *base* after the adapter makes ``is_base_model``
resolution serve the untrained base weights under the adapter's name — an
eval that runs, scores, and silently measures nothing. So the factory mounts
and addresses from one (name, path) spelling, verifies the adapter is
actually served before returning a client, and refuses loudly otherwise.

No GPU and no engine here: the seams under test are the spec validation, the
engine/server argument application, and the verify-or-refuse step — driven
against a stub HTTP server, which also exercises the health check's async
requirement (a blocking call would deadlock the serving loop it shares).
"""

from __future__ import annotations

import asyncio
import json

import pytest
import vllm
from ullm.backends import vllm_factory


def _factory(**kwargs: object) -> vllm_factory.InprocessServerFactory:
    """A factory over dummy engine args; nothing here starts an engine."""
    return vllm_factory.InprocessServerFactory(
        engine_args=vllm.AsyncEngineArgs(model="dummy/base"),
        **kwargs,  # type: ignore[arg-type]
    )


async def _stub_models_server(ids: list[str]) -> tuple[asyncio.Server, str]:
    """A minimal HTTP server answering ``/v1/models`` with the given ids."""
    body = json.dumps({"data": [{"id": i} for i in ids]}).encode()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s" % (len(body), body)
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}"


def test_an_adapter_spec_missing_either_half_is_refused() -> None:
    """Path without name (and name without path) fail before any spend."""
    with pytest.raises(ValueError):
        _factory(lora_adapter_path="/ckpt/adapter")._validate_lora()
    with pytest.raises(ValueError):
        _factory(lora_adapter_name="ckpt-1500")._validate_lora()
    _factory()._validate_lora()
    _factory(lora_adapter_path="/ckpt/adapter", lora_adapter_name="ckpt-1500")._validate_lora()


def test_the_adapter_is_mounted_from_the_single_spelling() -> None:
    """Engine args and server flags both derive from the one (name, path)."""
    factory = _factory(lora_adapter_path="/ckpt/adapter", lora_adapter_name="ckpt-1500", max_lora_rank=64)
    overrides: dict[str, object] = {}

    factory._apply_lora(overrides)

    assert factory.engine_args.enable_lora is True
    assert factory.engine_args.max_loras == 1
    assert factory.engine_args.max_lora_rank == 64
    (module,) = overrides["lora_modules"]  # type: ignore[misc]
    assert (module.name, module.path) == ("ckpt-1500", "/ckpt/adapter")


def test_no_adapter_leaves_the_engine_args_alone() -> None:
    """The default path is byte-identical to a factory without the feature."""
    factory = _factory()
    overrides: dict[str, object] = {}

    factory._apply_lora(overrides)

    assert overrides == {}
    assert not factory.engine_args.enable_lora


async def test_served_models_reads_the_server_list() -> None:
    """The verification half, against a real (stub) HTTP server."""
    server, base = await _stub_models_server(["dummy/base", "ckpt-1500"])
    try:
        served = await vllm_factory.InprocessServerFactory._served_models(base)
    finally:
        server.close()
        await server.wait_closed()

    assert served == {"dummy/base", "ckpt-1500"}


async def test_a_failed_start_tears_down_what_it_acquired() -> None:
    """The factory owns cleanup on its own failure path.

    ``BackendFactory.start()`` marks the factory started only when ``_start``
    returns, so a failure after the server task exists would otherwise leave
    ``close()`` a no-op — the server running with no owner, and the FlashInfer
    workspace on disk. ``_abort_start`` must await the cancellation (a
    fire-and-forget cancel keeps the port bound while a caller retries) and
    release the workspace.
    """
    import tempfile
    from pathlib import Path

    factory = _factory()
    server_alive = asyncio.Event()

    async def _pretend_server() -> None:
        server_alive.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(_pretend_server())
    await server_alive.wait()
    workspace = tempfile.mkdtemp(prefix="lora-abort-test-")
    factory._serve_task = task
    factory._flashinfer_workspace = workspace

    await factory._abort_start()

    assert task.cancelled()
    assert not Path(workspace).exists()
    assert factory._serve_task is None and factory._flashinfer_workspace is None


async def test_a_missing_adapter_is_refused_by_name() -> None:
    """The verification names what it wanted and what the server has."""
    factory = _factory(lora_adapter_path="/ckpt/adapter", lora_adapter_name="ckpt-1500")
    server, base = await _stub_models_server(["dummy/base"])
    try:
        with pytest.raises(RuntimeError) as caught:
            await factory._verify_adapter_served(base)
    finally:
        server.close()
        await server.wait_closed()
    assert "ckpt-1500" in str(caught.value) and "dummy/base" in str(caught.value)


async def test_a_served_adapter_passes_verification() -> None:
    """The healthy half of the same check."""
    factory = _factory(lora_adapter_path="/ckpt/adapter", lora_adapter_name="ckpt-1500")
    server, base = await _stub_models_server(["dummy/base", "ckpt-1500"])
    try:
        await factory._verify_adapter_served(base)
    finally:
        server.close()
        await server.wait_closed()
