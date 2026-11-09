"""Full-weight controls fence native draws and publish only completed updates."""

import asyncio
import json
from types import SimpleNamespace as NS

from aiohttp.test_utils import TestClient, TestServer
import pytest
import torch

from engine_wire import generation
from test_native_vllm_server import Engine, request, service
from training.native_vllm_server import create_app
from training.policy_sync.vllm_nccl import WeightManifest


MANIFEST = WeightManifest(
    ("model.language_model.layers.0.weight",), ("bfloat16",), ((2, 3),)
)
PUBLICATION = {
    "kind": "full_weight",
    "weight_version": "a" * 32,
    "manifest_sha256": MANIFEST.sha256,
}
INIT = {
    "init_info": {
        "master_address": "trainer",
        "master_port": 12345,
        "rank_offset": 1,
        "world_size": 3,
    }
}


class WeightEngine(Engine):
    def __init__(self):
        super().__init__()
        self.vllm_config.weight_transfer_config = NS(backend="nccl")
        self.vllm_config.lora_config = None
        self.vllm_config.parallel_config = NS(world_size_across_dp=2)
        self.vllm_config.model_config.dtype = torch.bfloat16
        self.paused = False
        self.fail_operation = None

    def record(self, operation, payload=None):
        self.controls.append((operation, payload))
        if self.fail_operation == operation:
            raise RuntimeError(f"failed {operation}")

    async def init_weight_transfer_engine(self, request):
        self.record("init", request.init_info)

    async def pause_generation(self, *, mode, clear_cache):
        self.record("pause", (mode, clear_cache))
        self.paused = True

    async def is_paused(self):
        return self.paused

    async def start_weight_update(self):
        self.record("start")

    async def update_weights(self, request):
        self.record("update", request.update_info)

    async def finish_weight_update(self):
        self.record("finish")

    async def resume_generation(self):
        self.record("resume")
        self.paused = False


@pytest.fixture(autouse=True)
def native_weight_dependencies(monkeypatch):
    monkeypatch.setattr(
        "training.native_vllm_server._weight_request",
        lambda kind, info: NS(**{f"{kind}_info": info}),
    )
    monkeypatch.setattr(
        "training.native_vllm_server.importlib.metadata.version", lambda _: "0.25.1"
    )


async def begin(server):
    await server.init_weights(INIT)
    await server.pause_weights({})
    await server.start_weights(PUBLICATION)


def test_http_publication_preserves_base_generation_and_exact_identity(monkeypatch):
    pytest.importorskip("ullm.backends.engine_proxy")
    monkeypatch.setattr(
        "ullm.backends.engine_validation.validate_output_text", lambda *args: None
    )

    async def scenario():
        server, engine = service(WeightEngine(), require_decoder=False)
        app = create_app(server, api_key="secret")
        async with TestClient(
            TestServer(app), headers={"Authorization": "Bearer secret"}
        ) as client:
            denied = await client.post(
                "/weights/init", json=INIT, headers={"Authorization": "wrong"}
            )
            assert denied.status == 401 and not engine.controls
            metadata = await (await client.get("/metadata")).json()
            assert (
                metadata["adapter"]["id"] == 0
                and "weight_version" not in metadata["adapter"]
            )
            info = await (await client.get("/weights/info")).json()
            assert info["world_size"] == 2

            async def draw(stamp):
                return await client.post(
                    "/generate",
                    json=generation(
                        request(user=stamp), server.adapter("base").identity()
                    ),
                )

            assert (await draw("run#before")).status == 400
            for operation, body in (
                ("init", INIT),
                ("pause", {}),
                ("start", PUBLICATION),
            ):
                assert (
                    await client.post(f"/weights/{operation}", json=body)
                ).status == 200
            assert (await draw("run#fenced")).status == 400 and not engine.calls
            assert (
                await client.post(
                    "/weights/update",
                    json={"update_info": MANIFEST.update_info(packed=True)},
                )
            ).status == 200
            assert (await client.post("/weights/finish", json={})).status == 200
            assert "weight_version" not in server.adapter("base").identity()
            response = await client.post("/weights/resume", json={})
            assert response.status == 200
            identity = (await response.json())["adapter"]
            assert identity == {
                "name": "base",
                "id": 0,
                "path": None,
                "server_instance_id": server.instance_id,
                **PUBLICATION,
            }
            response = await draw("run#published")
            assert response.status == 200
            output = await response.json()
            assert output["adapter"] == identity and output["lora_request"] is None
            assert output["prompt_token_ids"] == [1, 2]
            assert output["outputs"][0]["token_ids"] == [3, 4]
            assert [
                row[str(token)]["logprob"]
                for token, row in zip(
                    [3, 4], output["outputs"][0]["logprobs"], strict=True
                )
            ] == [-0.4, -0.4]
            assert [event[0] for event in engine.controls] == [
                "init",
                "pause",
                "start",
                "update",
                "finish",
                "reset",
                "resume",
            ]
            assert engine.controls[1] == ("pause", ("wait", True))
            assert engine.calls[0][3] is None and not server.mutating

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["start", "update", "finish", "resume", "cache"])
def test_partial_update_failure_never_exposes_weights(operation):
    async def scenario():
        server, engine = service(WeightEngine(), require_decoder=False)
        engine.fail_operation = operation
        if operation == "cache":
            engine.reset_ok = False
        with pytest.raises(RuntimeError):
            await begin(server)
            await server.update_weights(
                {"update_info": MANIFEST.update_info(packed=True)}
            )
            await server.finish_weights({})
            await server.resume_weights({})
        assert server.failure is not None and server.mutating
        assert "weight_version" not in server.adapter("base").identity()
        with pytest.raises(RuntimeError, match="poisoned"):
            async with server._draw("base", "run#failed"):
                pytest.fail("a failed publication admitted a draw")

    asyncio.run(scenario())


def test_full_publication_requires_drained_engine_and_fresh_version():
    async def scenario():
        server, engine = service(WeightEngine(), require_decoder=False)
        await server.init_weights(INIT)
        server.inflight.add("run#held")
        with pytest.raises(ValueError, match="drained"):
            await server.pause_weights({})
        assert not server.mutating and server.failure is None
        server.inflight.clear()
        await server.pause_weights({})
        await server.start_weights(PUBLICATION)
        await server.update_weights({"update_info": MANIFEST.update_info(packed=False)})
        await server.finish_weights({})
        await server.resume_weights({})
        await server.pause_weights({})
        with pytest.raises(ValueError, match="cannot be reused"):
            await server.start_weights(PUBLICATION)
        assert server.mutating and engine.paused

    asyncio.run(scenario())


def test_wrong_manifest_poisoned_before_receiver_dispatch():
    async def scenario():
        server, engine = service(WeightEngine(), require_decoder=False)
        await begin(server)
        info = MANIFEST.update_info(packed=True) | {"shapes": [[3, 2]]}
        with pytest.raises(ValueError, match="manifest differs"):
            await server.update_weights({"update_info": info})
        assert server.failure and server.mutating
        assert all(operation != "update" for operation, _ in engine.controls)

    asyncio.run(scenario())


def test_cancelled_control_keeps_engine_poisoned_and_fenced():
    async def scenario():
        server, engine = service(WeightEngine(), require_decoder=False)
        await server.init_weights(INIT)
        entered = asyncio.Event()

        async def blocked_pause(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        engine.pause_generation = blocked_pause
        operation = asyncio.create_task(server.pause_weights({}))
        await entered.wait()
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert server.failure and server.mutating

    asyncio.run(scenario())


def test_full_weight_controls_are_unavailable_on_lora_engines():
    server, _ = service(require_decoder=False)
    with pytest.raises(ValueError, match="not configured"):
        server.weight_info()


@pytest.mark.parametrize("environment", ["arc2", "arc3"])
@pytest.mark.parametrize("image_mode", ["", "current_grid", "level_start"])
def test_full_weight_launcher_selects_nccl_without_lora(
    monkeypatch, tmp_path, environment, image_mode
):
    from test_native_launcher import (
        test_local_model_override_preserves_public_alias as launch,
    )

    monkeypatch.setenv("SERVER_WEIGHT_MODE", "full_weight")
    launch(tmp_path, "base", None, environment, image_mode, "full_weight")
    arguments = json.loads((tmp_path / "argv.json").read_text())
    assert json.loads(arguments[arguments.index("--weight-transfer-config") + 1]) == {
        "backend": "nccl"
    }
    assert not {"--enable-lora", "--max-lora-rank", "--max-loras"}.intersection(
        arguments
    )
