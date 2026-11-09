from __future__ import annotations

import datetime
from pathlib import Path
import subprocess
import sys
import textwrap
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from training.policy_sync.full_weight import FullWeightParameterSource
from training.policy_sync.vllm_nccl import (
    VllmControlClient,
    VllmNcclWeightTransfer,
    WeightManifest,
)


def test_lazy_nccl_import_preserves_fresh_file_logging_and_environment(tmp_path):
    # Exercise real dictConfig/FileHandler behavior in isolation: the broken
    # import closes every handler, including pytest's own capture handlers.
    program = textwrap.dedent(
        """
        import builtins
        import logging
        import logging.config
        import os
        from pathlib import Path
        import sys
        from types import SimpleNamespace

        from training.policy_sync import vllm_trainer
        from training.run_artifacts import setup_logging_for_experiment

        root = logging.getLogger()
        directory = Path(sys.argv[1])
        # Pinned vLLM uses non-incremental dictConfig with this logger/handler
        # scope. disable_existing_loggers=False still closes existing handlers.
        config = {
            'version': 1, 'disable_existing_loggers': False,
            'handlers': {'vllm': {'class': 'logging.StreamHandler'}},
            'loggers': {'vllm': {'handlers': ['vllm'], 'propagate': False}},
        }
        control = directory / 'unprotected.log'
        setup_logging_for_experiment(control, '%(message)s')
        root.info('before')
        logging.config.dictConfig(config)
        root.info('after')
        assert control.read_text().splitlines() == ['before']

        original_import = builtins.__import__
        api = SimpleNamespace(
            NCCLWeightTransferEngine=object(), NCCLTrainerSendWeightsArgs=object()
        )
        fail_import = False

        def import_with_vllm_logging(name, *args, **kwargs):
            if name == 'vllm.distributed.weight_transfer.nccl_engine':
                # The logger is reached through nested imports of the API.
                import_with_vllm_logging('vllm.logger')
                if fail_import:
                    raise ImportError('nested import failed')
                return api
            if name == 'vllm.logger':
                configure = bool(int(os.environ.get('VLLM_CONFIGURE_LOGGING', '1')))
                if not configure and os.environ.get('VLLM_LOGGING_CONFIG_PATH'):
                    raise RuntimeError('disabled logging with a config path')
                if configure:
                    logging.config.dictConfig(config)
                return SimpleNamespace()
            return original_import(name, *args, **kwargs)

        vllm_trainer.importlib.metadata.version = lambda name: '0.25.1'
        builtins.__import__ = import_with_vllm_logging
        names = ('VLLM_CONFIGURE_LOGGING', 'VLLM_LOGGING_CONFIG_PATH')
        for case, previous in enumerate(((None, None), ('1', None), ('0', None), ('1', 'custom.json'))):
            for name, value in zip(names, previous):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            for fail_import in (False, True):
                path = directory / f'protected-{case}-{fail_import}.log'
                setup_logging_for_experiment(path, '%(message)s')
                handlers = root.handlers[:]
                handler = next(item for item in handlers if isinstance(item, logging.FileHandler))
                stream = handler.stream
                root.info('before')
                try:
                    loaded = vllm_trainer._load_nccl_api()
                    assert not fail_import
                    assert loaded == (api.NCCLWeightTransferEngine, api.NCCLTrainerSendWeightsArgs)
                except ImportError:
                    assert fail_import
                root.info('after')
                assert root.handlers == handlers
                assert handler.stream is stream and not stream.closed
                assert not handler._closed and handler.mode == 'w'
                assert path.read_text().splitlines() == ['before', 'after']
                assert tuple(os.environ.get(name) for name in names) == previous
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = torch.nn.Linear(3, 2, bias=False)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.visual = torch.nn.Linear(3, 2, bias=False).requires_grad_(False)


def _source_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor
        from torch.distributed.fsdp import fully_shard

        torch.manual_seed(7)
        policy = _Policy()
        expected_decoder = policy.decoder.weight.detach().to(torch.bfloat16).clone()
        expected_head = policy.lm_head.weight.detach().to(torch.bfloat16).clone()
        mesh = init_device_mesh("cpu", (world_size,))
        fully_shard(policy.decoder, mesh=mesh)
        source = FullWeightParameterSource(policy)
        source.validate_manifest()

        gathered_dtypes: list[torch.dtype] = []
        original_full_tensor = DTensor.full_tensor

        def record_full_tensor(tensor, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            gathered_dtypes.append(tensor.dtype)
            return original_full_tensor(tensor, *args, **kwargs)

        DTensor.full_tensor = record_full_tensor

        assert source.manifest.names == (
            "decoder.weight",
            "lm_head.weight",
        )
        assert source.manifest.dtype_names == ("bfloat16", "bfloat16")
        assert source.manifest.shapes == ((2, 3), (4, 2))
        if rank == 0:
            tensors = dict(source.rank_zero_iterator())
            torch.testing.assert_close(tensors["decoder.weight"], expected_decoder)
            torch.testing.assert_close(tensors["lm_head.weight"], expected_head)
        else:
            source.participate()
        assert gathered_dtypes == [torch.bfloat16]
        DTensor.full_tensor = original_full_tensor
        dist.barrier()
    finally:
        if "original_full_tensor" in locals():
            DTensor.full_tensor = original_full_tensor
        dist.destroy_process_group()


def test_two_rank_source_gathers_shards_and_streams_native_head(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _source_worker,
        args=(2, str(tmp_path / "source_init")),
        nprocs=2,
        join=True,
    )


def test_source_requires_distributed_initialization() -> None:
    if dist.is_initialized():
        pytest.skip("test requires no ambient process group")
    with pytest.raises(RuntimeError, match="torch.distributed initialization"):
        FullWeightParameterSource(_Policy())


def test_source_has_no_decoupled_process_group_option() -> None:
    with pytest.raises(TypeError, match="process_group"):
        FullWeightParameterSource(_Policy(), process_group=object())  # type: ignore[call-arg]


@pytest.mark.parametrize("change", ["count", "extra", "name", "shape", "dtype"])
def test_tensor_stream_must_match_manifest(change):
    manifest = WeightManifest(("decoder.weight",), ("bfloat16",), ((2, 3),))
    tensors = [("decoder.weight", torch.zeros(2, 3, dtype=torch.bfloat16))]
    if change == "count":
        tensors.clear()
    elif change == "extra":
        tensors *= 2
    elif change == "name":
        tensors[0] = ("lm_head.weight", tensors[0][1])
    elif change == "shape":
        tensors[0] = ("decoder.weight", torch.zeros(3, 2, dtype=torch.bfloat16))
    else:
        tensors[0] = ("decoder.weight", tensors[0][1].float())
    with pytest.raises(ValueError):
        list(manifest.validate_parameters(tensors))


def test_all_servers_receive_one_version_before_any_resume():
    manifest = WeightManifest(("decoder.weight",), ("bfloat16",), ((2, 3),))
    send_started = Event()
    events = []
    publications = []

    class Client:
        def __init__(self, name):
            self.name = name

        def world_size(self):
            return 2

        def initialize_transfer(self, **kwargs):
            assert kwargs["world_size"] == 5
            assert kwargs["rank_offset"] == (1 if self.name == "a" else 3)

        def pause_and_drain(self):
            events.append((self.name, "pause"))

        def start_update(self, publication):
            publications.append(publication)
            events.append((self.name, "start"))

        def receive_update(self, supplied, *, packed):
            assert supplied == manifest and packed is True
            assert send_started.wait(5)
            events.append((self.name, "receive"))

        def finish_update(self):
            assert sum(event == "receive" for _, event in events) == 2
            events.append((self.name, "finish"))

        def resume(self, publication):
            assert sum(event == "finish" for _, event in events) == 2
            assert publication == publications[0]
            events.append((self.name, "resume"))

    trainer = NS(
        rendezvous=lambda: ("trainer", 12345), initialize=lambda **kwargs: None
    )
    transfer = VllmNcclWeightTransfer([Client("a"), Client("b")], trainer=trainer)
    result = transfer.transfer(
        manifest, send_weights=send_started.set, weight_version="a" * 32
    )
    assert result == publications[0] == publications[1]
    assert result["manifest_sha256"] == manifest.sha256
    assert sum(event == "resume" for _, event in events) == 2


def test_failed_receive_never_finishes_or_resumes_any_server():
    sent = Event()

    def receive(*args, **kwargs):
        assert sent.wait(5)
        raise RuntimeError("receiver failed")

    def forbidden(*args):
        pytest.fail("failed transfer resumed")

    client = NS(
        world_size=lambda: 1,
        initialize_transfer=lambda **kwargs: None,
        pause_and_drain=lambda: None,
        start_update=lambda publication: None,
        receive_update=receive,
        finish_update=forbidden,
        resume=forbidden,
    )
    trainer = NS(
        rendezvous=lambda: ("trainer", 12345), initialize=lambda **kwargs: None
    )
    transfer = VllmNcclWeightTransfer([client], trainer=trainer)
    manifest = WeightManifest(("decoder.weight",), ("bfloat16",), ((2, 3),))
    with pytest.raises(RuntimeError, match="receiver failed"):
        transfer.transfer(manifest, send_weights=sent.set)


def test_native_control_client_pins_server_instance_and_has_no_v1_alias():
    responses = iter(
        [
            {"world_size": 1, "server_instance_id": "original"},
            {"status": "paused", "server_instance_id": "replacement"},
        ]
    )
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        payload = next(responses)
        return NS(raise_for_status=lambda: None, json=lambda: payload)

    client = VllmControlClient(
        "http://localhost:8000",
        auth_headers={"Authorization": "Bearer test"},
        session=NS(request=request),
    )
    assert client.world_size() == 1
    with pytest.raises(RuntimeError, match="instance changed"):
        client.pause_and_drain()
    assert [call[1] for call in calls] == [
        "http://localhost:8000/weights/info",
        "http://localhost:8000/weights/pause",
    ]
    assert all(call[2]["headers"] == {"Authorization": "Bearer test"} for call in calls)
    with pytest.raises(ValueError, match="without /v1"):
        VllmControlClient("http://localhost:8000/v1", auth_headers={})


POLICY = {"kind": "policy_weights", "policy_sha256": "c" * 64, "policy_iteration": 4}


def test_policy_publication_starts_with_its_manifest_and_resumes_its_label():
    """A Flash publication runs the same fenced sequence; only its start carries the manifest."""
    manifest = WeightManifest(("model.language_model.layers.0.mlp.gate.weight",), ("bfloat16",), ((7, 4),))
    details = {"names": list(manifest.names), "iteration": 4}
    sent = Event()
    calls = []
    responses = {
        "/weights/info": {"world_size": 1},
        "/weights/pause": {"status": "paused"},
        "/weights/start": {"status": "started"},
        "/weights/update": {"status": "received"},
        "/weights/finish": {"status": "finished"},
        "/weights/resume": {"status": "resumed", "adapter": {
            "name": "base", "id": 0, "path": None, "server_instance_id": "one", **POLICY}},
    }

    def request(method, url, **kwargs):
        path = url.removeprefix("http://server:1")
        calls.append((path, kwargs["json"]))
        if path == "/weights/update":
            assert sent.wait(5)
        payload = {**responses.get(path, {"status": "initialized"}), "server_instance_id": "one"}
        return NS(raise_for_status=lambda: None, json=lambda: payload)

    client = VllmControlClient("http://server:1", auth_headers={}, session=NS(request=request))
    trainer = NS(rendezvous=lambda: ("learner", 23456), initialize=lambda **kwargs: None)
    transfer = VllmNcclWeightTransfer([client], trainer=trainer, packed=False)
    assert transfer.transfer(manifest, send_weights=sent.set, policy=(POLICY, details)) == POLICY
    bodies = dict(calls)
    assert bodies["/weights/start"] == {**POLICY, "manifest": details}
    assert bodies["/weights/update"] == {"update_info": manifest.update_info(packed=False)}
    assert client.adapter_identity["policy_sha256"] == "c" * 64
    # Another label resumed is refused, and a full-weight label never carries a manifest.
    responses["/weights/resume"]["adapter"] = {**responses["/weights/resume"]["adapter"], "policy_iteration": 5}
    with pytest.raises(RuntimeError, match="different publication"):
        client.resume(POLICY)
    with pytest.raises(ValueError, match="carries a manifest"):
        client.start_update({"kind": "full_weight", "weight_version": "a" * 32, "manifest_sha256": "b" * 64},
                            manifest=details)
    with pytest.raises(ValueError, match="carries a manifest"):
        client.start_update(POLICY)


@pytest.mark.parametrize(
    "label",
    [
        {**POLICY, "policy_sha256": "C" * 64},
        {**POLICY, "policy_sha256": "c" * 63},
        {**POLICY, "policy_iteration": -2},
        {**POLICY, "policy_iteration": "4"},
        {**POLICY, "weight_version": "a" * 32},
        {key: value for key, value in POLICY.items() if key != "policy_iteration"},
    ],
)
def test_policy_labels_are_exact(label):
    from training.policy_sync.vllm_nccl import validate_publication

    assert validate_publication(dict(POLICY)) == POLICY
    with pytest.raises(ValueError):
        validate_publication(label)
