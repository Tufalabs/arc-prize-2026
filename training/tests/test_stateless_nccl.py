"""The learner's vLLM-free rank 0 of the weight-transfer group (CPU, with fakes)."""

import contextlib
import ctypes
import pickle
import sys
import types

import pytest
import torch

from training.policy_sync import stateless_nccl
from training.policy_sync.stateless_nccl import StatelessNcclTrainerEndpoint, _unique_id_pickle


@pytest.fixture
def vllm_unique_id(monkeypatch):
    """vLLM's ncclUniqueId, where the workers' pickle names it."""
    names = ["vllm", "vllm.distributed", "vllm.distributed.device_communicators",
             "vllm.distributed.device_communicators.pynccl_wrapper"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class ncclUniqueId(ctypes.Structure):  # noqa: N801 - vLLM's name
        _fields_ = [("internal", ctypes.c_byte * 128)]

    ncclUniqueId.__module__ = names[-1]
    ncclUniqueId.__qualname__ = "ncclUniqueId"
    sys.modules[names[-1]].ncclUniqueId = ncclUniqueId
    return ncclUniqueId


def test_unique_id_unpickles_as_vllms_structure(vllm_unique_id):
    raw = bytes(range(128))
    value = pickle.loads(_unique_id_pickle(raw))
    assert type(value) is vllm_unique_id
    assert bytes(value) == raw == bytes(bytearray(value.internal))
    # What vLLM's own rank 0 writes for the same id reads back the same.
    reference = vllm_unique_id()
    ctypes.memmove(ctypes.addressof(reference), raw, 128)
    assert pickle.dumps(reference, protocol=4).replace(b"\x94", b"")[11:] == _unique_id_pickle(raw)[2:]
    assert bytes(pickle.loads(pickle.dumps(reference))) == bytes(value)
    with pytest.raises(ValueError, match="128 bytes"):
        _unique_id_pickle(raw[:-1])


class _Fakes:
    def __init__(self):
        self.events = []
        self.store_args = None
        self.values = {}


@pytest.fixture
def fakes(monkeypatch):
    state = _Fakes()

    class Store:
        def __init__(self, **kwargs):
            state.store_args = kwargs
            state.events.append("store")

        def set(self, key, value):
            state.values[key] = value
            state.events.append(("set", key))

    class Nccl:
        def unique_id(self):
            return b"\x07" * 128

        def init_rank(self, world, uid, rank):
            state.events.append(("init_rank", world, uid, rank))
            return "comm"

        def all_reduce_sum(self, tensor, comm, stream):
            state.events.append(("all_reduce", tensor.tolist(), comm, stream))

        def broadcast(self, tensor, comm, stream):
            state.events.append(("broadcast", tuple(tensor.shape), comm, stream))

        def abort(self, comm):
            state.events.append(("abort", comm))

    monkeypatch.setattr("torch.distributed.TCPStore", Store, raising=False)
    monkeypatch.setattr(stateless_nccl, "_Nccl", Nccl)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    stream = types.SimpleNamespace(synchronize=lambda: None, cuda_stream=11)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    # Tensors "on the device": CPU tensors stand in for cuda:0.
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *shape, device=None: zeros(*shape))
    return state


def test_initialize_hosts_the_store_publishes_the_id_and_warms_up(fakes):
    endpoint = StatelessNcclTrainerEndpoint(host="127.0.0.1", port=0 or 41234)
    assert endpoint.rendezvous() == ("127.0.0.1", 41234)
    with pytest.raises(ValueError, match="advertised rendezvous"):
        endpoint.initialize(master_address="other", master_port=41234, world_size=3)
    with pytest.raises(ValueError, match="at least one receiver"):
        endpoint.initialize(master_address="127.0.0.1", master_port=41234, world_size=1)
    endpoint.initialize(master_address="127.0.0.1", master_port=41234, world_size=3)
    args = fakes.store_args
    assert (args["host_name"], args["port"], args["world_size"], args["is_master"], args["use_libuv"]) == (
        "127.0.0.1", 41234, 3, True, False)
    assert isinstance(args["master_listen_fd"], int)
    # The id is set before rank 0 blocks in ncclCommInitRank, then the warm-up all-reduce.
    assert fakes.events == ["store", ("set", "broadcast_from/0/0"), ("init_rank", 3, b"\x07" * 128, 0),
                            ("all_reduce", [0.0], "comm", 11)]
    assert fakes.values["broadcast_from/0/0"] == _unique_id_pickle(b"\x07" * 128)
    # Idempotent for the same group; another group is refused.
    endpoint.initialize(master_address="127.0.0.1", master_port=41234, world_size=3)
    with pytest.raises(RuntimeError, match="different transfer group"):
        endpoint.initialize(master_address="127.0.0.1", master_port=41234, world_size=4)


def test_send_broadcasts_each_tensor_from_rank_zero_in_order(fakes, monkeypatch):
    endpoint = StatelessNcclTrainerEndpoint(host="127.0.0.1", port=41235)
    with pytest.raises(RuntimeError, match="not initialized"):
        endpoint.send_weights([])
    endpoint.initialize(master_address="127.0.0.1", master_port=41235, world_size=2)
    endpoint._device = torch.device("cpu")
    fakes.events.clear()
    endpoint.send_weights([("a", torch.zeros(2, 3)), ("b", torch.zeros(4, 1))])
    assert fakes.events == [("broadcast", (2, 3), "comm", 11), ("broadcast", (4, 1), "comm", 11)]
    with pytest.raises(ValueError, match="contiguous"):
        endpoint.send_weights([("t", torch.ones(3, 2).t())])
    endpoint.close()
    assert not endpoint.initialized and fakes.events[-1] == ("abort", "comm")


def test_host_defaults_to_the_configured_or_local_name(monkeypatch):
    monkeypatch.setenv("ARC3_WEIGHT_TRANSFER_HOST", "learner-node")
    assert StatelessNcclTrainerEndpoint(port=1).rendezvous() == ("learner-node", 1)
    monkeypatch.delenv("ARC3_WEIGHT_TRANSFER_HOST")
    monkeypatch.setattr(stateless_nccl.socket, "gethostname", lambda: "dgx09")
    host, port = StatelessNcclTrainerEndpoint().rendezvous()
    assert host == "dgx09" and 0 < port < 65536
