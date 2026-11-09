"""Rank 0 of vLLM's NCCL weight-transfer group, spoken without vLLM.

The Flash learner's environment never contains vLLM (docs/environments.md),
yet its servers join the publication group through vLLM's own code:
``init_weight_transfer_engine`` reaches ``worker_init_process_group``, which
builds a ``StatelessProcessGroup`` over a ``torch.distributed.TCPStore`` and a
``PyNcclCommunicator`` on it (the fork at 3bda1ba09,
``vllm/distributed/weight_transfer/nccl_common.py``). This endpoint plays rank
0 of that protocol with torch alone:

1. it hosts the TCPStore on the rendezvous port of every interface (rank 0
   is the store's server; ``use_libuv=False`` as vLLM's), which waits for
   every rank; the servers reach it by the learner's host name;
2. it publishes NCCL's unique id under ``broadcast_from/0/0``, pickled as the
   ``ncclUniqueId`` ctypes structure the workers unpickle and hand to
   ``ncclCommInitRank`` (``_unique_id_pickle``);
3. it joins the communicator as rank 0 and takes part in the communicator's
   warm-up, an all-reduce of one float32 zero;
4. each publication is then one ``ncclBroadcast`` from rank 0 per tensor, in
   manifest order, which each worker receives into a buffer of the declared
   shape (training/vllm_blackwell_worker.py).

NCCL is called through ctypes on the ``libnccl.so.2`` torch already loaded,
as vLLM's ``pynccl_wrapper`` does: torch 2.13's own ``torch.cuda.nccl.init_rank``
fails on Python 3.12 (``PY_SSIZE_T_CLEAN macro must be defined``, probe job
2925910). Torch and NCCL must match the servers': the learner and the harness
lock both pin torch 2.13.0 with nvidia-nccl-cu13 2.29.7.
"""

from __future__ import annotations

from collections.abc import Iterable
import ctypes
from datetime import timedelta
import logging
import os
import socket
import struct
import threading
from typing import Any

import torch


log = logging.getLogger(__name__)
_STORE_TIMEOUT_SECONDS = 300.0
_DEFAULT_SEND_TIMEOUT_SECONDS = 30.0 * 60.0
# Exit codes 70 and 71 are the 27B trainer's stalled send and initialization.
_STALLED_SEND_EXIT_CODE = 72
_UNIQUE_ID_BYTES = 128
# Where vLLM keeps the structure its workers unpickle the unique id as.
_VLLM_UNIQUE_ID = ("vllm.distributed.device_communicators.pynccl_wrapper", "ncclUniqueId")
_UNIQUE_ID_KEY = "broadcast_from/0/0"


def _unicode(text: str) -> bytes:
    data = text.encode()
    return b"\x8c" + struct.pack("<B", len(data)) + data


def _unique_id_pickle(unique_id: bytes) -> bytes:
    """``pickle.dumps(ncclUniqueId)`` as vLLM's rank 0 would write it, without importing vLLM.

    A ctypes structure pickles as ``_ctypes._unpickle(cls, (__dict__, raw
    bytes))``, its class named by module and qualified name; the workers
    resolve that name in their own vLLM.
    """
    unique_id = bytes(unique_id)
    if len(unique_id) != _UNIQUE_ID_BYTES:
        raise ValueError(f"an NCCL unique id has {_UNIQUE_ID_BYTES} bytes, not {len(unique_id)}")
    module, name = _VLLM_UNIQUE_ID
    return b"".join(
        (
            b"\x80\x04",  # PROTO 4
            _unicode("_ctypes"), _unicode("_unpickle"), b"\x93",  # STACK_GLOBAL
            _unicode(module), _unicode(name), b"\x93",
            b"}",  # EMPTY_DICT: the structure's __dict__
            b"C" + struct.pack("<B", _UNIQUE_ID_BYTES) + unique_id,  # SHORT_BINBYTES
            b"\x86\x86",  # TUPLE2 (state), TUPLE2 (class, state)
            b"R.",  # REDUCE, STOP
        )
    )


# NCCL's enums (nccl.h).
_NCCL_FLOAT32, _NCCL_BFLOAT16, _NCCL_SUM = 7, 9, 0


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * _UNIQUE_ID_BYTES)]


class _Nccl:
    """The few libnccl entry points rank 0 needs; every call checks its result."""

    def __init__(self, path: str = "libnccl.so.2") -> None:
        library = ctypes.CDLL(path)
        signatures = {
            "ncclGetErrorString": (ctypes.c_char_p, [ctypes.c_int]),
            "ncclGetUniqueId": (ctypes.c_int, [ctypes.POINTER(_UniqueId)]),
            "ncclCommInitRank": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId,
                                                ctypes.c_int]),
            "ncclAllReduce": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]),
            "ncclBroadcast": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]),
            "ncclCommAbort": (ctypes.c_int, [ctypes.c_void_p]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(library, name)
            function.restype, function.argtypes = restype, argtypes
        self._library = library

    def _check(self, result: int, call: str) -> None:
        if result != 0:
            message = self._library.ncclGetErrorString(result).decode(errors="replace")
            raise RuntimeError(f"{call} failed: NCCL error {result}: {message}")

    def unique_id(self) -> bytes:
        value = _UniqueId()
        self._check(self._library.ncclGetUniqueId(ctypes.byref(value)), "ncclGetUniqueId")
        return bytes(value)

    def init_rank(self, world_size: int, unique_id: bytes, rank: int) -> ctypes.c_void_p:
        value, comm = _UniqueId(), ctypes.c_void_p()
        ctypes.memmove(ctypes.addressof(value), unique_id, _UNIQUE_ID_BYTES)
        # ctypes releases the GIL: the servers' HTTP threads keep running.
        self._check(self._library.ncclCommInitRank(ctypes.byref(comm), world_size, value, rank), "ncclCommInitRank")
        return comm

    def all_reduce_sum(self, tensor: torch.Tensor, comm, stream: int) -> None:
        if tensor.dtype != torch.float32:
            raise ValueError("the warm-up all-reduce is float32")
        self._check(self._library.ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), _NCCL_FLOAT32,
                                                _NCCL_SUM, comm, stream), "ncclAllReduce")

    def broadcast(self, tensor: torch.Tensor, comm, stream: int) -> None:
        if tensor.dtype != torch.bfloat16:
            raise ValueError("policy weights are broadcast as BF16")
        self._check(self._library.ncclBroadcast(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
                                                _NCCL_BFLOAT16, 0, comm, stream), "ncclBroadcast")

    def abort(self, comm) -> None:
        self._check(self._library.ncclCommAbort(comm), "ncclCommAbort")


def _available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        return int(listener.getsockname()[1])


def _terminate_stalled_send() -> None:
    log.critical(
        "Flash policy-weights broadcast exceeded its timeout; terminating rank zero "
        "so torchrun can fail-stop all trainer ranks."
    )
    os._exit(_STALLED_SEND_EXIT_CODE)


class StatelessNcclTrainerEndpoint:
    """The learner's rank 0 in a vLLM weight-transfer group: rendezvous, init, broadcasts."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        device: torch.device | str | None = None,
        send_timeout_seconds: float = _DEFAULT_SEND_TIMEOUT_SECONDS,
    ) -> None:
        configured = str(host or os.environ.get("ARC3_WEIGHT_TRANSFER_HOST", "") or "").strip()
        self._host = configured or socket.gethostname()
        self._port = int(port) if port is not None else _available_tcp_port()
        if not self._host:
            raise ValueError("NCCL weight-transfer host must not be empty.")
        if not (0 < self._port < 65536):
            raise ValueError("NCCL weight-transfer port must be in 1..65535.")
        if float(send_timeout_seconds) <= 0.0:
            raise ValueError("NCCL weight-transfer send timeout must be positive.")
        self._send_timeout_seconds = float(send_timeout_seconds)
        # The GPU the publication's tensors live on (rank zero's); the current one if unset.
        self._requested_device = None if device is None else torch.device(device)
        self._nccl_factory = _Nccl
        self._nccl: _Nccl | None = None
        self._store: Any | None = None
        self._comm: Any | None = None
        self._device: torch.device | None = None
        self._world_size: int | None = None

    def rendezvous(self) -> tuple[str, int]:
        return self._host, self._port

    @property
    def initialized(self) -> bool:
        return self._comm is not None

    def initialize(self, *, master_address: str, master_port: int, world_size: int) -> None:
        if self._comm is not None:
            if (str(master_address), int(master_port), int(world_size)) != (*self.rendezvous(), self._world_size):
                raise RuntimeError("NCCL trainer endpoint was already initialized for a different transfer group.")
            return
        if (str(master_address), int(master_port)) != self.rendezvous():
            raise ValueError("NCCL trainer initialization must use the advertised rendezvous.")
        if int(world_size) < 2:
            raise ValueError("NCCL weight transfer requires at least one receiver.")
        from torch.distributed import TCPStore

        nccl = self._nccl_factory()
        device = self._requested_device
        if device is None or device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Every interface: on Kander a node's own name resolves to 127.0.1.1
        # locally (/etc/hosts) and to its cluster address elsewhere, so a
        # listener bound to the advertised name is unreachable from the
        # rollout nodes (probe job 2925914).
        listener.bind(("", self._port))
        listener.listen()
        descriptor = listener.detach()
        try:
            # Returns once every rank has connected (each server worker joins
            # from its init_weight_transfer_engine).
            store = TCPStore(
                host_name=self._host,
                port=self._port,
                world_size=int(world_size),
                is_master=True,
                timeout=timedelta(seconds=_STORE_TIMEOUT_SECONDS),
                use_libuv=False,
                master_listen_fd=descriptor,
            )
        except BaseException:
            socket.close(descriptor)
            raise
        unique_id = nccl.unique_id()
        store.set(_UNIQUE_ID_KEY, _unique_id_pickle(unique_id))
        with torch.cuda.device(device):
            comm = nccl.init_rank(int(world_size), unique_id, 0)
            # PyNcclCommunicator's warm-up on every rank.
            stream = torch.cuda.current_stream(device)
            nccl.all_reduce_sum(torch.zeros(1, device=device), comm, stream.cuda_stream)
            stream.synchronize()
        self._nccl, self._store, self._comm = nccl, store, comm
        self._device, self._world_size = device, int(world_size)
        log.info("joined the NCCL weight-transfer group of %d ranks at %s:%d as rank 0",
                 world_size, self._host, self._port)

    def send_weights(self, parameters: Iterable[tuple[str, torch.Tensor]]) -> None:
        """One broadcast per tensor, in order; returns once all have left this GPU."""
        if self._comm is None or self._device is None or self._nccl is None:
            raise RuntimeError("NCCL trainer endpoint is not initialized.")
        watchdog = threading.Timer(self._send_timeout_seconds, _terminate_stalled_send)
        watchdog.daemon = True
        watchdog.start()
        try:
            with torch.cuda.device(self._device):
                stream = torch.cuda.current_stream(self._device)
                for name, tensor in parameters:
                    if tensor.device != self._device or not tensor.is_contiguous():
                        raise ValueError(f"{name} must be a contiguous tensor on {self._device} to broadcast")
                    self._nccl.broadcast(tensor, self._comm, stream.cuda_stream)
                stream.synchronize()
        finally:
            watchdog.cancel()

    def close(self) -> None:
        comm, nccl = self._comm, self._nccl
        self._nccl = self._comm = self._store = self._device = self._world_size = None
        if comm is not None and nccl is not None:
            nccl.abort(comm)
