from __future__ import annotations

from collections.abc import Iterable
import logging
import importlib.metadata
import os
import socket
import threading
from typing import Any

import torch


log = logging.getLogger(__name__)
_DEFAULT_SEND_TIMEOUT_SECONDS = 30.0 * 60.0
_STALLED_SEND_EXIT_CODE = 70


def _terminate_stalled_send() -> None:
    log.critical(
        "vLLM NCCL weight publication exceeded its timeout; terminating rank "
        "zero so torchrun can fail-stop all trainer ranks."
    )
    os._exit(_STALLED_SEND_EXIT_CODE)


def _load_nccl_api() -> tuple[Any, Any]:
    """Import vLLM lazily so CPU/config-only processes need not install it."""
    if importlib.metadata.version("vllm") != "0.25.1":
        raise RuntimeError("Full-weight publication requires pinned vLLM 0.25.1.")
    # vLLM's import-time dictConfig closes the application's file handlers,
    # including fresh mode-w handlers that cannot reopen after being closed.
    # Keep the override scoped so separately launched engines retain logging.
    logging_environment = {
        name: os.environ.get(name)
        for name in ("VLLM_CONFIGURE_LOGGING", "VLLM_LOGGING_CONFIG_PATH")
    }
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ.pop("VLLM_LOGGING_CONFIG_PATH", None)
    try:
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )
    finally:
        for name, value in logging_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    return NCCLWeightTransferEngine, NCCLTrainerSendWeightsArgs


def _available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        return int(listener.getsockname()[1])


class VllmNcclTrainerEndpoint:
    """Trainer rank in vLLM 0.25.1's stateless NCCL transfer group."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        packed: bool = True,
        send_timeout_seconds: float = _DEFAULT_SEND_TIMEOUT_SECONDS,
    ) -> None:
        configured_host = str(
            host or os.environ.get("ARC3_WEIGHT_TRANSFER_HOST", "") or ""
        ).strip()
        self._host = configured_host or socket.gethostname()
        self._port = int(port) if port is not None else _available_tcp_port()
        if not self._host:
            raise ValueError("NCCL weight-transfer host must not be empty.")
        if not (0 < self._port < 65536):
            raise ValueError("NCCL weight-transfer port must be in 1..65535.")
        if float(send_timeout_seconds) <= 0.0:
            raise ValueError("NCCL weight-transfer send timeout must be positive.")
        self._packed = bool(packed)
        self._send_timeout_seconds = float(send_timeout_seconds)
        self._group: Any | None = None
        self._engine: Any | None = None
        self._send_args_type: Any | None = None
        self._world_size: int | None = None

    def rendezvous(self) -> tuple[str, int]:
        return self._host, self._port

    def initialize(
        self, *, master_address: str, master_port: int, world_size: int
    ) -> None:
        if self._group is not None:
            if (
                str(master_address),
                int(master_port),
                int(world_size),
            ) != (*self.rendezvous(), self._world_size):
                raise RuntimeError(
                    "NCCL trainer endpoint was already initialized for a different "
                    "transfer group."
                )
            return
        if (str(master_address), int(master_port)) != self.rendezvous():
            raise ValueError(
                "NCCL trainer initialization must use the advertised rendezvous."
            )
        if int(world_size) < 2:
            raise ValueError("NCCL weight transfer requires at least one receiver.")
        engine, send_args_type = _load_nccl_api()
        self._group = engine.trainer_init(
            {
                "master_address": self._host,
                "master_port": self._port,
                "world_size": int(world_size),
            }
        )
        self._engine = engine
        self._send_args_type = send_args_type
        self._world_size = int(world_size)

    def send_weights(self, parameters: Iterable[tuple[str, torch.Tensor]]) -> None:
        if self._group is None or self._engine is None or self._send_args_type is None:
            raise RuntimeError("NCCL trainer endpoint is not initialized.")
        args = self._send_args_type(group=self._group, packed=self._packed)
        watchdog = threading.Timer(
            self._send_timeout_seconds,
            _terminate_stalled_send,
        )
        watchdog.daemon = True
        watchdog.start()
        try:
            self._engine.trainer_send_weights(iter(parameters), args)
        finally:
            watchdog.cancel()

    def close(self) -> None:
        group = self._group
        self._group = None
        self._engine = None
        self._send_args_type = None
        self._world_size = None
        destroy = getattr(group, "destroy", None)
        if callable(destroy):
            destroy()
