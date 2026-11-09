from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import logging
import hashlib
import json
import os
from queue import Queue
from threading import Event, Thread, Timer
from typing import Any, Protocol
import uuid

import requests
import torch

from training.vllm_runtime import native_url


_CONTROL_TIMEOUT_SECONDS = 120.0
_TRANSFER_TIMEOUT_SECONDS = 30.0 * 60.0
# Exit 70 is reserved for a stalled trainer send in vllm_trainer.py.
_STALLED_INITIALIZATION_EXIT_CODE = 71
log = logging.getLogger(__name__)


def _terminate_blocked_initialization(error: BaseException | None = None) -> None:
    detail = "timed out" if error is None else f"lost a server: {error}"
    log.critical(
        "vLLM NCCL transfer initialization %s; terminating rank zero so "
        "torchrun can fail-stop all trainer ranks.",
        detail,
    )
    os._exit(_STALLED_INITIALIZATION_EXIT_CODE)


@dataclass(frozen=True, slots=True)
class WeightManifest:
    names: tuple[str, ...]
    dtype_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        size = len(self.names)
        if size == 0:
            raise ValueError("Full-weight publication requires at least one parameter.")
        if len(self.dtype_names) != size or len(self.shapes) != size:
            raise ValueError("Weight manifest names, dtypes, and shapes must align.")
        if len(set(self.names)) != size:
            raise ValueError("Weight manifest parameter names must be unique.")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("Weight manifest names must be nonempty strings.")
        if any(dtype != "bfloat16" for dtype in self.dtype_names):
            raise ValueError("Full-weight publication requires bfloat16 tensors.")
        if any(
            type(size) is not int or size <= 0
            for shape in self.shapes
            for size in shape
        ):
            raise ValueError("Weight manifest dimensions must be positive integers.")

    @classmethod
    def from_named_parameters(
        cls, parameters: Iterable[tuple[str, torch.Tensor]]
    ) -> WeightManifest:
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[tuple[int, ...]] = []
        for name, parameter in parameters:
            names.append(str(name))
            dtype_names.append(str(parameter.dtype).removeprefix("torch."))
            shapes.append(tuple(int(size) for size in parameter.shape))
        return cls(tuple(names), tuple(dtype_names), tuple(shapes))

    def update_info(self, *, packed: bool) -> dict[str, object]:
        return {
            "names": list(self.names),
            "dtype_names": list(self.dtype_names),
            "shapes": [list(shape) for shape in self.shapes],
            "packed": packed,
        }

    @property
    def sha256(self) -> str:
        """Hash tensor coordinates, independently of transport packing."""
        value = self.update_info(packed=False)
        del value["packed"]
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_update_info(cls, value: Any) -> WeightManifest:
        if not isinstance(value, dict) or set(value) != {
            "names",
            "dtype_names",
            "shapes",
            "packed",
        }:
            raise ValueError("Weight update requires the exact tensor manifest.")
        if (
            type(value["packed"]) is not bool
            or any(
                not isinstance(value[key], list)
                for key in ("names", "dtype_names", "shapes")
            )
            or any(not isinstance(shape, list) for shape in value["shapes"])
        ):
            raise ValueError("Invalid weight manifest fields.")
        return cls(
            tuple(value["names"]),
            tuple(value["dtype_names"]),
            tuple(tuple(shape) for shape in value["shapes"]),
        )

    def validate_parameters(
        self, parameters: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Refuse missing, reordered, or differently typed transfer tensors."""
        count = 0
        for name, tensor in parameters:
            if count >= len(self.names) or (
                name != self.names[count]
                or str(tensor.dtype).removeprefix("torch.") != self.dtype_names[count]
                or tuple(tensor.shape) != self.shapes[count]
            ):
                raise ValueError(
                    "Publication tensor differs from the declared manifest."
                )
            count += 1
            yield name, tensor
        if count != len(self.names):
            raise ValueError("Publication is missing declared tensors.")


def _is_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_publication(value: Any) -> dict[str, Any]:
    """Validate an immutable publication label.

    ``full_weight``: a version and the tensor coordinates' digest, not a
    content digest. ``policy_weights`` (Flash merged weights): the sha256 of
    the publication's manifest, which carries every tensor's checksum, and its
    trainer iteration (training/flash_policy_weights.py).
    """
    if isinstance(value, dict) and value.get("kind") == "policy_weights":
        if (
            set(value) != {"kind", "policy_sha256", "policy_iteration"}
            or not _is_digest(value["policy_sha256"], 64)
            or type(value["policy_iteration"]) is not int
            or value["policy_iteration"] < -1
        ):
            raise ValueError(
                "Policy-weights publication requires kind, policy_sha256 and policy_iteration."
            )
        return dict(value)
    if (
        not isinstance(value, dict)
        or set(value) != {"kind", "weight_version", "manifest_sha256"}
        or value["kind"] != "full_weight"
    ):
        raise ValueError(
            "Full-weight publication requires kind, version, and manifest digest."
        )
    for key, length in (("weight_version", 32), ("manifest_sha256", 64)):
        if not _is_digest(value[key], length):
            raise ValueError(f"Invalid full-weight {key}.")
    return dict(value)


class NcclTrainerEndpoint(Protocol):
    def rendezvous(self) -> tuple[str, int]: ...

    def initialize(
        self, *, master_address: str, master_port: int, world_size: int
    ) -> None: ...


class _Response(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class _Session(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> _Response: ...


class VllmControlClient:
    """Single-attempt controls owned by the checked native vLLM service."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_headers: Mapping[str, str],
        session: _Session | None = None,
    ) -> None:
        self.root_url = native_url(base_url, "")
        self._headers = dict(auth_headers)
        self._session = session or requests.Session()
        self.server_instance_id: str | None = None
        self.adapter_identity: dict[str, Any] | None = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        params: Mapping[str, object] | None = None,
        timeout: float = _CONTROL_TIMEOUT_SECONDS,
    ) -> Any:
        response = self._session.request(
            method,
            self.root_url + path,
            headers=self._headers,
            json=json,
            params=params,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        instance = payload.get("server_instance_id")
        if not isinstance(instance, str) or not instance:
            raise RuntimeError("Native weight controls omitted the server instance.")
        if self.server_instance_id is not None and instance != self.server_instance_id:
            raise RuntimeError("Native weight-control server instance changed.")
        self.server_instance_id = instance
        expected_status = {
            "/weights/init": "initialized",
            "/weights/start": "started",
            "/weights/update": "received",
            "/weights/finish": "finished",
        }.get(path)
        if expected_status is not None and payload.get("status") != expected_status:
            raise RuntimeError(
                f"Native weight control did not confirm {expected_status}."
            )
        return payload

    def world_size(self) -> int:
        payload = self._request("GET", "/weights/info")
        world_size = payload["world_size"]
        if type(world_size) is not int or world_size < 1:
            raise RuntimeError(f"vLLM returned invalid world size {world_size}.")
        return world_size

    def pause_and_drain(self) -> None:
        payload = self._request(
            "POST",
            "/weights/pause",
            json={},
            timeout=_TRANSFER_TIMEOUT_SECONDS,
        )
        if payload.get("status") != "paused":
            raise RuntimeError(f"vLLM did not confirm pause: {payload!r}")

    def initialize_transfer(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        self._request(
            "POST",
            "/weights/init",
            json={
                "init_info": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                }
            },
            timeout=_TRANSFER_TIMEOUT_SECONDS,
        )

    def start_update(
        self, publication: dict[str, Any], *, manifest: dict[str, Any] | None = None
    ) -> None:
        """Begin an update; a policy-weights start also carries its manifest."""
        body = validate_publication(publication)
        if (manifest is not None) != (body["kind"] == "policy_weights"):
            raise ValueError("Only a policy-weights publication carries a manifest.")
        if manifest is not None:
            body["manifest"] = manifest
        self._request("POST", "/weights/start", json=body)

    def receive_update(self, manifest: WeightManifest, *, packed: bool) -> None:
        self._request(
            "POST",
            "/weights/update",
            json={"update_info": manifest.update_info(packed=packed)},
            timeout=_TRANSFER_TIMEOUT_SECONDS,
        )

    def finish_update(self) -> None:
        self._request("POST", "/weights/finish", json={})

    def resume(self, publication: dict[str, Any]) -> None:
        payload = self._request("POST", "/weights/resume", json={})
        if payload.get("status") != "resumed":
            raise RuntimeError(f"vLLM did not confirm resume: {payload!r}")
        adapter = payload.get("adapter")
        if (
            not isinstance(adapter, dict)
            or set(adapter)
            != {"name", "id", "path", "server_instance_id", *publication}
            or type(adapter["id"]) is not int
            or adapter["id"] != 0
            or adapter["path"] is not None
            or not isinstance(adapter["name"], str)
            or not adapter["name"]
            or adapter["server_instance_id"] != self.server_instance_id
            or any(adapter[key] != value for key, value in publication.items())
        ):
            raise RuntimeError("Native engine resumed a different publication.")
        self.adapter_identity = adapter


class _DaemonCallGroup:
    """Run terminal control calls without keeping a failed trainer alive."""

    def __init__(self, actions: Sequence[Callable[[], None]]) -> None:
        self._count = len(actions)
        self._completed: Queue[BaseException | None] = Queue()
        self._errors: Queue[BaseException] = Queue()
        for action in actions:
            Thread(target=self._run, args=(action,), daemon=True).start()

    def _run(self, action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as exc:  # noqa: BLE001 - propagate worker failures
            self._errors.put(exc)
            self._completed.put(exc)
        else:
            self._completed.put(None)

    def wait(self, *, fail_fast: bool) -> list[BaseException]:
        errors: list[BaseException] = []
        for _ in range(self._count):
            error = self._completed.get()
            if error is not None:
                errors.append(error)
                if fail_fast:
                    break
        return errors

    def wait_for_error(self) -> BaseException:
        return self._errors.get()


def _run_on_all(
    clients: Sequence[VllmControlClient],
    action: Callable[[VllmControlClient], None],
) -> None:
    calls = _DaemonCallGroup(
        [lambda client=client: action(client) for client in clients]
    )
    errors = calls.wait(fail_fast=True)
    if errors:
        raise RuntimeError(
            "vLLM control operation failed; all servers remain fenced."
        ) from errors[0]


class VllmNcclWeightTransfer:
    """Rank-zero control plane for one persistent multi-server NCCL group."""

    def __init__(
        self,
        clients: Sequence[VllmControlClient],
        *,
        trainer: NcclTrainerEndpoint,
        packed: bool = True,
    ) -> None:
        if not clients:
            raise ValueError("NCCL weight transfer requires at least one vLLM server.")
        self._clients = tuple(clients)
        self._trainer = trainer
        self._packed = packed
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        server_world_sizes = [client.world_size() for client in self._clients]
        transfer_world_size = 1 + sum(server_world_sizes)
        master_address, master_port = self._trainer.rendezvous()
        rank_offsets: list[int] = []
        next_rank = 1
        for server_world_size in server_world_sizes:
            rank_offsets.append(next_rank)
            next_rank += server_world_size

        calls = _DaemonCallGroup(
            [
                lambda client=client, rank_offset=rank_offset: (
                    client.initialize_transfer(
                        master_address=master_address,
                        master_port=master_port,
                        rank_offset=rank_offset,
                        world_size=transfer_world_size,
                    )
                )
                for client, rank_offset in zip(self._clients, rank_offsets, strict=True)
            ]
        )
        trainer_finished = Event()

        def fail_stop_on_server_error() -> None:
            error = calls.wait_for_error()
            if not trainer_finished.is_set():
                _terminate_blocked_initialization(error)

        Thread(target=fail_stop_on_server_error, daemon=True).start()
        watchdog = Timer(
            _TRANSFER_TIMEOUT_SECONDS,
            _terminate_blocked_initialization,
        )
        watchdog.daemon = True
        watchdog.start()
        try:
            self._trainer.initialize(
                master_address=master_address,
                master_port=master_port,
                world_size=transfer_world_size,
            )
        finally:
            trainer_finished.set()
            watchdog.cancel()
        errors = calls.wait(fail_fast=True)
        if errors:
            raise RuntimeError(
                "vLLM transfer initialization failed; servers remain fenced."
            ) from errors[0]
        self._initialized = True

    def transfer(
        self,
        manifest: WeightManifest,
        *,
        send_weights: Callable[[], None],
        weight_version: str | None = None,
        policy: tuple[dict[str, Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Drain, update, and resume every server, leaving them paused on error.

        ``policy`` is a Flash publication's (label, manifest); without it the
        publication is a full-weight version of ``manifest``.
        """
        if policy is None:
            publication = validate_publication(
                {
                    "kind": "full_weight",
                    "weight_version": weight_version or uuid.uuid4().hex,
                    "manifest_sha256": manifest.sha256,
                }
            )
            details = None
        else:
            publication, details = validate_publication(policy[0]), policy[1]
            if publication["kind"] != "policy_weights":
                raise ValueError("A policy transfer requires a policy-weights label.")
        self.initialize()
        _run_on_all(self._clients, lambda client: client.pause_and_drain())
        if details is None:
            _run_on_all(self._clients, lambda client: client.start_update(publication))
        else:
            _run_on_all(
                self._clients,
                lambda client: client.start_update(publication, manifest=details),
            )

        calls = _DaemonCallGroup(
            [
                lambda client=client: client.receive_update(
                    manifest, packed=self._packed
                )
                for client in self._clients
            ]
        )
        try:
            send_weights()
        except BaseException:
            # Failed transfers have no recovery. Receiver calls are daemonized so a
            # terminal trainer failure cannot keep the process alive until their
            # 30-minute HTTP timeout expires.
            raise
        errors = calls.wait(fail_fast=True)
        if errors:
            raise errors[0]

        _run_on_all(self._clients, lambda client: client.finish_update())
        _run_on_all(self._clients, lambda client: client.resume(publication))
        return publication
