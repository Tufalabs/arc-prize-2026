"""The engine surface consumed by vllmx, without importing the GPU runtime."""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable


class EngineLike(Protocol):
    """Render locally and generate through either a local engine or a transport.

    Remote transports expose their additional contract as ``RemoteEngineLike``.
    """

    model_config: Any

    def get_tokenizer(self) -> Any:
        """Return the tokenizer used to render inputs for this engine."""
        ...

    def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        *,
        lora_request: Any = None,
    ) -> AsyncIterator[Any]:
        """Yield engine outputs carrying the consumed IDs and sampled evidence."""
        ...

    async def abort(self, request_id: str) -> None:
        """Finish ownership of a cancelled request under the engine's policy."""
        ...


@runtime_checkable
class RemoteEngineLike(EngineLike, Protocol):
    """An explicit transport capability, including opaque request context."""

    remote_engine: Literal[True]
    metadata: dict[str, Any]

    async def connect(self) -> None:
        """Load and verify the serving engine's rendering identity."""
        ...

    async def aclose(self) -> None:
        """Finish owned requests and surface any terminal transport failure."""
        ...

    @property
    def lora_request(self) -> Any:
        """Return the immutable adapter selected by the connected model alias."""
        ...

    async def allows_generation_replay(self) -> bool:
        """Declare whether another generation attempt is permitted."""
        ...

    def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        *,
        lora_request: Any = None,
        request_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[Any]:
        """Preserve opaque owner context alongside exact engine inputs."""
        ...


def adapter_identity(
    lora_request: Any, *, model: str | None = None, server_instance_id: str | None = None
) -> dict[str, Any]:
    """Use one trace identity for base/LoRA policies and local/remote engines."""
    if lora_request is None:
        return {"name": model, "id": 0, "path": None, "server_instance_id": server_instance_id}
    coordinates = (
        lora_request
        if isinstance(lora_request, dict)
        else {name: getattr(lora_request, name) for name in ("lora_name", "lora_int_id", "lora_path")}
    )
    return {
        "name": coordinates["lora_name"],
        "id": coordinates["lora_int_id"],
        "path": coordinates["lora_path"],
        "server_instance_id": server_instance_id,
    }
