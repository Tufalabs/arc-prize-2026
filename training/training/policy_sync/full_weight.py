from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.distributed as dist

from training.policy_sync.vllm_nccl import WeightManifest


class FullWeightParameterSource:
    """Collectively materialize canonical full weights one tensor at a time."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        transfer_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Full-weight publication requires torch.distributed initialization."
            )
        self._model = model
        self._transfer_dtype = transfer_dtype
        self._parameters = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ),
                key=lambda item: item[0],
            )
        )
        self.manifest = WeightManifest(
            names=tuple(name for name, _parameter in self._parameters),
            dtype_names=tuple(
                str(transfer_dtype).removeprefix("torch.")
                for _name, _parameter in self._parameters
            ),
            shapes=tuple(
                tuple(int(size) for size in parameter.shape)
                for _name, parameter in self._parameters
            ),
        )

    def validate_manifest(self) -> None:
        manifests: list[WeightManifest | None] = [None] * dist.get_world_size()
        dist.all_gather_object(
            manifests,
            self.manifest,
        )
        if any(candidate != self.manifest for candidate in manifests):
            raise RuntimeError(
                "Full-weight publication ranks do not share one parameter manifest."
            )

    def rank_zero_iterator(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield tensors on rank zero; every rank must exhaust this iterator."""
        parameters = self._gather_parameters()
        if dist.get_rank() == 0:
            yield from self.manifest.validate_parameters(parameters)
        else:
            yield from parameters

    def _gather_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Gather in manifest order without retaining a whole GPU state dict."""
        from torch.distributed.tensor import DTensor

        rank = dist.get_rank()
        for name, parameter in self._parameters:
            if isinstance(parameter, DTensor):
                materialized = (
                    parameter.detach().to(dtype=self._transfer_dtype).full_tensor()
                )
            elif rank == 0:
                materialized = parameter.detach()
            else:
                continue
            if rank == 0:
                transfer_tensor = materialized.to(
                    dtype=self._transfer_dtype,
                    copy=(materialized.dtype != self._transfer_dtype),
                ).contiguous()
                yield name, transfer_tensor
            del materialized

    def participate(self) -> None:
        """Enter every DTensor gather on a nonzero trainer rank."""
        if dist.get_rank() == 0:
            raise RuntimeError(
                "Rank zero must stream rank_zero_iterator() into the transfer endpoint."
            )
        for _item in self.rank_zero_iterator():
            raise AssertionError(
                "Nonzero publication rank unexpectedly yielded a tensor."
            )
