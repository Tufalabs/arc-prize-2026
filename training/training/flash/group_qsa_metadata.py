"""Linear-space layout for group-packed QSA.

Block IDs address documents concatenated in units of ceil(length / 4) blocks.
This equals the selector's global block IDs for the model's aligned documents.
Groups never cross documents, including documents shorter than a group.
"""

from dataclasses import dataclass

import torch

GROUP = 8
SLOTS = GROUP * 513


@dataclass
class GroupLayout:
    starts: torch.Tensor
    ends: torch.Tensor
    document_starts: torch.Tensor
    block_offsets: torch.Tensor
    key_starts: torch.Tensor
    key_ends: torch.Tensor


def group_layout(n, device, packed_lengths=None):
    lengths = (n,) if packed_lengths is None else tuple(packed_lengths)
    if not lengths or any(x <= 0 for x in lengths) or sum(lengths) != n:
        raise ValueError("Invalid group QSA document lengths")
    if len(lengths) == 1:
        starts = torch.arange(0, n, GROUP, device=device, dtype=torch.int32)
        keys = torch.arange(0, n, 4, device=device, dtype=torch.int32)
        return GroupLayout(starts, torch.full_like(starts, n),
                           torch.zeros_like(starts), torch.zeros_like(starts),
                           keys, torch.full_like(keys, n))
    starts, ends, documents, offsets, keys, key_ends = [], [], [], [], [], []
    start, offset = 0, 0
    for length in lengths:
        groups = range(start, start + length, GROUP)
        starts.extend(groups)
        ends.extend([start + length] * len(groups))
        documents.extend([start] * len(groups))
        offsets.extend([offset] * len(groups))
        local_keys = range(start, start + length, 4)
        keys.extend(local_keys)
        key_ends.extend([start + length] * len(local_keys))
        start += length
        offset += (length + 3) // 4
    return GroupLayout(*(
        torch.tensor(x, device=device, dtype=torch.int32)
        for x in (starts, ends, documents, offsets, keys, key_ends)
    ))


def reference_union(chosen, packed_lengths=None):
    """CPU oracle for metadata tests, independent of the Triton sorting kernel."""
    layout = group_layout(len(chosen), "cpu", packed_lengths)
    unions = []
    for start, end, document, offset in zip(
        layout.starts.tolist(), layout.ends.tolist(),
        layout.document_starts.tolist(), layout.block_offsets.tolist(), strict=True,
    ):
        members = {}
        for lane, query in enumerate(range(start, min(start + GROUP, end))):
            local = query - document
            if local < 2051:
                blocks = range(offset, offset + local // 4 + 1)
            else:
                blocks = set(chosen[query].tolist()) - {-1}
                if (local + 1) % 4:
                    blocks.add(offset + (local + 1) // 4)
            for block in blocks:
                if offset <= block <= offset + local // 4:
                    members[block] = members.get(block, 0) | (1 << lane)
        unions.append(sorted(members.items()))
    return unions


def reverse_groups(ids, counts, blocks):
    """Block-to-union-edge CSR, with a stable increasing-group reduction order."""
    if ids.dtype != torch.int32 or counts.dtype != torch.int32 or ids.device != counts.device:
        raise ValueError("group QSA reverse_groups: requires int32 metadata on the same device")
    slots = torch.arange(ids.shape[1], device=ids.device)
    keys = torch.where(slots[None] < counts[:, None], ids, blocks).flatten()
    ordered, order = torch.sort(keys, stable=True)
    offsets = torch.searchsorted(ordered, torch.arange(blocks + 1, device=ids.device, dtype=keys.dtype))
    return order.to(torch.int32), offsets
