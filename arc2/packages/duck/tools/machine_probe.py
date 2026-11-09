"""The machine a traffic run lands on, measured before its engine starts: each
GPU pair's peer access and copy speed (large copies for bandwidth, small ones
for the latency the tensor pairs' all-reduces live at), and each card's
host-to-device speed (the host tier's path). One JSON object on stdout.

    python machine_probe.py > machine_p2p.json
"""

from __future__ import annotations

import json
import time

import torch


def _copy_rate(src: torch.Tensor, dst: torch.Tensor, reps: int) -> dict[str, float]:
    """Time ``reps`` copies of ``src`` into ``dst`` after a short warm-up."""
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    for device in {src.device, dst.device}:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(reps):
        dst.copy_(src, non_blocking=True)
    for device in {src.device, dst.device}:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    size = src.numel() * src.element_size()
    return {"GBps": round(size * reps / elapsed / 1e9, 3), "us_per_copy": round(elapsed / reps * 1e6, 2)}


def main() -> None:
    """Probe every pair of the run's GPUs and every card's host link."""
    count = torch.cuda.device_count()
    result: dict[str, object] = {"devices": [torch.cuda.get_device_name(i) for i in range(count)]}
    pairs = [(a, b) for a in range(count) for b in range(count) if a != b]
    result["peer_access"] = {f"{a}->{b}": torch.cuda.can_device_access_peer(a, b) for a, b in pairs}
    for label, size, reps in (("256MiB", 256 << 20, 10), ("128KiB", 128 << 10, 2000)):
        rates = {}
        for a, b in pairs:
            src = torch.empty(size, dtype=torch.uint8, device=a)
            dst = torch.empty(size, dtype=torch.uint8, device=b)
            rates[f"{a}->{b}"] = _copy_rate(src, dst, reps)
            del src, dst
        result[f"peer_copy_{label}"] = rates
    host = {}
    for i in range(count):
        pinned = torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True)
        on_card = torch.empty(256 << 20, dtype=torch.uint8, device=i)
        host[f"host->{i}"] = _copy_rate(pinned, on_card, 10)
        host[f"{i}->host"] = _copy_rate(on_card, pinned, 10)
        del pinned, on_card
    result["host_copy_256MiB"] = host
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
