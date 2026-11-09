"""Read-only cgroup-v2 and pinned-allocator telemetry; no allocator changes."""

import json
import os
import threading
import time
from pathlib import Path


def find_memory_cgroup(
    proc_cgroup=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")
):
    for line in proc_cgroup.read_text().splitlines():
        if line.startswith("0::"):
            path = root / line.split("::", 1)[1].lstrip("/")
            while path != root and root in path.parents:
                limit = path / "memory.max"
                if limit.is_file() and limit.read_text().strip().isdigit():
                    return path
                path = path.parent
    return None


def sample(cgroup):
    result = {"timestamp": time.time(), "pid": os.getpid()}
    if cgroup is None:
        return {**result, "unavailable": True}
    for name, key in [
        ("memory.current", "current_bytes"),
        ("memory.peak", "peak_bytes"),
        ("memory.max", "limit_bytes"),
    ]:
        p = cgroup / name
        if p.is_file():
            value = p.read_text().strip()
            if value.isdigit():
                result[key] = int(value)
    for name in ("memory.stat", "memory.events"):
        p = cgroup / name
        if p.is_file():
            result[name] = {
                k: int(v)
                for k, v in (line.split() for line in p.read_text().splitlines())
            }
    return result


class HostMemoryMonitor:
    def __init__(self, output, interval=5):
        self.output = Path(output)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with self.output.open("a") as f:
            try:
                cgroup = find_memory_cgroup()
                while True:
                    f.write(json.dumps(sample(cgroup)) + "\n")
                    f.flush()
                    if self.stop_event.wait(self.interval):
                        break
            except (OSError, ValueError) as exc:
                f.write(
                    json.dumps({"timestamp": time.time(), "error": str(exc)}) + "\n"
                )
                f.flush()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval + 1)


def record_pinned(out, rank, step, phase):
    import torch

    stats = torch.cuda.memory.host_memory_stats()
    keys = (
        "allocated_bytes.current",
        "allocated_bytes.peak",
        "active_bytes.current",
        "active_bytes.peak",
        "num_host_alloc",
        "num_host_free",
    )
    row = {
        "timestamp": time.time(),
        "step": step,
        "phase": phase,
        **{k: stats[k] for k in keys if k in stats},
    }
    with (Path(out) / f"pinned-memory-rank{rank}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
