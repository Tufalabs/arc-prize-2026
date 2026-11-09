"""What an engine's KV pool holds at the contexts we run, from its own log.

    uv run python tools/kv_capacity.py LOG [LOG ...] [--contexts 20000,40000,60000]

vLLM's "GPU KV cache size" line is max concurrency x max_model_len: it spreads
each request's fixed state over the whole window, where it weighs a few
percent. At the contexts we run it is a quarter of a request or more, and a
drafter enlarges it (``packages/ullm/flash_optimization.md``, "Speculative
decoding"). This reads the pool the way the engine prices it:

- all KV groups draw block ids from one pool, each block costing the widest
  group's bytes;
- a running request holds ``ceil(context / block)`` blocks in each attention
  group, plus fixed state blocks: 1.31 per Mamba group under RecoverSSM's
  decode tails (the running state, and the next tail's landing block about
  31% of the time: VLLM_KV_BLOCKS_TRACE on a B200 replay at k=3, 2026-10-11;
  ``--mamba-blocks`` otherwise) and 1 per circular-buffer group, none when
  the engine logs a transient pool (the rings' own, outside it);
- vLLM's concurrency line reserves 2 per Mamba group in align mode instead,
  so the pool is recovered from it with that reservation;
- the pool's block count follows from the logged concurrency at the window.

For each log: the groups, the fixed blocks a request, the pool in blocks, and
the requests that fit at each context. Where the log has engine minutes with
capacity waits (the pool binding), it also reads the running set there and
the context a running request holds on average. LOG is a job's stdout.log or a
Kaggle live log (``kaggle kernels logs -f OWNER/SLUG``).
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass, field

GROUP = re.compile(
    r"KV offloading group (\d+): (MambaSpec|CircularBufferSpec|[A-Za-z]*AttentionSpec|SlidingWindowSpec), "
    r"(\d+) tokens per block, (\d+) layers(, transient)?"
)
POOL = re.compile(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
MEMORY = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
VLLM_MAMBA_RESERVATION = 2  # MambaSpec.max_memory_usage_bytes in align mode, in pages
TRANSIENT_POOL = re.compile(r"Transient pool: (\d+) blocks")
MINUTE = re.compile(r"^\[engine\].*running=(\d+) waiting=(\d+)(?:\(([^)]*)\))? kv_cache=([\d.]+)%.*\| 60s:")


@dataclass
class Engine:
    """What one log says about its engine's KV pool."""

    path: str
    groups: dict[int, tuple[str, int, int]] = field(
        default_factory=lambda: dict[int, tuple[str, int, int]]()
    )  # id -> (kind, block, layers)
    pool_tokens: int = 0
    window: int = 0
    concurrency: float = 0.0
    memory_gib: float | None = None
    transient_pool: int = 0  # blocks of the rings' own pool, 0 without one
    minutes: list[tuple[int, int, float]] = field(
        default_factory=lambda: list[tuple[int, int, float]]()
    )  # (running, capacity waits, kv %)

    @property
    def block(self) -> int:
        """Tokens a block, the attention groups' (a ring's is its own, smaller)."""
        return max(block for kind, block, _ in self.groups.values() if kind != "CircularBufferSpec")

    @property
    def attention_groups(self) -> int:
        """Groups that hold a block per block's worth of context."""
        return sum(1 for kind, _, _ in self.groups.values() if kind not in ("MambaSpec", "CircularBufferSpec"))

    @property
    def widest_layers(self) -> int:
        """Layers in the widest group, whose bytes price every block."""
        return max(layers for _, _, layers in self.groups.values())

    def fixed_blocks(self, mamba_blocks: float) -> float:
        """State blocks a running request holds whatever its length."""
        mamba = sum(1 for kind, _, _ in self.groups.values() if kind == "MambaSpec")
        rings = sum(1 for kind, _, _ in self.groups.values() if kind == "CircularBufferSpec")
        return mamba * mamba_blocks + (0 if self.transient_pool else rings)

    def blocks_a_request(self, context: int, mamba_blocks: float) -> float:
        """Blocks a request of ``context`` tokens holds while it runs."""
        return self.attention_groups * math.ceil(context / self.block) + self.fixed_blocks(mamba_blocks)

    def pool_blocks(self) -> float:
        """The pool in blocks, from the concurrency vLLM logged at its window,
        which reserves 2 blocks per Mamba group in align mode."""
        return self.concurrency * self.blocks_a_request(self.window, VLLM_MAMBA_RESERVATION)


def read(path: str) -> Engine:
    """The groups, pool and engine minutes one log records."""
    engine = Engine(path)
    for line in open(path, errors="replace").read().replace("\r", "\n").splitlines():
        if m := GROUP.search(line):
            engine.groups[int(m.group(1))] = (m.group(2), int(m.group(3)), int(m.group(4)))
        elif m := POOL.search(line):
            engine.pool_tokens = int(m.group(1).replace(",", ""))
            engine.window = int(m.group(2).replace(",", ""))
            engine.concurrency = float(m.group(3))
        elif m := MEMORY.search(line):
            engine.memory_gib = float(m.group(1))
        elif m := TRANSIENT_POOL.search(line):
            engine.transient_pool = int(m.group(1))
        elif m := MINUTE.search(line):
            labels = m.group(3) or ""
            capacity = re.search(r"capacity:(\d+)", labels)
            engine.minutes.append((int(m.group(1)), int(capacity.group(1)) if capacity else 0, float(m.group(4))))
    if not engine.groups or not engine.concurrency:
        raise SystemExit(f"{path}: no KV group or pool lines (an engine log from start-up on is needed)")
    return engine


def main() -> None:
    """Print each log's pool as requests that fit, and its pool-bound running set."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--contexts", default="20000,40000,60000,100000")
    parser.add_argument(
        "--mamba-blocks", type=float, default=1.31, help="state blocks a running request holds per Mamba group"
    )
    args = parser.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    engines = [read(path) for path in args.logs]
    first = engines[0]
    for engine in engines:
        blocks = engine.pool_blocks()
        fixed = engine.fixed_blocks(args.mamba_blocks)
        memory = f", {engine.memory_gib:g} GiB" if engine.memory_gib is not None else ""
        print(f"{engine.path}")
        print(
            f"  pool {blocks:,.0f} blocks of {engine.block:,} tokens{memory}; a block is priced at the widest group "
            f"({engine.widest_layers} layers); vLLM's line: {engine.pool_tokens:,} tokens at {engine.window:,}"
        )
        kinds = ", ".join(f"{kind} {layers}x{block}" for kind, block, layers in engine.groups.values())
        print(f"  groups: {kinds}")
        print(
            f"  a request holds {engine.attention_groups} x ceil(context / {engine.block:,}) + {fixed:.2f} fixed blocks"
        )
        for context in contexts:
            need = engine.blocks_a_request(context, args.mamba_blocks)
            fits = blocks / need
            vs = ""
            if engine is not first:
                fits_first = first.pool_blocks() / first.blocks_a_request(context, args.mamba_blocks)
                # Blocks differ in bytes between engines; the requests that fit compare directly.
                vs = f"  ({fits / fits_first - 1:+.1%} against the first)"
            print(
                f"    at {context:>7,} tokens: {fits:7.1f} requests fit, fixed state {fixed / need:5.1%} of a request{vs}"
            )
        bound = [(run, kv) for run, cap, kv in engine.minutes if cap > 0]
        if bound:
            running = statistics.mean(run for run, _ in bound)
            kv = statistics.mean(kv for _, kv in bound)
            per_request = kv / 100 * blocks / running
            context = (per_request - fixed) / engine.attention_groups * engine.block
            print(
                f"  pool-bound ({len(bound)} of {len(engine.minutes)} minutes waited on capacity): running {running:.1f}, "
                f"KV {kv:.1f}%, so ~{per_request:.1f} blocks, ~{context:,.0f} tokens of context a running request"
            )
        print()


if __name__ == "__main__":
    main()
