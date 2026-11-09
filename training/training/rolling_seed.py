"""JAX-style rolling seeds for rollout / eval / training windows.

A single root seed (``config.training.seed``) deterministically rolls a fresh
seed for every window. Each window's seed is **split into two independent
branches** (the way ``jax.random.split`` separates a key from its subkeys):

* a ``main_seed`` -- the seed the main process would consume for that iteration,
* a ``child_root`` -- the branch that spawns per-rollout *child* seeds.

The two branches use distinct domain tags, so the child-seed sequence lives in a
different derivation domain than the main-seed sequence. A child seed therefore
never coincides with a main-process seed from any iteration -- *including future
ones* -- which is the key safety property of JAX-style key splitting (a child
must not be able to see a future main key).

Usage mirrors the JAX rolling-key pattern::

    rolling = RollingSeed(config.training.seed)
    it = rolling.iteration("train", schedule_index)   # rolls every iteration
    slot_seeds = it.children(num_rollouts)            # child branch -> per-rollout seeds
    # it.main_seed is the separate main-process branch for this iteration

or, generator-style::

    seeds = rolling.stream("train")
    use = next(seeds)                                  # next_iter_seed

It is keyed by ``(root_seed, kind, index)`` rather than call order, so the same
iteration index always yields the same seeds -- a resumed run reproduces the
exact rollout seeds, and the ``train`` and ``eval`` streams never collide.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterator
from dataclasses import dataclass

# re_arc draws instance seeds from ``randrange(2**31)``; keep derived seeds in
# the same non-negative 31-bit range so every consumer accepts them.
_SEED_BITS = 31
_SEED_MASK = (1 << _SEED_BITS) - 1

# Domain tags. Every derivation is prefixed by exactly one role tag, so the
# main-seed branch and the child-seed branch are disjoint derivation domains and
# cannot produce the same value (other than by hash collision, ~2**-31/pair).
_TAG_ITER = "arc3.rolling.iter"
_TAG_MAIN = "arc3.rolling.main"
_TAG_CHILD_ROOT = "arc3.rolling.child_root"
_TAG_CHILD_LEAF = "arc3.rolling.child_leaf"


def _derive(tag: str, *parts: object) -> int:
    """Stable hash of ``(tag, *parts)`` -> non-negative 31-bit int.

    Uses BLAKE2b so the value does not depend on ``PYTHONHASHSEED`` (unlike the
    builtin ``hash``), keeping runs reproducible across processes/restarts.
    """
    hasher = hashlib.blake2b(digest_size=8)
    for part in (tag, *parts):
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return int.from_bytes(hasher.digest(), "big") & _SEED_MASK


@dataclass(frozen=True)
class IterationSeed:
    """One window's seed, pre-split into a main branch and a child branch."""

    main_seed: int
    child_root: int

    @property
    def value(self) -> int:
        """The main-process seed for this iteration."""
        return self.main_seed

    def child(self, *parts: object) -> int:
        """Derive a child seed (e.g. one per rollout slot) on the child branch."""
        return _derive(_TAG_CHILD_LEAF, self.child_root, *parts)

    def children(self, count: int) -> list[int]:
        """The first ``count`` per-slot child seeds for this iteration."""
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.child(slot) for slot in range(count)]


class RollingSeed:
    """Rolls a fresh seed per window and splits it into main / child branches."""

    def __init__(self, root_seed: int) -> None:
        self._root = int(root_seed) & _SEED_MASK

    @property
    def root(self) -> int:
        return self._root

    def iteration(self, kind: str, index: int) -> IterationSeed:
        """The rolling seed for window ``kind`` at iteration ``index``.

        Every distinct ``index`` yields a different seed (each iteration sees
        fresh seeds); ``kind`` keeps the train/eval streams independent. The
        per-iteration master is then split two ways so child seeds and main
        seeds come from disjoint branches.
        """
        master = _derive(_TAG_ITER, self._root, str(kind), int(index))
        return IterationSeed(
            main_seed=_derive(_TAG_MAIN, master),
            child_root=_derive(_TAG_CHILD_ROOT, master),
        )

    def stream(self, kind: str) -> Iterator[IterationSeed]:
        """Successive iteration seeds for ``kind`` -- use with ``next()``."""
        for index in itertools.count():
            yield self.iteration(kind, index)
