"""Deterministic seed derivation.

The search's move ordering is random: each node it expands shuffles its
candidate moves with a fresh stream derived from the search's seed plus the
node's index, rather than advancing a long-lived stream. Deriving rather
than streaming is what makes the ordering reproducible across the process
boundary (the search runs in a subprocess, so any stream it advanced dies
with the child) and independent of how many LLM calls the game made, since
sampling draws from ``ctx.rng`` and the search never does.

``blake2b`` and not the builtin ``hash``: the latter is
``PYTHONHASHSEED``-dependent for ``str`` and ``bytes``. Parts must be ints
or short ASCII strings so their ``str()`` rendering is unambiguous.
``tests/data/rng_contract_vector.json`` pins the derivation and the orderings
it produces.
"""

from __future__ import annotations

import hashlib
import random


def derive_seed(*parts: object) -> int:
    """A stable integer from a coordinate tuple.

    Returns the *int*, which is what anything used as a further coordinate
    must be: passing a ``random.Random`` back in would stringify to
    ``<random.Random object at 0x...>``, an address that varies between
    processes.
    """
    payload = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def derive_rng(*parts: object) -> random.Random:
    """A fresh stream for a leaf consumer. Never use its result as a
    coordinate for a further derivation — see ``derive_seed``."""
    return random.Random(derive_seed(*parts))
