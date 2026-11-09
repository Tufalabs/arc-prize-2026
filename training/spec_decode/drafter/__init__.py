"""Training Flash-Next's MTP head (the drafter) on recorded duck traffic.

The flow: ``chains`` cuts ids-only recordings into chains, ``extract`` runs
them through the served target and caches its pre-final-mixer stream
(``capture`` is the engine-side half), ``train`` fits the head against the
target's own distribution, and ``jobs`` puts each step on a B200. See
``spec_decode/README.md`` in arc-training, which owns this package.
"""
