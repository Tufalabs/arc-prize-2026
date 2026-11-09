"""Shared rollout-slot identities and model RNG seeds for every harness."""


def _slot_game_ids(names: list[str]) -> list[str]:
    """Give each full-game rollout an identity independent of repeated env names."""
    return [f"{name}#s{slot_index}" for slot_index, name in enumerate(names)]


def _rng_seed_per_game(names: list, slot_seeds: list) -> dict:
    """Pin each slot's model RNG by its unique external game id.

    Platypus uses this RNG for per-call sampling seeds. Distinct external
    identities let repeated full games receive independent streams without
    changing the engine environment or the normal full-game lifecycle.
    """
    if len(slot_seeds) != len(names):
        raise ValueError(
            f"slot_seeds lists {len(slot_seeds)} seeds for {len(names)} "
            "game slots; the spec requires one seed per slot"
        )
    seeds: dict = {}
    for name, seed in zip(names, slot_seeds):
        seed = int(seed)
        if seeds.setdefault(str(name), seed) != seed:
            raise ValueError(
                f"conflicting model RNG seeds for slot {name!r} in one window "
                f"({seeds[str(name)]} and {seed})"
            )
    return seeds
