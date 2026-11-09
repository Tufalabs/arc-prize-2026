"""How a sweep allocates its seed space.

Two rules, each of which fails silently when broken — no error, no wrong
number in any output, just an experiment that answers a different question
than the one asked. They live here rather than in each experiment because
they are properties of sweeping, not of any particular sweep.
"""

from __future__ import annotations

#: Default spacing between one job's play seeds and the next job's. Large
#: because the only cost of a too-large stride is unused integers.
DEFAULT_PLAY_SEED_STRIDE = 1_000_000


def arm_seed_block(run_seed: int, arms_per_job: int, *, base: int = 0) -> tuple[int, ...]:
    """This job's arm seeds — its own block, so no two jobs share an arm.

    Reusing one block across jobs would replay the same configurations rather
    than drawing new ones: ten jobs of five arms would then measure five
    design points ten times each, which is a measurement of the noise floor
    wearing the shape of a sweep.
    """
    if arms_per_job < 1:
        raise ValueError(f"arms_per_job must be at least 1; got {arms_per_job}")
    start = base + arms_per_job * run_seed
    return tuple(range(start, start + arms_per_job))


def play_seed(run_seed: int, *, stride: int = DEFAULT_PLAY_SEED_STRIDE, entries: int | None = None) -> int:
    """The ``run_config.rng_seed`` for one job, spaced so no two jobs collide.

    A game's RNG seed is ``rng_seed`` plus its position in the run's list, so
    a job hands out ``rng_seed .. rng_seed + entries``. Two jobs whose bases
    sit closer together than that share randomness — and nothing records the
    seeds a run used, so the overlap is invisible afterwards.

    ``entries`` (games x passes) turns the rule into a check rather than a
    convention. It is optional only because a caller may not know the game
    count yet; passing it is strictly better.
    """
    if stride < 1:
        raise ValueError(f"stride must be positive; got {stride}")
    if entries is not None and entries > stride:
        raise ValueError(
            f"a job of {entries} entries needs a play-seed stride above {entries}, not {stride}: "
            "seeds are handed out by list position, so consecutive jobs would overlap"
        )
    return stride * run_seed


__all__ = ["DEFAULT_PLAY_SEED_STRIDE", "arm_seed_block", "play_seed"]
