from __future__ import annotations

from training.rolling_seed import RollingSeed

_SEED_CEIL = 1 << 31


def test_iteration_seed_rolls_every_iteration() -> None:
    rolling = RollingSeed(42)
    seeds = [rolling.iteration("train", i).value for i in range(64)]
    # Every iteration gets a distinct main seed.
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < _SEED_CEIL for s in seeds)


def test_train_and_eval_streams_do_not_collide() -> None:
    rolling = RollingSeed(42)
    train = {rolling.iteration("train", i).value for i in range(64)}
    eval_ = {rolling.iteration("eval", i).value for i in range(64)}
    assert train.isdisjoint(eval_)


def test_reproducible_across_instances() -> None:
    # A resumed run rebuilds RollingSeed from the same root and must reproduce
    # the exact per-iteration and per-child seeds.
    a = RollingSeed(123)
    b = RollingSeed(123)
    assert a.iteration("train", 7).value == b.iteration("train", 7).value
    assert a.iteration("train", 7).children(8) == b.iteration("train", 7).children(8)


def test_children_are_distinct_and_in_range() -> None:
    children = RollingSeed(5).iteration("train", 3).children(32)
    assert len(children) == 32
    assert len(set(children)) == 32
    assert all(0 <= c < _SEED_CEIL for c in children)


def test_children_differ_across_iterations() -> None:
    rolling = RollingSeed(5)
    first = rolling.iteration("train", 0).children(8)
    second = rolling.iteration("train", 1).children(8)
    assert set(first).isdisjoint(second)


def test_different_root_seeds_diverge() -> None:
    assert (
        RollingSeed(1).iteration("train", 0).value
        != RollingSeed(2).iteration("train", 0).value
    )


def test_stream_matches_indexed_iterations() -> None:
    rolling = RollingSeed(99)
    stream = rolling.stream("train")
    for i in range(10):
        assert next(stream).value == rolling.iteration("train", i).value


def test_child_seeds_disjoint_from_main_seeds_including_future() -> None:
    # The core split guarantee: the child-seed branch and the main-seed branch
    # are disjoint domains, so a child seed (even from iteration 0) never equals
    # a main-process seed from ANY iteration, including future ones.
    rolling = RollingSeed(42)
    n_iters, n_children = 200, 32
    main_seeds = {rolling.iteration("train", i).value for i in range(n_iters)}
    child_seeds: set[int] = set()
    for i in range(n_iters):
        child_seeds.update(rolling.iteration("train", i).children(n_children))
    assert main_seeds.isdisjoint(child_seeds)
    # Also disjoint from the eval main stream.
    eval_main = {rolling.iteration("eval", i).value for i in range(n_iters)}
    assert eval_main.isdisjoint(child_seeds)


def test_main_and_child_branches_differ_within_iteration() -> None:
    it = RollingSeed(7).iteration("train", 0)
    assert it.main_seed != it.child_root
    assert it.main_seed not in set(it.children(32))
