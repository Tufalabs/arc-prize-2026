"""Rollout augmentation: the pure transforms, their seeding and their audit.

Harness-free: ARC-2 tasks are stood in for by dataclasses shaped like TAAF's
``GameSpec``/``Pair``/``Grid``. The real ARC-2 TAAF is exercised in
tests/test_arc2_augmentation.py and the ARC-3 engine in
tests/test_arc3_augmentation.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from harness_support import augmentation
from harness_support.augmentation import (
    ARC2_PALETTE_SIZE,
    ARC3_PALETTE_SIZE,
    DIHEDRAL_TRANSFORMS,
    augment_arc2_task,
    augmentation_seed,
    dihedral,
    permute_colors,
    sample_arc2,
    sample_arc3,
    verify_rollouts,
    window_spec,
)
from training.config import RolloutAugmentationConfig

GRID = np.arange(6, dtype=np.int8).reshape(2, 3)


def test_the_eight_dihedral_transforms():
    expected = {
        "identity": [[0, 1, 2], [3, 4, 5]],
        "rot90": [[2, 5], [1, 4], [0, 3]],
        "rot180": [[5, 4, 3], [2, 1, 0]],
        "rot270": [[3, 0], [4, 1], [5, 2]],
        "flip_horizontal": [[2, 1, 0], [5, 4, 3]],
        "flip_vertical": [[3, 4, 5], [0, 1, 2]],
        "transpose": [[0, 3], [1, 4], [2, 5]],
        "antitranspose": [[5, 2], [4, 1], [3, 0]],
    }
    assert set(expected) == set(DIHEDRAL_TRANSFORMS)
    for transform, cells in expected.items():
        result = dihedral(GRID, transform)
        assert result.tolist() == cells
        assert result.dtype == GRID.dtype and result.flags.c_contiguous
        # A symmetry moves cells; it neither drops nor duplicates one.
        assert sorted(result.ravel()) == sorted(GRID.ravel())
    with pytest.raises(ValueError, match="unknown dihedral"):
        dihedral(GRID, "shear")


def test_dihedral_transforms_form_the_group_of_the_square():
    square = np.arange(9).reshape(3, 3)
    images = {dihedral(square, t).tobytes() for t in DIHEDRAL_TRANSFORMS}
    assert len(images) == 8
    # Closed under composition: transforming twice is again one of the eight.
    for first, second in itertools.product(DIHEDRAL_TRANSFORMS, repeat=2):
        assert dihedral(dihedral(square, first), second).tobytes() in images


def test_color_permutation_is_a_bijection_on_cells():
    color_map = (3, 1, 0, 2, 4, 5, 6, 7, 8, 9)
    grid = np.array([[0, 1], [2, 3]], dtype=np.int8)
    result = permute_colors(grid, color_map)
    assert result.tolist() == [[3, 1], [0, 2]] and result.dtype == np.int8
    inverse = tuple(color_map.index(color) for color in range(10))
    assert np.array_equal(permute_colors(result, inverse), grid)
    with pytest.raises(ValueError, match="outside"):
        permute_colors(np.array([[10]]), color_map)


def test_seeds_are_pinned_functions_of_run_window_and_puzzle():
    # Pinned: the worker and the trainer (and any later audit) must derive the
    # same augmentation on any Python or numpy version. A change here is a new
    # AUGMENTATION_VERSION.
    pinned = augmentation_seed(0, 0, "x")
    assert pinned == 5851653629350218605
    arc2 = sample_arc2(pinned, [0, 1, 2, 3, 5])
    assert (arc2.transform, arc2.color_map) == (
        "flip_vertical", (1, 0, 5, 2, 4, 3, 6, 7, 8, 9)
    )
    assert sample_arc3(pinned).color_map == (
        11, 0, 4, 15, 6, 13, 3, 9, 8, 7, 14, 5, 12, 10, 1, 2
    )
    seed = augmentation_seed(17, 3, "0a1d4ef5")
    assert seed == augmentation_seed(17, 3, "0a1d4ef5")
    assert len({
        seed,
        augmentation_seed(17, 4, "0a1d4ef5"),
        augmentation_seed(18, 3, "0a1d4ef5"),
        augmentation_seed(17, 3, "0a1d4ef6"),
    }) == 4
    for bad in ((1.0, 0, "x"), (1, -1, "x"), (1, 0, ""), (True, 0, "x")):
        with pytest.raises(ValueError):
            augmentation_seed(*bad)


def test_same_seed_same_augmentation_new_window_new_one():
    palette = [0, 2, 5, 7]
    first = sample_arc2(augmentation_seed(1, 0, "p"), palette)
    assert first == sample_arc2(augmentation_seed(1, 0, "p"), palette)
    windows = [sample_arc2(augmentation_seed(1, w, "p"), palette) for w in range(64)]
    assert len({(a.transform, a.color_map) for a in windows}) > 32
    assert {a.transform for a in windows} == set(DIHEDRAL_TRANSFORMS)
    arc3 = [sample_arc3(augmentation_seed(1, w, "ls20")) for w in range(64)]
    assert len({a.color_map for a in arc3}) == 64
    assert sample_arc3(augmentation_seed(1, 0, "ls20")) == arc3[0]


def test_arc2_colors_absent_from_the_puzzle_stay_fixed():
    palette = [0, 3, 4, 9]
    for window in range(50):
        sampled = sample_arc2(augmentation_seed(5, window, "p"), palette)
        assert sampled.palette == tuple(palette)
        assert sorted(sampled.color_map) == list(range(ARC2_PALETTE_SIZE))
        assert {sampled.color_map[c] for c in palette} == set(palette)
        for color in set(range(ARC2_PALETTE_SIZE)) - set(palette):
            assert sampled.color_map[color] == color
    with pytest.raises(ValueError, match="sorted and distinct"):
        sample_arc2(1, [3, 0])
    with pytest.raises(ValueError, match="0..9"):
        sample_arc2(1, [10])


def test_arc3_permutes_all_sixteen_colors_and_never_moves_cells():
    moved = set()
    for window in range(50):
        sampled = sample_arc3(augmentation_seed(5, window, "ls20"))
        assert sampled.transform == "identity"
        assert sampled.palette == tuple(range(ARC3_PALETTE_SIZE))
        assert sorted(sampled.color_map) == list(range(ARC3_PALETTE_SIZE))
        moved |= {c for c in range(16) if sampled.color_map[c] != c}
        frame = np.arange(16, dtype=np.int8).reshape(4, 4)
        assert sampled.apply(frame).tolist() == [
            [sampled.color_map[c] for c in row] for row in frame.tolist()
        ]
    assert moved == set(range(16))


def test_the_two_arc2_draws_are_independent_and_roughly_uniform():
    palette = [1, 2, 6]
    transforms, permutations = Counter(), Counter()
    for window in range(4800):
        seed = augmentation_seed(9, window, "p")
        both = sample_arc2(seed, palette)
        assert sample_arc2(seed, palette, colors=False).transform == both.transform
        assert sample_arc2(seed, palette, colors=False).color_map == tuple(range(10))
        assert sample_arc2(seed, palette, dihedral=False).color_map == both.color_map
        assert sample_arc2(seed, palette, dihedral=False).transform == "identity"
        transforms[both.transform] += 1
        permutations[tuple(both.color_map[c] for c in palette)] += 1
    # 600 and 800 expected; these bounds are over five standard deviations wide.
    assert len(transforms) == 8 and all(480 < n < 720 for n in transforms.values())
    assert len(permutations) == 6 and all(660 < n < 940 for n in permutations.values())


# --- ARC-2 tasks, with stand-ins shaped like TAAF's -------------------------


@dataclass(frozen=True, eq=False)
class Grid:
    data: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "data", np.array(self.data, dtype=np.int8))


@dataclass(frozen=True)
class Pair:
    question: Grid
    answer: Grid | None
    given: bool


@dataclass(frozen=True)
class Task:
    game_id: str
    pairs: tuple[Pair, ...]


def _task(slot="0a1d4ef5#s0"):
    return Task(
        game_id=slot,
        pairs=(
            Pair(Grid([[0, 1, 1], [0, 0, 2]]), Grid([[2, 2], [1, 0]]), True),
            Pair(Grid([[1, 0, 0]]), Grid([[0, 1, 1]]), True),
            Pair(Grid([[0, 2], [1, 1]]), Grid([[7, 2]]), False),
            Pair(Grid([[2, 0]]), None, False),
        ),
    )


def _spec(tmp_path, environment="arc2", window=3, **settings):
    config = RolloutAugmentationConfig.from_dict(
        {"enabled": True, **settings}, environment=environment
    )
    return {
        "job_dir": str(tmp_path),
        "augmentation": window_spec(
            config, environment=environment, run_seed=17, window=window, kind="train"
        ),
    }


def test_arc2_task_is_augmented_consistently_with_its_scored_outputs(tmp_path):
    task = _task()
    spec = _spec(tmp_path)
    augmented = augment_arc2_task(task, puzzle="0a1d4ef5", spec=spec)
    expected = sample_arc2(augmentation_seed(17, 3, "0a1d4ef5"), [0, 1, 2, 7])
    assert augmented.game_id == task.game_id
    for before, after in zip(task.pairs, augmented.pairs, strict=True):
        assert after.given == before.given
        # One augmentation for every grid: train and test, inputs and outputs.
        assert np.array_equal(after.question.data, expected.apply(before.question.data))
        assert isinstance(after.question, Grid)
        if before.answer is None:
            assert after.answer is None
        else:
            assert np.array_equal(after.answer.data, expected.apply(before.answer.data))
    # The test output's own color (7) belongs to the permuted palette.
    record = json.loads(
        (Path(tmp_path) / "augmentation" / "0a1d4ef5#s0.json").read_text()
    )
    assert record == expected.record(puzzle="0a1d4ef5", params=spec["augmentation"])
    assert record["palette"] == [0, 1, 2, 7]
    assert augmentation.applied_record(spec, "0a1d4ef5#s0") == record


def test_every_slot_of_a_puzzle_shares_the_window_augmentation(tmp_path):
    spec = _spec(tmp_path)
    slots = [augment_arc2_task(_task(f"p#s{i}"), puzzle="p", spec=spec) for i in range(4)]
    first = [pair.question.data.tobytes() for pair in slots[0].pairs]
    for slot in slots[1:]:
        assert [pair.question.data.tobytes() for pair in slot.pairs] == first
    records = [augmentation.applied_record(spec, f"p#s{i}") for i in range(4)]
    assert all(record == records[0] for record in records)
    # The next window redraws (and overwrites nothing of this window's: each
    # window has its own job directory).
    later_dir = Path(tmp_path) / "later"
    augment_arc2_task(_task("p#s0"), puzzle="p", spec=_spec(later_dir, window=4))
    later = augmentation.applied_record(_spec(later_dir, window=4), "p#s0")
    assert later["seed"] == augmentation_seed(17, 4, "p") != records[0]["seed"]


def test_arc2_sub_augmentations_can_be_disabled(tmp_path):
    task = _task()
    flat = augment_arc2_task(task, puzzle="p", spec=_spec(tmp_path, dihedral=False))
    assert all(
        a.question.data.shape == b.question.data.shape
        for a, b in zip(task.pairs, flat.pairs, strict=True)
    )
    recolorless = augment_arc2_task(task, puzzle="p", spec=_spec(tmp_path, colors=False))
    assert {int(c) for p in recolorless.pairs for c in np.unique(p.question.data)} == {0, 1, 2}
    record = augmentation.applied_record(_spec(tmp_path, colors=False), task.game_id)
    assert record["palette"] == [] and record["color_map"] == list(range(10))


def test_no_augmentation_returns_the_task_and_journals_nothing(tmp_path):
    task = _task()
    spec = {"job_dir": str(tmp_path), "augmentation": None}
    assert augment_arc2_task(task, puzzle="p", spec=spec) is task
    assert augment_arc2_task(task, puzzle="p", spec={"job_dir": str(tmp_path)}) is task
    assert not (Path(tmp_path) / "augmentation").exists()
    assert augmentation.applied_record(spec, task.game_id) is None


def test_arc3_spec_never_reaches_the_arc2_worker(tmp_path):
    with pytest.raises(ValueError, match="not an arc2 augmentation"):
        augment_arc2_task(_task(), puzzle="p", spec=_spec(tmp_path, environment="arc3"))


def test_an_augmented_slot_without_a_record_fails_the_report(tmp_path):
    with pytest.raises(RuntimeError, match="without an augmentation"):
        augmentation.applied_record(_spec(tmp_path), "never#s0")


def test_unaugmented_arc3_game_is_the_worker_class_itself(tmp_path):
    made = augmentation.arc3_game(
        SimpleNamespace, {"job_dir": str(tmp_path)}, env_name="ls20", external_game_id="ls20#s0"
    )
    assert made == SimpleNamespace(env_name="ls20", external_game_id="ls20#s0")


# --- the setting, the worker spec and the trainer's audit -------------------


@pytest.mark.parametrize("environment", ["arc2", "arc3"])
def test_window_spec_only_for_enabled_training_windows(environment):
    on = RolloutAugmentationConfig.from_dict({"enabled": True}, environment=environment)
    off = RolloutAugmentationConfig.from_dict(None, environment=environment)
    kwargs = dict(environment=environment, run_seed=5, window=12)
    assert window_spec(off, kind="train", **kwargs) is None
    assert window_spec(on, kind="eval", **kwargs) is None
    assert window_spec(on, kind="train", **kwargs) == {
        "version": 1,
        "environment": environment,
        "run_seed": 5,
        "window": 12,
        "dihedral": environment == "arc2",
        "colors": True,
    }
    with pytest.raises(ValueError, match="never flips"):
        window_spec(
            SimpleNamespace(enabled=True, dihedral=True, colors=True),
            kind="train",
            **{**kwargs, "environment": "arc3"},
        )


def _rollout(record, puzzle="p", trajectory_id="t"):
    return SimpleNamespace(
        trajectory_id=trajectory_id, metadata={"game_id": puzzle, "augmentation": record}
    )


@pytest.mark.parametrize("environment", ["arc2", "arc3"])
def test_trainer_rederives_each_rollout_augmentation(tmp_path, environment):
    spec = _spec(tmp_path, environment=environment)
    params = spec["augmentation"]
    seed = augmentation_seed(17, 3, "p")
    sampled = sample_arc2(seed, [0, 4]) if environment == "arc2" else sample_arc3(seed)
    record = sampled.record(puzzle="p", params=params)
    verify_rollouts([_rollout(record), _rollout(dict(record), trajectory_id="u")], params)
    with pytest.raises(ValueError, match="unaugmented window"):
        verify_rollouts([_rollout(record)], None)
    with pytest.raises(ValueError, match="reported no augmentation"):
        verify_rollouts([_rollout(None)], params)
    with pytest.raises(ValueError, match="another puzzle"):
        verify_rollouts([_rollout(record, puzzle="q")], params)
    tampered = dict(record, color_map=list(reversed(record["color_map"])))
    with pytest.raises(ValueError, match="differs from its window"):
        verify_rollouts([_rollout(tampered)], params)
    with pytest.raises(ValueError, match="differs from its window"):
        verify_rollouts([_rollout(record)], dict(params, window=4))
    verify_rollouts([_rollout(None)], None)


def test_trainer_refuses_an_invalid_arc2_palette(tmp_path):
    params = _spec(tmp_path)["augmentation"]
    record = sample_arc2(augmentation_seed(17, 3, "p"), [0, 4]).record(puzzle="p", params=params)
    with pytest.raises(ValueError, match="sorted and distinct"):
        verify_rollouts([_rollout(dict(record, palette=[4, 0]))], params)


# --- the inference.augmentation setting -------------------------------------

ROOT = Path(__file__).resolve().parents[1]
ARC2_RECIPE = ROOT / "configs/grpo_3gpu_arc2_overfit.json"
ARC3_RECIPE = ROOT / "configs/ppo_2gpu_arc3_games25_platypus.json"
FLASH_RECIPE = ROOT / "configs/flash_8gpu_arc2_grpo_tokmean.json"


def _load(tmp_path, recipe, settings):
    from training.config import ExperimentConfig

    payload = json.loads(recipe.read_text())
    if settings is not None:
        payload["inference"]["augmentation"] = settings
    path = Path(tmp_path) / "config.json"
    path.write_text(json.dumps(payload))
    return ExperimentConfig.load(path)


@pytest.mark.parametrize("recipe", [ARC2_RECIPE, ARC3_RECIPE])
def test_shipped_recipes_do_not_augment(recipe):
    from training.config import ExperimentConfig

    assert "augmentation" not in json.loads(recipe.read_text())["inference"]
    settings = ExperimentConfig.load(recipe).inference.augmentation
    assert settings.enabled is False and settings.identity() is None


def test_the_flash_recipe_augments_both_ways():
    from training.config import ExperimentConfig

    settings = ExperimentConfig.load(FLASH_RECIPE).inference.augmentation
    assert (settings.enabled, settings.dihedral, settings.colors) == (True, True, True)
    assert settings.identity() is not None


def test_arc2_augments_both_ways_by_default_and_the_frozen_config_keeps_it(tmp_path):
    from training.config import ExperimentConfig

    config = _load(tmp_path, ARC2_RECIPE, {"enabled": True})
    assert config.inference.augmentation == RolloutAugmentationConfig(
        enabled=True, dihedral=True, colors=True
    )
    frozen = config.to_dict()
    assert frozen["inference"]["augmentation"] == {
        "enabled": True, "dihedral": True, "colors": True,
    }
    path = Path(tmp_path) / "effective_config.json"
    path.write_text(json.dumps(frozen))
    assert ExperimentConfig.load(path).inference.augmentation == config.inference.augmentation
    for settings in ({"dihedral": False}, {"colors": False}):
        loaded = _load(tmp_path, ARC2_RECIPE, {"enabled": True, **settings})
        assert loaded.inference.augmentation.identity() == {
            "version": 1, "dihedral": True, "colors": True, **settings,
        }


def test_arc3_permutes_colors_and_refuses_flips_and_rotations(tmp_path):
    config = _load(tmp_path, ARC3_RECIPE, {"enabled": True})
    assert config.inference.augmentation == RolloutAugmentationConfig(
        enabled=True, dihedral=False, colors=True
    )
    for settings in ({"enabled": True, "dihedral": True}, {"dihedral": True}):
        with pytest.raises(ValueError, match="dihedral is ARC-2 only"):
            _load(tmp_path, ARC3_RECIPE, settings)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"enabled": True, "dihedral": False, "colors": False}, "neither dihedral nor colors"),
        ({"enabled": "yes"}, "augmentation.enabled must be true or false"),
        ({"enabled": True, "colors": 1}, "augmentation.colors must be true or false"),
        ({"enabled": True, "rotations": True}, "inference.augmentation.rotations"),
        (["enabled"], "inference.augmentation must be a JSON object"),
    ],
)
def test_invalid_settings_are_refused(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, ARC2_RECIPE, settings)
