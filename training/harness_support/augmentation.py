"""Data augmentation for training rollouts, ARC-2 and ARC-3 (off by default).

One augmentation per puzzle per window: every attempt at a puzzle in a window
(the GRPO group whose rewards are compared) sees the same augmentation, and
the next window draws a new one. The seed is a stable hash of (run seed,
window index, puzzle id), and every draw from it is a BLAKE2b hash rather than
a library RNG stream, so the worker that applies an augmentation and the
trainer that audits it derive the same one on any Python or numpy version.

ARC-2: one of the eight dihedral transforms and a permutation of the colors
the puzzle uses anywhere (train and test, inputs and outputs; colors absent
from the puzzle stay fixed), applied to every grid of the task before TAAF
sees it. The model reads and answers in augmented space, and TAAF scores the
answer against the augmented test outputs.

ARC-3: a permutation of all 16 palette colors applied to every frame the
engine returns for the whole run, every level
(``harness_support/arc3/augmented_game.py``). No flips or rotations: they
would change what the directional and click actions mean. Actions reach the
engine untouched, and the score is the engine's own level progress, which
does not depend on how frames are colored.

No harness imports: the trainer loads this module for configuration, the
worker spec and verification; the workers call ``augment_arc2_task`` and
``arc3_game`` where they build each task or game.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Part of the training identity: a change to the sampling or the transforms
# is a different augmentation.
AUGMENTATION_VERSION = 1
DIHEDRAL_TRANSFORMS = (
    "identity",
    "rot90",  # counter-clockwise quarter turn, as numpy.rot90
    "rot180",
    "rot270",
    "flip_horizontal",  # mirror left-right
    "flip_vertical",  # mirror top-bottom
    "transpose",  # mirror about the main diagonal
    "antitranspose",  # mirror about the anti-diagonal
)
ARC2_PALETTE_SIZE = 10
ARC3_PALETTE_SIZE = 16
# Per-slot records of what was applied, under the window's job directory.
JOURNAL_DIR = "augmentation"
_TAG = "arc.rollout_augmentation"


def dihedral(grid: Any, transform: str) -> np.ndarray:
    """``grid`` under one of the eight symmetries of the square."""
    array = np.asarray(grid)
    if array.ndim != 2:
        raise ValueError(f"a grid must be two-dimensional, got shape {array.shape}")
    match transform:
        case "identity":
            result = array
        case "rot90":
            result = np.rot90(array, 1)
        case "rot180":
            result = np.rot90(array, 2)
        case "rot270":
            result = np.rot90(array, 3)
        case "flip_horizontal":
            result = array[:, ::-1]
        case "flip_vertical":
            result = array[::-1, :]
        case "transpose":
            result = array.T
        case "antitranspose":
            result = array[::-1, ::-1].T
        case _:
            raise ValueError(f"unknown dihedral transform {transform!r}")
    return np.ascontiguousarray(result)


def permute_colors(grid: Any, color_map: Sequence[int]) -> np.ndarray:
    """Every cell ``c`` of ``grid`` replaced by ``color_map[c]``, same dtype."""
    array = np.asarray(grid)
    lookup = np.asarray(color_map, dtype=array.dtype)
    if array.size and (array.min() < 0 or array.max() >= len(lookup)):
        raise ValueError(f"grid colors fall outside a {len(lookup)}-color map")
    return lookup[array]


def _draw(seed: int, *parts: object) -> int:
    """A uniform 64-bit integer, a stable function of the seed and ``parts``."""
    hasher = hashlib.blake2b(digest_size=8)
    for part in (_TAG, seed, *parts):
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return int.from_bytes(hasher.digest(), "big")


def augmentation_seed(run_seed: int, window: int, puzzle: str) -> int:
    """The seed of one puzzle's augmentation in one training window."""
    if type(run_seed) is not int or type(window) is not int or window < 0:
        raise ValueError("augmentation seeds need an integer run seed and window index")
    if not isinstance(puzzle, str) or not puzzle:
        raise ValueError("augmentation seeds need a puzzle id")
    return _draw(run_seed, "seed", window, puzzle) & ((1 << 63) - 1)


def _shuffled(seed: int, colors: Sequence[int]) -> list[int]:
    """Fisher-Yates over ``colors`` with hashed draws (bias below 2**-58)."""
    items = list(colors)
    for index in range(len(items) - 1, 0, -1):
        other = _draw(seed, "colors", index) % (index + 1)
        items[index], items[other] = items[other], items[index]
    return items


def _palette(colors: Sequence[int], size: int) -> tuple[int, ...]:
    palette = tuple(colors)
    if any(type(color) is not int or not 0 <= color < size for color in palette):
        raise ValueError(f"palette colors must be integers in 0..{size - 1}")
    if list(palette) != sorted(set(palette)):
        raise ValueError("palette colors must be sorted and distinct")
    return palette


@dataclass(frozen=True)
class Augmentation:
    """One drawn augmentation: a dihedral transform, then a color map.

    ``color_map[c]`` is the color that replaces ``c`` over the whole palette;
    ``palette`` lists the colors it was drawn over (the others map to
    themselves).
    """

    seed: int
    transform: str
    palette: tuple[int, ...]
    color_map: tuple[int, ...]

    def apply(self, grid: Any) -> np.ndarray:
        """``grid`` transformed, then recolored."""
        return permute_colors(dihedral(grid, self.transform), self.color_map)

    def record(self, *, puzzle: str, params: dict[str, Any]) -> dict[str, Any]:
        """What a rollout's metadata keeps of it, enough to audit it."""
        return {
            "version": AUGMENTATION_VERSION,
            "environment": params["environment"],
            "puzzle": puzzle,
            "run_seed": params["run_seed"],
            "window": params["window"],
            "seed": self.seed,
            "transform": self.transform,
            "palette": list(self.palette),
            "color_map": list(self.color_map),
        }


def sample_arc2(
    seed: int, palette: Sequence[int], *, dihedral: bool = True, colors: bool = True
) -> Augmentation:
    """A dihedral transform and a permutation of the puzzle's own colors.

    The two draws are independent, so disabling one leaves the other as it
    would have been.
    """
    palette = _palette(palette, ARC2_PALETTE_SIZE)
    transform = (
        DIHEDRAL_TRANSFORMS[_draw(seed, "dihedral") % len(DIHEDRAL_TRANSFORMS)]
        if dihedral
        else "identity"
    )
    color_map = list(range(ARC2_PALETTE_SIZE))
    if not colors:
        palette = ()
    for color, image in zip(palette, _shuffled(seed, palette)):
        color_map[color] = image
    return Augmentation(seed, transform, palette, tuple(color_map))


def sample_arc3(seed: int) -> Augmentation:
    """A permutation of all 16 palette colors and no geometric transform.

    The whole palette, because a run's first frame need not show every color
    a later level uses.
    """
    palette = tuple(range(ARC3_PALETTE_SIZE))
    return Augmentation(seed, "identity", palette, tuple(_shuffled(seed, palette)))


def window_spec(
    settings: Any, *, environment: str, run_seed: int, window: int, kind: str
) -> dict[str, Any] | None:
    """The worker spec's ``augmentation`` entry for one window.

    ``None`` unless augmentation is enabled and the window is a training
    window: evaluation never augments.
    """
    if not settings.enabled or kind != "train":
        return None
    if environment == "arc3" and settings.dihedral:
        raise ValueError("ARC-3 augmentation never flips or rotates frames")
    return {
        "version": AUGMENTATION_VERSION,
        "environment": environment,
        "run_seed": int(run_seed),
        "window": int(window),
        "dihedral": bool(settings.dihedral),
        "colors": bool(settings.colors),
    }


def _params(spec: dict[str, Any], environment: str) -> dict[str, Any] | None:
    params = spec.get("augmentation")
    if params is None:
        return None
    if (
        not isinstance(params, dict)
        or params.get("version") != AUGMENTATION_VERSION
        or params.get("environment") != environment
    ):
        raise ValueError(f"the worker spec's augmentation is not an {environment} augmentation")
    return params


def _journal(spec: dict[str, Any], slot: str, record: dict[str, Any]) -> None:
    """Record what one slot's task or game was built with."""
    if not slot or "/" in slot or slot.startswith("."):
        raise ValueError(f"invalid rollout slot id {slot!r}")
    root = Path(spec["job_dir"]) / JOURNAL_DIR
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{slot}.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def augment_arc2_task(task: Any, *, puzzle: str, spec: dict[str, Any]) -> Any:
    """The TAAF ``GameSpec`` with every grid augmented, or ``task`` itself.

    ``task.game_id`` is the rollout slot; ``puzzle`` the dataset id that, with
    the window, picks the augmentation, so every slot of a puzzle in one
    window gets the same one. Test outputs are augmented too: they are what
    TAAF scores the submission against.
    """
    params = _params(spec, "arc2")
    if params is None:
        return task
    grids = [
        grid for pair in task.pairs for grid in (pair.question, pair.answer) if grid is not None
    ]
    present = sorted({int(color) for grid in grids for color in np.unique(grid.data)})
    augmentation = sample_arc2(
        augmentation_seed(params["run_seed"], params["window"], puzzle),
        present,
        dihedral=params["dihedral"],
        colors=params["colors"],
    )

    def augmented(grid: Any) -> Any:
        return None if grid is None else type(grid)(augmentation.apply(grid.data))

    result = replace(
        task,
        pairs=tuple(
            replace(pair, question=augmented(pair.question), answer=augmented(pair.answer))
            for pair in task.pairs
        ),
    )
    _journal(spec, task.game_id, augmentation.record(puzzle=puzzle, params=params))
    return result


def arc3_game(
    game_class: Any, spec: dict[str, Any], *, env_name: str, external_game_id: str
) -> Any:
    """One ARC-3 rollout slot's game: ``game_class`` itself when not
    augmenting, else the TAAF ``GameAPI`` that recolors every frame."""
    params = _params(spec, "arc3")
    if params is None:
        return game_class(env_name=env_name, external_game_id=external_game_id)
    if params["dihedral"] or not params["colors"]:
        raise ValueError("ARC-3 augmentation permutes colors only")
    from harness_support.arc3.augmented_game import ColorPermutedGameAPI

    augmentation = sample_arc3(augmentation_seed(params["run_seed"], params["window"], env_name))
    game = ColorPermutedGameAPI(
        env_name=env_name,
        external_game_id=external_game_id,
        color_map=augmentation.color_map,
    )
    _journal(spec, external_game_id, augmentation.record(puzzle=env_name, params=params))
    return game


def applied_record(spec: dict[str, Any], slot: str) -> dict[str, Any] | None:
    """The augmentation a slot's task or game was built with, for its result."""
    if spec.get("augmentation") is None:
        return None
    path = Path(spec["job_dir"]) / JOURNAL_DIR / f"{slot}.json"
    if not path.is_file():
        raise RuntimeError(f"augmented window built slot {slot!r} without an augmentation")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_record(params: dict[str, Any], record: dict[str, Any], puzzle: str) -> dict[str, Any]:
    """The record the trainer derives itself for ``puzzle`` in this window.

    ARC-2's palette (the colors the puzzle uses) is the one fact taken from
    the worker's record: the trainer does not load the grids. It must still
    be a valid palette, and everything else is re-derived from it.
    """
    seed = augmentation_seed(params["run_seed"], params["window"], puzzle)
    if params["environment"] == "arc2":
        palette = record.get("palette")
        if not isinstance(palette, list):
            raise ValueError("ARC-2 augmentation record lacks its palette")
        augmentation = sample_arc2(
            seed,
            palette if params["colors"] else [],
            dihedral=params["dihedral"],
            colors=params["colors"],
        )
    else:
        augmentation = sample_arc3(seed)
    return augmentation.record(puzzle=puzzle, params=params)


def verify_rollouts(rollouts: Sequence[Any], params: dict[str, Any] | None) -> None:
    """Require each rollout to report exactly the augmentation its window and
    puzzle derive, and none at all from an unaugmented window."""
    for rollout in rollouts:
        record = rollout.metadata.get("augmentation")
        label = f"rollout {rollout.trajectory_id}"
        if params is None:
            if record is not None:
                raise ValueError(f"{label}: an unaugmented window reported an augmentation")
            continue
        if not isinstance(record, dict):
            raise ValueError(f"{label}: an augmented window reported no augmentation")
        puzzle = rollout.metadata.get("game_id")
        if record.get("puzzle") != puzzle:
            raise ValueError(f"{label}: augmentation was drawn for another puzzle")
        if record != expected_record(params, record, puzzle):
            raise ValueError(f"{label}: augmentation differs from its window's derivation")
