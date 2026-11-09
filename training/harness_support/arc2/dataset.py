"""Read the pinned ARC-2 puzzle manifest without importing its harness."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

# The two packed puzzle sets and their sizes; a section names these or DERIVED_DATASETS.
DATASET_SIZES = {"arc2-train": 1000, "arc2-eval": 120}


def dataset_manifest(name: str) -> dict[str, Any]:
    """Identify the complete pinned puzzle set and its packed contents.

    A derived set is its packed set's content restricted to its puzzles.
    """
    if name in DERIVED_DATASETS:
        source, ids = DERIVED_DATASETS[name]
        packed = dataset_manifest(source)
        missing = sorted(set(ids) - set(packed["puzzle_ids"]))
        if missing or len(set(ids)) != len(ids):
            raise ValueError(f"{name}: duplicate puzzles or {missing} not in {source}")
        return {
            "name": name,
            "source": source,
            "sha256": packed["sha256"],
            "puzzle_ids": list(ids),
            "test_outputs": {game: packed["test_outputs"][game] for game in ids},
        }
    expected = DATASET_SIZES
    if name not in expected:
        raise ValueError(f"Unsupported ARC-2 dataset: {name!r}")
    path = (
        Path(__file__).resolve().parents[2]
        / "arc2-platypus/packages/taaf/src/taaf/data"
        / f"{name}.npz"
    )
    content = path.read_bytes()
    with np.load(path, allow_pickle=False) as packed:
        ids = bytes(packed["ids"]).decode("utf-8").splitlines()
        index = packed["index"]
        outputs = {
            game_id: int(
                sum((index[:, 0] == i) & (index[:, 2] == 1) & (index[:, 5] == 0))
            )
            for i, game_id in enumerate(ids)
        }
    if len(ids) != expected[name] or len(set(ids)) != len(ids):
        raise ValueError(f"{name}: unexpected puzzle count or duplicate identities")
    if any(count < 1 for count in outputs.values()):
        raise ValueError(f"{name}: every puzzle needs scored test outputs")
    return {
        "name": name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "puzzle_ids": ids,
        "test_outputs": outputs,
    }


# Named subsets of a pinned set, copied from the ARC-2 harness (taaf's
# ``DERIVED_SETS`` at arc2-platypus f380ffd; tests/test_arc2_dataset_selection.py
# compares them with ``taaf.datasets.load``): the trainer imports no harness
# package, and the job stages only the packed sets. Name -> (packed set,
# puzzle ids in taaf's order).
DERIVED_DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    # The arc2-train puzzles Flash-Next e307 did not solve on both of two
    # passes (taaf's HARD_TRAIN_TASK_IDS, 2026-09-15).
    "arc2-train-hard": (
        "arc2-train",
        (
            "09629e4f", "0d87d2a6", "1190bc91", "1acc24af", "1b8318e3", "1c56ad9f", "1e5d6875", "22425bda",
            "234bbc79", "2ccd9fef", "2dd70a9a", "31adaf00", "396d80d7", "4290ef0e", "456873bc", "50f325b5",
            "5e6bbc0b", "5ecac7f7", "6a1e5592", "84ba50d3", "891232d6", "8cb8642d", "8dab14c2", "8fbca751",
            "9473c6fb", "963c33f8", "9aec4887", "a64e4611", "a6953f00", "aa62e3f4", "b5bb5719", "b74ca5d1",
            "b7cb93ac", "b942fd60", "bf32578f", "c3fa4749", "d017b73f", "d6e50e54", "df9fd884", "e2092e0c",
            "e5062a87", "e73095fd", "f3b10344", "f8cc533f", "f8f52ecc", "f9d67f8b", "fe45cba4",
        ),
    ),
}
# Every name a section's ``dataset`` may hold, alone or in a list.
DATASET_NAMES = (*DATASET_SIZES, *DERIVED_DATASETS)


def dataset_names(dataset: str | list[str]) -> list[str]:
    """A section's ``dataset``, one name or a list of them, as a list."""
    return [dataset] if isinstance(dataset, str) else list(dataset)
