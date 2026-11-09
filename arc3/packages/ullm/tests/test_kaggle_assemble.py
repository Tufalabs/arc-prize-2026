"""``ullm.kaggle_assemble``: a checkpoint put together from the datasets that
hold its parts, as a kernel's setup does before the engine reads it.

The datasets here are directories of tiny safetensors files laid out as a
kernel mounts them (``<root>/<owner>/<slug>``); a shard's header is all the
assembler reads of it.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
import ullm.kaggle
import ullm.kaggle_assemble as kaggle_assemble

PRUNE = "owner/prune"
BASE = "owner/base"


def _shard(path: pathlib.Path, tensors: list[str], *, truncate: int = 0) -> None:
    """A safetensors file holding one float32 per tensor."""
    header = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, name in enumerate(tensors)
    }
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(4 * len(tensors) - truncate))


def _dataset(
    root: pathlib.Path, ref: str, shards: dict[str, list[str]], *, index: dict[str, list[str]] | None = None
) -> pathlib.Path:
    """A mounted dataset holding ``shards``, with an index over ``index``
    (no index file when it is None)."""
    directory = root / ref
    directory.mkdir(parents=True)
    for name, tensors in shards.items():
        _shard(directory / name, tensors)
    if index is not None:
        weight_map = {tensor: shard for shard, tensors in index.items() for tensor in tensors}
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (directory / "config.json").write_text("{}")
    return directory


#: Two shards: the experts, which a prune rewrites, and a table every prune shares.
LAYOUT = {"model-00001-of-00002.safetensors": ["experts.0", "experts.1"], "model-00002-of-00002.safetensors": ["ple.0"]}


@pytest.fixture
def mounts(tmp_path: pathlib.Path) -> pathlib.Path:
    """A prune that carries only its first shard, and a base that carries both."""
    root = tmp_path / "input"
    _dataset(
        root, PRUNE, {"model-00001-of-00002.safetensors": LAYOUT["model-00001-of-00002.safetensors"]}, index=LAYOUT
    )
    _dataset(root, BASE, LAYOUT, index=LAYOUT)
    return root


def test_the_prune_s_own_files_and_the_shards_it_lacks_make_one_checkpoint(
    mounts: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "assembled"
    taken = kaggle_assemble.assemble(out, PRUNE, [BASE], root=str(mounts))
    assert taken == [("model-00002-of-00002.safetensors", BASE)]
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["config.json", "model.safetensors.index.json", *LAYOUT]
    )
    assert (out / "model-00001-of-00002.safetensors").resolve() == mounts / PRUNE / "model-00001-of-00002.safetensors"
    assert (out / "model-00002-of-00002.safetensors").resolve() == mounts / BASE / "model-00002-of-00002.safetensors"
    assert (out / "model.safetensors.index.json").resolve().parent == mounts / PRUNE, (
        "the prune's index, not the base's"
    )


def test_assembling_again_changes_nothing(mounts: pathlib.Path, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "assembled"
    kaggle_assemble.assemble(out, PRUNE, [BASE], root=str(mounts))
    before = {path.name: os.readlink(path) for path in out.iterdir()}
    kaggle_assemble.assemble(out, PRUNE, [BASE], root=str(mounts))
    assert {path.name: os.readlink(path) for path in out.iterdir()} == before


def test_a_shard_comes_from_the_first_dataset_that_has_it(mounts: pathlib.Path, tmp_path: pathlib.Path) -> None:
    _dataset(mounts, "owner/other", {"model-00002-of-00002.safetensors": ["ple.0"]})
    taken = kaggle_assemble.assemble(tmp_path / "a", PRUNE, ["owner/other", BASE], root=str(mounts))
    assert taken == [("model-00002-of-00002.safetensors", "owner/other")]


def test_a_shard_no_dataset_holds_is_refused(mounts: pathlib.Path, tmp_path: pathlib.Path) -> None:
    _dataset(mounts, "owner/empty", {"unrelated.safetensors": ["x"]})
    with pytest.raises(SystemExit, match="no dataset holds model-00002-of-00002.safetensors"):
        kaggle_assemble.assemble(tmp_path / "a", PRUNE, ["owner/empty"], root=str(mounts))


def test_a_borrowed_shard_laid_out_otherwise_is_refused(mounts: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A dataset re-uploaded under the same name with another layout: the
    shard is there by name, and its tensors are not the index's."""
    _dataset(mounts, "owner/reshaped", {"model-00002-of-00002.safetensors": ["ple.0", "ple.1"]})
    with pytest.raises(SystemExit, match="holds other tensors than the index places in it"):
        kaggle_assemble.assemble(tmp_path / "a", PRUNE, ["owner/reshaped"], root=str(mounts))


def test_a_truncated_borrowed_shard_is_refused(mounts: pathlib.Path, tmp_path: pathlib.Path) -> None:
    directory = mounts / "owner/truncated"
    directory.mkdir(parents=True)
    _shard(directory / "model-00002-of-00002.safetensors", ["ple.0"], truncate=2)
    with pytest.raises(SystemExit, match="its header says"):
        kaggle_assemble.assemble(tmp_path / "a", PRUNE, ["owner/truncated"], root=str(mounts))


def test_a_dataset_that_is_not_mounted_is_refused(mounts: pathlib.Path, tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit, match="owner/missing is not mounted"):
        kaggle_assemble.assemble(tmp_path / "a", PRUNE, ["owner/missing"], root=str(mounts))


def test_a_shard_comes_from_a_model_version_where_the_kernel_mounts_it(
    mounts: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A base whose files moved to a Kaggle model: TAAF's input map lists no
    models, so the version is read at its mount, the framework in lower case."""
    models = tmp_path / "models"
    monkeypatch.setattr(ullm.kaggle, "KAGGLE_MODEL_MOUNT", str(models))
    _dataset(models, "owner/base/transformers/default/1", LAYOUT, index=LAYOUT)
    model = "owner/base/Transformers/default/1"
    taken = kaggle_assemble.assemble(tmp_path / "a", PRUNE, [model], root=str(mounts))
    assert taken == [("model-00002-of-00002.safetensors", model)]
    assert (tmp_path / "a" / taken[0][0]).resolve() == models / "owner/base/transformers/default/1" / taken[0][0]


def test_a_prune_whose_own_part_is_a_model_version_takes_the_rest_from_another(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both parts Kaggle models: the prune's index, configs and own shard from
    its version, the shard it lacks from the base's."""
    models = tmp_path / "models"
    monkeypatch.setattr(ullm.kaggle, "KAGGLE_MODEL_MOUNT", str(models))
    first = "model-00001-of-00002.safetensors"
    _dataset(models, "owner/prune/transformers/default/1", {first: LAYOUT[first]}, index=LAYOUT)
    _dataset(models, "owner/base/transformers/default/1", LAYOUT, index=LAYOUT)
    prune, base = "owner/prune/Transformers/default/1", "owner/base/Transformers/default/1"
    out = tmp_path / "a"
    taken = kaggle_assemble.assemble(out, prune, [base], root=str(tmp_path / "unused"))
    assert taken == [("model-00002-of-00002.safetensors", base)]
    assert (out / first).resolve() == models / "owner/prune/transformers/default/1" / first
    assert (out / "model.safetensors.index.json").resolve().parent == models / "owner/prune/transformers/default/1"


def test_a_model_version_that_is_not_mounted_is_refused(
    mounts: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ullm.kaggle, "KAGGLE_MODEL_MOUNT", str(tmp_path / "models"))
    with pytest.raises(SystemExit, match="owner/base/Transformers/default/1 is not mounted"):
        kaggle_assemble.assemble(tmp_path / "a", PRUNE, ["owner/base/Transformers/default/1"], root=str(mounts))


def test_a_dataset_without_an_index_is_refused(mounts: pathlib.Path, tmp_path: pathlib.Path) -> None:
    _dataset(mounts, "owner/bare", LAYOUT)
    with pytest.raises(SystemExit, match="has no model.safetensors.index.json"):
        kaggle_assemble.assemble(tmp_path / "a", "owner/bare", [BASE], root=str(mounts))


def test_a_mount_that_landed_elsewhere_is_found_through_taaf_s_input_map(
    mounts: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kaggle mounts a dataset at /kaggle/input/<slug> or under datasets/,
    by account and era; TAAF records where each landed."""
    elsewhere = tmp_path / "flat" / "base"
    elsewhere.parent.mkdir()
    (mounts / BASE).rename(elsewhere)
    monkeypatch.setenv("TAAF_KAGGLE_INPUT_PATHS", json.dumps({BASE: str(elsewhere)}))
    taken = kaggle_assemble.assemble(tmp_path / "a", PRUNE, [BASE], root=str(mounts))
    assert (tmp_path / "a" / taken[0][0]).resolve() == elsewhere / "model-00002-of-00002.safetensors"


def test_the_command_line_assembles_and_says_what_came_from_where(
    mounts: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "assembled"
    kaggle_assemble.main([str(out), PRUNE, BASE, "--root", str(mounts)])
    assert (out / "model-00002-of-00002.safetensors").is_symlink()
    assert BASE in capsys.readouterr().out
