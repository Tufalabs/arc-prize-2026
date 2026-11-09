"""Kaggle's versioned offline wheel utility mount resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import ullm.kaggle as ullm_kaggle


def _populate(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "package.whl").write_bytes(b"wheel")
    return path


def test_mapped_wheel_directory_wins_when_populated(tmp_path: Path) -> None:
    mapped_root = tmp_path / "mapped"
    mapped_wheels = _populate(mapped_root / "wheels")
    private_wheels = _populate(tmp_path / "notebooks" / "owner" / "utility" / "wheels")

    resolved = ullm_kaggle.resolve_kaggle_wheel_dir(
        "owner/utility",
        input_paths={"owner/utility": str(mapped_root)},
        notebooks_root=tmp_path / "notebooks",
    )

    assert resolved == mapped_wheels
    assert resolved != private_wheels


def test_private_underscore_mount_skips_empty_shadow_directories(tmp_path: Path) -> None:
    ref = "owner/my-wheel-utility"
    (tmp_path / "mapped" / "wheels").mkdir(parents=True)
    (tmp_path / "notebooks" / "owner" / "my-wheel-utility" / "wheels").mkdir(parents=True)
    normalized_wheels = _populate(tmp_path / "notebooks" / "owner" / "my_wheel_utility" / "wheels")

    resolved = ullm_kaggle.resolve_kaggle_wheel_dir(
        ref,
        input_paths={ref: str(tmp_path / "mapped")},
        notebooks_root=tmp_path / "notebooks",
    )

    assert resolved == normalized_wheels


def test_missing_wheel_mount_fails_with_every_checked_path(tmp_path: Path) -> None:
    ref = "owner/my-wheel-utility"

    with pytest.raises(RuntimeError, match="populated wheel directory missing") as caught:
        ullm_kaggle.resolve_kaggle_wheel_dir(
            ref,
            input_paths={ref: str(tmp_path / "mapped")},
            notebooks_root=tmp_path / "notebooks",
        )

    message = str(caught.value)
    assert str(tmp_path / "mapped" / "wheels") in message
    assert str(tmp_path / "notebooks" / "owner" / "my-wheel-utility" / "wheels") in message
    assert str(tmp_path / "notebooks" / "owner" / "my_wheel_utility" / "wheels") in message


def test_every_mount_layout_is_probed_whatever_the_declared_kind(tmp_path: Path) -> None:
    """Kaggle has attached the same source as a dataset and as a utility
    output at different times, so the wheels are looked for under every
    layout: the flat input mount, the owner-nested dataset mount, the flat
    and nested utility mounts, and the notebooks mount, hyphens kept or
    normalised."""
    ref = "owner/my-wheel-utility"
    roots = dict(
        input_root=tmp_path / "input", usr_lib_root=tmp_path / "usr_lib", notebooks_root=tmp_path / "notebooks"
    )
    layouts = [
        tmp_path / "input" / "my-wheel-utility",
        tmp_path / "input" / "datasets" / "owner" / "my-wheel-utility",
        tmp_path / "usr_lib" / "my-wheel-utility",
        tmp_path / "usr_lib" / "notebooks" / "owner" / "my_wheel_utility",
        tmp_path / "notebooks" / "owner" / "my-wheel-utility",
    ]
    for mount in layouts:
        wheels = _populate(mount / "wheels")
        assert ullm_kaggle.resolve_kaggle_wheel_dir(ref, input_paths={}, **roots) == wheels
        (wheels / "package.whl").unlink()
        wheels.rmdir()

    with pytest.raises(RuntimeError, match="populated wheel directory missing") as caught:
        ullm_kaggle.resolve_kaggle_wheel_dir(ref, input_paths={ref: str(tmp_path / "mapped")}, **roots)
    message = str(caught.value)
    assert str(tmp_path / "mapped" / "wheels") in message
    for mount in layouts:
        assert str(mount / "wheels") in message
