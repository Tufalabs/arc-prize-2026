"""Crash-safety of the tb_gap_collapse event-file swap.

The happy path (gaps rewritten, idempotent second pass) is pinned by
test_tb_gap_collapse_rewrites_large_gaps in tests/test_inference.py; here we
pin the swap ORDER (install replacements before unlinking originals) and the
manifest-driven recovery, so no crash point loses a run's event history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Two run segments at 600s cadence separated by a 5-hour queue wait.
_POINTS = ((0, 1000.0), (1, 1600.0), (2, 2200.0), (3, 20200.0), (4, 20800.0))


def _write_run_events(run_dir: Path, points=_POINTS) -> None:
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(run_dir))
    for step, wall in points:
        event = event_pb2.Event(wall_time=wall, step=step)
        event.summary.value.append(
            summary_pb2.Summary.Value(tag="score", simple_value=float(step))
        )
        writer.add_event(event)
    writer.close()


def _summary_wall_times(run_dir: Path) -> list[float]:
    from tensorboard.backend.event_processing.event_file_loader import (
        EventFileLoader,
    )

    times: list[float] = []
    for path in sorted(run_dir.iterdir()):
        if "tfevents" not in path.name:
            continue
        for event in EventFileLoader(str(path)).Load():
            if event.HasField("summary"):
                times.append(event.wall_time)
    return sorted(times)


def _assert_collapsed_history(run_dir: Path) -> None:
    times = _summary_wall_times(run_dir)
    assert len(times) == 5
    deltas = [b - a for a, b in zip(times, times[1:])]
    assert max(deltas) == pytest.approx(600.0)


def test_replacements_install_before_originals_are_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training import tb_gap_collapse

    run = tmp_path / "run"
    run.mkdir()
    _write_run_events(run)
    original_names = {p.name for p in run.iterdir() if "tfevents" in p.name}

    original_unlink = tb_gap_collapse.os.unlink

    def exploding_unlink(path, *args, **kwargs):
        if Path(path).name in original_names:
            raise RuntimeError("crash injected between install and unlink")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tb_gap_collapse.os, "unlink", exploding_unlink)
    with pytest.raises(RuntimeError, match="crash injected"):
        tb_gap_collapse.collapse_run_dir(run, gap_threshold=2400.0)
    monkeypatch.undo()

    # The crash hit AFTER the replacement set was installed: the originals
    # are all still present alongside the replacements — nothing was lost.
    names = {p.name for p in run.iterdir() if "tfevents" in p.name}
    assert original_names < names

    # The next invocation finishes the interrupted swap from the manifest,
    # then finds nothing left to collapse.
    assert tb_gap_collapse.collapse_run_dir(run, gap_threshold=2400.0) == 0
    assert {p.name for p in run.iterdir() if "tfevents" in p.name} == (
        names - original_names
    )
    _assert_collapsed_history(run)
    assert not (tmp_path / "run.gap-staging").exists()


@pytest.mark.parametrize("in_place", [True, False])
def test_rewrite_copies_complete_local_files_to_durable_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_place: bool
) -> None:
    from training import tb_gap_collapse

    run = tmp_path / "run"
    _write_run_events(run)
    originals = {p.name: p.read_bytes() for p in run.iterdir()}
    output = run if in_place else tmp_path / "output"
    staging = output.with_name(output.name + ".gap-staging")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tb_gap_collapse.tempfile, "tempdir", str(scratch))
    original_copy = tb_gap_collapse.shutil.copyfile
    copied = []

    def copy_complete_file(source, destination):
        assert source.is_relative_to(scratch)
        assert destination.parent == staging
        assert not (staging / "superseded.json").exists()
        _assert_collapsed_history(source.parent)
        assert {p.name: p.read_bytes() for p in run.iterdir()} == originals
        copied.append(source)
        return original_copy(source, destination)

    monkeypatch.setattr(tb_gap_collapse.shutil, "copyfile", copy_complete_file)
    assert tb_gap_collapse.collapse_run_dir(run, output_dir=output) == 1
    assert len(copied) == 1
    assert not copied[0].parent.exists()
    assert not staging.exists()
    _assert_collapsed_history(output)
    if not in_place:
        assert {p.name: p.read_bytes() for p in run.iterdir()} == originals


def test_interrupted_copy_preserves_originals_and_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training import tb_gap_collapse

    run = tmp_path / "run"
    _write_run_events(run)
    originals = {p.name: p.read_bytes() for p in run.iterdir()}
    staging = tmp_path / "run.gap-staging"
    copied = []

    def partial_copy(source, destination):
        copied.append(source)
        destination.write_bytes(source.read_bytes()[:20])
        raise OSError("copy interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(tb_gap_collapse.shutil, "copyfile", partial_copy)
        with pytest.raises(OSError, match="copy interrupted"):
            tb_gap_collapse.collapse_run_dir(run)
    assert {p.name: p.read_bytes() for p in run.iterdir()} == originals
    assert not (staging / "superseded.json").exists()
    assert not copied[0].parent.exists()
    assert tb_gap_collapse.collapse_run_dir(run) == 1
    _assert_collapsed_history(run)
    assert not staging.exists()


def test_staging_without_manifest_is_discarded_and_originals_survive(
    tmp_path: Path,
) -> None:
    from training.tb_gap_collapse import collapse_run_dir

    run = tmp_path / "run"
    run.mkdir()
    _write_run_events(run)
    staging = tmp_path / "run.gap-staging"
    staging.mkdir()
    # A crash before the manifest was written: nothing was installed, so the
    # half-written staging output is discarded and the pass restarts cleanly.
    (staging / "events.out.tfevents.1.host.1.0").write_bytes(b"partial write")

    assert collapse_run_dir(run, gap_threshold=2400.0) == 1
    assert not staging.exists()
    _assert_collapsed_history(run)


def test_recovery_completes_a_swap_interrupted_mid_install(
    tmp_path: Path,
) -> None:
    from training.tb_gap_collapse import collapse_run_dir

    # Produce a complete collapsed replacement set via the offline mode.
    source = tmp_path / "source"
    source.mkdir()
    _write_run_events(source)
    collapsed_dir = tmp_path / "collapsed"
    assert collapse_run_dir(source, output_dir=collapsed_dir, gap_threshold=2400.0) == 1

    # Reconstruct the on-disk state of a crash after the manifest was written
    # but before any staged file was installed: originals intact, the full
    # replacement set still staged.
    run = tmp_path / "run"
    run.mkdir()
    _write_run_events(run)
    original_names = sorted(p.name for p in run.iterdir() if "tfevents" in p.name)
    staging = tmp_path / "run.gap-staging"
    staging.mkdir()
    for path in collapsed_dir.iterdir():
        if "tfevents" in path.name:
            (staging / path.name).write_bytes(path.read_bytes())
    (staging / "superseded.json").write_text(
        json.dumps(original_names), encoding="utf-8"
    )

    # Recovery installs the staged set, unlinks the superseded originals, and
    # the normal pass then finds nothing left to collapse.
    assert collapse_run_dir(run, gap_threshold=2400.0) == 0
    assert not staging.exists()
    for name in original_names:
        assert not (run / name).exists()
    _assert_collapsed_history(run)
