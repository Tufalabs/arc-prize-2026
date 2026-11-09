"""Packed-dataset cache + parallel trace staging (sft/dataset_cache.py).

Unlike test_sft_selection.py these need `datasets` + `pillow` (the sft extra):
    uv run --extra sft --with pytest pytest tests/test_dataset_cache.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("datasets")
from PIL import Image

from sft.dataset_cache import cache_key, load_or_build, stage_trace_dirs


def _write_trace(root: Path, name: str, *, with_image: bool = True) -> Path:
    """Minimal finalized TAAF trace folder with one retemplate-able step."""
    trace = root / name
    (trace / "media").mkdir(parents=True, exist_ok=True)
    content = [{"type": "text", "text": "observe"}]
    if with_image:
        Image.new("RGB", (4, 4), (255, 0, 0)).save(trace / "media" / "f0.png")
        content.append({"type": "image", "path": "media/f0.png"})
    step = {
        "step_index": 0,
        "messages": [{"role": "user", "content": content}],
        "output_text": "ACTION1",
    }
    (trace / "steps.jsonl").write_text(json.dumps(step) + "\n", encoding="utf-8")
    (trace / "_SUCCESS").write_text("", encoding="utf-8")
    (trace / "trace.json").write_text("{}", encoding="utf-8")
    return trace


# --- cache key ----------------------------------------------------------------


def test_cache_key_order_insensitive_but_content_sensitive(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    assert cache_key([a, b], "auto") == cache_key([b, a], "auto")
    assert cache_key([a], "auto") != cache_key([a, b], "auto")
    assert cache_key([a], "auto") != cache_key([a], "retemplate")


def test_cache_key_rejects_unknown_empty_completion_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SFT_ON_EMPTY_COMPLETION", "erorr")
    with pytest.raises(ValueError, match="SFT_ON_EMPTY_COMPLETION"):
        cache_key([tmp_path / "trace"], "auto")


# --- staging --------------------------------------------------------------------


def test_stage_copies_only_pipeline_inputs(tmp_path: Path) -> None:
    src = _write_trace(tmp_path / "run", "t1")
    (src / "intermediate_states.pkl").write_bytes(b"x" * 16)  # bulky extra: skip
    [staged] = stage_trace_dirs([src], tmp_path / "stage", workers=4)
    assert staged.parent == tmp_path / "stage"
    assert staged.name.startswith("t1-")  # basename + resolved-source-path hash
    for name in ("steps.jsonl", "_SUCCESS", "trace.json", "media/f0.png", ".staged"):
        assert (staged / name).exists(), name
    assert not (staged / "intermediate_states.pkl").exists()


def test_stage_skips_already_staged(tmp_path: Path) -> None:
    src = _write_trace(tmp_path / "run", "t1")
    stage_trace_dirs([src], tmp_path / "stage", workers=4)
    (src / "steps.jsonl").write_text("poison\n", encoding="utf-8")
    [staged] = stage_trace_dirs([src], tmp_path / "stage", workers=4)
    assert "poison" not in (staged / "steps.jsonl").read_text(encoding="utf-8")


def test_stage_rebuilds_unmarked_partial_destination(tmp_path: Path) -> None:
    src = _write_trace(tmp_path / "run", "t1")
    [staged] = stage_trace_dirs([src], tmp_path / "stage", workers=4)
    (staged / ".staged").unlink()
    (staged / "stale-file").write_text("must disappear", encoding="utf-8")

    [restaged] = stage_trace_dirs([src], tmp_path / "stage", workers=4)

    assert restaged == staged
    assert (restaged / ".staged").is_file()
    assert not (restaged / "stale-file").exists()


def test_stage_rejects_incomplete_trace_without_publishing_marker(
    tmp_path: Path,
) -> None:
    src = tmp_path / "run" / "incomplete"
    src.mkdir(parents=True)
    (src / "steps.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        stage_trace_dirs([src], tmp_path / "stage", workers=2)

    assert list((tmp_path / "stage").glob("*/.staged")) == []


def test_stage_duplicate_source_copies_once_but_preserves_weighting(
    tmp_path: Path,
) -> None:
    src = _write_trace(tmp_path / "run", "t1")

    staged = stage_trace_dirs([src, src], tmp_path / "stage", workers=4)

    assert len(staged) == 2
    assert staged[0] == staged[1]
    assert (staged[0] / ".staged").is_file()


def test_stage_cross_split_same_basename_not_conflated(tmp_path: Path) -> None:
    # Sequential invocations (train then eval) share the stage root; a shared
    # trace basename must not resolve to the same dest, or the second split
    # hits the first's `.staged` marker and silently trains on its data.
    train = _write_trace(tmp_path / "train", "same-id", with_image=True)
    eval_src = _write_trace(tmp_path / "eval", "same-id", with_image=False)
    assert (train / "steps.jsonl").read_text() != (eval_src / "steps.jsonl").read_text()

    [staged_train] = stage_trace_dirs([train], tmp_path / "stage", workers=4)
    [staged_eval] = stage_trace_dirs([eval_src], tmp_path / "stage", workers=4)
    assert staged_train != staged_eval
    assert (staged_eval / "steps.jsonl").read_text() == (
        eval_src / "steps.jsonl"
    ).read_text()
    assert (staged_train / "steps.jsonl").read_text() == (
        train / "steps.jsonl"
    ).read_text()


def test_streamed_build_text_first_image_second(tmp_path: Path, monkeypatch) -> None:
    # Schema regression: with a text-only probe trace, `images` must still be
    # typed as Sequence(Image()) or the first image-bearing trace fails Arrow
    # generation (cast-to-null) during the streamed build.
    from sft.taaf_adapter import build_dataset_from_trace_dirs

    monkeypatch.setenv("SFT_BUILD_NUM_PROC", "1")
    text = _write_trace(tmp_path / "run", "t-text", with_image=False)
    image = _write_trace(tmp_path / "run", "t-image", with_image=True)
    dataset, resolved = build_dataset_from_trace_dirs(
        [text, image], mode="retemplate", work_dir=tmp_path / "work"
    )
    assert resolved == "retemplate"
    assert dataset.num_rows == 2
    rows = {len(images or []) for images in dataset["images"]}
    assert rows == {0, 1}  # one text-only row, one row carrying its PIL image


def test_explicit_token_exact_rejects_media_even_with_token_ids(tmp_path: Path) -> None:
    from sft.taaf_adapter import build_dataset_from_trace_dirs

    trace = _write_trace(tmp_path / "run", "with-image", with_image=True)
    step = json.loads((trace / "steps.jsonl").read_text(encoding="utf-8"))
    step["input_token_ids"] = [1, 2]
    step["output_token_ids"] = [3]
    (trace / "steps.jsonl").write_text(json.dumps(step) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contain media"):
        build_dataset_from_trace_dirs([trace], mode="token_exact")


# --- load_or_build ---------------------------------------------------------------


def test_no_cache_dir_builds_directly(tmp_path: Path) -> None:
    trace = _write_trace(tmp_path / "run", "t1")
    dataset, mode = load_or_build("train", [trace], mode="auto", cache_dir=None)
    assert mode == "retemplate"
    assert len(dataset) == 1


def test_pack_round_trip_and_cache_hit(tmp_path: Path) -> None:
    trace = _write_trace(tmp_path / "run", "t1")
    cache = tmp_path / "cache"

    built, mode = load_or_build("train", [trace], mode="auto", cache_dir=cache)
    assert mode == "retemplate"
    packs = list(cache.glob("train-*"))
    assert len(packs) == 1 and (packs[0] / "sft_pack.json").exists()

    # Second call must hit the pack, not the raw trace: poison the source.
    (trace / "steps.jsonl").unlink()
    loaded, mode2 = load_or_build("train", [trace], mode="auto", cache_dir=cache)
    assert mode2 == "retemplate"
    assert len(loaded) == len(built) == 1
    row = loaded[0]
    assert row["completion"][0]["content"][0]["text"] == "ACTION1"
    assert len(row["images"]) == 1  # media bytes embedded in the pack


def test_miss_with_stage_dir_builds_from_staged_copies(tmp_path: Path) -> None:
    trace = _write_trace(tmp_path / "run", "t1")
    dataset, _ = load_or_build(
        "train",
        [trace],
        mode="auto",
        cache_dir=tmp_path / "cache",
        stage_dir=tmp_path / "stage",
    )
    staged = list((tmp_path / "stage").glob("t1-*/.staged"))
    assert len(staged) == 1
    assert len(dataset) == 1


def test_load_or_build_streams_and_cleans_build_dir(tmp_path: Path) -> None:
    """A cache miss builds via the streaming path and leaves only the pack."""
    traces = [_write_trace(tmp_path / "run", f"t{i}") for i in range(3)]
    cache = tmp_path / "cache"
    dataset, resolved = load_or_build("train", traces, mode="auto", cache_dir=cache)
    assert resolved == "retemplate"
    assert len(dataset) == 3
    leftovers = [d.name for d in cache.iterdir() if d.name.startswith(".")]
    assert leftovers == [], leftovers
    packs = [d.name for d in cache.iterdir()]
    assert len(packs) == 1 and packs[0].startswith("train-")
    # The returned dataset must be usable after the build dir is gone.
    assert dataset[0]["images"] is not None


def test_load_or_build_recovers_stale_build_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _write_trace(tmp_path / "run", "t1")
    cache = tmp_path / "cache"
    work = tmp_path / "work"
    packed_name = f"train-{cache_key([trace], 'auto')}"
    stale_build = work / f".{packed_name}.build.{os.getpid()}"
    stale_build.mkdir(parents=True)
    (stale_build / "partial.arrow").write_bytes(b"partial")
    monkeypatch.setenv("SFT_BUILD_WORK_DIR", str(work))
    monkeypatch.setenv("SFT_BUILD_NUM_PROC", "1")

    dataset, resolved = load_or_build("train", [trace], mode="auto", cache_dir=cache)

    assert resolved == "retemplate"
    assert len(dataset) == 1
    assert not stale_build.exists()
