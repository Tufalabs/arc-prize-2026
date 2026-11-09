# type: ignore
"""Tests for taaf.trace (R5.*) — trace data structures, validation, write/read,
the dataframe bridge, and retained-bundle export."""

import hashlib
import json
import pickle
import tarfile
from pathlib import Path

import matplotlib
import pytest
import zstandard

import taaf
import taaf.solver_examples

matplotlib.use("Agg")


def _submission(game: taaf.game.Game, flip: bool = True) -> taaf.game.Submission:
    """A submission for ``game``: the mirrored question, or the unmirrored one
    when ``flip`` is False (so two submissions differ)."""
    if game.game_run is None:
        game.start_game()
    view = game.query()
    if flip:
        return taaf.solver_examples.flip_lr(view)
    return taaf.game.Submission(attempts=tuple((q, q) for q in view.questions))


def _played_run(trace_root: Path | None = None) -> taaf.game.GameRun:
    """A flip game played to its single answer; returns its GameRun."""
    game = taaf.game_examples.flip_game()
    game.start_game(taaf.game.RunSession(trace_root=trace_root))
    game.execute_action(_submission(game), generated_tokens=2, uncached_input_tokens=5)
    assert game.game_run is not None
    return game.game_run


def _provider_validation() -> dict:
    return {
        "provider_name": "vllm",
        "provider_version": "0.17",
        "api_kind": "chat.completions",
        "provider_model_id": "m",
        "request_args": {"temperature": 0.0},
        "validation_method": "adapter-v1",
        "original_payload_sha256": "a" * 64,
        "reconstructed_payload_sha256": "a" * 64,
        "input_token_reconstruction_status": "ok",
        "output_token_reconstruction_status": "ok",
        "media_reconstruction_status": "unavailable",
    }


def _tool_call_output_message() -> dict:
    return {
        "role": "assistant",
        "content": None,
        "reasoning": "look first\n",
        "tool_calls": [
            {
                "id": "call0",
                "type": "function",
                "function": {"name": "python", "arguments": '{"code": "print(1)"}'},
            }
        ],
    }


def _trace_for(run: taaf.game.GameRun, *, with_media: bool = False) -> taaf.trace.Trace:
    step0 = taaf.trace.TraceStep(
        run_id=run.run_id,
        step_index=0,
        messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
        input_token_ids=[1, 2, 3],
        output_text="ACTION1",
        output_message=_tool_call_output_message(),
        output_token_ids=[9],
        provider_validation=_provider_validation(),
        action_index=0,
        action=run.history[0].submission,
        uncached_input_tokens=5,
    )
    step1 = taaf.trace.TraceStep(
        run_id=run.run_id,
        step_index=1,
        messages=[{"role": "user", "content": "plan"}],
        input_token_ids=[4],
        output_text="thinking",
        output_token_ids=[7],
        provider_validation=_provider_validation(),
    )
    if with_media:
        step0.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "frame:"},
                    {"type": "image_url", "image_url": {"url": "media/0.png"}},
                ],
            }
        ]
        step0.media = [
            taaf.trace.MediaRef(media_id="m0", path="media/0.png", role="current_frame", source_bytes=b"\x89PNG-fake")
        ]
    return taaf.trace.Trace(
        run_id=run.run_id,
        source_kind="sft_collection",
        repo_name="ARC3-Inference",
        repo_commit_hash="deadbeef",
        model_id="Qwen",
        tokenizer_id="Qwen-tok",
        steps=[step0, step1],
        total_uncached_input_tokens=5,
    )


# --- run_id ------------------------------------------------------------------


def test_game_run_has_run_id_and_round_trips():
    run = _played_run()
    assert run.run_id and isinstance(run.run_id, str)
    d = run.to_json_dict()
    assert d["run_id"] == run.run_id
    restored = taaf.game.GameRun.from_json_dict(d)
    assert restored.run_id == run.run_id


def test_run_ids_are_unique_per_run():
    assert _played_run().run_id != _played_run().run_id


# --- JSON round-trip ---------------------------------------------------------


def test_trace_step_json_round_trip():
    run = _played_run()
    step = _trace_for(run).steps[0]
    d = step.to_json_dict()
    assert d["action"] == run.history[0].submission.to_json_list()
    restored = taaf.trace.TraceStep.from_json_dict(d)
    assert restored.action == run.history[0].submission
    assert restored.input_token_ids == [1, 2, 3]
    assert restored.action_index == 0


def test_trace_json_round_trip():
    run = _played_run()
    trace = _trace_for(run)
    d = trace.to_json_dict()
    assert d["schema_version"] == taaf.trace.SCHEMA_VERSION
    assert d["n_steps"] == 2
    assert "steps" not in d  # steps live in steps.jsonl, not trace.json
    restored = taaf.trace.Trace.from_json_dict(d, steps=trace.steps)
    assert restored.run_id == run.run_id
    assert restored.model_id == "Qwen"
    assert restored.message_format_source.endswith(taaf.trace.OPENAI_OPENAPI_COMMIT)


# --- validation --------------------------------------------------------------


def test_validate_trace_accepts_valid_trace():
    run = _played_run()
    taaf.trace.validate_trace(_trace_for(run), run)  # no raise


def test_validate_trace_rejects_run_id_mismatch():
    run = _played_run()
    trace = _trace_for(run)
    trace.run_id = "other"
    with pytest.raises(ValueError, match="run_id"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_non_contiguous_step_index():
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[1].step_index = 5
    with pytest.raises(ValueError, match="contiguous"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_action_mismatch():
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].action = _submission(taaf.game_examples.flip_game(), flip=False)  # not what was played
    with pytest.raises(ValueError, match="does not match"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_action_index_out_of_range():
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].action_index = 99
    with pytest.raises(ValueError, match="out of range"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_missing_provider_evidence():
    run = _played_run()
    trace = _trace_for(run)
    del trace.steps[0].provider_validation["original_payload_sha256"]
    with pytest.raises(ValueError, match="missing required key"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_failed_status():
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].provider_validation["input_token_reconstruction_status"] = "failed"
    with pytest.raises(ValueError, match="failed"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_empty_output_text_with_tokens():
    """output_text is a faithful render of the tokens, so empty text
    alongside non-empty output_token_ids is a capture bug (the empty-completion
    class that trained instant-EOS models), not a legitimate completion."""
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].output_text = ""
    with pytest.raises(ValueError, match="output_text is empty"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_allows_empty_output_text_for_empty_generation():
    """An immediate end-of-turn (no output tokens at all) is legitimately
    empty — the check only fires when tokens exist."""
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[1].output_text = ""
    trace.steps[1].output_token_ids = []
    taaf.trace.validate_trace(trace, run)  # no raise


def test_validate_trace_logprobs_alignment():
    """A present logprobs list must align one-to-one to
    output_token_ids; None means alternatives unavailable."""
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].logprobs = [[{"token": "A", "logprob": -0.1}]]  # len 1 == len([9])
    taaf.trace.validate_trace(trace, run)  # no raise
    trace.steps[0].logprobs = [[], []]  # misaligned: 2 != 1
    with pytest.raises(ValueError, match="logprobs length"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_missing_output_status():
    run = _played_run()
    trace = _trace_for(run)
    del trace.steps[0].provider_validation["output_token_reconstruction_status"]
    with pytest.raises(ValueError, match="missing required key"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_accepts_output_mismatch_status():
    """A lossy output_message round-trip is recorded evidence, not a
    validity failure — output_text stays exact, so the trace is not discarded."""
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].provider_validation["output_token_reconstruction_status"] = "mismatch"
    taaf.trace.validate_trace(trace, run)  # no raise


def test_validate_trace_accepts_whitespace_mismatch_status():
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].provider_validation["output_token_reconstruction_status"] = "whitespace_mismatch"
    taaf.trace.validate_trace(trace, run)  # no raise


def test_validate_trace_rejects_unknown_output_status():
    """A truly unknown status string (not just 'failed') is rejected."""
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].provider_validation["output_token_reconstruction_status"] = "bogus"
    with pytest.raises(ValueError, match="bogus"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_input_mismatch_status():
    """An input-token mismatch means the trace is not reproducible for
    SFT — unlike the output status (where output_text stays authoritative),
    it is a validation failure, never tolerated evidence."""
    run = _played_run()
    for status in ("mismatch", "whitespace_mismatch"):
        trace = _trace_for(run)
        trace.steps[0].provider_validation["input_token_reconstruction_status"] = status
        with pytest.raises(ValueError, match="input_token_reconstruction_status"):
            taaf.trace.validate_trace(trace, run)


def test_validate_trace_media_fingerprint_structure():
    """Fingerprint records, when recorded, must be structurally sound
    (shape + dtype + at least one of values/sha256); absent stays fine."""
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].provider_validation["media_fingerprints"] = {
        "pixel_values": {"shape": [1, 3, 448, 448], "dtype": "float32", "sha256": "b" * 64},
        "image_grid_thw": {"shape": [1, 3], "dtype": "int64", "values": [[1, 32, 32]]},
    }
    taaf.trace.validate_trace(trace, run)  # no raise
    trace.steps[0].provider_validation["media_fingerprints"] = {
        "pixel_values": {"shape": [1, 3], "dtype": "float32"}  # neither values nor sha256
    }
    with pytest.raises(ValueError, match="values.*sha256|sha256.*values"):
        taaf.trace.validate_trace(trace, run)
    trace.steps[0].provider_validation["media_fingerprints"] = {
        "pixel_values": {"shape": "not-a-list", "dtype": "float32", "sha256": "b" * 64}
    }
    with pytest.raises(ValueError, match="shape"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_non_assistant_output_message():
    run = _played_run()
    trace = _trace_for(run)
    trace.steps[0].output_message = {"role": "user", "content": "nope"}
    with pytest.raises(ValueError, match="output_message.role"):
        taaf.trace.validate_trace(trace, run)


def test_output_message_json_round_trip():
    run = _played_run()
    step = _trace_for(run).steps[0]
    d = step.to_json_dict()
    assert d["output_message"]["tool_calls"][0]["function"]["name"] == "python"
    restored = taaf.trace.TraceStep.from_json_dict(d)
    assert restored.output_message == _tool_call_output_message()


def test_load_trace_reads_v1_traces(tmp_path: Path):
    """v1 traces (no output_message, old schema_version) stay readable."""
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    folder = taaf.trace.write_trace(_trace_for(run), run, trace_root)

    meta = json.loads((folder / "trace.json").read_text())
    meta["schema_version"] = "taaf.trace.v1"
    (folder / "trace.json").write_text(json.dumps(meta))
    steps = []
    for line in (folder / "steps.jsonl").read_text().splitlines():
        d = json.loads(line)
        d.pop("output_message", None)
        d["schema_version"] = "taaf.trace.v1"
        d["provider_validation"].pop("output_token_reconstruction_status", None)
        steps.append(json.dumps(d))
    (folder / "steps.jsonl").write_text("\n".join(steps) + "\n")

    loaded = taaf.trace.load_trace(folder)
    assert loaded.schema_version == "taaf.trace.v1"
    assert loaded.steps[0].output_message is None
    assert loaded.steps[0].output_text == "ACTION1"


def test_load_trace_rejects_unknown_schema_version(tmp_path: Path):
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    folder = taaf.trace.write_trace(_trace_for(run), run, trace_root)
    meta = json.loads((folder / "trace.json").read_text())
    meta["schema_version"] = "taaf.trace.v99"
    (folder / "trace.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unknown trace schema_version"):
        taaf.trace.load_trace(folder)


def test_validate_trace_rejects_media_path_not_in_refs():
    run = _played_run()
    trace = _trace_for(run, with_media=True)
    trace.steps[0].media = []  # message still references media/0.png
    with pytest.raises(ValueError, match="media paths not in media refs"):
        taaf.trace.validate_trace(trace, run)


def test_validate_trace_rejects_absolute_media_path():
    run = _played_run()
    trace = _trace_for(run, with_media=True)
    trace.steps[0].media[0].path = "/etc/passwd"
    with pytest.raises(ValueError, match="trace-relative path"):
        taaf.trace.validate_trace(trace, run)


# --- write / read / _SUCCESS -------------------------------------------------


def test_write_and_load_trace_round_trip(tmp_path: Path):
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    folder = taaf.trace.write_trace(_trace_for(run, with_media=True), run, trace_root)
    assert (folder / "_SUCCESS").exists()
    assert (folder / "trace.json").exists()
    assert (folder / "steps.jsonl").exists()

    loaded = taaf.trace.load_trace(folder)
    assert loaded.run_id == run.run_id
    assert loaded.n_steps == 2
    assert loaded.steps[0].action == run.history[0].submission
    # Media are stored by content: the producer's name is replaced by the
    # bytes' sha256, and the message part citing it follows.
    path = loaded.steps[0].media[0].path
    assert path == f"media/{hashlib.sha256(b'\x89PNG-fake').hexdigest()}.png"
    assert (folder / path).read_bytes() == b"\x89PNG-fake"
    assert path in taaf.trace._message_media_paths(loaded.steps[0].messages)


def _media_step(step: taaf.trace.TraceStep, refs: list[tuple[str, str, bytes | None]]) -> None:
    """Give ``step`` the media refs (media_id, path, bytes) and one user message citing each."""
    step.media = [taaf.trace.MediaRef(media_id=i, path=p, role="current_frame", source_bytes=b) for i, p, b in refs]
    step.messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": p}} for _, p, _ in refs]}]


def _written_media(folder: Path) -> list[str]:
    return sorted(p.name for p in (folder / "media").iterdir())


def test_write_trace_stores_equal_media_bytes_once(tmp_path: Path):
    """An agent whose pictures ride every step's record cites them once per
    step per step; on disk that is one file, and every ref and message part
    resolves to it."""
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    trace = _trace_for(run, with_media=True)
    same = b"\x89PNG-same"
    for k, step in enumerate(trace.steps):
        _media_step(step, [(f"step{k}_img{j}", f"media/step{k}_img{j}.png", same) for j in range(3)])
    folder = taaf.trace.write_trace(trace, run, trace_root)
    assert _written_media(folder) == [f"{hashlib.sha256(same).hexdigest()}.png"]
    loaded = taaf.trace.load_trace(folder)
    for step in loaded.steps:
        assert {m.path for m in step.media} == {f"media/{_written_media(folder)[0]}"}
        assert taaf.trace._message_media_paths(step.messages) == {f"media/{_written_media(folder)[0]}"}
        assert [m.media_id for m in step.media] == [f"step{step.step_index}_img{j}" for j in range(3)]


def test_write_trace_shared_ref_instance_rewrites_every_citing_step(tmp_path: Path):
    """A MediaRef instance shared between steps is renamed once, and every
    step's message part that cited its original path follows — the plan is
    built from the original paths, not from refs mid-mutation."""
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    trace = _trace_for(run, with_media=True)
    shared = taaf.trace.MediaRef(
        media_id="shared", path="media/shared.png", role="current_frame", source_bytes=b"\x89PNG-shared"
    )
    for step in trace.steps:
        step.media = [shared]
        step.messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "media/shared.png"}}]}]
    folder = taaf.trace.write_trace(trace, run, trace_root)
    content = f"media/{hashlib.sha256(b'\x89PNG-shared').hexdigest()}.png"
    for step in taaf.trace.load_trace(folder).steps:
        assert [m.path for m in step.media] == [content]
        assert taaf.trace._message_media_paths(step.messages) == {content}


def test_write_trace_remaps_a_byteless_ref_aliasing_a_renamed_path(tmp_path: Path):
    """A ref without bytes whose path a byte-backed ref also names is an alias
    of those bytes; it follows the rename instead of dangling at the old,
    never-written path."""
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    trace = _trace_for(run, with_media=True)
    _media_step(trace.steps[0], [("a", "media/shared.png", b"\x89PNG-alias")])
    _media_step(trace.steps[1], [("b", "media/shared.png", None)])
    folder = taaf.trace.write_trace(trace, run, trace_root)
    content = f"media/{hashlib.sha256(b'\x89PNG-alias').hexdigest()}.png"
    loaded = taaf.trace.load_trace(folder)
    assert [m.path for m in loaded.steps[1].media] == [content]
    assert taaf.trace._message_media_paths(loaded.steps[1].messages) == {content}
    assert _written_media(folder) == [content.split("/")[1]]


def test_write_trace_dedups_by_digest_across_suffixes_and_names(tmp_path: Path):
    """Equal bytes are one file whatever the producer called them — differing
    extensions, case, a name that already looks content-addressed. Each
    message part is rewritten once from its original url, so a producer name
    equal to another ref's content name cannot cascade."""
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    trace = _trace_for(run, with_media=True)
    same = b"\x89PNG-same"
    digest = hashlib.sha256(same).hexdigest()
    other = b"\x89PNG-other"
    _media_step(trace.steps[0], [("a", "media/a.jpg", same), ("b", "media/b.JPEG", same)])
    # A producer name that IS another ref's content name, carrying other bytes.
    _media_step(trace.steps[1], [("c", f"media/{digest}.jpg", other), ("d", "media/d.png", same)])
    folder = taaf.trace.write_trace(trace, run, trace_root)
    assert _written_media(folder) == sorted([f"{digest}.jpg", f"{hashlib.sha256(other).hexdigest()}.jpg"])
    loaded = taaf.trace.load_trace(folder)
    assert {m.path for m in loaded.steps[0].media} == {f"media/{digest}.jpg"}
    assert [m.path for m in loaded.steps[1].media] == [
        f"media/{hashlib.sha256(other).hexdigest()}.jpg",
        f"media/{digest}.jpg",
    ]
    for step in loaded.steps:
        assert taaf.trace._message_media_paths(step.messages) == {m.path for m in step.media}


def test_write_trace_rejects_two_payloads_at_one_path(tmp_path: Path):
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    trace = _trace_for(run, with_media=True)
    _media_step(trace.steps[0], [("a", "media/x.png", b"\x89PNG-1")])
    _media_step(trace.steps[1], [("b", "media/x.png", b"\x89PNG-2")])
    with pytest.raises(ValueError, match="two different payloads"):
        taaf.trace.write_trace(trace, run, trace_root)
    assert not (trace_root / run.run_id / "_SUCCESS").exists()


def test_write_trace_rejects_missing_media_file(tmp_path: Path):
    """_SUCCESS asserts every referenced media file is on disk. A ref
    without source_bytes and no pre-existing file must fail the write and
    leave no completion marker."""
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    trace = _trace_for(run, with_media=True)
    trace.steps[0].media[0].source_bytes = None
    with pytest.raises(FileNotFoundError, match="media/0.png"):
        taaf.trace.write_trace(trace, run, trace_root)
    assert not (trace_root / run.run_id / "_SUCCESS").exists()


def test_load_trace_requires_success_marker(tmp_path: Path):
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    folder = taaf.trace.write_trace(_trace_for(run), run, trace_root)
    (folder / "_SUCCESS").unlink()
    with pytest.raises(FileNotFoundError):
        taaf.trace.load_trace(folder)


def test_load_trace_rejects_truncated_steps(tmp_path: Path):
    """A truncated steps.jsonl inside a _SUCCESS folder (partial write that
    slipped past finalization) must not load silently."""
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    folder = taaf.trace.write_trace(_trace_for(run), run, trace_root)
    steps_file = folder / "steps.jsonl"
    first_line = steps_file.read_text().splitlines()[0]
    steps_file.write_text(first_line + "\n")  # drop the second step
    with pytest.raises(ValueError, match="n_steps"):
        taaf.trace.load_trace(folder)


def test_register_trace_writes_and_validates(tmp_path: Path):
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    folder = run.register_trace(_trace_for(run))
    assert folder == trace_root / run.run_id
    assert (folder / "_SUCCESS").exists()


def test_register_trace_invalid_writes_no_success(tmp_path: Path):
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    trace = _trace_for(run)
    trace.steps[0].action = _submission(taaf.game_examples.flip_game(), flip=False)  # mismatched
    with pytest.raises(ValueError):
        run.register_trace(trace)
    assert not (trace_root / run.run_id / "_SUCCESS").exists()


# --- dataframe bridge --------------------------------------------------------


def test_iter_trace_rows_denormalizes_and_skips_incomplete(tmp_path: Path):
    trace_root = tmp_path / "traces"
    run = _played_run(trace_root=trace_root)
    taaf.trace.write_trace(_trace_for(run), run, trace_root)

    # An incomplete folder (no _SUCCESS) must be ignored.
    incomplete = trace_root / "incomplete"
    incomplete.mkdir()
    (incomplete / "trace.json").write_text("{}")

    rows = list(taaf.trace.iter_trace_rows(trace_root))
    assert len(rows) == 2  # only the completed trace's two steps
    row = rows[0]
    assert row["run_id"] == run.run_id
    assert row["step_index"] == 0
    assert row["source_kind"] == "sft_collection"  # trace-level metadata denormalized
    assert row["model_id"] == "Qwen"
    assert "trace_created_at" in row  # collision-renamed trace field


# --- retained-bundle export --------------------------------------------------


def test_export_run_bundle_contents_and_manifest(tmp_path: Path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trace_root = job_dir / "traces"
    run = _played_run(trace_root=trace_root)
    taaf.trace.write_trace(_trace_for(run, with_media=True), run, trace_root)
    (job_dir / "benchmark.json").write_text(json.dumps({"game_runs": [run.to_json_dict()]}))

    out = taaf.trace.export_run_bundle(job_dir, trace_root, job_dir / "retained_traces" / "trace_bundle.tar.zst")
    assert out.exists()

    with open(out, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            names = set(tar.getnames())
    assert "benchmark.json" in names
    assert f"traces/{run.run_id}/trace.json" in names
    assert f"traces/{run.run_id}/_SUCCESS" in names
    assert f"traces/{run.run_id}/media/{hashlib.sha256(b'\x89PNG-fake').hexdigest()}.png" in names

    manifest = json.loads((job_dir / "retained_traces" / "trace_manifest.json").read_text())
    assert manifest["rows"][0]["run_id"] == run.run_id
    assert manifest["rows"][0]["export_member_prefix"] == f"traces/{run.run_id}"


def test_export_run_bundle_skips_incomplete_traces(tmp_path: Path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trace_root = job_dir / "traces"
    run = _played_run(trace_root=trace_root)
    taaf.trace.write_trace(_trace_for(run), run, trace_root)
    (job_dir / "benchmark.json").write_text("{}")

    incomplete = trace_root / "halfwritten"
    incomplete.mkdir()
    (incomplete / "trace.json").write_text("{}")

    out = taaf.trace.export_run_bundle(job_dir, trace_root, job_dir / "retained_traces" / "trace_bundle.tar.zst")
    with open(out, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            names = set(tar.getnames())
    assert not any("halfwritten" in n for n in names)


# --- Benchmark integration ---------------------------------------------------


async def test_benchmark_registers_and_retains_traces(tmp_path: Path):
    bm = taaf.benchmark.Benchmark(
        label="trace-test",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverTracing(),
        n_passes=1,
        job_dir=tmp_path,
        retain_traces=True,
        periodic_save_interval_s=3600.0,
    )
    await bm.run()

    run_id = bm.game_runs[0].run_id
    trace_dir = tmp_path / "traces" / run_id
    assert (trace_dir / "_SUCCESS").exists()
    assert (tmp_path / "retained_traces" / "trace_bundle.tar.zst").exists()

    # The trace joins back to benchmark.json by run_id.
    benchmark = json.loads((tmp_path / "benchmark.json").read_text())
    assert benchmark["game_runs"][0]["run_id"] == run_id

    rows = list(taaf.trace.iter_trace_rows(tmp_path / "traces"))
    assert rows and all(r["run_id"] == run_id for r in rows)


async def test_benchmark_rejects_retain_traces_with_minimal_diagnostics(tmp_path: Path):
    """Retention + minimal diagnostics is a config error — minimal mode
    records no run-level metadata for traces to join to, so the
    combination fails loudly at run() start instead of producing an orphaned
    bundle."""
    bm = taaf.benchmark.Benchmark(
        label="trace-minimal",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverTracing(),
        n_passes=1,
        job_dir=tmp_path,
        retain_traces=True,
        periodic_save_interval_s=3600.0,
    )
    with pytest.raises(ValueError, match="retain_traces.*minimal_diagnostics"):
        await bm.run(minimal_diagnostics=True)
    assert not (tmp_path / "retained_traces").exists()


async def test_benchmark_env_trace_root_resolves_on_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A symbolic trace root (pickled at launch, before the scheduler
    assigns scratch) resolves against the worker's environment; the bundle
    still lands in the durable job_dir."""
    scratch = tmp_path / "scratch"
    job_dir = tmp_path / "job"
    monkeypatch.setenv("TAAF_NODE_LOCAL_ROOT", str(scratch))
    bm = taaf.benchmark.Benchmark(
        label="trace-env-root",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverTracing(),
        job_dir=job_dir,
        trace_root=Path("${TAAF_NODE_LOCAL_ROOT}/traces"),
        retain_traces=True,
        periodic_save_interval_s=3600.0,
    )
    # The symbolic path is what pickles across to the worker.
    assert str(pickle.loads(pickle.dumps(bm)).trace_root) == "${TAAF_NODE_LOCAL_ROOT}/traces"
    await bm.run()
    assert (scratch / "traces" / bm.game_runs[0].run_id / "_SUCCESS").exists()
    assert (job_dir / "retained_traces" / "trace_bundle.tar.zst").exists()


def test_benchmark_unresolved_trace_root_variable_is_fatal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAAF_NODE_LOCAL_ROOT", raising=False)
    bm = taaf.benchmark.Benchmark(trace_root=Path("${TAAF_NODE_LOCAL_ROOT}/traces"), retain_traces=True)
    with pytest.raises(RuntimeError, match="unresolved"):
        bm._effective_trace_root()


async def test_benchmark_rejects_env_trace_root_without_retention(tmp_path: Path):
    bm = taaf.benchmark.Benchmark(
        label="trace-env-no-retain",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverTracing(),
        job_dir=tmp_path,
        trace_root=Path("${TAAF_NODE_LOCAL_ROOT}/traces"),
        periodic_save_interval_s=3600.0,
    )
    with pytest.raises(ValueError, match="retain_traces"):
        await bm.run()


async def test_benchmark_default_trace_root_is_job_dir_traces(tmp_path: Path):
    bm = taaf.benchmark.Benchmark(
        label="trace-default-root",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverTracing(),
        job_dir=tmp_path,
        periodic_save_interval_s=3600.0,
    )
    await bm.run()
    assert (tmp_path / "traces" / bm.game_runs[0].run_id / "_SUCCESS").exists()
    # retain_traces defaults False → no bundle.
    assert not (tmp_path / "retained_traces").exists()
