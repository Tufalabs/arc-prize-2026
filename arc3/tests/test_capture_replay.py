"""Capture/replay: reference generation, bit-exact replay, and the
failure modes that make references useful (U3/U4/I13).

Runs entirely in the cheap tier: ArithmeticStubBackend actually answers
the agent's random sums, so the recorded traffic is request-dependent
without a model.

``test_generate_and_replay_roundtrip`` is the topology proof: it captures
and replays in subprocess mode at concurrency > 1, establishing that
per-game streams make scheduling irrelevant. Everything the other tests
assert — ordinal matching, content verification, exhaustion, the
comparison surface, the file format — is per-game backend behavior that
the process boundary cannot influence, so they run in-process. That is a
deliberate ~5s-per-test trade, not an oversight: a spawned child costs
~1s of imports before it does any work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.factory
from taaf.game_examples import ExampleGame

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from platypus.reference import generate_references, load_reference_seeds, load_references, replay_references
from reference_agents import PromptedRandomAgent

N_GAMES = 3


def _capture_config() -> LLMConfig:
    return LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.ArithmeticStubBackend()),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
        capture=True,
    )


def _replay_config(replay_dir: Path) -> LLMConfig:
    return LLMConfig(
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
        replay_dir=str(replay_dir),
    )


def _benchmark(
    llm_config: LLMConfig,
    job_dir: Path | None,
    *,
    seed: int | None = 11,
    seed_per_game: Mapping[str, int] | None = None,
    process_mode: str = "inprocess",
    **agent_kwargs: Any,
) -> taaf.benchmark.Benchmark:
    agent = PromptedRandomAgent(
        max_moves=5,
        llm_config=llm_config,
        run_config=RunConfig(
            rng_seed=seed,
            rng_seed_per_game=seed_per_game if seed_per_game is not None else {},
            process_mode=process_mode,  # type: ignore[arg-type]
        ),
        **agent_kwargs,
    )
    return taaf.benchmark.Benchmark(
        games=[ExampleGame(label=f"refgame{i}") for i in range(N_GAMES)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=job_dir,
    )


# The captured corpus a replay test receives: files, expected outcomes, dir.
Corpus = tuple[list[Path], dict[str, dict[str, Any]], Path]


async def _generate(tmp_path: Path, **kwargs: Any) -> Corpus:
    benchmark = _benchmark(_capture_config(), tmp_path / "job", **kwargs)
    archives = await generate_references(benchmark, tmp_path / "refs", "arith", minimal_diagnostics=True)
    replay_dir = tmp_path / "unpacked"
    expected = load_references(archives, replay_dir)
    return archives, expected, replay_dir


@pytest.fixture(scope="module")
async def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    """One captured corpus shared by the tests that only differ in how
    they REPLAY it (default generation parameters, no mutation of the
    files). Tests that need different capture-time parameters generate
    their own."""
    return await _generate(tmp_path_factory.mktemp("corpus"))


async def test_generate_and_replay_roundtrip(tmp_path: Path) -> None:
    # I13's validate-by-immediate-replay: a fresh agent (same config, same
    # seed) over the recorded traffic reproduces every game bit-exact —
    # actions, per-action tokens, final outcome. The topology proof: both
    # halves run subprocess-mode at concurrency > 1, so this also pins
    # that concurrent scheduling cannot perturb per-game streams.
    archives, expected, replay_dir = await _generate(tmp_path, process_mode="subprocess")
    assert len(archives) == N_GAMES and len(expected) == N_GAMES
    for game_expected in expected.values():
        assert len(game_expected["actions"]) > 0
        assert any(tokens[0] > 0 for tokens in game_expected["per_action_tokens"])
    replay = _benchmark(_replay_config(replay_dir), None, process_mode="subprocess")
    await replay_references(replay, expected)  # raises on any divergence


async def test_recorded_seeds_are_per_game_and_suffice_to_replay(
    corpus: Corpus,
) -> None:
    # What a reference records is the seed the game's RNG was actually
    # built from, not the run-level knob it was derived from: generation
    # ran rng_seed=11 across three games, so the three recorded seeds are
    # 11, 12, 13.
    archives, expected, replay_dir = corpus
    seeds = load_reference_seeds(archives)
    assert seeds == {f"refgame{i}": 11 + i for i in range(N_GAMES)}
    # And they are sufficient on their own. Replaying with NO rng_seed at
    # all, seeds fed back through the per-game override, still reproduces
    # every game bit-exact — which is what makes a reference replayable
    # without knowing how its seeds were originally derived (and hence
    # independent of the run's game count and ordering).
    replay = _benchmark(_replay_config(replay_dir), None, seed=None, seed_per_game=seeds)
    await replay_references(replay, expected)


async def test_replay_of_diverging_agent_fails_with_mismatch(
    corpus: Corpus,
) -> None:
    # A different seed changes the sums and the action stream — the replay
    # client hard-errors at the first divergent call, the game crashes,
    # and the outcome comparison names the divergence.
    _, expected, replay_dir = corpus
    replay = _benchmark(_replay_config(replay_dir), None, seed=12)
    with pytest.raises(AssertionError, match="diverged on"):
        await replay_references(replay, expected)
    notes = [run.solver_note or "" for run in replay.game_runs]
    assert any("replay mismatch" in note for note in notes)


async def test_replay_exhausts_when_agent_makes_more_calls(tmp_path: Path) -> None:
    # Captured with 2 seam checks per game; replayed with unlimited — the
    # third call has no record and must fail loudly, not silently invent.
    _, expected, replay_dir = await _generate(tmp_path, max_llm_calls=2)
    replay = _benchmark(_replay_config(replay_dir), None, max_llm_calls=None)
    with pytest.raises(AssertionError):
        await replay_references(replay, expected)
    notes = [run.solver_note or "" for run in replay.game_runs]
    assert any("replay exhausted" in note for note in notes)


async def test_replay_of_agent_making_fewer_calls_diverges(tmp_path: Path) -> None:
    # Fewer LLM calls leaves recorded traffic unconsumed AND (because the
    # skipped seam checks skip their rng draws) shifts the action stream —
    # the expected-outcome comparison catches it either way.
    _, expected, replay_dir = await _generate(tmp_path, max_llm_calls=3)
    replay = _benchmark(_replay_config(replay_dir), None, max_llm_calls=1)
    with pytest.raises(AssertionError, match="diverged on"):
        await replay_references(replay, expected)


async def test_replay_leaving_records_unread_is_caught(tmp_path: Path) -> None:
    # The comparison is field by field, so a trailing record that no field
    # reflects — a token_count, in practice — can go missing and still replay
    # green. The consumption check is what sees it, and it needs a job_dir:
    # the count travels out of the process that hosted the game through the
    # diagnostics stream it writes.
    _, expected, replay_dir = await _generate(tmp_path, max_llm_calls=3)
    replay = _benchmark(_replay_config(replay_dir), tmp_path / "replay_job", max_llm_calls=1)
    with pytest.raises(AssertionError, match="unread"):
        await replay_references(replay, expected)


async def test_tampered_expected_is_detected(corpus: Corpus) -> None:
    # The comparison surface itself must bite: a reference whose expected
    # outcome no longer matches the replay is a failure, not a pass.
    _, expected, replay_dir = corpus
    tampered = {game_id: dict(fields) for game_id, fields in expected.items()}
    tampered[next(iter(tampered))]["final_score"] = 99.9
    replay = _benchmark(_replay_config(replay_dir), None)
    with pytest.raises(AssertionError, match="final_score"):
        await replay_references(replay, tampered)


async def test_reference_files_are_wellformed(corpus: Corpus) -> None:
    # One flat JSONL per game with typed lines in order; the record lines
    # are the diffable regression surface.
    archives, _, _ = corpus
    for path in archives:
        assert path.name.endswith(".ref.jsonl")
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        line_types = [entry["line"] for entry in entries]
        assert line_types[0] == "info" and "expected" in line_types and "solver_pkl" in line_types
        records = [entry for entry in entries if entry["line"] == "record"]
        assert len(records) > 0  # empty record sections would replay vacuously
        assert [rec["index"] for rec in records] == list(range(len(records)))
        assert all(rec["request"]["sampling"]["seed"] is not None for rec in records)  # U8 seed stamped


async def test_reference_roundtrip_with_frame_images(tmp_path: Path) -> None:
    # The multimodal seam end to end: frames attached as images ride the
    # subprocess pipe + proxy socket into capture blocks, land in the flat
    # reference file, and replay bit-exact. Subprocess mode on purpose —
    # like the text roundtrip this is a topology proof (images crossing
    # the real wire), not per-game backend behavior.
    archives, expected, replay_dir = await _generate(tmp_path, attach_frame_image=True, process_mode="subprocess")
    for path in archives:
        text = path.read_text()
        assert '"c": "image"' in text  # every game made image-bearing calls
        # And the pixels are elided at collection. The request carries the
        # hash, which is what the comparison sees, so the payload is
        # committed weight nothing reads back — the replay below is what
        # says the elision cost the reference nothing.
        assert '"image_elided": true' in text
        assert "image_b64" not in text
    replay = _benchmark(_replay_config(replay_dir), None, attach_frame_image=True, process_mode="subprocess")
    await replay_references(replay, expected)


async def test_replay_verifies_image_pixels(tmp_path: Path) -> None:
    # A doubled image_scale changes only the pixels — prompt text, rng
    # stream, and call order are identical — so the mismatch proves image
    # content participates in replay verification. In-process: verification
    # is per-game backend behavior (see module docstring).
    _, expected, replay_dir = await _generate(tmp_path, attach_frame_image=True)
    replay = _benchmark(_replay_config(replay_dir), None, attach_frame_image=True, image_scale=2)
    with pytest.raises(AssertionError, match="diverged on"):
        await replay_references(replay, expected)
    notes = [run.solver_note or "" for run in replay.game_runs]
    assert any("replay mismatch" in note for note in notes)


def test_a_reference_that_cannot_replay_is_refused(tmp_path: Path) -> None:
    # The guard every collected reference passes through. Its failure mode is
    # silence: a prompt that quotes an absolute path replays nowhere, and
    # without this the discovery happens in CI as a prompt diff whose cause
    # is nowhere in the diff. Planted the way it actually happens — a path
    # inside recorded text, not as a whole line.
    import sys

    from platypus.reference import _assert_replayable

    clean = tmp_path / "clean.ref.jsonl"
    clean.write_text('{"line": "record", "request": {"elements": ["File \\"world_model.py\\", line 3"]}}\n')
    _assert_replayable(clean, tmp_path)  # a stripped path is fine

    leaky = tmp_path / "leaky.ref.jsonl"
    leaky.write_text(f'{{"line": "record", "request": {{"elements": ["cwd is {sys.prefix}/lib"]}}}}\n')
    with pytest.raises(AssertionError, match="python prefix"):
        _assert_replayable(leaky, tmp_path)

    # And a process address, which does not reproduce even on this machine.
    # A real one: a model ran help() on a function with an object default.
    addressed = tmp_path / "addressed.ref.jsonl"
    addressed.write_text(
        '{"line": "record", "request": {"elements": '
        '["test_world_model(_hc=<function host_call at 0x778162c19120>)"]}}\n'
    )
    with pytest.raises(AssertionError, match="process address"):
        _assert_replayable(addressed, tmp_path)


def test_solver_provenance_blob_redacts_api_keys() -> None:
    # Reference files get committed, so the pickled solver template must
    # not carry backend credentials — and redaction must not touch the
    # live solver it was made from.
    import pickle

    import ullm.backends.anthropic as _anthropic

    from platypus.reference import solver_provenance_blob

    secret = "sk-test-not-a-real-key-1234"
    backend = _anthropic.Anthropic(api_key=secret, model="claude-haiku-4-5")
    agent = PromptedRandomAgent(
        llm_config=LLMConfig(backend_factory=ullm.factory.PrebuiltBackend(backend=backend)),
        run_config=RunConfig(rng_seed=0),
    )
    solver = LLMAgentSolver(agent=agent)
    blob = solver_provenance_blob(solver)
    assert secret.encode() not in blob
    restored = pickle.loads(blob)
    assert restored.agent.llm_config.backend_factory.backend.api_key == "REDACTED"
    assert backend.api_key == secret  # original untouched
