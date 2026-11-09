"""Shared scaffolding for the committed-reference tests.

Every reference test is the same moves: replay the committed file in the
cheap tier, regenerate it behind an opt-in, and refresh its viewer slice
behind another. What differs between cases is a configuration, so that is
what a reference test should consist of; the moves live here.

The one rule worth stating out loud: **the replay and the regeneration must
build the same benchmark**. A test that assembled its agent twice would
eventually assemble it two ways, and the reference would stop being a claim
about the agent it says it is. Hence :class:`ReferenceCase`, whose ``build``
is called for both.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import pytest
import ullm.backend
import ullm.factory

import taaf.benchmark
from platypus import LLMConfig
from platypus.reference import (
    REF_SUFFIX,
    ReferenceFile,
    generate_references,
    load_reference_seeds,
    load_references,
    read_reference,
    refresh_viewer_html,
    replay_references,
    require_pinned_hash_seed,
)

REFERENCE_DIR = Path(__file__).parent / "references"


def replay_job_dir(tmp_path: Path) -> Path:
    """Where :func:`replay_case` writes the replay's diagnostics — named so
    a test that wants to assert on the run's *structure* can read the span
    stream from here."""
    return tmp_path / "replay_job"


@dataclass(frozen=True)
class ReferenceRun:
    """What a case's benchmark builder is handed.

    ``seeds`` is the per-game seed each recording was generated with, read
    back off the committed file — empty during generation, since there is
    nothing to read yet. A builder that feeds it to
    ``RunConfig.rng_seed_per_game`` alongside its own ``rng_seed`` gets both
    halves right: generation seeds from the declared constant, replay from
    what was actually used. That is what keeps a reference replayable after
    the run's game list changes length, which the derived
    ``rng_seed + index`` scheme does not survive.
    """

    llm_config: LLMConfig
    job_dir: Path | None
    seeds: Mapping[str, int]


@dataclass(frozen=True)
class ReferenceCase:
    """One reference scenario: how to build its benchmark, and under which
    sampling profiles. Both are single-sourced across capture and replay."""

    name: str
    build: Callable[[ReferenceRun], taaf.benchmark.Benchmark]
    profiles: Mapping[str, ullm.backend.SamplingParameters]
    directory: Path = REFERENCE_DIR

    def paths(self) -> list[Path]:
        """The committed files for this case, one per game."""
        return sorted(self.directory.glob(f"{self.name}.*{REF_SUFFIX}"))


async def replay_case(case: ReferenceCase, tmp_path: Path) -> list[ReferenceFile]:
    """Replay the committed reference and require it to reproduce bit-exact.

    Returns the committed files, read back, so a test can go straight on to
    its own assertions.
    """
    require_pinned_hash_seed(f"replay of {case.name}")
    paths = case.paths()
    assert paths, (
        f"no committed reference for {case.name!r} — run the regenerate test "
        f"(see the module docstring); looked in {case.directory}"
    )
    expected = load_references(paths, tmp_path / "unpacked")
    benchmark = case.build(
        ReferenceRun(
            llm_config=LLMConfig(sampling_profiles=case.profiles, replay_dir=str(tmp_path / "unpacked")),
            # Always a job_dir: a reference carries a deterministic slice of
            # the viewer HTML, and the replay has to render its own to
            # compare. Harmless for a reference that carries none — the
            # comparison only looks at keys the reference has.
            job_dir=replay_job_dir(tmp_path),
            seeds=load_reference_seeds(paths),
        )
    )
    await replay_references(benchmark, expected)
    return [read_reference(path) for path in paths]


async def regenerate_case(case: ReferenceCase, *, backend: ullm.backend.Backend, tmp_path: Path) -> list[ReferenceFile]:
    """Record a fresh reference from ``backend`` and validate it by replay.

    The validation is not optional politeness: a reference that cannot be
    replayed is not a reference, and the cheapest moment to find that out
    is before it lands.
    """
    require_pinned_hash_seed(f"regeneration of {case.name}")
    before = set(case.paths())
    capture = case.build(
        ReferenceRun(
            llm_config=LLMConfig(
                backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
                sampling_profiles=case.profiles,
                capture=True,
            ),
            job_dir=tmp_path / "capture_job",
            seeds={},
        )
    )
    paths = await generate_references(capture, case.directory, case.name)
    expected = load_references(paths, tmp_path / "validate")
    validation = case.build(
        ReferenceRun(
            llm_config=LLMConfig(sampling_profiles=case.profiles, replay_dir=str(tmp_path / "validate")),
            job_dir=tmp_path / "validate_job",
            seeds=load_reference_seeds(paths),
        )
    )
    try:
        await replay_references(validation, expected)
    except BaseException:
        # A reference that failed its own validation is not one, and leaving
        # it on disk is worse than having none: the next `git add` sweeps it
        # up and it becomes the committed baseline for behaviour nothing can
        # reproduce. The generating run's job_dir is still there to read.
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    # Only now that a fresh set exists and replays: sweep the files the old
    # set left behind (a game that dropped out). Deleting before generating
    # would leave the tree with no reference at all whenever the capture run
    # fails partway — recoverable only from git, and easy to commit blind.
    for stale in before - set(paths):
        stale.unlink(missing_ok=True)
    print(f"regenerated {len(paths)} reference file(s) in {case.directory} — commit them")
    return [read_reference(path) for path in paths]


async def refresh_case(case: ReferenceCase, tmp_path: Path) -> None:
    """Rewrite the case's committed ``viewer_html`` slices from a replay
    render — the sanctioned repair for a viewer-only change.

    The everything-except-viewer_html comparison inside
    :func:`platypus.reference.refresh_viewer_html` is what makes the rewrite
    trustworthy; the validating replay afterwards proves the rewritten files
    reproduce, fresh pages included. Live regeneration
    (:func:`regenerate_case`) remains the canonical re-baseline."""
    require_pinned_hash_seed(f"viewer refresh of {case.name}")
    paths = case.paths()
    assert paths, f"no committed reference for {case.name!r} — looked in {case.directory}"
    expected = load_references(paths, tmp_path / "unpacked")
    benchmark = case.build(
        ReferenceRun(
            llm_config=LLMConfig(sampling_profiles=case.profiles, replay_dir=str(tmp_path / "unpacked")),
            job_dir=tmp_path / "refresh_job",
            seeds=load_reference_seeds(paths),
        )
    )
    await refresh_viewer_html(benchmark, expected, paths)
    validation = case.build(
        ReferenceRun(
            llm_config=LLMConfig(sampling_profiles=case.profiles, replay_dir=str(tmp_path / "validate")),
            job_dir=tmp_path / "validate_job",
            seeds=load_reference_seeds(paths),
        )
    )
    await replay_references(validation, load_references(paths, tmp_path / "validate"))


def require_viewer_refresh() -> None:
    """Gate a viewer-refresh test: it REWRITES the committed references
    (viewer slice only), so like regeneration it must never happen as a
    side effect of running a tier. Free — no provider credits."""
    if os.environ.get("PLATYPUS_REFRESH_VIEWER_HTML") != "1":
        pytest.skip(
            "viewer refresh REWRITES the committed references' viewer_html: opt in with PLATYPUS_REFRESH_VIEWER_HTML=1"
        )


@overload
def require_regeneration(*, spends_api: Literal[True]) -> str: ...


@overload
def require_regeneration(*, spends_api: bool = ...) -> str | None: ...


def require_regeneration(*, spends_api: bool = False) -> str | None:
    """Gate a regeneration test; returns the Anthropic key when it spends one.

    Two opt-ins rather than one, because ``make test-api`` has to stay
    read-only: regenerating re-baselines what "correct" means, and that can
    never happen as a side effect of running a tier.
    """
    if os.environ.get("PLATYPUS_REGENERATE_REFERENCES") != "1":
        pytest.skip("regeneration REWRITES the committed references: opt in with PLATYPUS_REGENERATE_REFERENCES=1")
    if not spends_api:
        return None
    if os.environ.get("ULLM_API_TESTS") != "1":
        pytest.skip("regeneration spends provider credits: opt in with ULLM_API_TESTS=1")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.fail("ULLM_API_TESTS=1 but ANTHROPIC_API_KEY is not set")
    return key
