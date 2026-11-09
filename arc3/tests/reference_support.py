"""Shared scaffolding for the committed-reference tests.

Every reference test is the same three moves: replay the committed file in
the cheap tier, regenerate it behind an opt-in, and — for the scripted
ones — read the file back to prove the corners it was built to steer still
fire. What differs between them is a configuration and a list of
assertions, so that is what a reference test should consist of; the moves
live here.

The one rule worth stating out loud: **the replay and the regeneration must
build the same benchmark**. A test that assembled its agent twice would
eventually assemble it two ways, and the reference would stop being a claim
about the agent it says it is. Hence :class:`ReferenceCase`, whose ``build``
is called for both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, overload

import pytest
import taaf.benchmark
import ullm.backend
import ullm.conv
import ullm.factory

from platypus import LLMConfig
from platypus.diagnostics import SpanNode, build_span_tree, game_stream_path, read_events
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
    """Where :func:`replay_case` writes the replay's diagnostics.

    Named rather than left implicit because a test that wants to assert on
    the run's *structure* — which spans opened, in what order — reads the
    span stream from here. The reference itself carries only a slice of the
    rendered viewer, so structural claims need the live stream, and the
    replay is a live run of exactly the recorded behaviour.
    """
    return tmp_path / "replay_job"


def replay_spans(tmp_path: Path) -> SpanNode:
    """The span tree of the single game :func:`replay_case` just replayed.

    What the reference pins is what the model was shown; what this reads is
    which branch the agent took to show it. Some branches have no prompt at
    all — a correction folded in without an LLM call, a rollback to a rung —
    and those are exactly the ones worth asserting.
    """
    (game_dir,) = sorted((replay_job_dir(tmp_path) / "games").iterdir())
    (root,) = build_span_tree(read_events(game_stream_path(game_dir)))
    return root


def descendants(node: SpanNode, kind: str) -> list[SpanNode]:
    """Every span of ``kind`` in the subtree, ``node`` included, in order."""
    found = [node] if node.kind == kind else []
    for child in node.children:
        found.extend(descendants(child, kind))
    return found


def span_events(node: SpanNode, kind: str) -> list[dict[str, Any]]:
    """Every event named ``kind`` anywhere in the subtree, in order."""
    found = [event for event in node.events if event["kind"] == kind]
    for child in node.children:
        found.extend(span_events(child, kind))
    return found


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
    job_dir: Optional[Path]
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

    Returns the committed files, read back, so a scripted test can go
    straight on to its corner assertions.
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


async def replay_once(
    case: ReferenceCase, tmp_path_factory: pytest.TempPathFactory
) -> tuple[list[ReferenceFile], Path]:
    """Replay ``case`` into a directory of its own and hand back both views.

    A scripted scenario is read two ways — as the prompt text the model was
    shown, and as the span tree the run wrote — and a test per view is the
    right shape for what each pins. The replay itself is deterministic, so
    running it once per view buys nothing and costs a second full game;
    these were the two slowest tests in the suite. Back the pair with one
    module-scoped fixture over this instead.

    Returns ``(references, tmp_path)``: the second is what
    :func:`replay_spans` reads the run's structure from.
    """
    tmp_path = tmp_path_factory.mktemp(case.name)
    return await replay_case(case, tmp_path), tmp_path


async def regenerate_case(case: ReferenceCase, *, backend: ullm.backend.Backend, tmp_path: Path) -> list[ReferenceFile]:
    """Record a fresh reference from ``backend`` and validate it by replay.

    The validation is not optional politeness (I13): a reference that cannot
    be replayed is not a reference, and the cheapest moment to find that out
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
    reproduce, fresh pages included (I13 — the same principle as
    :func:`regenerate_case`, whose live-backend flow remains the canonical
    re-baseline)."""
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
def require_regeneration(*, spends_api: bool = ...) -> Optional[str]: ...


def require_regeneration(*, spends_api: bool = False) -> Optional[str]:
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


def tool_call(tool_name: str, arguments: dict[str, Any], *notes: str) -> ullm.conv.LLMResponse:
    """A scripted assistant turn calling one tool, optionally preceded by
    text the model "said" first.

    Always a ``ListContent``, even with no notes: the response is re-sent as
    a prior assistant message on every later call, so its shape is part of
    what a committed reference pins.
    """
    elements: list[ullm.conv.Content] = [ullm.conv.StringContent(text=note) for note in notes]
    elements.append(ullm.conv.ToolCallContent(tool_name=tool_name, arguments=arguments))
    return ullm.conv.LLMResponse(final_content=ullm.conv.ListContent(elements=elements))


def parallel_tool_calls(*calls: tuple[str, dict[str, Any]]) -> ullm.conv.LLMResponse:
    """One scripted turn calling several tools at once — the shape that
    makes a harness decide what to do with the calls it will not run."""
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name=name, arguments=args) for name, args in calls]
        )
    )
