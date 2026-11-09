"""Viewer: pages render from a real run's streams, the exporter
materializes the same canonical paths with identical bytes, the request
resolver renders viewer paths fresh and falls through to static files
(the TAAF-central-diagnostics link contract)."""

from __future__ import annotations

import contextlib
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin

import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.factory
from taaf.game_examples import ExampleGame

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from platypus.diagnostics import build_span_tree, game_stream_path, read_events
from platypus.reference import game_run_dir
from platypus.viewer import export_viewer, render_viewer_path, resolve_request, serve
from reference_agents import PromptedRandomAgent


async def _run(job_dir: Path) -> taaf.benchmark.Benchmark:
    config = LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.ArithmeticStubBackend()),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
    )
    agent = PromptedRandomAgent(
        max_moves=3,
        attach_frame_image=True,  # exercises image blocks end to end
        llm_config=config,
        run_config=RunConfig(rng_seed=3, process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[ExampleGame(label="viewgame")],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=job_dir,
    )
    await benchmark.run()
    return benchmark


async def test_pages_render_and_export_matches_server(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    benchmark = await _run(job_dir)
    (run,) = benchmark.game_runs
    assert run.solver_analysis_html == f"viewer/g_{run.run_id}.html"

    # Index page: names the game, links its page.
    index, content_type = render_viewer_path(job_dir, "index.html") or (b"", "")
    assert content_type.startswith("text/html")
    assert b"viewgame" in index and f"g_{run.run_id}.html".encode() in index

    # Game page: move strip, frame-thumbnail column, and a link per move span.
    game_page, _ = render_viewer_path(job_dir, f"g_{run.run_id}.html") or (b"", "")
    assert b"<th>frame</th>" in game_page and b"frame-thumb" in game_page
    root = build_span_tree(read_events(game_stream_path(game_run_dir(job_dir, run.run_id))))[0]
    moves = [c for c in root.children if c.kind == "infra/move"]
    for move in moves:
        assert f"g_{run.run_id}_m{move.span_id}.html".encode() in game_page

    # Move page: llm/turn chunk with the rendered conversation, the
    # UNREGISTERED agent/seam_check span via the generic fallback, and an
    # image reference into the png route.
    llm_move = next(m for m in moves if m.children)
    move_page, _ = render_viewer_path(job_dir, f"g_{run.run_id}_m{llm_move.span_id}.html") or (b"", "")
    assert b"agent/seam_check" in move_page  # generic fallback renders unknown kinds
    assert b"llm/turn" in move_page and b"assistant" in move_page
    assert b"What is " in move_page  # conversation content rendered
    assert b"png/" in move_page  # frame artifact and/or prompt image by reference
    # The frame row: input + result frames side by side, labeled.
    assert b'class="move-frames"' in move_page
    assert b"<figcaption>at</figcaption>" in move_page and b"<figcaption>after</figcaption>" in move_page
    # Single-child spans auto-open (seam_check wraps exactly the llm/turn).
    assert b'class="chunk chunk-agent-seam_check" open>' in move_page
    # prev/next navigation, including the non-trivial variant.
    assert b"move-nav" in move_page and b"non-trivial" in move_page

    # Every referenced image resolves through the png route as real PNG.
    import re as _re

    keys = set(_re.findall(rb"png/([0-9a-f]{64})\.png", move_page))
    assert keys
    for key in keys:
        png, png_type = render_viewer_path(job_dir, f"png/{key.decode()}.png") or (b"", "")
        assert png_type == "image/png" and png.startswith(b"\x89PNG")

    # Static export: same canonical paths, byte-identical renderings.
    out = export_viewer(job_dir)
    assert (out / "index.html").read_bytes() == index
    assert (out / f"g_{run.run_id}.html").read_bytes() == game_page
    assert (out / f"g_{run.run_id}_m{llm_move.span_id}.html").read_bytes() == move_page
    for key in keys:
        assert (out / "png" / f"{key.decode()}.png").read_bytes().startswith(b"\x89PNG")


async def test_in_flight_and_aborted_rendering(tmp_path: Path) -> None:
    """The live-viewing contract: a stream mid-write (spans still open)
    and a stream from an untidily dead child (torn tail, missing closes)
    both render — with in-flight/aborted marking and child-aggregated
    totals — rather than erroring."""
    from platypus.diagnostics import BlockStore, SpanLogger, blocks_dir

    job_dir = tmp_path / "job"
    await _run(job_dir)  # a real finished run: index + blocks exist

    # An in-flight game: open spans, events flushed, nothing closed.
    live_dir = job_dir / "games" / "live0000"
    logger = SpanLogger(game_stream_path(live_dir), BlockStore(blocks_dir(job_dir)))
    with logger.span("infra/game", "livegame"):
        with logger.span("infra/move", "move 0") as move:
            move.event(
                "infra/action", name="ACTION1", row=None, col=None, outcome="playing", level=0, levels_completed=0
            )
            move.add_usage(11, 7)
            move.set_outcome("executed ACTION1")
        with logger.span("infra/move", "move 1"):
            # Mid-move: render everything NOW, while spans are open.
            index, _ = render_viewer_path(job_dir, "index.html") or (b"", "")
            assert b"livegame" in index and b"in flight" in index
            game_page, _ = render_viewer_path(job_dir, "g_live0000.html") or (b"", "")
            assert b"in flight" in game_page and b"11 / 7" in game_page  # closed move totals visible
            move_page, _ = render_viewer_path(job_dir, "g_live0000_m1.html") or (b"", "")
            assert b"executed UP" in move_page  # display names, not engine names
    logger.close()
    # The index caches a game's row by its stream's version: the closed
    # stream is a new version, so the row is no longer in flight.
    index, _ = render_viewer_path(job_dir, "index.html") or (b"", "")
    assert b"livegame" in index and b"in flight" not in index

    # An aborted game: strip the trailing close events and tear the last
    # line (the SIGKILL signature). Pages still render, marked aborted.
    lines = game_stream_path(live_dir).read_text().splitlines()
    while '"e":"close"' in lines[-1]:
        lines.pop()
    game_stream_path(live_dir).write_text("\n".join(lines) + '\n{"e":"clo')
    game_page, _ = render_viewer_path(job_dir, "g_live0000.html") or (b"", "")
    assert b"in flight / aborted" in game_page
    assert b"g_live0000_live.html" in game_page  # game page links its live streamer
    move_page, _ = render_viewer_path(job_dir, "g_live0000_m2.html") or (b"", "")
    assert b"in flight" in move_page
    # An in-flight span carries the "watch live" link right in its header.
    assert b"live-link" in move_page and b"g_live0000_live.html" in move_page


async def test_resolve_request_dynamic_vs_static(tmp_path: Path) -> None:
    job_dir = tmp_path / "runs" / "job1"
    await _run(job_dir)

    root = tmp_path / "runs"
    # Viewer path under a nested run dir renders dynamically.
    rendered = resolve_request(root, "/job1/viewer/index.html")
    assert rendered is not None and b"viewgame" in rendered[0]
    # Dynamic beats disk: an exported (stale) file never shadows a fresh render.
    export_viewer(job_dir)
    (job_dir / "viewer" / "index.html").write_bytes(b"STALE")
    fresh = resolve_request(root, "/job1/viewer/index.html")
    assert fresh is not None and fresh[0] != b"STALE"

    # Static fallthrough: TAAF's central page (and anything else on disk)
    # serves as a plain file, so its relative links into viewer/ resolve.
    (job_dir / "diagnostics.html").write_text("<a href='viewer/index.html'>solver</a>")
    static = resolve_request(root, "/job1/diagnostics.html")
    assert static is not None and b"viewer/index.html" in static[0]

    # Directory listings link run dirs to their viewers; traversal denied.
    listing = resolve_request(root, "/")
    assert listing is not None and b"job1/viewer/index.html" in listing[0]
    assert resolve_request(root, "/../" + tmp_path.name) is None


async def test_viewer_route_is_confined_to_the_serving_root(tmp_path: Path) -> None:
    """A run dir outside the serving root must not render, however the URL
    reaches it: ".." is a valid route segment, and the live page WRITES a
    request flag into the run dir it resolves."""
    outside = tmp_path / "elsewhere" / "job9"
    await _run(outside)
    root = tmp_path / "runs"
    root.mkdir(parents=True, exist_ok=True)

    run_id = next(p.name for p in (outside / "games").iterdir() if p.is_dir())
    assert resolve_request(root, "/../elsewhere/job9/viewer/index.html") is None
    assert resolve_request(root, f"/../elsewhere/job9/viewer/g_{run_id}.html") is None
    # The live variant is the one with a side effect; it must not have run it.
    assert resolve_request(root, f"/../elsewhere/job9/viewer/g_{run_id}_live.html") is None
    assert not (outside / "games" / run_id / "live_request").exists()


async def test_viewer_never_deserializes_a_pickle_block(tmp_path: Path) -> None:
    """The codec travels in the stream, so honoring "pickle" would make
    rendering a run directory execute what that directory holds."""
    from platypus.chunks import ViewContext
    from platypus.diagnostics import BlockStore

    store = BlockStore(tmp_path / "blocks")
    key = store.store({"payload": "harmless"}, "pickle")
    view = ViewContext(blocks=store)
    with pytest.raises(ValueError):
        view.load_block(key, "pickle")
    assert "not rendered" in view.text_of({"$block": key, "codec": "pickle"})


@contextlib.contextmanager
def _serving(root: Path) -> Generator[str]:
    """A viewer on a free port, for the handler behaviour no in-process call
    reaches: ``resolve_request`` returns a body, and a redirect is not one."""
    root.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=lambda: serve(root, port=port), daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):  # the thread binds asynchronously
        try:
            urllib.request.urlopen(base, timeout=0.1)
            break
        except (urllib.error.HTTPError, OSError):
            pass
    yield base


def test_a_directory_url_redirects_to_its_slash(tmp_path: Path) -> None:
    """Without the redirect, every relative link on a listing is fetched from
    the parent directory.

    A browser resolves ``d.html`` on a page served at ``/a/b/c`` against
    ``/a/b/`` — so the link lands on a sibling of the directory being viewed,
    or 404s. Clicking through never shows it, because a listing writes its own
    directory links with the slash on; it takes a typed or pasted URL, which is
    why it presents as occasional rather than constant.
    """
    root = tmp_path / "runs"
    (root / "a" / "b" / "c").mkdir(parents=True)
    (root / "a" / "b" / "c" / "d.html").write_text("target")
    (root / "a" / "b" / "d.html").write_text("the sibling it used to reach")

    with _serving(root) as base:
        opened = urllib.request.urlopen(f"{base}/a/b/c")

        assert opened.url == f"{base}/a/b/c/", "no redirect: relative links resolve one level up"
        landing = urllib.request.urlopen(urljoin(opened.url, "d.html")).read().decode()
        assert landing == "target"


def test_a_directory_url_needing_percent_encoding_redirects_to_its_slash(tmp_path: Path) -> None:
    """The redirect target is built from the raw URL, not the decoded path.

    A name with a space arrives percent-encoded, so the decoded path and the
    request's own URL are different strings. A target built by substituting one
    into the other matches nothing and sends the browser back to the URL it
    just asked for — a redirect loop where the old behaviour merely served the
    listing with links resolving one level up.
    """
    root = tmp_path / "runs"
    (root / "a dir").mkdir(parents=True)
    (root / "a dir" / "d.html").write_text("target")

    with _serving(root) as base:
        opened = urllib.request.urlopen(f"{base}/a%20dir")

        assert opened.url == f"{base}/a%20dir/", opened.url
        assert urllib.request.urlopen(urljoin(opened.url, "d.html")).read().decode() == "target"


def test_a_directory_url_that_already_has_its_slash_is_served_directly(tmp_path: Path) -> None:
    """The redirect must not fire on the common path, or every listing click
    costs a round trip."""
    root = tmp_path / "runs"
    (root / "a").mkdir(parents=True)

    with _serving(root) as base:
        connection = urllib.request.urlopen(f"{base}/a/")

        assert connection.status == 200
        assert connection.url == f"{base}/a/"


def test_the_redirect_does_not_escape_the_serving_root(tmp_path: Path) -> None:
    """It resolves a path to decide whether to redirect, so it needs the same
    confinement the static fallthrough has."""
    (tmp_path / "outside").mkdir()
    root = tmp_path / "runs"
    root.mkdir()

    with _serving(root) as base:
        try:
            status = urllib.request.urlopen(f"{base}/../outside").status
        except urllib.error.HTTPError as refused:
            status = refused.code

        assert status == 404, "a path outside the root must not be redirected into"


async def test_resolve_request_with_relative_root(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    """The CLI passes a relative --out, so a relative root must resolve
    exactly as an absolute one does — for the listing, a subdirectory, and a
    rendered page alike."""
    await _run(tmp_path / "runs" / "job1")
    monkeypatch.chdir(tmp_path)
    listing = resolve_request(Path("runs"), "/")
    assert listing is not None and b"job1/viewer/index.html" in listing[0]
    sub = resolve_request(Path("runs"), "/job1/games/")
    assert sub is not None
    page = resolve_request(Path("runs"), "/job1/viewer/index.html")
    assert page is not None and b"viewgame" in page[0]
