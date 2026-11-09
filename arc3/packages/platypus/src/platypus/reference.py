"""Reference files: capture packaging, collection, and replay (I13, U3/U4).

A reference is **one flat JSONL file per (game, pass)** — no archive, no
directory tree — so a deliberate behavior change reviews as a line diff
(the changed request/response records and the expected-outcome delta),
and inspection is ``less``/``jq``, not an unpack step. Typed lines, in
order:

    {"line": "info", ...}          run_id / game_id / the per-game seed
                                   (the seed is replayable input, not just
                                   provenance — see load_reference_seeds)
    {"line": "block", ...}         content blocks by hash (one line each)
    {"line": "record", ...}        recorded LLM calls, in call order
                                   (each record keeps its own "kind" field)
    {"line": "expected", ...}      actions, per-action tokens, final outcome
    {"line": "solver_pkl", ...}    generating solver template (base64, one line)
    {"line": "git_status", ...}    code identity at generation (when present)

The per-game file is written BY THE SOLVER at game end
(``games/<run_id>/reference.jsonl``, assembled from the game's capture
streams) — test code only *collects* finished files into a references
directory under stable names (``<name>.<game_id>.ref.jsonl``) after the
clean-state guard. A reference is a *behavioral claim*: replay pairs it
with whatever agent code claims to reproduce the behavior; the
provenance lines are never a gate.
"""

from __future__ import annotations

import base64
import copy
import json
import pickle
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import taaf.benchmark
import taaf.game

import platypus.llm
from platypus import viewer
from platypus.diagnostics import SpanNode, build_span_tree, read_game_events

REF_SUFFIX = ".ref.jsonl"


def require_pinned_hash_seed(where: str) -> None:
    """Raise unless this interpreter started with hash randomisation off.

    Iterating a ``set`` of strings is ordered by their hashes, and CPython
    randomises those per process unless ``PYTHONHASHSEED`` is pinned. Any
    such order that reaches a prompt makes a reference unreplayable: the
    committed file holds whatever order the generating process produced,
    and a replay under a different seed sends different bytes. The agent's
    own code is only half of the exposure — model-written code runs in the
    REPL and prints whatever it likes, which no amount of sorting on our
    side can pin.

    The seed is read at interpreter startup, so a process cannot set its
    own — only its children's. That is the failure worth asserting rather
    than papering over: exporting it from inside Python pins the children
    and leaves the setter unpinned, with nothing to report the asymmetry.
    ``where`` names the process, because the whole point is to catch the
    case where some of them are pinned and others are not.
    """
    if sys.flags.hash_randomization != 0:
        raise RuntimeError(
            f"{where}: PYTHONHASHSEED is not pinned (sys.flags.hash_randomization="
            f"{sys.flags.hash_randomization}). Set it before the interpreter starts — the make "
            f"targets do; a bare `pytest` does not. Exporting it from inside Python pins only "
            f"child processes, which is how a run ends up half-deterministic with nothing to show for it."
        )


def solver_provenance_blob(solver: Any) -> bytes:
    """Pickle a solver template for reference provenance, with secrets
    scrubbed: every ``api_key`` string in the object graph becomes
    ``"REDACTED"``. The blob documents the generating configuration; it is
    never unpickled to make live calls, so the key must not ride along
    into a committed reference file. Deepcopy detaches (backends deepcopy
    as configuration clones), so the caller's own solver stays intact."""
    clone = copy.deepcopy(solver)
    _redact_secrets(clone, seen=set())
    return pickle.dumps(clone)


def _redact_secrets(obj: Any, seen: set[int]) -> None:
    """Replace every ``api_key`` found in an object graph with a marker, in
    place. Walks dicts and ``__dict__``s, tracking visited ids so a cycle
    cannot loop; ``seen`` is that record."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        mapping = cast("dict[Any, Any]", obj)
        for key, value in mapping.items():
            if key == "api_key" and isinstance(value, str):
                mapping[key] = "REDACTED"
            else:
                _redact_secrets(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:  # pyright: ignore[reportUnknownVariableType]
            _redact_secrets(value, seen)
    else:
        # Mutating __dict__ directly also reaches into frozen dataclasses.
        instance_dict = getattr(obj, "__dict__", None)
        if isinstance(instance_dict, dict):
            _redact_secrets(instance_dict, seen)


def game_run_dir(job_dir: Path, run_id: str) -> Path:
    """The designated home for one game-run's durable content (capture
    today; diagnostics/trace/snapshot in later phases). Keyed by
    ``GameRun.run_id`` — unique across passes, parts, and future branches
    (elevates TAAF's R5.41 trace-folder pattern)."""
    return job_dir / "games" / run_id


#: The digest the viewer prints for a block it cannot display — a pickle or
#: raw bytes, named so a human can fetch it from the block store.
#:
#: **Masked in the comparison, unlike an image key.** A block address is the
#: hash of its content, and a pickle's content includes the layout of whatever
#: classes it holds. An artifact pickling an object graph from a third-party
#: package moves its digest whenever that package refactors a field — an event
#: that says nothing about the pipeline this check exists to protect, and one
#: that reaches anyone who re-syncs a path dependency. A PNG grid's key
#: is left compared, because its bytes come from a numpy array we produced.
#:
#: Deliberately narrow: the artifact's name, its codec, and its position all
#: still have to match, so a *missing* or *renamed* artifact still fails.
#:
#: **Asserting the block's contents instead was considered and dropped**, so
#: that it stays dropped. Arbitrary pickle bytes are not a comparison surface
#: — the masking is the correct answer to them, not a concession — and a
#: format-aware checker (one that unpickled a scenedec graph and compared it
#: semantically) buys nothing the gate does not already have: a real
#: divergence in what the pipeline produced reaches the byte-compared *prompt*
#: within a few calls, because the next thing the agent does is show the model
#: what it built. Reach for this only if a divergence is ever found that never
#: reaches a prompt.
_OPAQUE_BLOCK_DIGEST = re.compile(r"(?<= block <code>)[0-9a-f]{6,}(?=…</code>)")


def _mask_opaque_blocks(pages: Any) -> Any:
    """``pages`` with every undisplayable block's digest replaced. Accepts the
    whole ``viewer_html`` mapping so both sides of a comparison can be put
    through it, whenever they were recorded."""
    if isinstance(pages, dict):
        return {k: _OPAQUE_BLOCK_DIGEST.sub("BLOCK", v) if isinstance(v, str) else v for k, v in pages.items()}  # pyright: ignore[reportUnknownVariableType]
    return pages


# Viewer pages captured into a reference. The index plus one move page cover
# the two things a storage-layer change (e.g. block packing) could break: the
# run summary, and a move whose conversation is rendered by resolving blocks.
# More pages would bloat the reference and churn on cosmetic viewer edits.
def _reference_html(job_dir: Path, run: taaf.game.GameRun) -> Optional[dict[str, str]]:
    """Deterministic viewer HTML for this run, keyed by a run-id-free page
    name, or None if diagnostics were not written (no span stream).

    Rendered with ``deterministic=True`` so wall-clock durations are omitted,
    and the run's own uuid — which names the on-disk dir and appears in every
    href — is replaced with a fixed token, since it differs between the run
    that generated the reference and the run that replays it. The content
    address of an undisplayable block goes the same way, for the reason given
    at :data:`_OPAQUE_BLOCK_DIGEST`. Everything left (tokens, conversation
    content, image keys) reproduces bit-exact."""
    roots = build_span_tree(read_game_events(job_dir, run.run_id))
    if not roots:
        return None

    # First and last NON-TRIVIAL move — one that ran an LLM turn, not a
    # trivial drain / auto-reset. The two ends bracket the run: the first
    # move renders a fresh conversation, the last a fully accumulated one, so
    # between them the block-resolving paths get their widest exercise.
    def _ran_llm_turn(node: SpanNode) -> bool:
        """Whether a move opened an llm/turn span anywhere beneath it — any
        child will not do: agents open spans of their own on every move (the
        whale observes), so that test selects observation-only pages whose
        conversation rendering is the very thing these goldens exist to pin."""
        return any(child.kind == "llm/turn" or _ran_llm_turn(child) for child in node.children)

    llm_moves = [c for c in roots[0].children if c.kind == "infra/move" and _ran_llm_turn(c)]
    pages = {"index": "index.html", "game": f"g_{run.run_id}.html"}
    if llm_moves:
        pages["move_first"] = f"g_{run.run_id}_m{llm_moves[0].span_id}.html"
        pages["move_last"] = f"g_{run.run_id}_m{llm_moves[-1].span_id}.html"
    out: dict[str, str] = {}
    for name, path in pages.items():
        rendered = viewer.render_viewer_path(job_dir, path, deterministic=True)
        if rendered is None:
            return None
        html = rendered[0].decode("utf-8", "replace")
        # Neutralize the per-run uuid: the full id in hrefs and its 12-char
        # prefix the viewer shows as a label. Longest first so the prefix
        # replace only catches the standalone label.
        html = html.replace(run.run_id, "RUN_ID").replace(run.run_id[:12], "RUN_ID")
        out[name] = _OPAQUE_BLOCK_DIGEST.sub("BLOCK", html)
    return out


def expected_from_run(run: taaf.game.GameRun, job_dir: Optional[Path] = None) -> dict[str, Any]:
    """The bit-exact comparison surface: actions, per-action tokens, final
    outcome. Wallclock fields are deliberately excluded.

    With ``job_dir``, a deterministic slice of the viewer HTML is included
    under ``viewer_html`` — an end-to-end check that the diagnostics -> viewer
    pipeline is output-invariant across a change. Omitted when no job_dir is
    given or diagnostics were not written, so references without it stay
    valid and this key is compared only when both sides carry it."""
    out: dict[str, Any] = {
        "game_id": run.game_id,
        "state": run.state,
        "levels_completed": run.levels_completed,
        "final_score": run.final_score,
        "actions": [{"id": rec.action.id.name, "data": rec.action.data or None} for rec in run.history],
        "per_action_tokens": [[rec.generated_tokens, rec.uncached_input_tokens] for rec in run.history],
        "final_tokens": [run.final_generated_tokens, run.final_uncached_input_tokens],
    }
    if job_dir is not None:
        html = _reference_html(job_dir, run)
        if html is not None:
            out["viewer_html"] = html
    return out


def write_game_reference(
    job_dir: Path, run: taaf.game.GameRun, game_seed: int | None, solver_blob: bytes, git_status_text: str | None
) -> Path:
    """Assemble one game's reference file from its capture streams — called
    by the solver when the game ends (any terminal state; whether the file
    is fit to be a reference is the COLLECTOR's judgment).

    ``game_seed`` is the seed this game's RNG was actually built from (
    ``AgentSolver._game_seed``), not the run-level ``rng_seed`` — see
    ``load_reference_seeds``."""
    game_dir = game_run_dir(job_dir, run.run_id)
    capture_dir = game_dir / "capture"
    assert (capture_dir / "records.jsonl").exists(), f"no capture recorded for {run.game_id!r}"
    lines: list[str] = [
        json.dumps(
            {"line": "info", "run_id": run.run_id, "game_id": run.game_id, "game_seed": game_seed}, sort_keys=True
        )
    ]
    blocks_path = capture_dir / "blocks.jsonl"
    if blocks_path.exists():
        seen: set[str] = set()
        for line in blocks_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["hash"] not in seen:  # duplicate block lines are legal in the stream
                seen.add(entry["hash"])
                lines.append(json.dumps({"line": "block", **entry}, sort_keys=True))
    for line in (capture_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
        lines.append(json.dumps({"line": "record", **json.loads(line)}, sort_keys=True))
    # No viewer_html here: this runs at game-end teardown, where the game
    # subprocess's span-close may not be flushed yet, so the render would read
    # "in flight". The golden is injected post-run by collect_references, from
    # the finished job_dir. (job_dir is still used above for capture paths.)
    lines.append(json.dumps({"line": "expected", **expected_from_run(run)}, sort_keys=True))
    lines.append(json.dumps({"line": "solver_pkl", "b64": base64.b64encode(solver_blob).decode()}))
    if git_status_text is not None:
        lines.append(json.dumps({"line": "git_status", "text": git_status_text}))
    path = game_dir / "reference.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def collect_references(benchmark: taaf.benchmark.Benchmark, out_dir: Path, name: str) -> list[Path]:
    """Copy the solver-written per-game reference files into ``out_dir``
    under stable names. Refuses unclean generation runs: a crashed or
    cancelled game would still replay bit-exact (its failure is
    deterministic), so it must be rejected here, not by validation (I13).

    All or nothing: a rejected collection removes what it had already
    written. Half a reference set on disk is worse than none — the next
    ``git add`` sweeps it up, and the file that got through is the one whose
    problem nobody looked at."""
    assert benchmark.job_dir is not None
    bad = {run.game_id: run.state for run in benchmark.game_runs if run.state not in ("won", "gave_up")}
    assert not bad, f"reference generation run did not finish cleanly: {bad} — fix the agent and regenerate"
    out_dir.mkdir(parents=True, exist_ok=True)
    collected: list[Path] = []
    try:
        for run in benchmark.game_runs:
            source = game_run_dir(benchmark.job_dir, run.run_id) / "reference.jsonl"
            assert source.exists(), f"solver wrote no reference file for {run.game_id!r}"
            target = out_dir / f"{name}.{run.game_id}{REF_SUFFIX}"
            assert target not in collected, (
                f"duplicate game_id {run.game_id!r} — multi-pass references not supported yet"
            )
            shutil.copy(source, target)
            collected.append(target)  # before the checks below, so a refusal cleans it up
            elide_image_payloads(target)
            # Only a single-game run can pin viewer_html. The index page
            # lists every run in the job, and the normalisation neutralises
            # this run's own uuid alone -- a sibling's is not this run's to
            # rewrite -- so on a multi-game reference the page carries an id
            # no replay reproduces, and the validation below rejects the
            # golden it was just handed. Recorded as absent instead, which
            # is the state refresh_viewer_html already knows to leave alone.
            if len(benchmark.game_runs) == 1:
                _inject_viewer_html(target, benchmark.job_dir, run)
            _assert_replayable(target, benchmark.job_dir)
    except BaseException:
        for path in collected:
            path.unlink(missing_ok=True)
        raise
    return collected


#: The tail of a default CPython object repr — ``<function f at 0x7f…>``.
#: The address is per-process, so anything quoting one cannot be reproduced
#: even by the same code on the same machine. Matched in this shape rather
#: than as a bare ``0x…`` so a colour constant in model-written code does
#: not trip it.
_OBJECT_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+>")


def _assert_replayable(reference_path: Path, job_dir: Path) -> None:
    """Refuse a reference carrying something no replay can reproduce.

    Two kinds, and both really happen. A **directory of this machine** — its
    temp root, checkout, venv or job dir — reproduces on no other machine;
    the usual source is a traceback rendered into a tool result, which an
    agent should be stripping to basenames, and the other is the model
    printing something like ``os.getcwd()`` in its own REPL cell. A
    **process address** in an object repr does not reproduce even here: a
    model that runs ``help()`` on a function with an object default gets one
    for free.

    Caught at collection, which is the one moment the answer is cheap: the
    run that produced it is fresh, so it can be fixed and re-recorded. The
    replay that validates the reference would catch it too, but as a
    thousand-line prompt diff with the cause somewhere inside it.

    The path half is coarse on purpose: a model that merely *writes*
    ``/tmp/scratch.pkl`` in its own code trips it even though that path is
    the same everywhere. Telling the two apart would mean deciding which
    absolute paths happen to be reproducible, and being wrong that way costs
    a mystifying CI failure; being wrong this way costs one re-record, and
    tells you the model is writing absolute paths.
    """
    text = reference_path.read_text(encoding="utf-8")
    roots = {
        "temp root": tempfile.gettempdir(),
        "python prefix": sys.prefix,
        "working directory": str(Path.cwd()),
        "job directory": str(job_dir.resolve()),
    }
    for what, root in roots.items():
        if root in ("/", "") or root not in text:
            continue
        _refuse(reference_path, f"names this machine's {what} ({root!r})", text, text.index(root))
    address = _OBJECT_ADDRESS.search(text)
    if address is not None:
        _refuse(reference_path, "quotes a process address", text, address.start())


def _refuse(reference_path: Path, what: str, text: str, offset: int) -> None:
    """Raise with the offending line quoted and located, so a malformed
    reference names where it went wrong rather than which parser gave up."""
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    line = text[line_start:] if line_end < 0 else text[line_start:line_end]
    within = offset - line_start
    raise AssertionError(
        f"{reference_path.name} {what}, so it will not replay:\n    …{line[max(0, within - 120) : within + 200]}…"
    )


def elide_image_payloads(reference_path: Path) -> int:
    """Drop the raw pixels from a reference's image blocks, keeping the hash
    line. Returns how many blocks were elided.

    An image block stores its pixels as base64 of the raw array, which is
    what a reference is mostly made of once an agent shows the model
    frames: a 64x64 grid upscaled 8x is ~1.2 MB per distinct image, and an
    agent that narrates a level frame by frame produces dozens.

    None of it participates in replay. The request carries the image as
    ``{"c": "image", "block": <sha256 of the pixels>}`` and that hash is
    what the comparison sees, so pixel content stays pinned bit-exactly;
    the payload is only ever read back to rebuild a *response*, and no
    response carries an image. The shape is kept because it costs nothing
    and says what was there.

    Also usable on already-committed references, which is how the ones
    written before this existed get slimmed.
    """
    lines = reference_path.read_text(encoding="utf-8").splitlines()
    elided = 0
    for i, line in enumerate(lines):
        entry = json.loads(line)
        if entry.get("line") != "block":
            continue
        block = entry.get("block")
        if not isinstance(block, dict) or "image_b64" not in block:
            continue
        entry["block"] = {"shape": cast("dict[str, Any]", block).get("shape"), "image_elided": True}
        lines[i] = json.dumps(entry, sort_keys=True)
        elided += 1
    if elided:
        reference_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return elided


def _inject_viewer_html(reference_path: Path, job_dir: Path, run: taaf.game.GameRun) -> None:
    """Rewrite the reference's ``expected`` line to carry viewer_html rendered
    from the FINISHED original run.

    The golden MUST come from the original run, not a replay: the reference
    exists so replay can be checked against the original, and a golden taken
    from replay would make replay-vs-replay always agree — blind to the very
    divergence it must catch. By now the original run is complete and its
    spans flushed, so the render is settled ("ok", not "in flight")."""
    html = _reference_html(job_dir, run)
    if html is None:
        return
    _rewrite_expected_viewer_html(reference_path, html)


def _rewrite_expected_viewer_html(reference_path: Path, html: dict[str, str]) -> None:
    """Rewrite the ``expected`` line's ``viewer_html`` in place, leaving
    every other line of the reference byte-identical."""
    lines = reference_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        entry = json.loads(line)
        if entry.get("line") == "expected":
            entry["viewer_html"] = html
            lines[i] = json.dumps(entry, sort_keys=True)
            break
    reference_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def generate_references(
    benchmark: taaf.benchmark.Benchmark, out_dir: Path, name: str, *, minimal_diagnostics: bool = False
) -> list[Path]:
    """Run a capture-enabled benchmark, then collect the solver-written
    reference files.

    ``minimal_diagnostics`` skips the run's HTML/plot generation, which
    costs over a second per run and has nothing to do with the reference.
    Leave it False when generating a reference you intend to commit — the
    generating run should look like a production run.
    """
    await benchmark.run(minimal_diagnostics=minimal_diagnostics)
    return collect_references(benchmark, out_dir, name)


def load_references(paths: list[Path], dest: Path) -> dict[str, dict[str, Any]]:
    """Split reference files back into the runtime capture layout
    (``dest/<game_id>/{records,blocks}.jsonl`` — what ``llm_config.replay_dir``
    expects) and return the expected outcomes by game_id."""
    expected: dict[str, dict[str, Any]] = {}
    for path in paths:
        info: dict[str, Any] | None = None
        blocks: list[str] = []
        records: list[str] = []
        game_expected: dict[str, Any] | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            line_type = entry.pop("line")
            if line_type == "info":
                info = entry
            elif line_type == "block":
                blocks.append(json.dumps(entry, sort_keys=True))
            elif line_type == "record":
                records.append(json.dumps(entry, sort_keys=True))
            elif line_type == "expected":
                game_expected = entry
        assert info is not None and game_expected is not None, f"malformed reference file {path}"
        game_id = info["game_id"]
        assert game_id not in expected, f"duplicate reference for game {game_id!r}"
        expected[game_id] = game_expected
        game_dir = dest / game_id
        game_dir.mkdir(parents=True, exist_ok=True)
        (game_dir / "records.jsonl").write_text("\n".join(records) + "\n" if records else "", encoding="utf-8")
        if blocks:
            (game_dir / "blocks.jsonl").write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return expected


def load_reference_seeds(paths: list[Path]) -> dict[str, int]:
    """``game_id -> the per-game seed the reference was generated with``,
    for feeding ``RunConfig.rng_seed_per_game`` on replay.

    This is the one place a reference's provenance becomes load-bearing.
    The rest of a reference is a behavioral claim checked against whatever
    agent code claims to reproduce it, and re-declaring the generating
    configuration in the test is enough — but only while the generating
    code IS the code under test. A reference produced elsewhere (another
    repo, or simply a run whose game list has since changed length) has
    generation-time inputs that can be carried and not reconstructed, and
    the per-game seed is one of them.

    References written before the seed was recorded are skipped rather
    than failing: their tests declare ``rng_seed`` themselves, which is
    exactly what they did before."""
    seeds: dict[str, int] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("line") != "info":
                continue
            seed = entry.get("game_seed")
            if seed is not None:
                seeds[entry["game_id"]] = int(seed)
            break
    return seeds


@dataclass(frozen=True)
class ReferenceFile:
    """A committed reference read back for inspection.

    Replay answers "does the agent still do this?"; this answers the other
    question a *scripted* reference needs — "is this still the run I built
    it to be?". A scenario steered through a corner (a give-up, a rollback,
    a repair) reaches that corner only as long as the script and the agent
    keep agreeing, and a regeneration can quietly lose one while replaying
    perfectly. So the corner assertions read the COMMITTED file, not a live
    run, and this is what they read it with.
    """

    path: Path
    lines: list[dict[str, Any]]

    @classmethod
    def read(cls, path: Path) -> "ReferenceFile":
        """Load a reference file: one JSON object per line, order preserved."""
        return cls(path=path, lines=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])

    def _one(self, line_type: str) -> dict[str, Any]:
        """The single line of ``line_type``; a reference carrying none, or more
        than one, is malformed."""
        for entry in self.lines:
            if entry.get("line") == line_type:
                return entry
        raise AssertionError(f"{self.path.name} has no {line_type!r} line")

    @property
    def info(self) -> dict[str, Any]:
        """The run identity line: run_id, game_id, and the per-game seed that
        makes the run replayable."""
        return self._one("info")

    @property
    def expected(self) -> dict[str, Any]:
        """The recorded outcome: actions, per-action tokens, final state."""
        return self._one("expected")

    @property
    def blocks(self) -> dict[str, Any]:
        """Every content block in the file, by hash."""
        return {entry["hash"]: entry.get("block") for entry in self.lines if entry.get("line") == "block"}

    def action_ids(self) -> list[str]:
        """The executed actions, in order."""
        return [action["id"] for action in self.expected["actions"]]

    def records(self, kind: Optional[str] = "response") -> list[dict[str, Any]]:
        """Recorded calls in call order. ``kind=None`` for every call —
        counting calls (``token_count``) and failures (``malformed``,
        ``prompt_too_long``) as well as answered ones."""
        return [
            entry
            for entry in self.lines
            if entry.get("line") == "record" and (kind is None or entry.get("kind") == kind)
        ]

    def request_text(self, record: dict[str, Any]) -> str:
        """All resolvable text in one record's request — inline strings plus
        the contents of any ``{"$block": hash}`` reference — flattened into
        one string. Images have no text to resolve and drop out.

        Flattened rather than structured because the callers are corner
        assertions: what they want to know is whether a phrase the agent
        only emits on a particular branch is in there at all.
        """
        blocks = self.blocks
        out: list[str] = []

        def walk(node: Any) -> None:
            """Collect block references from anywhere in a nested structure."""
            if isinstance(node, dict):
                ref = cast("dict[str, Any]", node).get("$block")
                if isinstance(ref, str) and isinstance(blocks.get(ref), str):
                    out.append(cast(str, blocks[ref]))
                    return
                for value in cast("dict[str, Any]", node).values():
                    walk(value)
            elif isinstance(node, list):
                for value in cast("list[Any]", node):
                    walk(value)
            elif isinstance(node, str):
                out.append(node)

        walk(record["request"])
        return "\n".join(out)

    def all_request_text(self) -> str:
        """Every request's text, in call order — the haystack a corner
        assertion searches for the phrase its branch produces."""
        return "\n".join(self.request_text(record) for record in self.records(kind=None))


def read_reference(path: Path) -> ReferenceFile:
    """Read one reference file (the module-level spelling of
    :meth:`ReferenceFile.read`)."""
    return ReferenceFile.read(path)


def _assert_stream_consumed(run: taaf.game.GameRun, job_dir: Optional[Path]) -> None:
    """Require the replay to have read the whole recorded stream.

    Every other comparison is field by field, so a trailing record that no
    field reflects — a ``token_count`` call, in practice — can be dropped by
    an agent change and still replay green. The count comes from the process
    that hosted the game (see ``platypus.llm.record_replay_consumption``),
    through the diagnostics stream it wrote; a run without diagnostics simply
    has nothing to check.
    """
    if job_dir is None:
        return
    sidecar = game_run_dir(job_dir, run.run_id) / platypus.llm.REPLAY_CONSUMPTION_FILE
    if not sidecar.exists():
        return
    unconsumed = json.loads(sidecar.read_text(encoding="utf-8")).get("unconsumed", 0)
    if unconsumed:
        raise AssertionError(
            f"replay of {run.game_id!r} left {unconsumed} recorded call(s) unread: the agent made "
            f"fewer backend calls than the recording did. Every compared field can still match — "
            f"a dropped token_count is invisible to all of them — so this is the check that sees it."
        )


def _divergence(wanted: Any, got: Any, *, window: int = 400) -> str:
    """The first difference between two compared values, small enough to read.

    Dumping both sides whole puts hundreds of kilobytes on one assertion
    line, which a CI log truncates — leaving the one place the divergent
    bytes existed saying only that they diverged. For dict-of-str values
    (``viewer_html``) this names the first differing page and quotes both
    sides around the first differing character; everything else falls back
    to a bounded repr.
    """
    if isinstance(wanted, dict) and isinstance(got, dict):
        want_map = cast(dict[str, Any], wanted)
        got_map = cast(dict[str, Any], got)
        if set(want_map) != set(got_map):
            return f"  keys differ: expected {sorted(want_map)}, got {sorted(got_map)}"
        for page in sorted(want_map):
            a, b = want_map[page], got_map[page]
            if a == b:
                continue
            if isinstance(a, str) and isinstance(b, str):
                at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                low = max(0, at - window // 4)
                return (
                    f"  page {page!r} first differs at char {at} (lengths {len(a)} vs {len(b)}):\n"
                    f"  expected: ...{a[low : at + window]!r}...\n"
                    f"  got:      ...{b[low : at + window]!r}..."
                )
            return f"  page {page!r}: expected {a!r:.{window}}, got {b!r:.{window}}"
        return "  (dicts compare unequal but no page differs — ordering only)"
    return f"  expected: {wanted!r:.{window}}\n  got:      {got!r:.{window}}"


def assert_matches_expected(benchmark: taaf.benchmark.Benchmark, expected: dict[str, dict[str, Any]]) -> None:
    """Compare a finished (replay) benchmark against per-game expected
    outcomes, field by field, with the first divergence named."""
    job_dir = benchmark.job_dir
    for run in benchmark.game_runs:
        want = expected.get(run.game_id)
        assert want is not None, f"no reference for game {run.game_id!r}"
        _assert_stream_consumed(run, job_dir)
        got = expected_from_run(run, job_dir)
        for key in want:
            if key == "viewer_html" and "viewer_html" not in got:
                raise AssertionError(
                    f"reference for {run.game_id!r} carries viewer_html but the replay produced none — "
                    f"the replay benchmark needs a job_dir so diagnostics are written and can be rendered."
                )
            # Applied to the recording as well as to the fresh render, so a
            # reference carrying a live digest still compares — re-recording
            # one to normalise a field neither side asserts buys nothing.
            wanted = _mask_opaque_blocks(want[key]) if key == "viewer_html" else want[key]
            assert got[key] == wanted, (
                f"replay of {run.game_id!r} diverged on {key!r}:\n{_divergence(wanted, got[key])}"
                + (f"\n  solver note: {run.solver_note}" if run.solver_note else "")
            )


async def replay_references(benchmark: taaf.benchmark.Benchmark, expected: dict[str, dict[str, Any]]) -> None:
    """Run a replay-configured benchmark and assert it reproduces the
    references bit-exact (I13's validation and the CI check are the same
    call)."""
    await benchmark.run()
    assert_matches_expected(benchmark, expected)


async def refresh_viewer_html(
    benchmark: taaf.benchmark.Benchmark, expected: dict[str, dict[str, Any]], paths: list[Path]
) -> None:
    """Re-baseline ONLY the ``viewer_html`` slice of committed references,
    from a replay's own render.

    For a change that touches nothing but viewer rendering — a new chunk
    class, a page-layout edit — the pinned pages move while every other
    compared surface stays put, and re-recording a live run to re-baseline
    a rendering that is ours by construction would spend provider credits
    and produce a fresh conversation to re-review. This runs the replay
    (recorded traffic, deterministic, free), requires **every surface
    except viewer_html** to reproduce bit-exact — which is what makes the
    rewrite trustworthy — and rewrites the committed ``expected`` lines
    with the replay's rendering.

    Deliberately NOT the true regeneration flow. A golden written by replay
    is blind, on the surface it writes, to a replay-backend divergence from
    the real backend — the very thing ``_inject_viewer_html`` insists the
    generation-time golden must be able to catch. Live regeneration
    (``regenerate_case``) therefore remains the canonical re-baseline; this
    is the sanctioned shortcut for viewer-only changes.
    """
    assert benchmark.job_dir is not None, "viewer refresh needs a job_dir to render from"
    await benchmark.run()
    stripped = {game: {k: v for k, v in want.items() if k != "viewer_html"} for game, want in expected.items()}
    assert_matches_expected(benchmark, stripped)
    runs = {run.game_id: run for run in benchmark.game_runs}
    for path in paths:
        reference = read_reference(path)
        if "viewer_html" not in reference.expected:
            # Only ever REWRITE the slice, never add it: a reference without
            # one carries a generation-time decision (e.g. a multi-game run,
            # whose shared index page names sibling run ids no replay can
            # reproduce), and overriding that is generation's call to make.
            continue
        run = runs[reference.info["game_id"]]
        html = _reference_html(benchmark.job_dir, run)
        assert html is not None, f"replay of {run.game_id!r} wrote no diagnostics to render viewer_html from"
        _rewrite_expected_viewer_html(path, html)
