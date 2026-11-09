#!/usr/bin/env python3
# pyright: basic
# (kagglesdk and requests carry no types, and the SDK call is the point of this script)
"""Pull one Kaggle kernel VERSION's output into a local run dir a comparison can use.

Usage:
    python pull_kaggle_run.py <owner/slug> <version_number> <out_dir> \
        [--label L] [--spans] [--movies] [--file F ...]

WHY THIS EXISTS. Kaggle's CLI and TAAF's KaggleHandle both pull only the LATEST
version of a kernel, which is the wrong run whenever the notebook was re-pushed
after the run you care about. The version is selectable one layer down, in
kagglesdk: DownloadKernelOutput takes a version_number and answers with a
redirect to

    https://www.kaggleusercontent.com/kf/<session_id>/<signed>/<file>

where <session_id> is exactly the `scriptVersionId` in the notebook page URL.
So one call both PROVES which version came back (check it against the URL you
were handed) and yields a signed base that serves any other file of that
version BY NAME. Everything here hangs off that.

WHAT YOU GET, AND WHAT YOU CANNOT.
  - Fixed-name files: all of them, always (the default set below).
  - games/<run_id>/spans.jsonl: derived from benchmark.json's run ids (--spans).
  - movies/: whatever diagnostics.html names (--movies).
  - diag_blocks/<xx>/<sha256>: NOT obtainable for an old version. The names are
    content hashes, so they cannot be derived, and the only listing endpoint is
    prefix-locked to the newest session (see the traps in SKILL.md). They are
    needed by `make viewer` drill-downs, not by comparisons or diagnostics.html.
    If you need them, either the version IS the latest (use `kaggle kernels
    output`) or download from the browser, which authenticates with cookies.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import pathlib
import re
import sys
from collections.abc import Iterable

import requests

# benchmark.json is what a comparison scores from. The task sidecar rides
# along because the score curves are computed offline from it: every recorded
# answer is scored against the task's own answers, which only a rehearsal
# carries -- a real submission withholds them, and its curves cannot be
# scored locally at all.
REQUIRED = ("benchmark.json", "game_specs.pkl")
# Cheap, and each answers a question a comparison raises: stdout.log has the
# engine and agent config dumps (temperature, connector, concurrency -- how you
# check two runs are comparable at all), git_status.txt the source revisions of
# every repo the run was built from.
USEFUL = (
    "game_slots.json",
    "stdout.log",
    "summary.txt",
    "git_status.txt",
    "run_spans.jsonl",
    "diagnostics.html",
    "solver.pkl",
    "games.pkl",
    "submission.json",
)
DEFAULT_FILES = REQUIRED + USEFUL


#: Provenance stamp this tool writes into every directory it pulls into.
MARKER = "KAGGLE_SOURCE.txt"


def unusable_out_dir(out: pathlib.Path, stamp: str) -> str | None:
    """Why ``out`` must not be pulled into under ``stamp``, or None if it may.

    The marker is a claim about every byte in the directory, and ``download``
    keeps whatever is already there under each name — so it has to be CHECKED
    before a pull, not merely written after one. Reusing a directory across
    versions otherwise leaves the previous version's files in place under the
    new version's label, and nothing downstream can tell. A directory this
    tool never stamped is refused for the same reason: its contents cannot be
    shown to belong to any version. An empty or absent one is fresh.
    """
    marker = out / MARKER
    if marker.is_file():
        found = marker.read_text()
        if found != stamp:
            return (
                f"{out} was pulled from a different source:\n"
                f"  it says:   {found.strip()}\n"
                f"  asked for: {stamp.strip()}\n"
                "Files already there would be kept and relabelled. Pull into a fresh directory."
            )
        return None
    if out.exists() and any(out.iterdir()):
        return (
            f"{out} is not empty and carries no {MARKER}, so its files cannot be shown to belong "
            "to any version; they would be kept and labelled with this one. Pull into a fresh "
            "directory."
        )
    return None


def download(session: requests.Session, url: str, target: pathlib.Path) -> int:
    """Stream ``url`` to ``target``; the HTTP status, or -1 when it was already there.

    Chunked to disk rather than read into memory: the sidecar alone runs to a
    gigabyte and twelve of these run at once. Written to a ``.part`` beside the
    target and renamed on completion, so an interrupted pull never leaves a
    non-empty file under the final name for the already-there check to trust.
    """
    if target.is_file() and target.stat().st_size > 0:
        return -1
    with session.get(url, stream=True, timeout=900) as resp:
        if not resp.ok:
            return resp.status_code
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        os.replace(part, target)
    return 200


_WS = re.compile(r"\s*")


def relabel(bench: pathlib.Path, label: str) -> None:
    """Rewrite the document's top-level ``label`` in place.

    Spliced rather than round-tripped -- the file runs to tens of MB and a
    dump would rewrite every byte of it -- but with the JSON parser doing the
    reading. The top-level object is walked one member at a time, so the key
    is found whatever its position or the file's whitespace, and a ``label``
    nested inside a game run is never mistaken for it. The old value is
    decoded as the string it is, so an escaped quote inside it ends nothing
    early and a second relabel finds exactly the value the first one wrote.
    A document with no top-level string ``label`` is refused, not guessed at.
    """
    text = bench.read_text()
    decoder = json.JSONDecoder()

    def skip_ws(at: int) -> int:
        matched = _WS.match(text, at)
        assert matched is not None  # ``\s*`` matches the empty string, so it always matches
        return matched.end()

    pos = skip_ws(0)
    if text[pos : pos + 1] != "{":
        raise SystemExit(f"{bench}: not a JSON object, cannot relabel")
    pos = skip_ws(pos + 1)
    while text[pos : pos + 1] != "}":
        key, pos = decoder.raw_decode(text, pos)
        pos = skip_ws(pos)
        if text[pos : pos + 1] != ":":
            raise SystemExit(f"{bench}: malformed object member at offset {pos}")
        pos = skip_ws(pos + 1)
        value, end = decoder.raw_decode(text, pos)
        if key == "label":
            if not isinstance(value, str):
                raise SystemExit(f"{bench}: top-level label is {type(value).__name__}, not a string")
            bench.write_text(text[:pos] + json.dumps(label) + text[end:])
            return
        pos = skip_ws(end)
        if text[pos : pos + 1] == ",":
            pos = skip_ws(pos + 1)
    raise SystemExit(f"{bench}: no top-level label to rewrite")


def main() -> int:
    """Resolve the version to its session, pull its files by name, relabel last."""
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", help="owner/slug")
    ap.add_argument("version", type=int, help="version number (v1 = 1)")
    ap.add_argument("out_dir")
    ap.add_argument("--label", help="rewrite benchmark.json's label (Kaggle runs all share the notebook's)")
    ap.add_argument(
        "--spans", action="store_true", help="also pull games/<run_id>/spans.jsonl (per-call LLM telemetry)"
    )
    ap.add_argument("--movies", action="store_true", help="also pull the movies diagnostics.html names")
    ap.add_argument("--file", action="append", dest="files", help="pull exactly these instead of the default set")
    args = ap.parse_args()

    from kagglesdk import KaggleClient
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

    owner, _, slug = args.kernel.partition("/")
    files = tuple(args.files) if args.files else DEFAULT_FILES
    out = pathlib.Path(args.out_dir)

    req = ApiDownloadKernelOutputRequest()
    req.owner_slug, req.kernel_slug = owner, slug
    req.file_path, req.version_number = "benchmark.json", args.version
    with KaggleClient() as client:
        redirect = client.kernels.kernels_api_client.download_kernel_output(req)
    url = str(redirect.url)
    session_id = url.split("/kf/")[1].split("/")[0]
    base = url.rsplit("/", 1)[0]
    print(f"{slug} v{args.version} -> scriptVersionId {session_id}")

    stamp = f"{args.kernel} version {args.version} scriptVersionId {session_id}\n"
    refusal = unusable_out_dir(out, stamp)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    # Written BEFORE the pull, so an interrupted pull leaves a correctly
    # labelled directory that a rerun resumes into rather than refuses.
    (out / MARKER).write_text(stamp)
    session = requests.Session()

    def fetch(name: str) -> tuple[str, int]:
        return name, download(session, f"{base}/{name}", out / name)

    def pull(names: Iterable[str], tag: str) -> None:
        ok = skip = 0
        missing = []
        with cf.ThreadPoolExecutor(max_workers=12) as pool:
            for name, code in pool.map(fetch, names):
                if code == 200:
                    ok += 1
                elif code == -1:
                    skip += 1
                else:
                    missing.append(name)
        note = f", missing {missing}" if missing else ""
        print(f"  {tag}: {ok} fetched, {skip} already there{note}")

    # deploy_meta.json is absent on Kaggle (SLURM writes it); a 404 is expected.
    pull(files, "files")
    bench = out / "benchmark.json"
    if not bench.is_file():
        print("no benchmark.json -- this version has no comparable run data", file=sys.stderr)
        return 1
    meta = json.loads(bench.read_text())

    if args.spans:
        pull([f"games/{g['run_id']}/spans.jsonl" for g in meta.get("game_runs", [])], "spans")
    if args.movies and (out / "diagnostics.html").is_file():
        html = (out / "diagnostics.html").read_text(errors="replace")
        pull(sorted(set(re.findall(r"movies/[^\"'\s>]+?\.(?:mp4|html)", html))), "movies")

    # Relabel LAST: any later re-pull of benchmark.json would overwrite it.
    if args.label:
        relabel(bench, args.label)
        print(f"  label -> {args.label}")

    start_time, end_time = meta.get("start_time"), meta.get("end_time")
    if start_time and end_time:
        hours = (
            datetime.datetime.fromisoformat(end_time) - datetime.datetime.fromisoformat(start_time)
        ).total_seconds() / 3600
        runs = len(meta.get("game_runs") or [])
        print(f"  ran {hours:.2f} h ({runs} game runs) -- check this is the version you meant")
    if not (out / "game_specs.pkl").is_file():
        print(
            "  WARNING: no game_specs.pkl -- the comparison cannot score this run's curves",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
