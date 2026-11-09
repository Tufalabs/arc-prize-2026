"""Archive a traffic run, so a row in the RTX table can be traced to it.

    uv run python tools/traffic_archive.py SOURCE --label NAME [--kind KIND]
    uv run python tools/traffic_archive.py --pull OWNER/SLUG --label NAME

``SOURCE`` is a run's output: a SLURM job dir, a Kaggle kernel's pulled
output, or whatever an interactive session saved -- a directory or a single
log. ``--pull`` fetches a Kaggle kernel's output first (``kaggle kernels
output``). The run is copied under ``ARCHIVE`` with a ``provenance.json``
saying what produced the number: the code, the workload, the configuration,
the reading and the checks that decide whether the reading counts. The row
it prints is the row the table takes.

A reading with no provenance is not archivable: a run whose log carries no
settled line is stored as a failure, with ``reading`` null and the reason.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import getpass
import hashlib
import json
import pathlib
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any, Optional

ARCHIVE = pathlib.Path("/data/shared/research/arc_3_results") / getpass.getuser() / "traffic" / "archive"
WORKLOADS = pathlib.Path("/data/shared/research/arc_3_results") / getpass.getuser() / "traffic"

# What a run leaves behind that says how it was configured and what it did.
# Anything else in the source (weights, game pickles, the answers) is bulk.
KEEP = (
    "stdout.log",
    "traffic_result.json",
    "traffic_calls.jsonl",
    "kv_offload_trace.jsonl",
    "git_status.txt",
    "taaf_setup_env.json",
    "benchmark.json",
    "solver.pkl",
)

HEADER = re.compile(r"\[traffic\] replaying (\d+) of (\d+) lanes from (\S+)")
WINDOW = re.compile(
    r"\[traffic\] (\d+) lanes, (\d+) s, ([\d.]+) s between a lane's calls; the rates are counted from (\d+) s in"
)
PROGRESS = re.compile(r"\[traffic\] (\d+)s: (\d+) in flight")
SETTLED = re.compile(r"\[traffic\] settled (\d+) s from (\d+) s in: gen ([\d.]+) tok/s, uncached prompt ([\d.]+) tok/s")
# The queue's labels come as one or both: ``waiting=39(capacity:2,deferred:37)``.
ENGINE = re.compile(
    r"\[engine\] \d\d:\d\d:\d\d running=(\d+) waiting=(\d+)(?:\(([^)]*)\))?.*?"
    r"60s: gen=(\d+)tok/s prompt=(\d+)tok/s\(new=(\d+)\) prefix_hit=([\d.]+)%"
    r"(?:.*?ext_hit=([\d.]+)%)?"
)
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
KV_CONCURRENCY = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
AGENT = re.compile(r"benchmark\.solver: (.+)")
# The block TAAF prints at start-up: every indented line under it.
GIT_STATUS = re.compile(r"git status:\n((?:  \S.*\n)+)")


@dataclasses.dataclass(frozen=True)
class EngineLine:
    """One engine line of the reading: what it says about the pool."""

    running: int
    waiting: int  # requests not admitted this second
    capacity: int  # of those, the ones the pool had no room for
    deferred: int  # of those, cold heads the budget set aside
    prefix_hit: float  # served from the pool
    ext_hit: float  # of the rest, served by the host tier rather than recomputed

    @property
    def combined_hit(self) -> float:
        """What the prompt did not have to recompute, pool and tier together."""
        return self.prefix_hit + (100.0 - self.prefix_hit) * self.ext_hit / 100.0


def _text(path: pathlib.Path) -> str:
    """A log's text, whether it is plain or a Kaggle kernel's JSONL wrapper:
    one object per line, the first opening a list and the last closing it,
    each carrying a ``data`` string with its own newline."""
    raw = path.read_text(errors="ignore")
    if not raw.lstrip().startswith(("{", ",{", "[{")):
        return raw
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("[,").removesuffix("]")
        if not line:
            continue
        try:
            out.append(str(json.loads(line).get("data", "")))
        except (json.JSONDecodeError, AttributeError):
            out.append(line + "\n")
    return "".join(out)


def _logs(source: pathlib.Path) -> list[pathlib.Path]:
    """The source's logs, largest last: a kernel's own log holds the engine's
    start-up where TAAF's stdout holds only what the run printed."""
    if source.is_file():
        return [source]
    return sorted(
        (p for p in source.iterdir() if p.suffix in (".log", ".txt") and p.name != "git_status.txt"),
        key=lambda p: p.stat().st_size,
    )


def _reading(text: str) -> dict[str, Any]:
    """The run's own numbers and the checks that say whether they count.

    The engine lines are attributed to the reading by where they sit in the
    log rather than by their clock: the replay prints its elapsed seconds as
    it goes, so a line after the ``settle_s`` mark and before the window's
    summary is a line from the reading.
    """
    out: dict[str, Any] = {"settled": None, "engine_lines": 0}
    if m := HEADER.search(text):
        out["lanes"] = int(m.group(1))
        out["workload_lanes"] = int(m.group(2))
        out["workload_path"] = m.group(3)
    if m := WINDOW.search(text):
        out["window_s"] = int(m.group(2))
        out["gap_s"] = float(m.group(3))
        out["settle_s"] = int(m.group(4))
    if m := SETTLED.search(text):
        out["settled"] = {
            "settled_s": int(m.group(1)),
            "settled_at_s": int(m.group(2)),
            "generated_tok_s": float(m.group(3)),
            "uncached_prompt_tok_s": float(m.group(4)),
        }
    if m := KV_TOKENS.search(text):
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    if m := KV_CONCURRENCY.search(text):
        out["kv_full_window_requests"] = float(m.group(2))
    if m := AGENT.search(text):
        out["agent"] = m.group(1).strip()

    settle_s = int(out.get("settle_s", 300))
    elapsed = 0
    rows: list[EngineLine] = []
    for line in text.splitlines():
        if "[traffic] window " in line:
            break  # the window's summary: what follows is the run winding down
        if p := PROGRESS.search(line):
            elapsed = int(p.group(1))
        elif (e := ENGINE.search(line)) and elapsed >= settle_s:
            labels = {
                kind: int(count)
                for kind, count in (part.split(":") for part in (e.group(3) or "").split(",") if ":" in part)
            }
            rows.append(
                EngineLine(
                    running=int(e.group(1)),
                    waiting=int(e.group(2)),
                    capacity=labels.get("capacity", 0),
                    deferred=labels.get("deferred", 0),
                    prefix_hit=float(e.group(7)),
                    ext_hit=float(e.group(8) or 0.0),
                )
            )
    out["engine_lines"] = len(rows)
    if rows:
        out["checks"] = {
            "running_mean": round(statistics.mean(r.running for r in rows), 1),
            # Waits for pool capacity alone: the waiting count also holds
            # requests the step budget deferred, which are not the reading's
            # "a request always waits for room".
            "capacity_waits": [
                min(r.capacity for r in rows),
                int(statistics.median(r.capacity for r in rows)),
                max(r.capacity for r in rows),
            ],
            "capacity_max": max(r.capacity for r in rows),
            "deferred_max": max(r.deferred for r in rows),
            "prefix_hit_pct": round(statistics.mean(r.prefix_hit for r in rows), 1),
            "combined_hit_pct": round(statistics.mean(r.combined_hit for r in rows), 1),
        }
    out["allocator_failures"] = len(re.findall("memory allocation failed with OOM", text))
    return out


def _code(source: pathlib.Path, text: str) -> dict[str, Any]:
    """What the run was built from: the bundle's ``git_status.txt`` where the
    run left one, otherwise the same block as the run printed it -- so a lone
    console log from an interactive session still names its commit."""
    status = source / "git_status.txt" if source.is_dir() else source.parent / "git_status.txt"
    if status.is_file():
        raw = status.read_text().strip()
    elif m := GIT_STATUS.search(text):
        raw = m.group(1).strip()
    else:
        return {}
    out: dict[str, Any] = {"raw": raw}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 3 and (parts[0].endswith("platypus") or parts[0] in ("arc2", "arc3")):
            out["repo"], out["commit"], out["state"] = parts[0], parts[1], parts[2]
        elif len(parts) >= 3 and "vllm" in parts[0]:
            out["vllm"] = parts[1]
    return out


def _workload(name: Optional[str]) -> dict[str, Any]:
    """The workload the run replayed, identified by content where we hold it:
    two runs compare only if this hash is the same."""
    if not name:
        return {}
    out: dict[str, Any] = {"name": pathlib.PurePath(name).name}
    local = WORKLOADS / out["name"]
    if local.is_file():
        out["path"] = str(local)
        out["sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()[:16]
        out["bytes"] = local.stat().st_size
    return out


def _row(record: dict[str, Any]) -> str:
    """The run as a row of the RTX table."""
    r: dict[str, Any] = record.get("reading") or {}
    c: dict[str, Any] = r.get("checks") or {}
    settled: dict[str, Any] = r.get("settled") or {}
    waits: list[Any] = c.get("capacity_waits") or ["-", "-", "-"]
    return (
        f"| {record['label']} | {record['kind']} | {r.get('lanes', '?')} | "
        f"**{settled.get('generated_tok_s', 0):.1f} / {settled.get('uncached_prompt_tok_s', 0):.1f}** | "
        f"{c.get('running_mean', '?')} | {waits[0]} / {waits[1]} / {waits[2]} | {c.get('deferred_max', '?')} | "
        f"{c.get('combined_hit_pct', '?')}% ({c.get('prefix_hit_pct', '?')}% pool) | "
        f"{record['reading'].get('allocator_failures', '?')} | "
        f"{record['note']} | `{pathlib.Path(record['archive']).name}` |"
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Archive one run and print its row."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=pathlib.Path)
    parser.add_argument("--pull", default=None, help="a Kaggle kernel (OWNER/SLUG) whose output to fetch first")
    parser.add_argument("--label", required=True, help="what the run is called in the table, e.g. baseline")
    parser.add_argument("--kind", default="kaggle-rtx", help="where it ran: kaggle-rtx, kaggle-rtx-interactive, b200")
    parser.add_argument("--note", default="", help="one line for the table's notes column")
    args = parser.parse_args(argv)

    source = args.source
    if args.pull:
        source = pathlib.Path(tempfile.mkdtemp(prefix="traffic_pull_"))
        subprocess.run(["kaggle", "kernels", "output", args.pull, "-p", str(source)], check=True)
    if source is None or not source.exists():
        raise SystemExit(f"no run to archive at {source}")

    logs = _logs(source)
    if not logs:
        raise SystemExit(f"{source} holds no log")
    text = "\n".join(_text(p) for p in logs)
    reading = _reading(text)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    target = ARCHIVE / f"{stamp}_{args.label}"
    target.mkdir(parents=True, exist_ok=False)
    kept: list[str] = []
    for path in [source] if source.is_file() else sorted(source.iterdir()):
        if path.is_file() and (path.name in KEEP or path in logs):
            shutil.copy2(path, target / path.name)
            kept.append(path.name)

    record = {
        "label": args.label,
        "kind": args.kind,
        "note": args.note,
        "archived_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "archived_by": getpass.getuser(),
        "source": str(source),
        "pulled_from": args.pull,
        "archive": str(target),
        "files": kept,
        "code": _code(source, text),
        "workload": _workload(reading.get("workload_path")),
        "reading": reading,
    }
    (target / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    row = _row(record)
    (target / "row.md").write_text(row + "\n")
    with (ARCHIVE / "index.jsonl").open("a") as f:
        f.write(
            json.dumps({k: record[k] for k in ("label", "kind", "archived_at", "archive", "workload", "code")}) + "\n"
        )

    print(f"archived {len(kept)} files to {target}")
    if reading["settled"] is None:
        print("NO READING: the log carries no settled line -- archived as a failure, not a row")
    else:
        print(row)


if __name__ == "__main__":
    main()
