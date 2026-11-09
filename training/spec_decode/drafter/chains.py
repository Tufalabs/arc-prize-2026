"""Chains: the unit the drafter trains on, cut from ids-only recordings.

A trace's steps form a **chain** while each prompt extends the previous
prompt plus its reply (ARC2 and ARC3 both keep the reasoning, so a continued
call's prompt is the last one, its reply, and what the harness appended).
One sequence -- the chain's last prompt plus its reply -- then holds every
call's reply at a known position, so one forward through the target covers
the whole chain. A compaction or a fresh render starts a new chain.

What trains is the replies: ``generated`` holds each call's ``[start, end)``
positions in the chain's sequence. Everything else (prompts, tool results,
the harness's own text, images) is context the head attends to but is never
asked to predict.

A dataset is a directory holding ``chains.jsonl``, one chain per line.

    python -m drafter.chains pick OUT_DIR RECORDING:TRACE:STEP [...]
    python -m drafter.chains build OUT_DIR RECORDING [--val-games N] [--sample N] [--seed S]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import re
import json
import time
from pathlib import Path
from typing import Iterator, Optional

#: The target's image placeholder, from the checkpoint's config. A reply
#: that sampled it would read as an image on the way back in, so such chains
#: are refused.
IMAGE_TOKEN_ID = 248056


@dataclasses.dataclass
class Chain:
    chain_id: str
    harness: str  # "arc2" or "arc3"
    recording: str  # the recording's job directory
    trace: str  # the trace's run id
    steps: list[int]  # the trace's step indices, in order
    ids: list[int]  # the last prompt plus its reply
    images: list[str]  # absolute paths, in placeholder order
    generated: list[tuple[int, int]]  # each call's reply, [start, end)
    game: str = ""  # the game or task played (clone suffix dropped), for splits

    @property
    def num_generated(self) -> int:
        return sum(end - start for start, end in self.generated)

    def ids_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.ids, separators=(",", ":")).encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "Chain":
        row = json.loads(line)
        row["generated"] = [tuple(span) for span in row["generated"]]
        return cls(**row)


def harness_of(recording: Path) -> str:
    """Which harness recorded a run directory, from its results tree."""
    parts = recording.resolve().parts
    if "arc_2_results" in parts:
        return "arc2"
    if "arc_3_results" in parts:
        return "arc3"
    raise ValueError(f"{recording}: not under arc_2_results or arc_3_results")


def placeholder_runs(ids: list[int]) -> int:
    """How many images a sequence carries: runs of the image placeholder."""
    runs, previous = 0, None
    for token in ids:
        if token == IMAGE_TOKEN_ID and previous != IMAGE_TOKEN_ID:
            runs += 1
        previous = token
    return runs


def trace_games(recording: Path) -> dict[str, str]:
    """Trace (run id) -> the game it played, from the recording's
    ``benchmark.json``; an official-110 clone (``ar25-0c556536_3``) counts as
    its game (``ar25-0c556536``)."""
    for _ in range(20):
        try:
            runs = json.loads((recording / "benchmark.json").read_text())["game_runs"]
            break
        except json.JSONDecodeError:
            time.sleep(3)  # TAAF rewrites it every ten minutes while the run plays
    else:
        raise ValueError(f"{recording}: benchmark.json stayed unreadable")
    return {run["run_id"]: re.sub(r"_\d+$", "", run["game_id"]) for run in runs if run.get("run_id")}


def recording_chains(recording: Path) -> tuple[list[Chain], dict[str, int]]:
    """Every valid chain of a recording, and why the rest were dropped."""
    games = trace_games(recording)
    chains: list[Chain] = []
    dropped: dict[str, int] = {}
    traces = sorted(p for p in (recording / "traces").iterdir() if (p / "steps.jsonl").exists())
    # A recording that seals its traces (``_SUCCESS``, as the long runs do)
    # may still be playing: a trace not sealed yet may be mid-write.
    if any((p / "_SUCCESS").exists() for p in traces):
        traces = [p for p in traces if (p / "_SUCCESS").exists()]
    for trace_dir in traces:
        if trace_dir.name not in games:
            # Sealed since TAAF last saved benchmark.json: its game is unknown.
            dropped["not in benchmark.json"] = dropped.get("not in benchmark.json", 0) + 1
            continue
        for chain in trace_chains(recording, trace_dir):
            chain.game = games.get(chain.trace, "")
            try:
                validate(chain)
            except ValueError as error:
                reason = str(error).split(": ", 1)[-1].split(" ")[0:3]
                dropped[" ".join(reason)] = dropped.get(" ".join(reason), 0) + 1
                continue
            if chain.num_generated:
                chains.append(chain)
    return chains, dropped


def trace_chains(recording: Path, trace_dir: Path) -> Iterator[Chain]:
    """The chains of one trace, in step order."""
    steps = [json.loads(line) for line in (trace_dir / "steps.jsonl").read_text().splitlines() if line]
    harness = harness_of(recording)

    def chain(members: list[dict]) -> Chain:
        last = members[-1]
        ids = list(last["input_token_ids"]) + list(last["output_token_ids"])
        generated = [
            (len(step["input_token_ids"]), len(step["input_token_ids"]) + len(step["output_token_ids"]))
            for step in members
        ]
        images = [str(trace_dir / item["path"]) for item in (last.get("media") or [])]
        first = members[0]["step_index"]
        return Chain(
            chain_id=f"{trace_dir.name}.{first}",
            harness=harness,
            recording=str(recording),
            trace=trace_dir.name,
            steps=[step["step_index"] for step in members],
            ids=ids,
            images=images,
            generated=generated,
        )

    members: list[dict] = []
    previous: Optional[list[int]] = None
    for step in steps:
        ids_in, ids_out = step["input_token_ids"], step["output_token_ids"]
        if not ids_in:
            continue
        if previous is not None and ids_in[: len(previous)] != previous:
            yield chain(members)
            members = []
        members.append(step)
        previous = list(ids_in) + list(ids_out)
    if members:
        yield chain(members)


def validate(chain: Chain) -> None:
    """Refuse a chain the extraction would serve wrongly."""
    for start, end in chain.generated:
        if not 0 < start <= end <= len(chain.ids):
            raise ValueError(f"{chain.chain_id}: reply span {start}-{end} outside {len(chain.ids)} tokens")
        if IMAGE_TOKEN_ID in chain.ids[start:end]:
            raise ValueError(f"{chain.chain_id}: a reply sampled the image placeholder")
    runs = placeholder_runs(chain.ids)
    if runs != len(chain.images):
        raise ValueError(f"{chain.chain_id}: {runs} placeholder runs against {len(chain.images)} images")
    for path in chain.images:
        if not Path(path).exists():
            raise ValueError(f"{chain.chain_id}: missing image {path}")


def write_dataset(out_dir: Path, chains: list[Chain]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for chain in chains:
        validate(chain)
    ids = [chain.chain_id for chain in chains]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chain ids")
    (out_dir / "chains.jsonl").write_text("".join(chain.to_json() + "\n" for chain in chains))


def read_dataset(path: Path) -> list[Chain]:
    return [Chain.from_json(line) for line in (Path(path) / "chains.jsonl").read_text().splitlines() if line]


def _summary(name: str, chains: list[Chain]) -> str:
    by_harness: dict[str, list[Chain]] = {}
    for chain in chains:
        by_harness.setdefault(chain.harness, []).append(chain)
    parts = [
        f"{harness} {len(group)} chains, {sum(len(c.ids) for c in group):,} tokens, "
        f"{sum(c.num_generated for c in group):,} generated, {len({c.game for c in group})} games"
        for harness, group in sorted(by_harness.items())
    ]
    return f"{name}: " + "; ".join(parts)


def build(out_dir: Path, recording: Path, val_games: int, sample: Optional[int], seed: int) -> None:
    """A recording's chains as a dataset: all of them, or a sample of
    ``sample`` chains; with ``val_games``, that many games (drawn with
    ``seed``) held out whole into ``val/`` and the rest in ``train/``."""
    import random

    chains, dropped = recording_chains(recording)
    rng = random.Random(seed)
    if dropped:
        print(f"dropped: {dropped}")
    if sample is not None and sample < len(chains):
        # Stratified by game: one chain of each game in turn, so a small
        # sample still covers every game the recording played.
        by_game: dict[str, list[Chain]] = {}
        for chain in chains:
            by_game.setdefault(chain.game, []).append(chain)
        pools = [rng.sample(group, len(group)) for _, group in sorted(by_game.items())]
        rng.shuffle(pools)
        picked: list[Chain] = []
        while len(picked) < sample:
            for pool in pools:
                if pool and len(picked) < sample:
                    picked.append(pool.pop())
        chains = sorted(picked, key=lambda c: c.chain_id)
    if not val_games:
        write_dataset(out_dir, chains)
        print(_summary(str(out_dir), chains))
        return
    games = sorted({chain.game for chain in chains})
    held = set(rng.sample(games, val_games))
    train = [c for c in chains if c.game not in held]
    val = [c for c in chains if c.game in held]
    write_dataset(out_dir / "train", train)
    write_dataset(out_dir / "val", val)
    (out_dir / "split.json").write_text(json.dumps({"recording": str(recording), "seed": seed, "val_games": sorted(held)}, indent=2) + "\n")
    print(_summary(str(out_dir / "train"), train))
    print(_summary(str(out_dir / "val"), val) + f" -- held out: {sorted(held)}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("pick", help="a dataset of named chains: RECORDING:TRACE:FIRST_STEP")
    pick.add_argument("out_dir", type=Path)
    pick.add_argument("chains", nargs="+")
    whole = sub.add_parser("build", help="a dataset of a whole recording's chains")
    whole.add_argument("out_dir", type=Path)
    whole.add_argument("recording", type=Path)
    whole.add_argument("--val-games", type=int, default=0, help="games held out whole into val/")
    whole.add_argument("--sample", type=int, default=None, help="keep this many chains, drawn at random")
    whole.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "build":
        build(args.out_dir, args.recording, args.val_games, args.sample, args.seed)
        return
    chains = []
    for spec in args.chains:
        recording, trace, step = spec.rsplit(":", 2)
        found = [
            chain
            for chain in trace_chains(Path(recording), Path(recording) / "traces" / trace)
            if chain.steps[0] == int(step)
        ]
        if len(found) != 1:
            raise SystemExit(f"{spec}: no chain starts at that step")
        chains.append(found[0])
    write_dataset(args.out_dir, chains)
    for chain in chains:
        print(
            f"{chain.chain_id} {chain.harness}: {len(chain.ids):,} tokens, {len(chain.generated)} calls, "
            f"{chain.num_generated:,} generated, {len(chain.images)} images"
        )
    print(f"wrote {args.out_dir / 'chains.jsonl'}")


if __name__ == "__main__":
    main()
