"""Convert the upstream ARC-AGI-2 checkout into our packed datasets.

Run via `make fetch-data`, which clones the upstream repo first. The output
is committed: it is a fifth of a megabyte, and shipping it means a run needs
no network, no checkout, and no JSON parse — which is what Kaggle's offline
sandbox requires.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import taaf.datasets


def main(argv: list[str] | None = None) -> int:
    """Build one `.npz` per named set from a local ARC-AGI-2 checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="ARC-AGI-2 checkout root")
    parser.add_argument("--out", type=Path, default=taaf.datasets.DATA_DIR, help="where the .npz files go")
    args = parser.parse_args(argv)

    for name, rel in taaf.datasets.NPZ_SETS.items():
        directory = args.source / rel
        if not directory.is_dir():
            raise SystemExit(f"{directory} not found — is --source an ARC-AGI-2 checkout?")
        specs = taaf.datasets.load_repo_dir(directory)
        out = args.out / f"{name}.npz"
        taaf.datasets.pack(specs, out)
        # Round-trip before declaring success: a packing bug that only shows
        # up on read would ship silently and corrupt every later run.
        assert taaf.datasets.unpack(out) == specs, f"{name}: pack/unpack round-trip differs"
        n_answer = sum(len(s.to_answer) for s in specs)
        print(f"{name}: {len(specs)} tasks, {n_answer} to-answer pairs -> {out} ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
