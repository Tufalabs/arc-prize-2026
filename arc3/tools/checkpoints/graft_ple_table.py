"""Stage a checkpoint whose PLE n-gram table comes from another checkpoint.

Intel's AutoRound W4A16 Flash-Next (competitor-tricks T1) keeps its PLE table
in BF16, 95 GiB, twice the FP8 table the served e307 pins on the host; the
Kaggle box cannot hold it beside the host tier. The table is not part of what
AutoRound tuned (its ``ple`` layers are in the ignore list), so this takes
every other tensor from ``--src`` and the table from ``--ple-from`` (e307's
FP8 table and its global scale), as a public notebook serves another author's
hybrid of the same two sources.

Nothing is copied: ``--out`` gets symlinks to both sources' shards (the
graft's under a ``ple-`` prefix), an index naming each tensor's shard, every
other file of ``--src``, and ``--src``'s config with the graft's
``ple_embedding_dtype``. ``prune_flashnext_experts.py`` then reads the index
and writes the real checkpoint. A ``--src`` shard that was not downloaded is
allowed only if every tensor the index puts in it is the PLE table or the
MTP block (every cell drafts from its own drafter checkpoint); ``graft.json``
records both sources and what was dropped.

Usage: graft_ple_table.py --src DIR --ple-from DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INDEX = "model.safetensors.index.json"
TABLE = ".ple.ple_embedding.ngram_embedding."


def main() -> None:
    """Write the staged source: symlinks, a merged index, the config, provenance."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--ple-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src_map: dict[str, str] = json.loads((args.src / INDEX).read_text())["weight_map"]
    graft_map: dict[str, str] = json.loads((args.ple_from / INDEX).read_text())["weight_map"]
    missing = {shard for shard in set(src_map.values()) if not (args.src / shard).is_file()}
    dropped_mtp = sorted(k for k, s in src_map.items() if s in missing and k.startswith("mtp."))
    dropped_table = sorted(k for k in src_map if TABLE in k)
    lost = sorted(k for k, s in src_map.items() if s in missing and not k.startswith("mtp.") and TABLE not in k)
    assert not lost, f"{len(lost)} tensors in shards not on disk, neither table nor MTP: {lost[:5]}"

    weight_map = {k: s for k, s in src_map.items() if s not in missing and TABLE not in k}
    grafted = {k: f"ple-{s}" for k, s in graft_map.items() if TABLE in k}
    assert grafted, f"no PLE table in {args.ple_from}"
    assert not set(grafted) & set(weight_map)
    weight_map.update(grafted)

    args.out.mkdir(parents=True, exist_ok=False)
    for shard in sorted({s for s in src_map.values() if s not in missing}):
        (args.out / shard).symlink_to((args.src / shard).resolve())
    for shard in sorted({s for k, s in graft_map.items() if TABLE in k}):
        (args.out / f"ple-{shard}").symlink_to((args.ple_from / shard).resolve())
    for p in args.src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (INDEX, "config.json"):
            (args.out / p.name).symlink_to(p.resolve())
    (args.out / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))

    config = json.loads((args.src / "config.json").read_text())
    dtype = json.loads((args.ple_from / "config.json").read_text())["text_config"]["ple_embedding_dtype"]
    config["text_config"]["ple_embedding_dtype"] = dtype
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "graft.json").write_text(
        json.dumps(
            {
                "src": str(args.src.resolve()),
                "ple_from": str(args.ple_from.resolve()),
                "ple_embedding_dtype": dtype,
                "table_tensors_grafted": len(grafted),
                "table_tensors_dropped": len(dropped_table),
                "mtp_tensors_dropped": len(dropped_mtp),
                "shards_not_on_disk": sorted(missing),
            },
            indent=1,
        )
    )
    print(
        f"staged {len(weight_map)} tensors: {len(grafted)} table tensors from {args.ple_from.name} "
        f"(dropped {len(dropped_table)} of the source's), {len(dropped_mtp)} MTP tensors left out; "
        f"shards not on disk: {sorted(missing)}"
    )


if __name__ == "__main__":
    main()
