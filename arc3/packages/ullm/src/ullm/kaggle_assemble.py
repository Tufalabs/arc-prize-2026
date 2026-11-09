"""Put a checkpoint together on a Kaggle kernel from the datasets that hold its parts.

A Kaggle dataset counts against its owner's private quota, and expert prunes
of one base share most of their bytes: Flash-Next's ten shards hold the
routed experts in the first six and the PLE table in the last four, which no
expert prune changes. So a prune's dataset can carry its index, its configs
and the shards it changed, and take the rest from a dataset that has them
(``ullm.kaggle.KaggleCheckpoint``); either part can be one version of a
Kaggle model instead.

    "$PYTHON" -m ullm.kaggle_assemble OUT DATASET SHARDS_FROM [SHARDS_FROM ...]

OUT becomes a directory of links: to every file of DATASET, then to every
shard DATASET's index names and DATASET lacks, from the first SHARDS_FROM that
has it. A DATASET or SHARDS_FROM of five parts is a model version, read where
the kernel mounts it (``ullm.kaggle.model_mount``). The engine serves OUT. A
shard taken from another dataset must be the
one the index expects: its safetensors header names exactly the tensors the
index places in it, and the file ends where the header's offsets say. A kernel
mounts a dataset's latest version, so this refuses one re-uploaded with
another layout rather than serving it. Bytes are not hashed: which shards two
checkpoints share is the uploader's to establish.

Run as a kernel setup command before the engine starts
(``KaggleEngine.setup_commands``); it imports nothing beyond the standard
library, since it runs before the engine is installed. A refusal exits
non-zero, which fails the kernel's setup.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections.abc import Sequence

import ullm.kaggle
import ullm.kaggle_prefetch

#: A safetensors header's own key, which names no tensor.
_METADATA = "__metadata__"


def mounted(ref: str, root: str = ullm.kaggle.KAGGLE_DATASET_MOUNT) -> pathlib.Path:
    """``ref``'s mount: a model version's where the kernel mounts it, which
    TAAF's input map does not list; a dataset's under ``root``, or wherever
    TAAF's input map put it."""
    if ullm.kaggle.is_model_version(ref):
        model = pathlib.Path(ullm.kaggle.model_mount(ref))
        directory = model if model.is_dir() else None
    else:
        directory = ullm.kaggle_prefetch.resolve_checkpoint_dir(f"{root}/{ref}")
    if directory is None or not any(directory.iterdir()):
        raise SystemExit(f"[assemble] {ref} is not mounted")
    return directory


def check_shard(path: pathlib.Path, tensors: set[str]) -> None:
    """Refuse a shard that is not the one the index expects: one whose header
    names other tensors than ``tensors``, or whose file does not end where its
    header's offsets say."""
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    named = set(header) - {_METADATA}
    if named != tensors:
        missing, extra = sorted(tensors - named), sorted(named - tensors)
        raise SystemExit(
            f"[assemble] {path} holds other tensors than the index places in it: "
            f"{len(missing)} missing (first {missing[:2]}), {len(extra)} extra (first {extra[:2]})"
        )
    end = max((header[name]["data_offsets"][1] for name in named), default=0)
    if path.stat().st_size != 8 + length + end:
        raise SystemExit(f"[assemble] {path} is {path.stat().st_size} bytes; its header says {8 + length + end}")


def _link(directory: pathlib.Path, name: str, target: pathlib.Path) -> None:
    """``directory/name`` as a link to ``target``, replacing a link elsewhere."""
    link = directory / name
    if os.path.islink(link):
        if os.readlink(link) == str(target):
            return
        link.unlink()
    elif os.path.lexists(link):
        raise SystemExit(f"[assemble] {link} exists and is not a link")
    link.symlink_to(target)


def assemble(
    out: pathlib.Path, dataset: str, shards_from: Sequence[str], root: str = ullm.kaggle.KAGGLE_DATASET_MOUNT
) -> list[tuple[str, str]]:
    """Make ``out`` the checkpoint ``dataset`` and ``shards_from`` hold
    together; return each shard taken from another dataset, with its dataset."""
    own = mounted(dataset, root)
    others = [(ref, mounted(ref, root)) for ref in shards_from]
    if not (own / ullm.kaggle_prefetch.INDEX).is_file():
        raise SystemExit(f"[assemble] {dataset} has no {ullm.kaggle_prefetch.INDEX}")
    tensors: dict[str, set[str]] = {}
    for tensor, shard in json.loads((own / ullm.kaggle_prefetch.INDEX).read_text())["weight_map"].items():
        tensors.setdefault(shard, set()).add(tensor)
    out.mkdir(parents=True, exist_ok=True)
    for source in sorted(own.iterdir()):
        if source.is_file():
            _link(out, source.name, source)
    taken: list[tuple[str, str]] = []
    for shard in sorted(tensors):
        if (own / shard).is_file():
            continue
        found = next(((ref, directory / shard) for ref, directory in others if (directory / shard).is_file()), None)
        if found is None:
            raise SystemExit(f"[assemble] no dataset holds {shard}: not {dataset}, nor {list(shards_from)}")
        ref, path = found
        check_shard(path, tensors[shard])
        _link(out, shard, path)
        taken.append((shard, ref))
    return taken


def main(argv: Sequence[str] | None = None) -> None:
    """Assemble the checkpoint the command line names, and say what came from where."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=pathlib.Path, help="the directory the engine serves")
    parser.add_argument(
        "dataset",
        help="OWNER/SLUG, or a model's OWNER/SLUG/FRAMEWORK/VARIATION/VERSION, holding the index, the configs and its own shards",
    )
    parser.add_argument(
        "shards_from",
        nargs="+",
        help="OWNER/SLUG, or a model's OWNER/SLUG/FRAMEWORK/VARIATION/VERSION, to take the shards it lacks from, in order",
    )
    parser.add_argument("--root", default=ullm.kaggle.KAGGLE_DATASET_MOUNT, help="where datasets are mounted")
    args = parser.parse_args(argv)
    taken = assemble(args.out, args.dataset, args.shards_from, args.root)
    sources = {ref: [shard for shard, by in taken if by == ref] for ref in dict.fromkeys(ref for _, ref in taken)}
    print(f"[assemble] {args.out}: {args.dataset}, with {sources or 'nothing taken'}", flush=True)


if __name__ == "__main__":
    main()
