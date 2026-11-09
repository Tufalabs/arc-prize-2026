"""Write a drafter checkpoint whose head drafts from a reduced vocabulary.

FR-Spec's frequency-ranked draft vocabulary, without retraining: the head's
LM head keeps the rows of the ``--size`` token ids generated most often in the
drafter fork's training chains (both harnesses), and the checkpoint takes
EAGLE-3's two conventions for it -- ``text_config.draft_vocab_size`` and a
``d2t`` tensor of draft-to-target id offsets (``target = draft + d2t[draft]``).
The fork's ``Qwen4ExpMTP`` then keeps that head as its own instead of the
target's and scatters its logits into the target vocabulary, where the draft
is sampled and verified; a token outside the ids is never drafted.

Everything else is copied as it is: the MTP block, the embedding (shared with
the target at load), the config's other fields and the tokenizer files. A
provenance file records the sources and the ids' coverage of held-out chains.

Usage: reduce_drafter_vocab.py --src DRAFTER --size N --chains JSONL... --heldout JSONL... [--out DIR]
       (default out: <DRAFTER>-dv<N // 1024>k)
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file as _raw_save_file  # pyright: ignore[reportUnknownVariableType]

_save_file = cast("Any", _raw_save_file)

LM_HEAD_KEY = "lm_head.weight"


def generated_counts(paths: list[Path]) -> collections.Counter[int]:
    """Generated token ids of the chains, counted (a chain's ``generated`` spans
    index its ``ids``)."""
    counts: collections.Counter[int] = collections.Counter()
    for path in paths:
        with path.open() as f:
            for line in f:
                chain = json.loads(line)
                ids = chain["ids"]
                for start, end in chain["generated"]:
                    counts.update(ids[start:end])
    return counts


def main() -> None:
    """Rank the generated ids, keep the head's rows for the top ``--size``, write the checkpoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="a drafter directory (extract_mtp_drafter.py's)")
    ap.add_argument("--size", type=int, default=16384)
    ap.add_argument("--chains", type=Path, nargs="+", required=True, help="training chains.jsonl files")
    ap.add_argument("--heldout", type=Path, nargs="*", default=[], help="chains.jsonl files to report coverage on")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.src.with_name(f"{args.src.name}-dv{args.size // 1024}k")

    counts = generated_counts(args.chains)
    ranked = [token for token, _ in counts.most_common()]
    if len(ranked) < args.size:
        raise SystemExit(f"only {len(ranked)} distinct generated ids, fewer than --size {args.size}")
    ids = torch.tensor(sorted(ranked[: args.size]), dtype=torch.long)
    chosen = set(ranked[: args.size])
    coverage: dict[str, float] = {}
    for path in [*args.chains, *args.heldout]:
        held = generated_counts([path])
        coverage[str(path)] = sum(c for t, c in held.items() if t in chosen) / sum(held.values())

    config = json.loads((args.src / "config.json").read_text())
    text_config = config.get("text_config", config)
    vocab_size = int(text_config["vocab_size"])
    if int(ids[-1]) >= vocab_size:
        raise SystemExit(f"id {int(ids[-1])} is outside the vocabulary of {vocab_size}")
    text_config["draft_vocab_size"] = args.size

    out.mkdir(parents=True, exist_ok=False)
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(args.src.glob("*.safetensors")):
        # safetensors ships no type information; the handle is Any by choice.
        reader = cast("Any", safe_open(shard, "pt"))
        with reader as f:
            for key in f.keys():
                tensors[key] = cast("torch.Tensor", f.get_tensor(key))
    head = tensors[LM_HEAD_KEY]
    if head.shape[0] != vocab_size:
        raise SystemExit(f"{LM_HEAD_KEY} has {head.shape[0]} rows, the config's vocabulary {vocab_size}")
    tensors[LM_HEAD_KEY] = head[ids].contiguous()
    tensors["d2t"] = ids - torch.arange(args.size, dtype=torch.long)
    _save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    # save_file creates its file 0600; the head is read by every user who serves it.
    (out / "model.safetensors").chmod(0o664)

    for path in args.src.iterdir():
        if path.suffix == ".safetensors" or path.name in ("config.json", "model.safetensors.index.json"):
            continue
        if path.is_file():
            shutil.copy2(path, out / path.name)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "draft_vocab_provenance.json").write_text(
        json.dumps(
            {
                "source": str(args.src),
                "draft_vocab_size": args.size,
                "ranked_by": "generated-token counts",
                "chains": [str(p) for p in args.chains],
                "distinct_generated_ids": len(ranked),
                "coverage_of_generated_tokens": coverage,
            },
            indent=2,
        )
        + "\n"
    )
    for path, share in coverage.items():
        print(f"{share:.4%} of generated tokens covered: {path}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
