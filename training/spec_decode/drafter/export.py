"""A trained head as a drafter checkpoint vLLM loads like the original.

The original drafter directory with its ``mtp.*`` tensors replaced by the
trained ones: same names, shapes and dtype (bf16; experts packed as
``gate_up_proj``/``down_proj``), the embedding and LM head copied unchanged
(they are the target's), config, tokenizer and the rest copied as they are.
Every tensor of the original must be accounted for.

    python -m drafter.export HEAD_SAFETENSORS OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

from drafter.head import DRAFTER


def export(head_path: Path, out_dir: Path, source: Path = DRAFTER) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file

    from drafter.files import save_file

    trained = load_file(str(head_path))
    out_dir.mkdir(parents=True, exist_ok=False)
    tensors = {}
    with safe_open(str(source / "model.safetensors"), framework="pt") as f:
        names = list(f.keys())
        for name in names:
            original = f.get_tensor(name)
            if name.startswith("mtp."):
                if name not in trained:
                    raise ValueError(f"the trained head has no {name}")
                tensor = trained.pop(name)
                if tensor.shape != original.shape or tensor.dtype != original.dtype:
                    raise ValueError(f"{name}: {tuple(tensor.shape)} {tensor.dtype} against {tuple(original.shape)} {original.dtype}")
                tensors[name] = tensor
            else:
                tensors[name] = original
    if trained:
        raise ValueError(f"trained tensors the checkpoint does not hold: {sorted(trained)}")
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for path in source.iterdir():
        if path.name != "model.safetensors" and path.is_file():
            shutil.copy2(path, out_dir / path.name)
    (out_dir / "drafter_provenance.json").write_text(
        json.dumps({"source": str(source), "head": str(head_path), "replaced": sum(n.startswith("mtp.") for n in names)}, indent=2) + "\n"
    )
    print(f"[export] {out_dir}: {len(tensors)} tensors, {sum(n.startswith('mtp.') for n in names)} trained", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("head", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    export(args.head, args.out_dir)


if __name__ == "__main__":
    main()
