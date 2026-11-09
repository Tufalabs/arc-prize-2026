"""Export serving-ready runtime LoRA adapters from SFT checkpoints (F1/F2).

    uv run python -m sft.export_runtime --run-dir runs/<name> --checkpoint 1832
    uv run python -m sft.export_runtime --run-dir runs/<name> --all
    uv run python -m sft.export_runtime --run-dir runs/<name> --all --allow-zero-delta

For each raw ``checkpoint-N`` dir (under ``<run-dir>/outputs`` or the run dir
itself) this writes a clean ``checkpoint-N-runtime`` sibling holding exactly
``adapter_config.json`` + ``adapter_model.safetensors`` +
``runtime_adapter_summary.json`` -- the two-file layout vLLM serves without
sniffing the raw trainer's optimizer state or transformers-5 tokenizer files.

The export validates vLLM load-invariants (bias/modules_to_save/DoRA/rank/target
modules) and, by default, refuses all-zero-B adapters (pass ``--allow-zero-delta``
to export a no-op). It is idempotent: a checkpoint is skipped only when its
``-runtime`` sibling has all three artifacts, the summary's ``adapter_sha256``
matches the runtime weights, and the summary's recorded source fingerprint
(source weights + config hashes, export format version) matches the current
checkpoint -- anything unverifiable re-exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from training.runtime_lora import export_runtime_lora_adapter

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
_RUNTIME_SUFFIX = "-runtime"
_SUMMARY_FILENAME = "runtime_adapter_summary.json"
_WEIGHTS_FILENAME = "adapter_model.safetensors"
_CONFIG_FILENAME = "adapter_config.json"
# Recorded in the summary and required to match for a skip; bump to force
# re-export of everything when the runtime layout or summary format changes.
_EXPORT_FORMAT_VERSION = 1


@dataclass(slots=True)
class ExportResult:
    step: int
    checkpoint_dir: Path
    runtime_dir: Path
    status: str  # "exported" | "skipped"
    summary: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_root(run_dir: Path) -> Path:
    """The dir that directly holds ``checkpoint-N`` subdirs (run dir or its outputs)."""
    for candidate in (run_dir, run_dir / "outputs"):
        if candidate.is_dir() and any(
            _CHECKPOINT_DIR_RE.match(child.name) for child in candidate.iterdir() if child.is_dir()
        ):
            return candidate
    raise SystemExit(
        f"No checkpoint-N dirs found under {run_dir} or {run_dir / 'outputs'}."
    )


def _discover_checkpoints(root: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for child in root.iterdir():
        match = _CHECKPOINT_DIR_RE.match(child.name)
        if child.is_dir() and match:
            step = int(match.group(1))
            if step in found:
                raise SystemExit(
                    f"Ambiguous checkpoint step {step} under {root}: "
                    f"{found[step].name} and {child.name}."
                )
            found[step] = child
    return found


def _source_fingerprint(checkpoint_dir: Path) -> dict[str, object]:
    """What the runtime export was derived from; recorded in the summary."""
    return {
        "source_weights_sha256": _sha256(checkpoint_dir / _WEIGHTS_FILENAME),
        "source_config_sha256": _sha256(checkpoint_dir / _CONFIG_FILENAME),
        "export_format_version": _EXPORT_FORMAT_VERSION,
    }


def _runtime_is_current(runtime_dir: Path, checkpoint_dir: Path) -> bool:
    """True only when every runtime artifact exists, the runtime weights match
    the summary's hash, and the recorded source fingerprint matches the current
    checkpoint -- a missing file, a stale/absent fingerprint, or an edited
    source all force a re-export."""
    summary_path = runtime_dir / _SUMMARY_FILENAME
    weights_path = runtime_dir / _WEIGHTS_FILENAME
    config_path = runtime_dir / _CONFIG_FILENAME
    if not (summary_path.is_file() and weights_path.is_file() and config_path.is_file()):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded = summary.get("adapter_sha256")
    if not recorded or recorded != _sha256(weights_path):
        return False
    try:
        expected = _source_fingerprint(checkpoint_dir)
    except OSError:
        return False
    if not all(summary.get(key) == value for key, value in expected.items()):
        return False
    # The runtime config is a copied serving artifact too. Checking only the
    # source fingerprint allowed a locally edited/corrupt runtime config to be
    # reported current even though vLLM would load different settings.
    try:
        return _sha256(config_path) == expected["source_config_sha256"]
    except OSError:
        return False


def export_run_checkpoints(
    *,
    run_dir: Path,
    checkpoint: int | None = None,
    export_all: bool = False,
    allow_zero_delta: bool = False,
) -> list[ExportResult]:
    """Export the selected checkpoints' ``-runtime`` siblings; return per-ckpt results."""
    if checkpoint is not None and export_all:
        raise ValueError("checkpoint and export_all are mutually exclusive")
    run_dir = Path(run_dir)
    root = _checkpoint_root(run_dir)
    checkpoints = _discover_checkpoints(root)
    if not checkpoints:
        raise SystemExit(f"No checkpoint-N dirs found under {root}.")

    if checkpoint is not None:
        if checkpoint not in checkpoints:
            available = ", ".join(str(n) for n in sorted(checkpoints))
            raise SystemExit(f"checkpoint-{checkpoint} not found under {root}; available: {available}.")
        selected = {checkpoint: checkpoints[checkpoint]}
    elif export_all:
        selected = checkpoints
    else:
        available = ", ".join(str(n) for n in sorted(checkpoints))
        raise SystemExit(f"Specify --checkpoint N or --all (available checkpoints: {available}).")

    results: list[ExportResult] = []
    for step in sorted(selected):
        checkpoint_dir = selected[step]
        runtime_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}{_RUNTIME_SUFFIX}"
        if _runtime_is_current(runtime_dir, checkpoint_dir):
            summary = json.loads((runtime_dir / _SUMMARY_FILENAME).read_text(encoding="utf-8"))
            results.append(ExportResult(step, checkpoint_dir, runtime_dir, "skipped", summary))
            continue
        export_runtime_lora_adapter(
            source_adapter_dir=checkpoint_dir,
            output_dir=runtime_dir,
            allow_zero_delta=allow_zero_delta,
            validate_load_invariants=True,
            compute_delta_stats=True,
            compute_adapter_sha256=True,
        )
        # Stamp the source fingerprint into the summary (SFT-side addition;
        # the shared exporter in training/runtime_lora.py stays RL-identical).
        summary_path = runtime_dir / _SUMMARY_FILENAME
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(_source_fingerprint(checkpoint_dir))
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        results.append(ExportResult(step, checkpoint_dir, runtime_dir, "exported", summary))
    return results


def _aggregate_delta(summary: dict) -> tuple[float, float]:
    """(max |delta| over module types, max frac_B_nonzero over module types)."""
    stats = summary.get("delta_stats") or {}
    max_abs = max((float(v.get("max_abs_delta", 0.0)) for v in stats.values()), default=0.0)
    frac_b = max((float(v.get("frac_B_nonzero", 0.0)) for v in stats.values()), default=0.0)
    return max_abs, frac_b


def _print_table(results: list[ExportResult]) -> None:
    header = f"{'checkpoint':<16} {'mode':<26} {'rank':>5} {'tensors':>8} {'max|Δ|':>11} {'fracB':>7} {'status':>9}"
    print(header)
    print("-" * len(header))
    for res in results:
        max_abs, frac_b = _aggregate_delta(res.summary)
        print(
            f"{res.checkpoint_dir.name:<16} "
            f"{str(res.summary.get('export_mode', '')):<26} "
            f"{int(res.summary.get('runtime_rank', 0)):>5} "
            f"{int(res.summary.get('converted_tensor_count', 0)):>8} "
            f"{max_abs:>11.3e} "
            f"{frac_b:>7.3f} "
            f"{res.status:>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export runtime LoRA adapters from SFT checkpoints")
    parser.add_argument("--run-dir", required=True, help="Run dir (holds outputs/checkpoint-* or checkpoint-*).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", type=int, help="Export only checkpoint-N.")
    group.add_argument("--all", action="store_true", help="Export every checkpoint-N.")
    parser.add_argument(
        "--allow-zero-delta",
        action="store_true",
        help="Export even when lora_B is all-zero (no effective delta); default refuses it.",
    )
    args = parser.parse_args()

    results = export_run_checkpoints(
        run_dir=Path(args.run_dir),
        checkpoint=args.checkpoint,
        export_all=args.all,
        allow_zero_delta=args.allow_zero_delta,
    )
    _print_table(results)
    exported = sum(1 for r in results if r.status == "exported")
    skipped = len(results) - exported
    print(f"\n{exported} exported, {skipped} skipped ({len(results)} checkpoint(s)).")


if __name__ == "__main__":
    main()
