"""Build or deploy the pinned PR #131 evaluation with explicit user flags."""
import argparse
import ast
import asyncio
import hashlib
import json
from pathlib import Path

from provenance import imported_harness_identity

HERE = Path(__file__).resolve().parent


def main():
    """Execute the official notebook's build cells and optionally its deployment cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--receipt", type=Path, help="Write the successful CPU build contract atomically")
    parser.add_argument("--fp8-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()
    harness=imported_harness_identity()
    checkpoint = args.checkpoint.expanduser().resolve()
    path = HERE / "official/sft_eval.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "official_sft_eval"}
    for index in (1, 3):
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "CHECKPOINT":
                    node.value = ast.Constant(str(checkpoint))
                elif node.targets[0].id == "FP8_KV":
                    node.value = ast.Constant(args.fp8_kv)
        if index == 3 and args.run_dir is not None:
            namespace["RUN_DIR"] = args.run_dir.expanduser().resolve()
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102 - Pinned official notebook.
    root = namespace["RESULTS_ROOT"].resolve()
    for benchmark in namespace["benchmarks"].values():
        if not benchmark.job_dir.resolve().is_relative_to(root):
            raise ValueError(f"evaluation job directory must be under {root}")
    contract = {"harness_commit": harness['harness_commit'], "harness_identity": harness,
                "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "checkpoint": str(checkpoint), "fp8_kv": args.fp8_kv, "student_profile": True,
                "holdout_passes": namespace["HOLDOUT_PASSES"], "official_passes": namespace["OFFICIAL_PASSES"],
                "slots": namespace["SLOTS"], "run_dir": str(namespace["RUN_DIR"]),
                "matched_int4_reference": not args.fp8_kv}
    contract["local_checkpoint_selected_before_engine_construction"] = True
    print(json.dumps(contract, indent=2))
    if args.receipt:
        receipt = args.receipt.expanduser().resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_suffix(receipt.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(contract, status="passed", cpu_only=True,
                                             gpu_inference_qualified=False), indent=2) + "\n")
        temporary.replace(receipt)
    if args.deploy:
        namespace["RUN_DIR"].mkdir(parents=True, exist_ok=False)
        (namespace["RUN_DIR"] / "EVALUATION_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
        code = compile("".join(notebook["cells"][5]["source"]), str(path), "exec",
                       flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        asyncio.run(eval(code, namespace))


if __name__ == "__main__":
    main()
