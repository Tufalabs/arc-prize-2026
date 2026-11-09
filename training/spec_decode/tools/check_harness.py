"""Check the installed ARC-2/ARC-3 main harness without starting an engine.

Run with that harness's interpreter and arc-training/spec_decode on PYTHONPATH:
    python /path/to/arc-training/spec_decode/tools/check_harness.py --harness arc3 --out report.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import pickle
from pathlib import Path


def check(harness: str) -> dict:
    """Construct and serialize the real benchmark and check its engine factories."""
    import duck.agent
    import platypus.deployment
    import taaf.standard_benchmarks
    from platypus import LLMAgentSolver

    from drafter.extract import build_factory as extract_factory
    from drafter.head import DRAFTER, TARGET
    from drafter.online import build_factory as online_factory

    source = Path(__file__).with_name(f"long_run_{harness}.py")
    spec = importlib.util.spec_from_file_location("long_run_check", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.build_agent(str(TARGET), str(DRAFTER), False)
    solver = LLMAgentSolver(agent=agent)
    benchmark = (module._benchmark("official_25", solver) if harness == "arc3" else
                 taaf.standard_benchmarks.make_benchmark_named("arc2-eval", solver, n_passes=1))
    # Submission pickles the benchmark; catch stale factory or agent contracts.
    payload = pickle.dumps(benchmark)
    _, extraction = extract_factory()
    online = online_factory(DRAFTER, 4, 1)
    if extraction.engine_args.model != str(TARGET) or online.engine_args.model != str(TARGET):
        raise RuntimeError("serving factories selected a different training target")
    return {
        "harness": harness,
        "duck_source": duck.agent.__file__,
        "main_project": platypus.deployment.detect_main_project(),
        "versions": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
        "games": len(benchmark.games),
        "benchmark_pickle_bytes": len(payload),
        "target": str(TARGET),
        "drafter": str(DRAFTER),
        "extraction_speculative_config": extraction.engine_args.speculative_config,
        "online_speculative_config": online.engine_args.speculative_config,
    }


def main() -> None:
    """Write a reviewable report for one independently installed harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=("arc2", "arc3"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.harness)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
