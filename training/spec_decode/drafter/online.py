"""A drafter checkpoint's acceptance on the served engine, over a dataset's calls.

The duck's B200 engine (tier off) with the drafter configured as the replay
launcher's ``mtp`` arms do, pointed at ``--drafter``. Every call of every
chain is served at once: its prompt (the chain up to the call's reply, with
the images that far), temperature 0.6 alone (``--sampling deployed``: the
recordings' top-p 0.8899, top-k 194 and presence penalty 1.5 as well, which
the draft side never applies, so they cap acceptance for any drafter) and the
recorded reply length, forced and capped, with a fixed seed per call so two
drafters see the same requests. Acceptance comes from the engine's own
spec-decode counters, overall and per draft position.

    python -m drafter.online DATASET_DIR OUT_JSON --drafter DIR [--k 3] [--max-tokens 1024] [--sampling deployed]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

from drafter.chains import Chain, placeholder_runs, read_dataset
from drafter.serving import moe_backend

SAMPLING = {
    # Every coordinate stated, so no engine or checkpoint default fills one in.
    "temperature": {"temperature": 0.6, "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0},
    "deployed": {"temperature": 0.6, "top_p": 0.8899009607194754, "top_k": 194, "presence_penalty": 1.5},
}


def calls(chain: Chain, cap: int) -> list[tuple[list[int], list[str], int]]:
    """Each call's prompt, its images and its forced reply length."""
    out = []
    for start, end in chain.generated:
        prompt = chain.ids[:start]
        out.append((prompt, chain.images[: placeholder_runs(prompt)], min(end - start, cap)))
    return out


def build_factory(
    drafter: Path, k: int, requests: int, drafter_kv_dtype: Optional[str] = None,
    drafter_moe_backend: Optional[str] = None,
) -> Any:
    """Use the selected harness cell and the same target used for training."""
    from duck.agent import DuckAgent

    agent = DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
    agent.run_config.max_concurrent = requests
    factory = agent.llm_config.backend_factory
    factory.cpu_kv_offload = False
    factory.profile = None
    factory.mode = "direct"
    engine_args = factory.engine_args
    from drafter.head import TARGET

    engine_args.model = str(TARGET)
    engine_args.speculative_config = {
        **(engine_args.speculative_config or {}),
        "method": "mtp",
        "num_speculative_tokens": k,
        "model": str(drafter),
        "quantization": engine_args.quantization,
        "moe_backend": moe_backend(drafter, drafter_moe_backend),
        "draft_sample_method": "probabilistic",
    }
    if drafter_kv_dtype:
        # The drafter's attention cache in its own dtype; by default it
        # follows the target's (FP8 per token and head).
        engine_args.speculative_config["kv_cache_dtype"] = drafter_kv_dtype
    agent.prepare_for_backend()
    return factory


async def serve(
    dataset: Path, out: Path, drafter: Path, k: int, cap: int, sampling: str,
    drafter_kv_dtype: Optional[str] = None, drafter_moe_backend: Optional[str] = None,
) -> None:
    from PIL import Image

    import ullm.backend
    import ullm.engine_monitor

    chains = read_dataset(dataset)
    requests = [(chain, index, call) for chain in chains for index, call in enumerate(calls(chain, cap))]
    factory = build_factory(drafter, k, len(requests), drafter_kv_dtype, drafter_moe_backend)
    started = time.monotonic()
    backend = await factory.start()
    print(f"[online] engine up in {time.monotonic() - started:.0f}s with drafter {drafter}", flush=True)
    before = await ullm.engine_monitor.inprocess_source()

    async def one(chain: Chain, index: int, call: tuple[list[int], list[str], int]) -> int:
        prompt, images, length = call
        config = dict(SAMPLING[sampling])
        params = ullm.backend.GenericSamplingParameters(
            temperature=config.pop("temperature"),
            top_p=config.pop("top_p"),
            max_tokens=length,
            seed=1000 + index,
            extra_params={**config, "min_tokens": length, "ignore_eos": True},
        )
        pictures = [Image.open(path).convert("RGB") for path in images]
        final: Any = None
        async for output in backend.serve_ids(prompt, pictures, params, f"online-{chain.chain_id}-{index}"):
            final = output
        return len(final.outputs[0].token_ids)

    tick = time.monotonic()
    generated = await asyncio.gather(*(one(chain, index, call) for chain, index, call in requests))
    seconds = time.monotonic() - tick
    after = await ullm.engine_monitor.inprocess_source()
    await factory.close()

    def delta(name: str) -> float:
        return after.scalars.get(name, 0.0) - before.scalars.get(name, 0.0)

    drafts = delta("vllm:spec_decode_num_drafts")
    draft_tokens = delta("vllm:spec_decode_num_draft_tokens")
    accepted = delta("vllm:spec_decode_num_accepted_tokens")
    by_position = {
        position: after.accepted_by_position.get(position, 0.0) - before.accepted_by_position.get(position, 0.0)
        for position in sorted(after.accepted_by_position, key=int)
    }
    result = {
        "target": factory.engine_args.model,
        "drafter": str(drafter),
        "k": k,
        "sampling": sampling,
        "drafter_kv_dtype": drafter_kv_dtype,
        "moe_backend": factory.engine_args.speculative_config["moe_backend"],
        "calls": len(requests),
        "generated": sum(generated),
        "seconds": round(seconds, 1),
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted": accepted,
        "acceptance": accepted / draft_tokens if draft_tokens else None,
        "accepted_per_draft": accepted / drafts if drafts else None,
        "acceptance_by_position": {p: v / drafts for p, v in by_position.items()} if drafts else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("[online]", json.dumps(result), flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--sampling", choices=sorted(SAMPLING), default="temperature")
    parser.add_argument("--drafter-kv-dtype", default=None, help="the drafter's KV cache dtype (default: the target's)")
    parser.add_argument("--moe-backend", choices=("triton", "marlin"), help="override the backend selected from the head's quantization config")
    args = parser.parse_args(argv)
    asyncio.run(serve(args.dataset, args.out, args.drafter, args.k, args.max_tokens, args.sampling,
                      args.drafter_kv_dtype, args.moe_backend))


if __name__ == "__main__":
    main()
