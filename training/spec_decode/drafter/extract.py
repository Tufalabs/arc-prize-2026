"""Run chains through the served target and cache its pre-final-mixer stream.

The engine is the duck's B200 column as ``DuckAgent.build`` makes it, with
only the changes extraction needs:

- **prefix caching off:** a cached position computes nothing, so it would
  have no hidden state;
- **no host tier, one sequence at a time, no drafter, no profiler:** the
  capture maps a step's rows to one prompt, and nothing else should run;
- **one reply token:** the prompt is the whole chain, so the prefill is
  the forward and the single sampled token is thrown away.

Everything else (checkpoint, quantization, KV and recurrent-state dtypes,
kernels, compile and graphs) is the column's, so the stream is the one the
head reads when serving, up to prefill-versus-decode numerics. ``capture``
does the reading inside the engine core; this drives it and writes the cache:

- ``<chain_id>.safetensors``: ``hidden`` ``[tokens, 10240]`` bf16 and
  ``ids``;
- ``manifest.json`` (``manifest-II-of-NN.json`` per shard): the target, the
  engine settings that shape the stream, and every chain's size and id hash;
- optionally ``<chain_id>.logprobs.safetensors``: vLLM's own top-k prompt
  logprobs, for checking the trainer's label head against the engine.

    python -m drafter.extract DATASET_DIR OUT_DIR [--shard I/N] [--limit N] [--prompt-logprobs K] [--drafter DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from drafter import capture
from drafter.chains import Chain, read_dataset
from drafter.serving import moe_backend

#: Engine settings recorded in the manifest: those that shape the stream.
_RECORDED_ENGINE_ARGS = (
    "model",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "mamba_ssm_cache_dtype",
    "mamba_cache_mode",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "enable_prefix_caching",
    "enforce_eager",
)


def build_factory(drafter: Optional[Path] = None, drafter_moe_backend: Optional[str] = None) -> tuple[Any, Any]:
    """The duck's B200 engine, with extraction's changes. With ``drafter``, a
    k=1 MTP drafter from that directory, configured as the replay's ``mtp``
    arms do, whose pass-0 output is captured too (parity runs)."""
    from duck.agent import DuckAgent

    agent = DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
    agent.run_config.max_concurrent = 1  # the derived max_num_seqs follows
    factory = agent.llm_config.backend_factory
    factory.cpu_kv_offload = False
    factory.profile = None
    factory.mode = "direct"
    engine_args = factory.engine_args
    # The deployment cell serves e307; the stream must be the head's target's.
    from drafter.head import TARGET

    engine_args.model = str(TARGET)
    engine_args.enable_prefix_caching = False
    deployed_draft = dict(engine_args.speculative_config or {})
    engine_args.speculative_config = None
    if drafter is not None:
        engine_args.speculative_config = {
            **deployed_draft,
            "method": "mtp",
            "num_speculative_tokens": 1,
            "model": str(drafter),
            "quantization": engine_args.quantization,
            "moe_backend": moe_backend(drafter, drafter_moe_backend),
            "draft_sample_method": "probabilistic",
        }
    agent.prepare_for_backend()
    return agent, factory


def _logprob_arrays(prompt_logprobs: Any, k: int) -> dict[str, Any]:
    """vLLM's per-position dicts as ``[tokens, k]`` id and logprob arrays;
    position 0 has none."""
    import torch

    ids = torch.full((len(prompt_logprobs), k), -1, dtype=torch.int32)
    values = torch.full((len(prompt_logprobs), k), float("-inf"), dtype=torch.float32)
    for position, entry in enumerate(prompt_logprobs):
        if not entry:
            continue
        ranked = sorted(entry.items(), key=lambda item: -item[1].logprob)[:k]
        for rank, (token, logprob) in enumerate(ranked):
            ids[position, rank] = token
            values[position, rank] = logprob.logprob
    return {"ids": ids, "logprobs": values}


async def _extract_chain(backend: Any, chain: Chain, capture_dir: Path, out_dir: Path, prompt_logprobs: int) -> dict:
    from PIL import Image
    from safetensors.torch import load_file

    import ullm.backend
    from drafter.files import save_file

    images = [Image.open(path).convert("RGB") for path in chain.images]
    extra = {"prompt_logprobs": prompt_logprobs} if prompt_logprobs else {}
    params = ullm.backend.GenericSamplingParameters(temperature=1.0, max_tokens=1, extra_params=extra)
    request_id = f"{capture.REQUEST_TAG}{chain.chain_id}"
    started = time.monotonic()
    final = None
    async for output in backend.serve_ids(chain.ids, images, params, request_id):
        final = output
    served_s = time.monotonic() - started

    stem = capture.capture_file(capture_dir, request_id).stem
    deadline = time.monotonic() + 120.0
    while True:
        found = sorted(capture_dir.glob(stem + "*.safetensors"))
        failed = sorted(capture_dir.glob(stem + "*.failed"))
        if failed:
            raise RuntimeError(f"{chain.chain_id}: {failed[0].read_text().strip()}")
        if found:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"{chain.chain_id}: no capture written for {request_id}")
        await asyncio.sleep(0.5)
    tensors = load_file(str(found[0]))
    if tensors["ids"].tolist() != chain.ids:
        raise RuntimeError(f"{chain.chain_id}: the engine served a different prompt than the chain")
    save_file(tensors, str(out_dir / f"{chain.chain_id}.safetensors"))
    found[0].unlink()
    if prompt_logprobs:
        if final is None or final.prompt_logprobs is None:
            raise RuntimeError(f"{chain.chain_id}: asked for prompt logprobs, got none")
        save_file(_logprob_arrays(final.prompt_logprobs, prompt_logprobs), str(out_dir / f"{chain.chain_id}.logprobs.safetensors"))
    hidden = tensors["hidden"]
    print(
        f"[extract] {chain.chain_id}: {len(chain.ids):,} tokens in {served_s:.1f}s "
        f"({len(chain.ids) / served_s:,.0f} tok/s), stream {tuple(hidden.shape)} "
        f"rms {hidden.float().pow(2).mean().sqrt().item():.3f}",
        flush=True,
    )
    return {
        "chain_id": chain.chain_id,
        "harness": chain.harness,
        "tokens": len(chain.ids),
        "generated": chain.num_generated,
        "ids_sha256": chain.ids_sha256(),
        "served_s": round(served_s, 2),
    }


async def extract(
    dataset: Path,
    out_dir: Path,
    limit: Optional[int],
    prompt_logprobs: int,
    drafter: Optional[Path] = None,
    shard: tuple[int, int] = (0, 1),
    drafter_moe_backend: Optional[str] = None,
) -> None:
    """Every ``count``-th chain from the ``index``-th into ``out_dir``, one
    manifest per shard. A chain whose file is already there is kept, so a job
    that ran out of time resumes where it stopped."""
    import vllm

    index, count = shard
    chains = read_dataset(dataset)[:limit][index::count]
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = out_dir / f"capture-{index}"
    capture_dir.mkdir(exist_ok=True)
    name = "manifest.json" if count == 1 else f"manifest-{index:02d}-of-{count:02d}.json"
    # Read by the plugin in the engine core, which the factory spawns: set
    # before it does.
    os.environ[capture.ENV_DIR] = str(capture_dir)
    if drafter is not None:
        os.environ[capture.ENV_DRAFTER] = "1"
    agent, factory = build_factory(drafter, drafter_moe_backend)
    engine_args = factory.engine_args
    manifest: dict[str, Any] = {
        "format": "drafter-stream-v1",
        "dataset": str(dataset),
        "shard": [index, count],
        "vllm": vllm.__version__,
        "engine_args": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in ((name_, getattr(engine_args, name_, None)) for name_ in _RECORDED_ENGINE_ARGS)
        },
        "prompt_logprobs": prompt_logprobs,
        "drafter": None if drafter is None else str(drafter),
        "speculative_config": engine_args.speculative_config,
        "chains": [],
    }

    def write_manifest() -> None:
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(out_dir / name)

    pending = []
    for chain in chains:
        if (out_dir / f"{chain.chain_id}.safetensors").exists():
            manifest["chains"].append(
                {
                    "chain_id": chain.chain_id,
                    "harness": chain.harness,
                    "tokens": len(chain.ids),
                    "generated": chain.num_generated,
                    "ids_sha256": chain.ids_sha256(),
                    "served_s": None,
                }
            )
        else:
            pending.append(chain)
    print(f"[extract] shard {index}/{count}: {len(chains)} chains, {len(chains) - len(pending)} already there", flush=True)
    write_manifest()
    if pending:
        started = time.monotonic()
        backend = await factory.start()
        print(f"[extract] engine up in {time.monotonic() - started:.0f}s", flush=True)
        try:
            for chain in pending:
                manifest["chains"].append(await _extract_chain(backend, chain, capture_dir, out_dir, prompt_logprobs))
                write_manifest()
        finally:
            await factory.close()
    capture_dir.rmdir()
    print(f"[extract] wrote {len(manifest['chains'])} chains to {out_dir}", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-logprobs", type=int, default=0)
    parser.add_argument("--drafter", type=Path, default=None, help="also capture this k=1 drafter's pass-0 output (parity)")
    parser.add_argument("--shard", default="0/1", help="INDEX/COUNT: every COUNT-th chain from the INDEX-th")
    parser.add_argument("--moe-backend", choices=("triton", "marlin"), help="override the drafter backend selected from the head's quantization config")
    args = parser.parse_args(argv)
    index, count = (int(x) for x in args.shard.split("/"))
    asyncio.run(extract(args.dataset, args.out_dir, args.limit, args.prompt_logprobs, args.drafter,
                        (index, count), args.moe_backend))


if __name__ == "__main__":
    main()
