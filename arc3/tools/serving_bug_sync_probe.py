"""The serving-bug check's exact procedure through the SYNC vllm.LLM entrypoint.

Diagnostic for the async-vs-sync question: every measurement that showed
the warm-hit perturbation (engine_probe grid) drove the engine through
``vllm.LLM``; production and the startup check drive ``AsyncLLMEngine``.
This runs the check's procedure — multi-chunk full prefill, 8 seeded warm
draws at an off-page high-entropy prefix, per-draw cache-reset colds —
on the sync entrypoint with the duck preset's production engine args.

Run on a B200 node:
    srun ... uv run python tools/serving_bug_sync_probe.py
"""

import dataclasses
from typing import Any

import vllm

from duck.agent import DuckAgent


def main() -> None:
    """Boot the sync engine with production args and print the verdict."""
    factory: Any = DuckAgent.build().llm_config.backend_factory
    ea = factory.engine_args
    ea.tensor_parallel_size = 1
    defaults = vllm.AsyncEngineArgs(model=ea.model)
    kwargs = {"model": ea.model}
    for f in dataclasses.fields(ea):
        v = getattr(ea, f.name)
        if f.name != "model" and v != getattr(defaults, f.name):
            kwargs[f.name] = v
    print(f"[sync-probe] LLM kwargs: {kwargs}", flush=True)
    llm = vllm.LLM(**kwargs)
    tokenizer = llm.get_tokenizer()

    entries = [
        f"Entry {i}: the courier crossed bridge {(i * 7) % 97} at dusk carrying parcel {(i * 13) % 89},"
        for i in range(1560)
    ]
    prefix_text = " ".join(entries[:1400]) + " At the junction the courier finally decided to"
    full_text = prefix_text + " open the parcel. " + " ".join(entries[1400:])
    prefix_ids = tokenizer.encode(prefix_text)
    full_ids = tokenizer.encode(full_text)
    print(f"[sync-probe] full {len(full_ids)} tokens, prefix {len(prefix_ids)}", flush=True)

    def prompt(ids: list[int]):
        try:
            return vllm.TokensPrompt(prompt_token_ids=ids)
        except AttributeError:
            from vllm.inputs import TokensPrompt

            return TokensPrompt(prompt_token_ids=ids)

    def params(seed: int):
        return vllm.SamplingParams(temperature=2.0, top_k=194, max_tokens=1, seed=seed)

    llm.generate([prompt(full_ids)], vllm.SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
    warm: list[int] = [
        int(o.outputs[0].token_ids[0])
        for o in llm.generate([prompt(prefix_ids)] * 8, [params(s) for s in range(8)], use_tqdm=False)
    ]
    cold: list[int] = []
    for s in range(8):
        llm.reset_prefix_cache()
        o = llm.generate([prompt(prefix_ids)], params(s), use_tqdm=False)[0]
        cold.append(int(o.outputs[0].token_ids[0]))

    def shown(ids: list[int]) -> str:
        return " ".join(repr(tokenizer.decode([t])) for t in ids)

    mism = sum(1 for c, w in zip(cold, warm) if c != w)
    print(f"[sync-probe] warm: {shown(warm)}", flush=True)
    print(f"[sync-probe] cold: {shown(cold)}", flush=True)
    print(f"[sync-probe] {mism}/8 pairs differ -> {'PERTURBED' if mism >= 2 else 'CONSISTENT'}", flush=True)


if __name__ == "__main__":
    main()
