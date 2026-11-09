"""The same prompts through Flash-Next's target on two or more MoE backends: per-position log-probs compared.

Each arm builds the seal's B200 engine (``seal.presets``) on CHECKPOINT with the drafter off, eager, and the
arm's MoE backend, and scores PROMPTS prompts of LENGTH tokens cut from the repo's own text (prompt
log-probs, top-5). Every arm runs in a process of its own, so the card is clear between them. The first arm
is the reference; each other arm is compared against it per position: the gap in the actual next token's
log-prob, and whether the top-1 token agrees.

    python tools/moe_logprob_check.py OUT_DIR [ARM ...]  # arms: marlin, marlin_repeat, cutedsl_w4a16, cutedsl_w4a4
"""

import dataclasses
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, cast

from ullm.backends.deployment import CUTEDSL_W4A16_ENV

CHECKPOINT = "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc3-reap307-dials-joint205-u800"
PROMPTS, LENGTH = 32, 3072
TEXT = [
    "design.md",
    "settings.md",
    "roadmap.md",
    "packages/ullm/flash_optimization.md",
    "packages/ullm/flash_optimization_arc2.md",
    "packages/ullm/kv_offload.md",
    "packages/ullm/design.md",
    "packages/duck/design.md",
    "packages/ullm/src/ullm/traffic.py",
    "packages/ullm/src/ullm/backends/vllm_factory.py",
]
# Each arm: (moe_backend, environment), over the cell's own environment.
ARMS: dict[str, tuple[str, dict[str, str]]] = {
    "marlin": ("marlin", {}),
    # Marlin again, in an engine of its own: the floor any other arm is read against.
    "marlin_repeat": ("marlin", {}),
    "cutedsl_w4a16": ("flashinfer_cutedsl", {CUTEDSL_W4A16_ENV: "1"}),
    "cutedsl_w4a4": ("flashinfer_cutedsl", {CUTEDSL_W4A16_ENV: "0"}),
}


def prompts() -> list[list[int]]:
    """PROMPTS consecutive cuts of LENGTH tokens from the TEXT files, in order."""
    from transformers import AutoTokenizer

    tokenizer = cast(Any, AutoTokenizer.from_pretrained(CHECKPOINT))  # pyright: ignore[reportUnknownMemberType]
    text = "\n\n".join(pathlib.Path(p).read_text() for p in TEXT)
    ids = [int(t) for t in tokenizer(text)["input_ids"]]
    if len(ids) < PROMPTS * LENGTH:
        raise SystemExit(f"{len(ids)} tokens of text, {PROMPTS * LENGTH} wanted")
    return [ids[i * LENGTH : (i + 1) * LENGTH] for i in range(PROMPTS)]


def _rank(entry: tuple[int, Any]) -> int:
    """A prompt log-prob entry's rank (1 is the top token)."""
    return int(entry[1].rank or 0)


def score(arm: str, out_dir: pathlib.Path) -> None:
    """One arm, in this process: the engine, the prompts, the log-probs to out_dir/<arm>.json."""
    from seal.agent import SealAgent

    backend, env = ARMS[arm]
    factory: Any = SealAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200").llm_config.backend_factory
    os.environ.update(factory.env_overrides)
    os.environ.update(env)
    import vllm

    engine_args = factory.engine_args
    kwargs = {f.name: getattr(engine_args, f.name) for f in dataclasses.fields(vllm.EngineArgs)}
    kwargs.update(
        model=CHECKPOINT,
        speculative_config=None,
        enforce_eager=True,
        max_model_len=LENGTH + 64,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
    )
    kwargs["kernel_config"].moe_backend = backend
    llm = vllm.LLM(**kwargs)
    params = vllm.SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=5)
    outputs = llm.generate([vllm.TokensPrompt(prompt_token_ids=p) for p in prompts()], params)
    rows: list[dict[str, list[Any]]] = []
    for out in outputs:
        ids = out.prompt_token_ids
        assert ids is not None
        actual: list[float] = []
        top1: list[int] = []
        for pos, lp in enumerate(out.prompt_logprobs or []):
            if lp is None:
                continue
            actual.append(lp[ids[pos]].logprob)
            top1.append(min(lp.items(), key=_rank)[0])
        rows.append({"actual": actual, "top1": top1})
    (out_dir / f"{arm}.json").write_text(json.dumps(rows))
    print(f"[{arm}] scored {len(rows)} prompts", flush=True)


def compare(out_dir: pathlib.Path, arms: list[str]) -> None:
    """Each arm against the first, position by position: the log-prob gap and top-1 agreement."""
    ref = json.loads((out_dir / f"{arms[0]}.json").read_text())
    ref_actual = [x for r in ref for x in r["actual"]]
    ref_top1 = [x for r in ref for x in r["top1"]]
    n = len(ref_actual)
    print(f"\n{n} positions; reference {arms[0]} mean log-prob of the actual token {sum(ref_actual) / n:.4f}")
    print(f"{'arm':16} {'mean logprob':>12} {'mean |gap|':>10} {'p99 |gap|':>9} {'max |gap|':>9} {'top-1 agrees':>12}")
    for arm in arms[1:]:
        rows = json.loads((out_dir / f"{arm}.json").read_text())
        actual = [x for r in rows for x in r["actual"]]
        top1 = [x for r in rows for x in r["top1"]]
        gaps = sorted(abs(a - b) for a, b in zip(actual, ref_actual))
        agree = sum(a == b for a, b in zip(top1, ref_top1)) / n
        print(
            f"{arm:16} {sum(actual) / n:>12.4f} {sum(gaps) / n:>10.4f} {gaps[int(0.99 * n)]:>9.4f} {gaps[-1]:>9.4f} "
            f"{agree:>12.2%}"
        )


def main() -> None:
    """Score each arm in a process of its own (or, as that process, the one it names), then compare."""
    out_dir = pathlib.Path(sys.argv[1])
    arms = sys.argv[2:] or list(ARMS)
    if os.environ.get("MOE_LOGPROB_ARM"):
        score(os.environ["MOE_LOGPROB_ARM"], out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        subprocess.run([sys.executable, __file__, str(out_dir)], env={**os.environ, "MOE_LOGPROB_ARM": arm}, check=True)
    compare(out_dir, arms)


if __name__ == "__main__":
    main()
