"""Train the drafter head on cached chains, several draft steps at once.

For a chain of T tokens with the target's stream at every position:

- **Step 0** (pass 0) runs the head over every slot t in 0..T-2: the target's
  stream at t and token t+1, at position t. Its single stream drafts token
  t+2, against the target's own distribution there.
- **Step j** runs every anchor t at once as a draft j tokens on: the head's
  own step-(j-1) multi-stream at t and token t+1+j (the data's, i.e. the
  target's sample, which is what the draft had to be to get this far), at
  position t+j, drafting token t+2+j. Attention as ``drafter.attention``
  describes, over FP8 per-token-head keys and values as the served cache
  holds them (``--kv-fp8``, straight-through).

A (t, j) pair trains only where tokens t+1 .. t+2+j all lie in one call's
reply: a draft starts from a sampled token and never runs past the call. The
loss is full-vocabulary KL(target || head) at ``--loss-temperature`` (1 by
default), per step averaged over its pairs, the steps weighted equally; with
``--draft-weighting``, each pair weighted instead by how often serving
drafts it: the chance a draft starts at its anchor times the chance the
draft gets as far as the step, from the head's own acceptance ratios.

The run: chains stream from their caches (a thread loads the next while one
trains), one per GPU per microbatch across ``--gpus`` data-parallel ranks,
``--accum`` microbatches an update, each draft step recomputed in the
backward so long chains fit. A chain longer than ``DRAFTER_MAX_TOKENS``
(unset: none) runs as windows of at most that many tokens (``windows``).
ARC2 and ARC3 are mixed 1:1 by trained tokens each epoch. Warmup then
cosine decay over the planned updates.

Monitoring: TensorBoard under ``OUT/tb`` and one JSON line per event in
``OUT/metrics.jsonl``: per update the loss, per step KL and acceptance,
gradient norm, learning rate, throughput; the validation set every
``--val-every`` updates, per harness. Every ``--save-minutes`` a checkpoint
lands in ``OUT/checkpoints/update-NNNNNN/head.safetensors`` with a line in
``OUT/checkpoints/index.jsonl`` (``drafter.watch`` tests each as it lands),
and the optimizer state in ``OUT/state/`` for ``--resume``. ``OUT/DONE``
marks the end.

    python -m drafter.train OUT --train CACHE:DATASET [...] [--val CACHE:DATASET] [--gpus 4] [--steps 3] [--lr 2e-5]
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint

from drafter import attention as qsa
from drafter.chains import read_dataset
from drafter.head import DrafterHead, LabelHead, export_state, load_head, load_trained
from drafter.loss import ChunkedDistillation, acceptance_metrics

#: Chains longer than this many tokens run as windows of at most as many
#: (``windows``); unset or 0 runs every chain whole. One B200 holds a chain
#: of about 85k tokens in training, and ARC3's grow to its 158k window.
MAX_TOKENS = int(os.environ.get("DRAFTER_MAX_TOKENS") or 0)


@dataclasses.dataclass
class ChainData:
    """A cached chain, its stream still on disk."""

    chain_id: str
    harness: str
    tokens: int
    trained: int  # step-0 trained positions: each reply less its first token
    path: Path
    logprobs_path: Optional[Path]
    ids: torch.Tensor  # [T] long
    span: torch.Tensor  # [T] long: the call a reply token belongs to, else -1
    starts: torch.Tensor  # [calls] long: where each call's reply starts
    offset: int = 0  # a window's first row in the cached chain
    cached: int = 0  # the cached chain's tokens (0: ``tokens``)


def windows(chain: ChainData, max_tokens: int) -> list[ChainData]:
    """``chain`` as windows of at most ``max_tokens`` tokens (itself when it
    fits, or with ``max_tokens`` 0). The first trains every position it
    holds; each later one opens a quarter window before the first position
    it trains, 1024-aligned so the indexer's blocks fall where they do in the
    whole chain, and holds that stretch as context: its reply tokens there
    are marked as a prompt's (span -1). So every position trains in exactly
    one window, over at least that much of what precedes it, and keeps its
    own position for the rotary (``offset``). (A draft does not cross a
    window's end, so the last few anchors before one lose their later draft
    steps: a handful of (position, step) pairs per boundary.)"""
    if not max_tokens or chain.tokens <= max_tokens:
        return [chain]
    context = max(1024, max_tokens // 4 // 1024 * 1024)
    if context + 1024 >= max_tokens:
        raise ValueError(f"windows of {max_tokens} tokens leave no room past their {context}-token context")
    pieces: list[ChainData] = []
    first = 0  # the next window's first trained position
    while first < chain.tokens:
        start = 0 if first == 0 else (first - context) // 1024 * 1024
        end = min(chain.tokens, start + max_tokens)
        span = chain.span[start:end].clone()
        span[: first - start] = -1
        lengths = torch.bincount(span[span >= 0], minlength=len(chain.starts))
        pieces.append(
            dataclasses.replace(
                chain,
                chain_id=f"{chain.chain_id}@{start}",
                tokens=end - start,
                trained=int((lengths - 1).clamp(min=0).sum()),
                logprobs_path=None,
                ids=chain.ids[start:end],
                span=span,
                # A reply begun in the context starts, for the window, where it trains.
                starts=(chain.starts.clamp(min=first) - start).clamp(max=end - start),
                offset=start,
                cached=chain.tokens,
            )
        )
        first = end
    return pieces


def scored_from(chain: ChainData, position: int) -> Optional[ChainData]:
    """``chain`` scoring only what lies at or past ``position`` of its cached
    chain: the reply tokens before it are marked as a prompt's (span -1), as
    a window's context is, so the head still attends over them. None when
    nothing is left to score. Whole chains and their windows then score the
    same positions over two lengths of what precedes them."""
    first = position - chain.offset
    if first <= 0:
        return chain
    if first >= chain.tokens:
        return None
    span = chain.span.clone()
    span[:first] = -1
    lengths = torch.bincount(span[span >= 0], minlength=len(chain.starts))
    trained = int((lengths - 1).clamp(min=0).sum())
    if not trained:
        return None
    return dataclasses.replace(
        chain, trained=trained, logprobs_path=None, span=span, starts=chain.starts.clamp(min=first)
    )


@dataclasses.dataclass
class Loaded:
    """A chain on the device, ready to run."""

    chain: ChainData
    ids: torch.Tensor
    span: torch.Tensor
    starts: torch.Tensor
    stream: torch.Tensor  # [T, 4H] bf16
    logprobs: Optional[dict[str, torch.Tensor]]


def load_chains(cache: Path, dataset: Path) -> list[ChainData]:
    """The dataset's chains that the cache holds (every ``manifest*.json`` in
    it: one per extraction shard). The dataset may be a subset of what the
    cache holds."""
    chains = {chain.chain_id: chain for chain in read_dataset(dataset)}
    manifests = sorted(Path(cache).glob("manifest*.json"))
    if not manifests:
        raise ValueError(f"{cache}: no manifest")
    out = []
    for manifest in manifests:
        for entry in json.loads(manifest.read_text())["chains"]:
            if entry["chain_id"] not in chains:
                continue  # a dataset that samples a cached set's chains
            chain = chains[entry["chain_id"]]
            if entry["ids_sha256"] != chain.ids_sha256():
                raise ValueError(f"{chain.chain_id}: cached for other ids than the dataset's")
            span = torch.full((len(chain.ids),), -1, dtype=torch.long)
            for call, (start, end) in enumerate(chain.generated):
                span[start:end] = call
            logprobs = Path(cache) / f"{chain.chain_id}.logprobs.safetensors"
            out.append(
                ChainData(
                    chain_id=chain.chain_id,
                    harness=chain.harness,
                    tokens=len(chain.ids),
                    trained=sum(max(0, end - start - 1) for start, end in chain.generated),
                    path=Path(cache) / f"{chain.chain_id}.safetensors",
                    logprobs_path=logprobs if logprobs.exists() else None,
                    ids=torch.tensor(chain.ids, dtype=torch.long),
                    span=span,
                    starts=torch.tensor([start for start, _ in chain.generated], dtype=torch.long),
                )
            )
    missing = len(chains) - len(out)
    if missing:
        print(f"[train] {cache}: {missing} of {len(chains)} dataset chains not cached, left out", flush=True)
    pieces = [piece for chain in out for piece in windows(chain, MAX_TOKENS)]
    if len(pieces) > len(out):
        long = sum(chain.tokens > MAX_TOKENS for chain in out)
        print(
            f"[train] {cache}: {long} chains over {MAX_TOKENS:,} tokens run as windows, {len(pieces)} in all",
            flush=True,
        )
    return pieces


def _read(chain: ChainData) -> tuple[ChainData, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(chain.path), framework="pt") as f:
        ids, hidden = f.get_slice("ids"), f.get_slice("hidden")
        if ids.get_shape()[0] != (chain.cached or chain.tokens):
            raise ValueError(f"{chain.chain_id}: cached ids differ")
        rows = slice(chain.offset, chain.offset + chain.tokens)
        if not torch.equal(ids[rows].long(), chain.ids):
            raise ValueError(f"{chain.chain_id}: cached ids differ")
        stream = hidden[rows]
    logprobs = load_file(str(chain.logprobs_path)) if chain.logprobs_path else None
    return chain, stream.pin_memory(), logprobs


def _to_device(item: tuple[ChainData, torch.Tensor, Optional[dict[str, torch.Tensor]]], device: str) -> Loaded:
    chain, stream, logprobs = item
    return Loaded(
        chain=chain,
        ids=chain.ids.to(device),
        span=chain.span.to(device),
        starts=chain.starts.to(device),
        stream=stream.to(device, non_blocking=True),
        logprobs=logprobs,
    )


def load(chain: ChainData, device: str) -> Loaded:
    return _to_device(_read(chain), device)


def prefetched(chains: Iterable[ChainData], device: str, depth: int = 2) -> Iterator[Loaded]:
    """The chains loaded one ahead on a thread: disk and NFS reads overlap
    the step before."""
    slots: queue.Queue = queue.Queue(maxsize=depth)

    def worker() -> None:
        try:
            for chain in chains:
                slots.put(_read(chain))
        except BaseException as error:  # surfaced on the consuming side
            slots.put(error)
        slots.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = slots.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        yield _to_device(item, device)


class Trainer:
    def __init__(
        self, head: DrafterHead, label: LabelHead, args: argparse.Namespace, optimize: bool = True, world: int = 1
    ) -> None:
        self.head, self.label, self.args, self.world = head, label, args, world
        self.steps = args.steps
        self.params = [p for _, p in head.trainable()]
        trainable = {id(p) for p in self.params}
        for _, p in head.named_parameters():
            p.requires_grad_(optimize and id(p) in trainable)
        if not optimize:
            return
        # bf16 weights for compute, as served; fp32 masters for the update.
        self.masters = [p.detach().float().clone() for p in self.params]
        router = {id(p) for name, p in head.trainable() if name.endswith("mlp.gate.weight")}
        groups = [
            {"params": [m for m, p in zip(self.masters, self.params) if id(p) not in router], "lr_scale": 1.0},
            {
                "params": [m for m, p in zip(self.masters, self.params) if id(p) in router],
                "lr_scale": args.router_lr_scale,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            [g for g in groups if g["params"]], lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )

    # --- the head over one chain ----------------------------------------------

    def _step(
        self,
        step: int,
        anchors: torch.Tensor,
        n0: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        context_key: Optional[torch.Tensor],
        context_value: Optional[torch.Tensor],
        blocks: Optional[torch.Tensor],
        *diagonal_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """One draft step's layer. Returns its multi-stream, its keys and
        values (pass 0's are every later step's context), and at step 0 the
        indexer's block keys."""
        head, args = self.head, self.args
        sa = head.layer.self_attn
        indexer = sa.indexer
        rows = x.shape[0]
        heads, kv_heads, dim = head.config.num_attention_heads, head.config.num_key_value_heads, sa.head_dim
        diagonal = list(zip(diagonal_flat[0::2], diagonal_flat[1::2]))
        saved: dict[str, Any] = {}

        def attend(mixed: torch.Tensor) -> torch.Tensor:
            query, gate = torch.chunk(sa.q_proj(mixed).view(rows, heads, 2 * dim), 2, dim=-1)
            query = qsa.rope(sa.q_norm(query), cos, sin)
            key = qsa.rope(sa.k_norm(sa.k_proj(mixed).view(rows, kv_heads, dim)), cos, sin)
            value = sa.v_proj(mixed).view(rows, kv_heads, dim)
            if args.kv_fp8:
                # What the served cache holds: the attention reads these.
                key, value = qsa.fake_fp8(key), qsa.fake_fp8(value)
            with torch.no_grad():
                index_q, raw = qsa.indexer_projections(indexer, mixed, cos, sin)
                keys = qsa.block_keys(indexer, raw, cos, sin) if step == 0 else blocks
                chosen = qsa.select_blocks(indexer, index_q, keys, anchors)
                mask = qsa.selection_mask(chosen, anchors, n0, indexer.compress_ratio, indexer.token_budget)
            context = (key, value) if step == 0 else (context_key, context_value)
            out, lse = qsa.sparse_attention(
                query.transpose(0, 1)[None],
                context[0].transpose(0, 1)[None],
                context[1].transpose(0, 1)[None],
                mask,
                sa.scaling,
                compiled=not args.eager_attention,
            )
            out = out[0].transpose(0, 1)
            if step > 0:
                extra = [(k[:rows], v[:rows]) for k, v in diagonal] + [(key, value)]
                out = qsa.merge_diagonal(out, lse[0].transpose(0, 1), query, extra, sa.scaling)
            saved.update(key=key, value=value, blocks=keys if step == 0 else None)
            return sa.o_proj(out.reshape(rows, heads * dim) * torch.sigmoid(gate.reshape(rows, heads * dim)))

        multi = head.block(x, attend)
        return multi, saved["key"], saved["value"], saved["blocks"]

    def chain_loss(
        self, chain: Loaded, *, train: bool, keep: Optional[dict[str, torch.Tensor]] = None
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        """The chain's loss (``train``) and metric sums. ``keep``, when
        given, receives pass 0's multi-stream at every slot."""
        head, label, args = self.head, self.label, self.args
        ids, span, stream = chain.ids, chain.span, chain.stream
        total = len(ids)
        n0 = total - 1
        with torch.no_grad():
            teacher_single = torch.cat(
                [label.single(stream[start : start + 4096]) for start in range(1, total, 4096)]
            )  # row p - 1 is the target's single stream at position p
        slots = torch.arange(n0, device=ids.device)
        context_key = context_value = blocks = None
        diagonal: list[torch.Tensor] = []
        previous: Optional[torch.Tensor] = None
        # Per anchor, the running product of the acceptance ratios of the
        # steps so far (``acceptance_metrics``): the chance a served draft
        # from there is accepted this far, exact where the data was sampled
        # by that p. Step j's anchors are a subset of step j-1's.
        reach: dict[str, torch.Tensor] = {}
        reach_steps: dict[str, list[torch.Tensor]] = {}
        pending: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        metrics: dict[str, float] = {}
        checkpointed = train and args.checkpoint_steps
        for step in range(self.steps):
            rows = n0 - step
            if rows <= 1:
                break
            anchors = slots[:rows]
            stream_in = stream[:rows] if step == 0 else previous[:rows]  # type: ignore[index]
            x = head.fuse(stream_in, F.embedding(ids[1 + step : 1 + step + rows], label.embed))
            cos, sin = head.rope(anchors + step + chain.chain.offset, stream)
            function = functools.partial(self._step, step, anchors, n0)
            inputs = (x, cos, sin, context_key, context_value, blocks, *[t[:rows] for t in diagonal])
            if checkpointed:
                multi, key, value, new_blocks = torch.utils.checkpoint.checkpoint(
                    function, *inputs, use_reentrant=False
                )
            else:
                multi, key, value, new_blocks = function(*inputs)
            if step == 0:
                context_key, context_value, blocks = key, value, new_blocks
                if keep is not None:
                    keep["multi"] = multi.detach()
            else:
                diagonal += [key, value]
            # Pairs whose tokens t+1 .. t+2+step all lie in one call's reply.
            anchor = anchors[: rows - 1]
            call = span[anchor + 1]
            picked = anchor[(call >= 0) & (call == span[anchor + 2 + step])]
            if picked.numel():
                single = head.sample_stream(multi[picked])
                teacher = teacher_single[picked + step]
                targets = ids[picked + 2 + step]
                if train:
                    pending.append((picked, single, teacher, targets))
                served = (ids, chain.starts[span[picked + 1]], picked + 2 + step)
                ratios: dict[str, torch.Tensor] = {}
                sums = acceptance_metrics(
                    single.detach(), teacher, label.lm_head, targets, served=served, ratios=ratios
                )
                for name, ratio in ratios.items():
                    running = reach.setdefault(name, torch.ones(n0, device=ids.device))
                    running[picked] *= ratio
                    sums[f"reach_{name}"] = float(running[picked].sum())
                    snapshot = torch.zeros(n0, device=ids.device)
                    snapshot[picked] = running[picked]
                    reach_steps.setdefault(name, []).append(snapshot)
                for name, value_ in sums.items():
                    metrics[f"step{step}/{name}"] = value_
            previous = multi
        lengths = torch.bincount(span[span >= 0], minlength=len(chain.starts))
        generated = [(int(a), int(a + n)) for a, n in zip(chain.starts.tolist(), lengths.tolist())]
        losses = []
        if pending:
            weights: list[Optional[torch.Tensor]] = [None] * len(pending)
            if args.draft_weighting:
                weights = self._serving_weights(reach_steps["t0.6"], generated, [picked for picked, *_ in pending])
            for (picked, single, teacher, targets), weight in zip(pending, weights):
                loss = ChunkedDistillation.apply(
                    single,
                    label.lm_head,
                    teacher,
                    label.lm_head,
                    targets,
                    args.loss_temperature,
                    args.ce_weight,
                    64,
                    weight,
                )
                losses.append(loss if weight is not None else loss / picked.numel())
        loss = torch.stack(losses).mean() if losses else None
        if not train and reach_steps:
            for name, steps in reach_steps.items():
                for key, value_ in _draft_weighted(steps, generated).items():
                    metrics[f"drafts_{name}/{key}"] = value_
        return loss, metrics

    @staticmethod
    def _serving_weights(
        reach: list[torch.Tensor], generated: list[tuple[int, int]], picked: list[torch.Tensor]
    ) -> list[Optional[torch.Tensor]]:
        """Per draft step, each row's weight as serving counts it: the chance
        a draft starts at its anchor (``_draft_starts``) times the chance the
        draft is accepted up to the step (the reach before it, 1 at step 0),
        over the chain's expected number of drafts. Step 0's weights then sum
        to one, a later step's to its mean reach before it. From the head's
        own acceptance ratios on the data's tokens at T=0.6 (exact where the
        data was sampled that way), detached."""
        starts = torch.from_numpy(_draft_starts(reach, generated)).to(reach[0].device)
        drafts = float(starts.sum())
        weights: list[Optional[torch.Tensor]] = []
        before = torch.ones_like(starts)
        for step, rows in enumerate(picked):
            weights.append((starts[rows] * before[rows] / max(drafts, 1e-9)).detach())
            before = reach[step]
        return weights

    # --- updates ---------------------------------------------------------------

    def accumulate(self, chain: Loaded) -> tuple[float, dict[str, float]]:
        """Forward and backward of one microbatch, gradients added up."""
        loss, metrics = self.chain_loss(chain, train=True)
        if loss is None:
            return float("nan"), metrics
        (loss / self.args.accum).backward()
        return float(loss.detach()), metrics

    def apply(self, lr: float, ratio: bool) -> dict[str, float]:
        """Average the gradients over ranks, clip, step AdamW on the fp32
        masters, and copy them back into the bf16 weights."""
        grads = []
        for p, m in zip(self.params, self.masters):
            g = p.grad.float() if p.grad is not None else torch.zeros_like(m)
            if self.world > 1:
                dist.all_reduce(g)
                g /= self.world
            grads.append(g)
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        clip = min(1.0, self.args.clip / (float(norm) + 1e-6))
        snapshot = [m.detach().clone() for m in self.masters] if ratio else None
        for m, g in zip(self.masters, grads):
            m.grad = g.mul_(clip)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        out = {"grad_norm": float(norm), "lr": lr}
        if snapshot is not None:
            before = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(s) for s in snapshot]))
            delta = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(m - s) for m, s in zip(self.masters, snapshot)])
            )
            out["update_ratio"] = float(delta / before)
            del snapshot
        with torch.no_grad():
            for p, m in zip(self.params, self.masters):
                p.copy_(m)
                p.grad = None
        return out

    def state(self) -> dict[str, Any]:
        return {"masters": [m.detach().cpu() for m in self.masters], "optimizer": self.optimizer.state_dict()}

    def restore(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            for m, saved, p in zip(self.masters, state["masters"], self.params):
                m.copy_(saved)
                p.copy_(m)
        self.optimizer.load_state_dict(state["optimizer"])


# --- metrics ---------------------------------------------------------------------


def _draft_starts(steps: list[torch.Tensor], generated: list[tuple[int, int]]) -> np.ndarray:
    """Per anchor, the chance the engine starts a draft there.

    The engine drafts from where the last verify ended: a reply's first
    anchor is the slot before its first token, and a draft from anchor t that
    keeps j tokens puts the next anchor at t + 1 + j. The chances of keeping
    0..k tokens at each anchor are read off ``steps`` (per step, each
    anchor's reach; 0 where the step has no row)."""
    reach = np.stack([step.float().cpu().numpy() for step in steps])  # [k, anchors]
    k = reach.shape[0]
    weight = np.zeros(reach.shape[1])
    for start, end in generated:
        last = end - 3  # the last anchor with a token to draft inside the reply
        if last < start - 1:
            continue
        weight[start - 1] = 1.0
        for t in range(start - 1, last + 1):
            w = weight[t]
            if w == 0.0:
                continue
            kept = reach[:, t]
            previous = 1.0
            for j in range(k):
                if t + 1 + j <= last:
                    weight[t + 1 + j] += w * (previous - kept[j])
                previous = kept[j]
            if t + 1 + k <= last:
                weight[t + 1 + k] += w * previous
    return weight


def _draft_weighted(steps: list[torch.Tensor], generated: list[tuple[int, int]]) -> dict[str, float]:
    """The reach per draft position as the engine counts it: averaged over
    the drafts it makes (``_draft_starts``), not over every position. Hard
    stretches take many short drafts and easy ones few long ones, so per
    draft the hard stretches weigh more than per position. Sums, with
    ``rows`` the expected number of drafts."""
    starts = _draft_starts(steps, generated)
    sums = {"rows": float(starts.sum())}
    for j, step in enumerate(steps):
        sums[f"pos{j}"] = float((starts * step.float().cpu().numpy()).sum())
    return sums


def _add(total: dict[str, float], part: dict[str, float]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0.0) + value


def _averaged(sums: dict[str, float]) -> dict[str, float]:
    """Per-step sums -> means over rows."""
    out = {}
    steps = sorted({key.split("/")[0] for key in sums if "/" in key})
    for step in steps:
        rows = sums.get(f"{step}/rows", 0.0)
        for key, value in sums.items():
            if key.startswith(step + "/") and not key.endswith("/rows") and rows:
                out[key] = value / rows
        out[f"{step}/rows"] = rows
    for key, value in sums.items():
        if "/" not in key:
            out[key] = value
    return out


def _agreed(flags: list[bool], world: int, device: str) -> list[bool]:
    """Rank 0's decisions for every rank. A clock read on each rank can
    disagree across a boundary, and ranks that disagree on whether to save
    or stop meet different collectives and hang."""
    if world == 1:
        return flags
    tensor = torch.tensor([int(flag) for flag in flags], device=device)
    dist.broadcast(tensor, src=0)
    return [bool(value) for value in tensor.tolist()]


def _gathered(local: Any, world: int) -> list[Any]:
    if world == 1:
        return [local]
    everyone: list[Any] = [None] * world
    dist.all_gather_object(everyone, local)
    return everyone


def evaluate(trainer: Trainer, chains: list[ChainData], device: str, rank: int = 0, world: int = 1) -> dict[str, float]:
    """No update: per step, means over all rows and per harness. Ranks share
    the chains; every rank gets the result."""
    groups: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for loaded in prefetched(chains[rank::world], device):
            _, metrics = trainer.chain_loss(loaded, train=False)
            for group in ("all", loaded.chain.harness):
                _add(groups.setdefault(group, {}), metrics)
    merged: dict[str, dict[str, float]] = {}
    for part in _gathered(groups, world):
        for group, sums in part.items():
            _add(merged.setdefault(group, {}), sums)
    return {f"{group}/{key}": value for group, sums in merged.items() for key, value in _averaged(sums).items()}


# --- the run ---------------------------------------------------------------------


def _sets(specs: list[str]) -> list[ChainData]:
    chains: list[ChainData] = []
    for spec in specs:
        cache, dataset = spec.split(":", 1)
        chains += load_chains(Path(cache), Path(dataset))
    return chains


def _epoch(chains: list[ChainData], seed: int, epoch: int) -> list[ChainData]:
    """One epoch's order: shuffled, the harness with more trained tokens cut
    to the other's so the two weigh the same."""
    rng = random.Random(seed * 1000 + epoch)
    by_harness: dict[str, list[ChainData]] = {}
    for chain in chains:
        by_harness.setdefault(chain.harness, []).append(chain)
    budget = min(sum(c.trained for c in group) for group in by_harness.values())
    order: list[ChainData] = []
    for group in by_harness.values():
        group = rng.sample(group, len(group))
        spent = 0
        for chain in group:
            if spent >= budget and len(by_harness) > 1:
                break
            order.append(chain)
            spent += chain.trained
    return rng.sample(order, len(order))


def _schedule(update: int, total: int, args: argparse.Namespace) -> float:
    warmup = max(1, math.ceil(args.warmup_frac * total))
    if update < warmup:
        return args.lr * (update + 1) / warmup
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return args.lr * (args.lr_final_scale + (1 - args.lr_final_scale) * 0.5 * (1 + math.cos(math.pi * progress)))


def run(rank: int, world: int, port: int, args: argparse.Namespace) -> None:
    from torch.utils.tensorboard import SummaryWriter

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    if world > 1:
        import datetime

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
            timeout=datetime.timedelta(minutes=60),
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    started = time.monotonic()
    lead = rank == 0
    out: Path = args.out
    writer = log = None
    if lead:
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(exist_ok=True)
        (out / "state").mkdir(exist_ok=True)
        (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")
        writer = SummaryWriter(str(out / "tb"))
        log = (out / "metrics.jsonl").open("a")

    def record(kind: str, update: int, values: dict[str, Any]) -> None:
        if not lead:
            return
        for key, value in values.items():
            if isinstance(value, float) and math.isfinite(value):
                writer.add_scalar(f"{kind}/{key}", value, update)
        log.write(
            json.dumps({"kind": kind, "update": update, "elapsed_s": round(time.monotonic() - started, 1), **values})
            + "\n"
        )
        log.flush()
        writer.flush()

    def say(text: str) -> None:
        if lead:
            print(f"[train] {text}", flush=True)

    train = _sets(args.train)
    val = _sets(args.val) if args.val else []
    per_harness = {h: sum(c.trained for c in train if c.harness == h) for h in sorted({c.harness for c in train})}
    say(
        f"train {len(train)} chains, {sum(c.tokens for c in train):,} tokens, trained positions by harness {per_harness}; val {len(val)} chains"
    )
    head = load_head(device=device)
    if args.init_head is not None:
        load_trained(head, args.init_head)
    label = LabelHead(device=device)
    trainer = Trainer(head, label, args, world=world)
    say(
        f"{sum(p.numel() for p in trainer.params) / 1e9:.2f}B trainable parameters; loaded at {time.monotonic() - started:.0f}s"
    )

    per_update = world * args.accum
    epoch_chains = len(_epoch(train, args.seed, 0))
    total_updates = max(1, math.ceil(args.epochs * epoch_chains / per_update))
    update, epoch, cursor = 0, 0, 0
    state_path = out / "state" / "latest.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        trainer.restore(state["trainer"])
        update, epoch, cursor = state["update"], state["epoch"], state["cursor"]
        say(f"resumed at update {update} (epoch {epoch}, chain {cursor})")
    say(f"{epoch_chains} chains an epoch, {per_update} an update: {total_updates} updates planned")

    def validate(at: int) -> None:
        if val:
            result = evaluate(trainer, val, device, rank, world)
            record("val", at, result)
            say(
                f"val at {at}: "
                + " ".join(
                    f"{k}={v:.4f}"
                    for k, v in result.items()
                    if k.startswith("all/") and ("accept_t0.6" in k or k.endswith("/kl"))
                )
            )

    def checkpoint(at: int) -> None:
        if world > 1:
            dist.barrier()
        if lead:
            from drafter.files import save_file

            directory = out / "checkpoints" / f"update-{at:06d}"
            directory.mkdir(parents=True, exist_ok=True)
            save_file(export_state(head), str(directory / "head.safetensors"))
            meta = {
                "update": at,
                "epoch": epoch,
                "cursor": cursor,
                "elapsed_s": round(time.monotonic() - started, 1),
                "path": str(directory / "head.safetensors"),
            }
            (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            with (out / "checkpoints" / "index.jsonl").open("a") as index:
                index.write(json.dumps(meta) + "\n")
            if args.resume_state:
                tmp = state_path.with_suffix(".tmp")
                torch.save({"trainer": trainer.state(), "update": at, "epoch": epoch, "cursor": cursor}, tmp)
                tmp.replace(state_path)
            say(f"checkpoint {directory}")
        if world > 1:
            dist.barrier()

    if lead:
        # A resumed run has more checkpoints coming: the watcher waits again.
        (out / "DONE").unlink(missing_ok=True)
    validated = -1
    if update == 0:
        validate(0)
        validated = 0
    last_save = time.monotonic()
    saved_at = -1
    deadline = started + args.minutes * 60
    stop = False
    while update < total_updates and not stop:
        order = _epoch(train, args.seed, epoch)
        mine = [order[i] for i in range(cursor, len(order)) if (i - cursor) % world == rank]
        stream = prefetched(mine, device)
        while update < total_updates and cursor + per_update <= len(order):
            tick = time.monotonic()
            sums: dict[str, float] = {}
            losses, tokens = [], 0
            for _ in range(args.accum):
                loaded = next(stream)
                loss, metrics = trainer.accumulate(loaded)
                losses.append(loss)
                tokens += loaded.chain.tokens
                _add(sums, metrics)
            ratio = args.ratio_every > 0 and update % args.ratio_every == 0
            applied = trainer.apply(_schedule(update, total_updates, args), ratio)
            torch.cuda.synchronize()
            update += 1
            cursor += per_update
            merged: dict[str, float] = {}
            all_losses: list[float] = []
            all_tokens = 0
            for part_sums, part_losses, part_tokens in _gathered((sums, losses, tokens), world):
                _add(merged, part_sums)
                all_losses += part_losses
                all_tokens += part_tokens
            seconds = time.monotonic() - tick
            metrics = _averaged(merged)
            finite = [x for x in all_losses if math.isfinite(x)]
            metrics.update(
                applied,
                loss=sum(finite) / max(1, len(finite)),
                seconds=seconds,
                tokens_per_s=all_tokens / seconds,
                epoch=epoch + cursor / len(order),
                max_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            record("train", update, metrics)
            say(
                f"update {update}/{total_updates}: loss {metrics['loss']:.4f} "
                + " ".join(f"a{s}={metrics.get(f'step{s}/accept_t0.6', float('nan')):.3f}" for s in range(args.steps))
                + f" grad {metrics['grad_norm']:.3f} lr {metrics['lr']:.2e} {seconds:.1f}s {metrics['tokens_per_s']:,.0f} tok/s"
            )
            if args.val_every and update % args.val_every == 0:
                validate(update)
                validated = update
            save, stop = _agreed(
                [time.monotonic() - last_save >= args.save_minutes * 60, time.monotonic() >= deadline], world, device
            )
            if save:
                checkpoint(update)
                last_save, saved_at = time.monotonic(), update
            if stop:
                break
        if cursor + per_update > len(order):
            epoch, cursor = epoch + 1, 0

    if validated != update:
        validate(update)
    if saved_at != update:
        checkpoint(update)
    if lead:
        (out / "DONE").write_text(
            json.dumps(
                {
                    "update": update,
                    "total_updates": total_updates,
                    "complete": update >= total_updates,
                    "elapsed_s": round(time.monotonic() - started, 1),
                }
            )
            + "\n"
        )
        say(f"done: {update} updates in {time.monotonic() - started:.0f}s")
        writer.close()
        log.close()
    if world > 1:
        dist.destroy_process_group()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--train", action="append", required=True, help="CACHE:DATASET, repeatable")
    parser.add_argument("--val", action="append", default=[], help="CACHE:DATASET, repeatable")
    parser.add_argument("--gpus", type=int, default=0, help="data-parallel ranks; 0: every visible GPU")
    parser.add_argument("--accum", type=int, default=1, help="microbatches (chains) per rank per update")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3, help="draft steps trained (k)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--lr-final-scale", type=float, default=0.1, help="cosine decays to lr times this")
    parser.add_argument("--router-lr-scale", type=float, default=0.1)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    # The distillation's temperature: serving drafts and verifies at 0.6, and
    # acceptance there reads the tempered pair alone. Its gradient scales
    # with the temperature, so an LR matched to a T=1 run is divided by it.
    parser.add_argument("--loss-temperature", type=float, default=1.0)
    parser.add_argument(
        "--draft-weighting",
        action="store_true",
        help="weight each (anchor, step) row as serving counts it: the chance a draft starts there x its reach before the step",
    )
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument(
        "--kv-fp8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="attend over FP8 per-token-head K/V, as the served drafter does",
    )
    parser.add_argument(
        "--checkpoint-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute each draft step in the backward",
    )
    parser.add_argument("--minutes", type=float, default=600.0, help="wall-clock budget for updates")
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--save-minutes", type=float, default=30.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the optimizer state with each checkpoint",
    )
    parser.add_argument("--ratio-every", type=int, default=25, help="updates between update-ratio readings")
    parser.add_argument("--init-head", type=Path, default=None, help="start from a trained head (head.safetensors)")
    parser.add_argument("--eager-attention", action="store_true", help="uncompiled FlexAttention (CPU tests)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    world = args.gpus or torch.cuda.device_count()
    if world <= 1:
        run(0, 1, 0, args)
        return
    import socket

    # Data-parallel training needs no serving engine or harness imports.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(run, args=(world, port, args), nprocs=world, join=True)


if __name__ == "__main__":
    main()
