"""Flash SFT expert_lora_sharded: rank-sliced routed adapters train as the replicated ones.

On CPU collectives (gloo) with 2 and 4 ranks, each rank feeding its own tokens, the sliced layout gives the
replicated layout's outputs and gradients, the trainer's reduction, clip and HostAdamW step give the replicated
update, and a checkpoint written by the trainer (full adapters.pt, shared optimizer.pt, one optimizer-rankR.pt per
rank) resumes to the same slices and optimizer state.
"""

import datetime
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch import nn

from tests.gloo_ranks import spawn_ranks
from training.flash import expert_lora_shard as shard
from training.flash.expert_lora import install_expert_lora, lora_delta

LORA = {"rank": 8, "alpha": 16.0, "dropout": 0.0}


def experts():
    torch.manual_seed(7)
    m = nn.Module()
    m.gate_up_proj = nn.Parameter(torch.randn(3, 32, 16), requires_grad=False)
    m.down_proj = nn.Parameter(torch.randn(3, 16, 16), requires_grad=False)
    adapters = install_expert_lora(m, LORA)
    with torch.no_grad():
        for name, p in adapters.items():
            if name.endswith("_B"):
                p.normal_(std=0.05, generator=torch.Generator().manual_seed(len(name)))
    dense = nn.Parameter(torch.randn(4, 5, generator=torch.Generator().manual_seed(1)))
    return m, {"dense": dense, **{f"layers.0.mlp.experts.{k}": p for k, p in adapters.items()}}


def grouped(inputs, w, offsets):
    starts = [0] + offsets.tolist()
    return torch.cat([inputs[starts[i]:starts[i + 1]] @ w[i].T for i in range(len(w))])


def forward_backward(m, dense, rank):
    """A routed-expert MLP over grouped rows (base GEMMs plus the LoRA deltas) and a dense adapter."""
    g = torch.Generator().manual_seed(100 + rank)  # every rank its own tokens
    counts = torch.randint(1, 5, (3,), generator=g)
    offsets = torch.cumsum(counts, 0).to(torch.int32)
    x = torch.randn(int(offsets[-1]), 16, generator=g)
    up = grouped(x, m.gate_up_proj, offsets) + torch.cat(
        [lora_delta(m, name, x, offsets, grouped) for name in ("gate", "up")], dim=-1)
    gate, value = up.chunk(2, -1)
    out = grouped(torch.nn.functional.silu(gate) * value, m.down_proj, offsets)
    out = out + lora_delta(m, "down", torch.nn.functional.silu(gate) * value, offsets, grouped)
    out = out.sum() * 40.0 + (dense * torch.randn(dense.shape, generator=g)).sum() * 40.0  # norm above 1: clips
    out.backward()
    return out.detach()


def trainer(trainables, rank, world, run_dir):
    """A FlashSFT with only what the update and checkpoint methods use."""
    from training.flash_sft import FlashSFT, HostAdamW

    t = object.__new__(FlashSFT)
    t.trainables, t.rank, t.world, t.run_dir = trainables, rank, world, Path(run_dir)
    t.sharded = {i for i, p in enumerate(trainables.values()) if shard.is_rank_shard(p)}
    t.config = {"optimizer": {"max_grad_norm": 1.0}, "seed": 1}
    t.device = torch.device("cpu")
    t.first_nll = None
    t.optimizer = HostAdamW(list(trainables.values()), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    return t


def step(t):
    """FlashSFT._train_update after the backward: reduce, clip, step."""
    t._reduce_gradients()
    if t.sharded:
        norm = shard.clip_grad_norm_(list(t.trainables.values()), 1.0)
    else:
        norm = torch.nn.utils.clip_grad_norm_(list(t.trainables.values()), 1.0)
    t.optimizer.step()
    t.optimizer.zero_grad(set_to_none=True)
    return norm


def expected_slice(full, name, sliced, rank):
    return shard.rank_slice(full, sliced, rank) if shard.is_rank_shard(sliced) else full


def _checks(rank, world, rendezvous, out_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=60))
    cuda_rng = mock.patch.multiple(torch.cuda, get_rng_state=lambda device=None: torch.zeros(1, dtype=torch.uint8),
                                   set_rng_state=lambda state, device=None: None)
    try:
        full_module, full = experts()
        reference = trainer(full, rank, world, Path(out_dir) / "replicated")
        sliced_module, sliced = experts()
        shard.shard_expert_adapters({k: p for k, p in sliced.items() if k != "dense"}, rank, world)
        t = trainer(sliced, rank, world, Path(out_dir) / "sliced")
        assert t.sharded == set(range(1, len(sliced)))
        for name, p in sliced.items():
            assert torch.equal(p.detach(), expected_slice(full[name].detach(), name, p, rank)), name

        for update in range(3):
            out_full = forward_backward(full_module, full["dense"], rank + 10 * update)
            out_sliced = forward_backward(sliced_module, sliced["dense"], rank + 10 * update)
            if update == 0:
                assert torch.equal(out_full, out_sliced)
            else:
                # The replicated layout sums the gradients in flattened buckets, the slices per tensor: the same
                # sums in another order, so later updates agree to FP32 rounding.
                torch.testing.assert_close(out_sliced, out_full, rtol=1e-5, atol=1e-5)
            norm_full, norm_sliced = step(reference), step(t)
            assert norm_full > 1
            torch.testing.assert_close(norm_sliced, norm_full, rtol=1e-6, atol=0)
            for name, p in sliced.items():
                torch.testing.assert_close(p.detach(), expected_slice(full[name].detach(), name, p, rank),
                                           rtol=1e-6, atol=1e-8, msg=name)

        t.first_nll = 1.366
        with cuda_rng:
            t._save(3)
            reference._save(3)
        checkpoint = t.run_dir / "checkpoints" / "update-000003"
        assert (checkpoint / f"optimizer-rank{rank}.pt").is_file()
        assert not list((reference.run_dir / "checkpoints" / "update-000003").glob("optimizer-rank*.pt"))
        # adapters.pt holds the full adapters, as the replicated layout writes them.
        saved = torch.load(checkpoint / "adapters.pt")
        for name, p in full.items():
            torch.testing.assert_close(saved[name], p.detach(), rtol=1e-6, atol=1e-8)

        _, fresh = experts()
        shard.shard_expert_adapters({k: p for k, p in fresh.items() if k != "dense"}, rank, world)
        with torch.no_grad():
            for p in fresh.values():
                p.zero_()
        resumed = trainer(fresh, rank, world, t.run_dir)
        with cuda_rng:
            resumed._resume()
        assert resumed.update == 3
        assert resumed.first_nll == 1.366
        for name, p in fresh.items():
            assert torch.equal(p.detach(), sliced[name].detach()), name
        before, after = t.optimizer.state_dict(), resumed.optimizer.state_dict()
        assert set(before["state"]) == set(after["state"]) == set(range(len(sliced)))
        for index, values in before["state"].items():
            for key, value in values.items():
                assert torch.equal(after["state"][index][key], value), (index, key)
        for master, p in zip(resumed.optimizer.masters, fresh.values()):
            assert torch.equal(master, p.detach())
        (Path(out_dir) / f"passed-{rank}").write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("world", [2, 4])
def test_sliced_adapters_train_as_replicated(world):
    with tempfile.TemporaryDirectory() as tmp:
        spawn_ranks(_checks, world, str(Path(tmp) / "rendezvous"), tmp, nprocs=world)
        assert sorted(p.name for p in Path(tmp).glob("passed-*")) == [f"passed-{r}" for r in range(world)]


def test_rank_must_divide_and_module_unchanged_without_flag():
    m, adapters = experts()
    routed = {k.rsplit(".", 1)[-1]: p for k, p in adapters.items() if k != "dense"}
    with pytest.raises(ValueError, match="divide"):
        shard.shard_expert_adapters(dict(routed), 0, 3)
    assert not any(shard.is_rank_shard(p) for p in m.parameters())
    assert shard.adapter_weight(m, "lora_gate_A") is m.lora_gate_A


def test_validation():
    shard.validate_expert_lora_sharded(False, None)
    shard.validate_expert_lora_sharded(False, LORA)
    shard.validate_expert_lora_sharded(True, LORA)
    for args in (("yes", LORA), (None, LORA), (True, None)):
        with pytest.raises(ValueError):
            shard.validate_expert_lora_sharded(*args)
