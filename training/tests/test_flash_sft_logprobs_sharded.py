"""Recorded CE+KL updates agree with replicated adapters on two and four ranks."""

import datetime
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sft.logprob.targets import ByteVocabulary, prepare_row
from tests.gloo_ranks import spawn_ranks
from tests.logprob import conftest as teacher_fixtures
from tests.logprob.test_targets_and_loss import row_and_steps
from tests.test_flash_sft_expert_lora_sharded import (
    expected_slice,
    experts,
    grouped,
    step,
    trainer,
)
from training.flash.expert_lora import lora_delta
from training.flash.expert_lora_shard import shard_expert_adapters
from training.flash_sft import parameter_digest


def hidden(module, dense, row, rank, update):
    generator = torch.Generator().manual_seed(100 + rank + update * 10)
    ids = torch.tensor(row["input_token_ids"] + row["output_token_ids"])
    x = torch.randn(len(ids), 16, generator=generator)
    offsets = torch.tensor([2, 4, 6], dtype=torch.int32)
    up = grouped(x, module.gate_up_proj, offsets) + torch.cat(
        [lora_delta(module, name, x, offsets, grouped) for name in ("gate", "up")],
        dim=-1,
    )
    gate, value = up.chunk(2, -1)
    activated = torch.nn.functional.silu(gate) * value
    out = grouped(activated, module.down_proj, offsets)
    out = out + lora_delta(module, "down", activated, offsets, grouped)
    return (out + dense.sum() * 0.01).unsqueeze(0), ids


def checks(rank, world, rendezvous, out_dir, tokenizer):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=world,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        vocabulary = ByteVocabulary(tokenizer)
        row, steps = row_and_steps()
        row = prepare_row(row, steps, vocabulary)
        spec = {"weight": 0.3, "tokenizer_sha256": vocabulary.sha256, "require_logprobs": True}
        reference_module, full = experts()
        sliced_module, sliced = experts()
        shard_expert_adapters({n: p for n, p in sliced.items() if n != "dense"}, rank, world)
        reference = trainer(full, rank, world, Path(out_dir) / "full")
        actual = trainer(sliced, rank, world, Path(out_dir) / "sliced")
        head = torch.randn(8, 16, generator=torch.Generator().manual_seed(19))
        for run in (reference, actual):
            run.recorded_logprobs = spec
            run.config["head_chunk"] = 2
            run.head = SimpleNamespace(weight=head)
        for update in range(3):
            losses = []
            for run, module in ((reference, reference_module), (actual, sliced_module)):
                run._hidden = lambda r, module=module, run=run, update=update: hidden(
                    module, run.trainables["dense"], r, rank, update
                )
                loss = run._row_loss(row)
                losses.append(loss.detach())
                (loss / (4 * world)).backward()
            torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-5)
            norms = [step(run) for run in (reference, actual)]
            torch.testing.assert_close(norms[0], norms[1], rtol=2e-6, atol=1e-6)
            for name, parameter in sliced.items():
                # Bucketed and per-tensor reductions sum FP32 gradients in different layouts;
                # three AdamW steps can differ by a few FP32 ulps, as in the baseline test.
                torch.testing.assert_close(
                    parameter, expected_slice(full[name], name, parameter, rank),
                    rtol=2e-6, atol=1e-7,
                    msg=lambda message, name=name, update=update: f"update {update+1}, {name}: {message}",
                )
            digest = parameter_digest(sliced)
            digests = [None] * world
            dist.all_gather_object(digests, digest)
            assert digests == [digest] * world
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("world", [2, 4])
def test_recorded_loss_shards_match_replicated(world, tmp_path):
    teacher_fixtures.vocab.__wrapped__(tmp_path)
    with tempfile.TemporaryDirectory() as temporary:
        spawn_ranks(
            checks, world, str(Path(temporary) / "rendezvous"), temporary,
            str(tmp_path / "tokenizer.json"), nprocs=world,
        )
