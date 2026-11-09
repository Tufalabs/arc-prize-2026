import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sft.distillation.host_memory import HostMemoryMonitor, find_memory_cgroup, sample
from sft.distillation.train import worker_command
from training.flash.offload import checkpoint_layer, should_offload


class OffloadTests(unittest.TestCase):
    def test_policy_counts_and_invalid_values(self):
        for every, count in [(0, 0), (1, 48), (2, 24), (4, 12)]:
            self.assertEqual(sum(should_offload(i, every) for i in range(48)), count)
        for invalid in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                should_offload(0, invalid)

    def test_checkpoint_loss_and_gradient_match(self):
        torch.manual_seed(19)
        base = torch.nn.Sequential(
            torch.nn.Linear(7, 11), torch.nn.GELU(), torch.nn.Linear(11, 7)
        )
        test = copy.deepcopy(base)
        wrapped = checkpoint_layer(test, 0, 0)
        x = torch.randn(13, 7, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        a = base(x).square().sum()
        b = wrapped(y).square().sum()
        a.backward()
        b.backward()
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(x.grad, y.grad)
        for p, q in zip(base.parameters(), test.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad)

    def test_chunked_offload_matches_the_whole_tensor_copy(self):
        from torch.autograd.graph import saved_tensors_hooks

        from training.flash.offload import chunked_save_on_cpu, slab_lengths, validate_offload_chunk

        self.assertIsNone(validate_offload_chunk(None))
        self.assertEqual(validate_offload_chunk(256), 256)
        for invalid in [0, 3, 1.0, True, "256"]:
            with self.assertRaises(ValueError):
                validate_offload_chunk(invalid)
        # A 112,089-token boundary: 2.14 GiB whole (rounded to 4), 9 slabs of 256 MiB.
        dim, lengths = slab_lengths((1, 112089, 10240), 2, 256 << 20)
        self.assertEqual((dim, len(lengths), lengths[0]), (1, 9, 13107))
        self.assertEqual(sum(lengths), 112089)
        self.assertEqual(slab_lengths((), 2, 8), (None, [1]))
        self.assertEqual(slab_lengths((3, 4), 2, 1 << 20), (None, [1]))
        self.assertEqual(slab_lengths((2, 11, 3), 4, 48), (1, [2] * 5 + [1]))
        self.assertEqual(slab_lengths((1, 5, 100), 8, 16), (2, [1] * 100))

        class whole_copy(saved_tensors_hooks):
            def __init__(self):
                def pack(t):
                    return torch.empty(t.size(), dtype=t.dtype).copy_(t)

                super().__init__(pack, lambda p: p)

        def run(context):
            torch.manual_seed(1)
            x = torch.randn(3, 37, 5, dtype=torch.float64, requires_grad=True)
            w = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
            with context:
                h = (x @ w).tanh()
                h = (h.transpose(0, 2).contiguous() ** 3).sum() + (h.transpose(0, 2) * x.transpose(0, 2)).sum()
            h.backward()
            return h, x.grad, w.grad

        reference = run(whole_copy())
        for chunk in (64, 200, 1 << 30):
            for a, b in zip(reference, run(chunked_save_on_cpu(chunk)), strict=True):
                self.assertTrue(torch.equal(a, b))
        hooks = chunked_save_on_cpu(64)
        tensor = torch.randn(4, 9, 7).transpose(0, 2)
        packed = hooks.pack_hook(tensor)
        self.assertEqual(len(packed[3]), 9)
        self.assertTrue(torch.equal(hooks.unpack_hook(packed), tensor))
        wrapped = checkpoint_layer(torch.nn.Linear(7, 7), 0, 1, offload_chunk_mib=1)
        y = torch.randn(13, 7, requires_grad=True)
        wrapped(y).square().sum().backward()
        self.assertEqual(list(wrapped.state_dict()), ["weight", "bias"])  # same keys as the framework wrapper
        with self.assertRaises(ValueError):
            checkpoint_layer(torch.nn.Linear(2, 2), 0, 1, offload_chunk_mib=3)

    def test_only_selected_layers_offload(self):
        with patch(
            "training.flash.offload.offload_wrapper",
            side_effect=lambda x: x,
        ) as offload:
            for i in range(4):
                checkpoint_layer(torch.nn.Linear(2, 2), i, 2)
        self.assertEqual(offload.call_count, 2)

    def test_host_memory_reads_job_limit_and_events(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job = root / "job"
            leaf = job / "task"
            leaf.mkdir(parents=True)
            proc = root / "cgroup"
            proc.write_text("0::/job/task\n")
            (leaf / "memory.max").write_text("max")
            for name, value in [
                ("memory.max", "1024"),
                ("memory.current", "512"),
                ("memory.peak", "768"),
                ("memory.stat", "anon 400\nfile 112\n"),
                ("memory.events", "oom_kill 1\n"),
            ]:
                (job / name).write_text(value)
            found = find_memory_cgroup(proc, root)
            self.assertEqual(found, job)
            result = sample(found)
            self.assertEqual(result["current_bytes"], 512)
            self.assertEqual(result["memory.events"]["oom_kill"], 1)
            out = root / "memory.jsonl"
            with patch(
                "sft.distillation.host_memory.find_memory_cgroup", return_value=job
            ):
                monitor = HostMemoryMonitor(out, interval=0.01)
                monitor.start()
                monitor.stop()
            self.assertEqual(
                json.loads(out.read_text().splitlines()[0])["limit_bytes"], 1024
            )

    def test_policy_reaches_worker(self):
        c = {
            "student": "s",
            "data": "d",
            "cache": "c",
            "steps": 8,
            "lr": 1e-5,
            "temperature": 1,
            "ce_weight": 0,
            "chunk": 64,
            "save_every": 2,
            "seed": 20260916,
            "offload_every": 2,
        }
        cmd = worker_command(c, Path("/result"))
        self.assertEqual(cmd[cmd.index("--offload-every") + 1], "2")
