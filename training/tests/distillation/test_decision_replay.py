import unittest
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from training.flash.adapters import FlashDecoderLayer, FP32TopKRouter
from training.flash.options import FlashOptions
from training.flash.model import router_forward
from training.flash.decision_replay import DecisionTape, expert_topk, qsa_blocks
from training.flash.qsa_selection import select_blocks, selection_contexts

from concurrent.futures import ThreadPoolExecutor
from training.flash.offload import checkpoint_layer
from training.flash.dense_lora import dense_lora_policy


class DecisionReplayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_default_and_live_router_gradients(self):
        logits = torch.randn(7, 19, requires_grad=True)
        probs = logits.softmax(-1)
        actual = expert_topk(probs, 3)
        expected = probs.topk(3, -1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        tape = DecisionTape()
        with tape.record():
            recorded, indices = expert_topk(probs, 3, 4, (3, 4))
        self.assertEqual(tape.nbytes, 7 * 3 * 2)
        self.assertEqual(tape.entries["experts", 4].indices.dtype, torch.int16)
        for normalize in (False, True):
            changed = (-logits.detach()).requires_grad_()
            with tape.replay():
                weights, forced = expert_topk(changed.softmax(-1), 3, 4, (3, 4))
            reference = changed.softmax(-1).gather(-1, indices)
            if normalize:
                weights = weights / weights.sum(-1, keepdim=True)
                reference = reference / reference.sum(-1, keepdim=True)
            torch.testing.assert_close(forced, indices, rtol=0, atol=0)
            torch.testing.assert_close(weights, reference, rtol=0, atol=0)
            a = torch.autograd.grad(weights.square().sum(), changed, retain_graph=True)[
                0
            ]
            b = torch.autograd.grad(reference.square().sum(), changed)[0]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertGreater(float(a.abs().sum()), 0)
        torch.testing.assert_close(recorded, expected.values, rtol=0, atol=0)

    def test_qsa_packed_padding_and_cache(self):
        indexer = SimpleNamespace(
            layer_idx=3, compress_ratio=4, token_budget=8, block_topk=2
        )
        x = torch.randn(1, 7, 4, requires_grad=True)
        chosen = torch.tensor([[0, -1]] * 7, dtype=torch.int32)
        calls = []

        def compute():
            calls.append(1)
            return chosen

        def layer(value):
            ids, _ = select_blocks(
                indexer, value, lambda: qsa_blocks(indexer, value, compute, (3, 4))
            )
            return value.sin().square().sum() * (ids[:, 0].float().sum() + 1)

        tape = DecisionTape()
        with tape.record():
            checkpoint(
                layer, x, use_reentrant=False, context_fn=selection_contexts
            ).backward()
        self.assertEqual(len(calls), 1)
        expected_grad = x.grad.clone()
        x.grad = None
        with tape.replay():
            checkpoint(
                layer, x, use_reentrant=False, context_fn=selection_contexts
            ).backward()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)
        torch.testing.assert_close(tape.entries["qsa", 3].indices, chosen)
        with tape.replay(), self.assertRaisesRegex(RuntimeError, "layout/shape"):
            qsa_blocks(indexer, x, compute, (4, 3))

    def test_expert_checkpoint_recompute_and_nested_scopes(self):
        x = torch.randn(7, 19, requires_grad=True)

        def layer(value):
            weights, _ = expert_topk(value.softmax(-1), 3, 0)
            return weights.square().sum()

        tape = DecisionTape()
        with tape.record():
            checkpoint(layer, x, use_reentrant=False).backward()
        grad = x.grad.clone()
        x.grad = None
        with tape.replay():
            with DecisionTape().record():
                expert_topk(x.softmax(-1), 3, 1)
            checkpoint(layer, x, use_reentrant=False).backward()
        torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
        self.assertEqual(set(tape.entries), {("experts", 0)})
        # Context restoration also applies on exceptions.
        with self.assertRaisesRegex(RuntimeError, "Missing replay"), tape.replay():
            expert_topk(x.softmax(-1), 3, 1)
        expert_topk(x.softmax(-1), 3)

    def test_checkpoint_tape_survives_worker_thread_and_scope_exit(self):

        class Router(torch.nn.Module):
            def forward(self, x):
                weights, _ = expert_topk(x.softmax(-1), 3, 0)
                return weights.square().sum()

        for reuse in (False, True):
            layer = checkpoint_layer(Router(), 0, 0, reuse_qsa_selection=reuse)
            x = torch.randn(7, 19, requires_grad=True)
            expected = torch.autograd.grad(Router()(x), x)[0]
            tape = DecisionTape()
            with tape.record():
                output = layer(x)
            # Mirror CUDA autograd: recompute on a different thread, after the
            # caller's tape scope has ended. The checkpoint owns its snapshot.
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(output.backward, retain_graph=True).result()
                pool.submit(output.backward).result()
            torch.testing.assert_close(x.grad, 2 * expected, rtol=0, atol=0)

    def test_reject_mismatch_and_mutation(self):
        tape = DecisionTape()
        probs = torch.randn(7, 19).softmax(-1)
        with tape.record():
            expert_topk(probs, 3, 1, (3, 4))
        with self.assertRaisesRegex(RuntimeError, "fresh"), tape.record():
            pass
        with tape.replay():
            with self.assertRaisesRegex(RuntimeError, "layout/shape"):
                expert_topk(probs, 3, 1, (4, 3))
            with self.assertRaisesRegex(ValueError, "Invalid packed"):
                expert_topk(probs, 3, 1, (3, 3))
            tape.entries["experts", 1].indices.add_(1)
            with self.assertRaisesRegex(RuntimeError, "modified in place"):
                expert_topk(probs, 3, 1, (3, 4))

    def test_actual_router_hook(self):
        for normalize in (False, True):
            router = FP32TopKRouter(
                Qwen4ExpTextConfig(
                    hidden_size=16,
                    num_experts=19,
                    num_experts_per_tok=3,
                    norm_topk_prob=normalize,
                )
            ).float()
            router.layer_idx = 7
            with torch.no_grad():
                router.weight.normal_()
            x = torch.randn(1, 7, 16)
            ordinary = router(x)
            tape = DecisionTape()
            with tape.record():
                recorded = router(x)
            for a, b in zip(ordinary, recorded):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            legacy_tape = DecisionTape()
            with legacy_tape.record():
                legacy = router_forward(router, x)
            self.assertEqual(set(legacy_tape.entries), {("experts", 7)})
            for a, b in zip(recorded, legacy):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with tape.replay():
                logits, weights, indices = router(-x)
                expected = logits.float().softmax(-1).gather(-1, ordinary[2])
                if normalize:
                    expected = expected / expected.sum(-1, keepdim=True)
                torch.testing.assert_close(weights, expected, rtol=0, atol=0)
                weights.square().sum().backward()
            self.assertGreater(float(router.weight.grad.abs().sum()), 0)
            torch.testing.assert_close(indices, ordinary[2], rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and the Flash runtime")
class DecisionReplayCudaTest(unittest.TestCase):
    def test_decoder_checkpoint_and_retained_attention(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        config = Qwen4ExpTextConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=24,
            num_key_value_heads=2,
            head_dim=256,
            intermediate_size=512,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=4,
            num_experts_per_tok=2,
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,
            },
            ple_layer_ids=[],
        )
        config._attn_implementation = "sdpa"
        config._experts_implementation = "grouped_mm"
        layer = (
            FlashDecoderLayer(
                config,
                3,
                flex_qsa=True,
                fp32_router=True,
                chunk_rule=None,
                causal_conv=None,
                fuse_norm_tail=False,
                options=FlashOptions(qsa_kernel="flex"),
            )
            .cuda()
            .bfloat16()
        )
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.normal_(std=0.02)
                parameter.requires_grad_(False)
        layer.mlp.gate.float()
        dense_lora_policy(4, 8).configure_layer(layer, 3)
        n = 256
        x = torch.randn(
            1, n, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        rotary = (
            torch.ones(1, n, 128, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, n, 128, device="cuda", dtype=torch.bfloat16),
        )
        for kernel in ("flex", "group_sparse"):
            layer.self_attn.qsa_kernel = kernel
            for lengths in (None, (128, 128)):
                for retain in (False, True):
                    # Exercise the offload wrapper together with packed replay
                    # and retained attention; document contexts are reconstructed
                    # from each checkpoint invocation's kwargs on CUDA workers.
                    offload = int(kernel == "group_sparse" and lengths is not None and retain)
                    wrapped = checkpoint_layer(
                        layer, 3, offload, reuse_qsa_selection=True, retain_qsa_attention=retain
                    )
                    tape = DecisionTape()
                    with tape.record():
                        output = wrapped(x, position_embeddings=rotary, packed_lengths=lengths)
                        self.assertEqual(set(tape.entries), {("qsa", 3), ("experts", 3)})
                        for entry in tape.entries.values():
                            self.assertEqual(entry.signature[1], lengths)
                        expected = torch.autograd.grad(output.float().square().sum(), x)[0]
                    with tape.replay():
                        actual_output = wrapped(x, position_embeddings=rotary, packed_lengths=lengths)
                        actual = torch.autograd.grad(actual_output.float().square().sum(), x)[0]
                    torch.testing.assert_close(actual_output, output, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)



if __name__ == "__main__":
    unittest.main()
