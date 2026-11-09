"""Production-width GDN + QSA + nonzero dense LoRA, with frozen decisions."""

import json
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from tests.distillation.flash_test_model import build_test_model, snapshot_kernels
from training.flash.decision_replay import DecisionTape, qsa_blocks

from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextExperts, apply_rotary_pos_emb,
)


class EagerAttention(nn.Module):
    """Unfused attention oracle; masks are independent of sparse kernel metadata."""

    def __init__(self, attention):
        super().__init__()
        self.inner = attention

    def forward(self, x, position_embeddings, **kwargs):
        attn = self.inner
        lengths = kwargs.get("packed_lengths")

        def missing():
            raise AssertionError("The FP32 oracle must replay BF16 decisions")

        chosen = qsa_blocks(attn.indexer, x, missing, lengths)
        shape = (*x.shape[:-1], -1, attn.head_dim)
        q, gate = attn.q_proj(x).view(*x.shape[:-1], -1, 2 * attn.head_dim).chunk(2, -1)
        q = attn.q_norm(q.reshape(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        outputs = []
        start = 0
        for length in lengths or (x.shape[1],):
            pos = torch.arange(length, device=x.device)
            ids = chosen[start:start + length].long() - start // 4
            selected = torch.zeros((length, (length + 3) // 4 + 1), device=x.device, dtype=torch.bool)
            selected.scatter_(1, torch.where(ids >= 0, ids, (length + 3) // 4), True)
            allow = (pos[None] <= pos[:, None]) & (
                (pos[:, None] < 2051) | selected[:, pos // 4]
                | (pos[None] >= (pos[:, None] + 1) // 4 * 4)
            )
            key = k[:, :, start:start + length].repeat_interleave(12, 1)
            value = v[:, :, start:start + length].repeat_interleave(12, 1)
            scores = (q[:, :, start:start + length] @ key.transpose(-1, -2)) * attn.scaling
            outputs.append(scores.masked_fill(~allow, -torch.inf).softmax(-1) @ value)
            start += length
        output = torch.cat(outputs, 2).transpose(1, 2).reshape(*x.shape[:-1], -1)
        return attn.o_proj(output * gate.reshape_as(output).sigmoid()), None


class EagerExperts(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.inner = experts

    def forward(self, x, indices, weights):
        # Bypass HF's grouped-GEMM dispatch: differentiate its unfused reference.
        return Qwen4ExpTextExperts.forward.__wrapped__(self.inner, x, indices, weights)


class EagerGDN(nn.Module):
    """FP32 convolution, recurrence and gated norm using only autograd ops."""

    def __init__(self, gdn):
        super().__init__()
        self.inner = gdn

    def forward(self, x, **kwargs):
        gdn = self.inner
        lengths = kwargs.get("packed_lengths") or (x.shape[1],)
        projected = gdn.in_proj_qkv(x).transpose(1, 2)
        convolved = [F.silu(F.conv1d(part, gdn.conv1d.weight, gdn.conv1d.bias,
                                   groups=projected.shape[1], padding=gdn.conv_kernel_size - 1))[..., :n]
                     for part, n in zip(projected.split(lengths, -1), lengths)]
        q, k, v = torch.cat(convolved, -1).transpose(1, 2).split((2048, 2048, 6144), -1)
        q, k = (t.unflatten(-1, (16, 128)).repeat_interleave(3, 2) for t in (q, k))
        q, k = (t * torch.rsqrt(t.square().sum(-1, keepdim=True) + 1e-6) for t in (q, k))
        q = q / 128**0.5
        v = v.unflatten(-1, (48, 128))
        beta = gdn.in_proj_b(x).sigmoid()
        decay = (-gdn.A_log.exp() * F.softplus(gdn.in_proj_a(x) + gdn.dt_bias)).exp()
        rows, start = [], 0
        for length in lengths:
            state = x.new_zeros(1, 48, 128, 128)
            for i in range(start, start + length):
                state = state * decay[:, i, :, None, None]
                delta = (v[:, i] - (state * k[:, i, :, :, None]).sum(-2)) * beta[:, i, :, None]
                state = state + k[:, i, :, :, None] * delta[..., None, :]
                rows.append((state * q[:, i, :, :, None]).sum(-2))
            start += length
        core = torch.stack(rows, 1)
        core = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + gdn.norm.variance_epsilon)
        core = core * gdn.norm.weight * gdn.in_proj_z(x).view_as(core).sigmoid()
        return gdn.out_proj(core.flatten(-2))


def run(bundle, tokens, upstream, packed):
    inputs = []

    def capture(module, args, output):
        output.requires_grad_().retain_grad()
        inputs.append(output)

    handle = bundle.model.embed_tokens.register_forward_hook(capture)
    kwargs = {"packed_lengths": (128, 128)} if packed else {}
    output = bundle.model(tokens, use_cache=False, **kwargs).last_hidden_state
    output.backward(upstream.to(output.dtype))
    handle.remove()
    result = {"output": output.detach().float(), "input_gradient": inputs[0].grad.float()}
    for name, parameter in bundle.trainable.items():
        if parameter.grad is None or not parameter.grad.count_nonzero():
            raise AssertionError(f"Missing or vacuous LoRA gradient: {name}")
        result[name] = parameter.grad.detach().float().clone()
    return result


@unittest.skipUnless(torch.cuda.is_available(), "CUDA and the pinned Flash runtime required")
class ComposedFlashTests(unittest.TestCase):
    def test_fast_options_against_fp32_with_packed_documents(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Keep FLA tuning identical between BF16 implementations; the FP32
        # recurrence remains independent pure Torch autograd.
        self.enterContext(snapshot_kernels())
        for packed in (False, True):
            with self.subTest(packed=packed):
                torch.manual_seed(20261001)
                tokens = torch.arange(256, device="cuda")[None]
                upstream = torch.randn(1, 256, 2560, device="cuda", dtype=torch.bfloat16)
                tape = DecisionTape()
                # Flex is the pre-existing attention path that supports decision replay.
                control = dict(attention_backend="flex_qsa", qsa_kernel="flex",
                               expert_optimization="none", fuse_norm_tail=False)
                eager = build_test_model(ple=True, **control)
                with tape.record():
                    baseline = run(eager, tokens, upstream, packed)
                del eager
                self.assertEqual(set(tape.entries), {("experts", 0), ("experts", 1), ("qsa", 1)})
                reference = build_test_model(ple=True, **control)
                reference.model.float()
                # Assign replacements to the decoder itself, not the forwarding
                # checkpoint wrapper. The oracle needs no activation checkpointing.
                reference.model.layers = nn.ModuleList([
                    layer._checkpoint_wrapped_module for layer in reference.model.layers
                ])
                for layer in reference.model.layers:
                    if layer.ple is not None:
                        # The file-backed embedding always returns BF16. Its
                        # frozen values must feed the FP32 reference projections.
                        layer.ple.ple_embedding.register_forward_hook(
                            lambda module, args, output: output.float())
                    layer.mlp.experts = EagerExperts(layer.mlp.experts)
                    if hasattr(layer, "linear_attn"):
                        layer.linear_attn = EagerGDN(layer.linear_attn)
                    else:
                        layer.self_attn = EagerAttention(layer.self_attn)
                with tape.replay():
                    oracle = run(reference, tokens, upstream, packed)
                del reference
                # The compatible maximal configuration uses HF rounding, repeated
                # GDN heads, BF16 storage and reference-precision decoder compilation.
                candidate = build_test_model(fast=True, ple=True, compile_layers=True)
                with tape.replay():
                    actual = run(candidate, tokens, upstream, packed)
                del candidate
                self.assertEqual(actual.keys(), oracle.keys())
                for name in oracle:
                    base = baseline[name] - oracle[name]
                    error = actual[name] - oracle[name]
                    for metric, reduce in (("rms", lambda x: x.square().mean().sqrt()),
                                           ("max", lambda x: x.abs().max())):
                        eb, ea = reduce(base).item(), reduce(error).item()
                        print(json.dumps(dict(composed=name, packed=packed, metric=metric,
                                              eager_error=eb, fast_error=ea)), flush=True)
                        with self.subTest(tensor=name, metric=metric):
                            self.assertLessEqual(ea, 1.5 * eb, (packed, name, metric, ea, eb))
