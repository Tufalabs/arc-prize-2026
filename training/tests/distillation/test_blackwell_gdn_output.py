"""GPU acceptance: the guarded FLA chunk rule is deterministic and exact on Blackwell.

FLA 0.5.1's eight-warp BK=BV=128 ``chunk_fwd_kernel_o`` configuration returns
nondeterministic, wrong outputs on B200 (Triton 3.7.1): about 40% relative error
against the exact recurrence, starting at token 32 of each 64-token chunk. The
autotuner picked it for the learner's packed (varlen) prefill, so two identical
learner forwards disagreed by ~0.3 nats on sampled tokens. The check runs in a
fresh interpreter because the guard must precede the first kernel launch.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch

SCRIPT = textwrap.dedent(
    """
    import json
    import torch
    from training.blackwell_gdn import configure_blackwell_gdn

    device = torch.device("cuda", 0)
    assert configure_blackwell_gdn("fla", device=device)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import fla.ops.common.chunk_o as chunk_o

    torch.manual_seed(0)
    T, H, HV, K = 512, 16, 48, 128
    q = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, T, HV, K, device=device, dtype=torch.bfloat16)
    beta = torch.rand(1, T, HV, device=device, dtype=torch.bfloat16)
    g = -torch.nn.functional.softplus(torch.randn(1, T, HV, device=device)) * 0.1
    cu = torch.tensor([0, T], device=device, dtype=torch.int32)
    scale = K ** -0.5

    def call():
        return chunk_gated_delta_rule(q, k, v, g=g, beta=beta, scale=scale, initial_state=None,
                                      output_final_state=False, use_qk_l2norm_in_kernel=True,
                                      cu_seqlens=cu)[0].float()

    # Exact recurrence in float64: S <- exp(g) S; S += k beta (v - S^T k)^T; o = scale q^T S.
    qn = torch.nn.functional.normalize(q.double(), dim=-1).repeat_interleave(HV // H, 2)[0]
    kn = torch.nn.functional.normalize(k.double(), dim=-1).repeat_interleave(HV // H, 2)[0]
    state = torch.zeros(HV, K, K, dtype=torch.float64, device=device)
    reference = torch.empty(T, HV, K, dtype=torch.float64, device=device)
    for t in range(T):
        state = state * g[0, t].double().exp()[:, None, None]
        delta = (v[0, t].double() - torch.einsum("hk,hkv->hv", kn[t], state)) * beta[0, t].double()[:, None]
        state = state + kn[t][:, :, None] * delta[:, None, :]
        reference[t] = torch.einsum("hk,hkv->hv", qn[t] * scale, state)
    reference = reference[None].float()

    def check():
        outputs = [call() for _ in range(4)]
        return dict(
            bitwise=all(torch.equal(outputs[0], o) for o in outputs[1:]),
            rel_err=max(float((o - reference).norm() / reference.norm()) for o in outputs),
        )

    tuner = chunk_o.chunk_fwd_kernel_o
    while not hasattr(tuner, "configs"):
        tuner = tuner.fn
    results = {"autotuned": check(), "warps": [c.num_warps for c in tuner.configs]}
    kept = list(tuner.configs)
    for config in kept:
        tuner.configs = [config]
        tuner.cache.clear()
        results[f"BK{config.kwargs['BK']}_w{config.num_warps}"] = check()
    print("RESULT " + json.dumps(results))
    """
)


def _blackwell_with_fla():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        return False
    try:
        import fla  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_blackwell_with_fla(), "requires a Blackwell GPU and FLA")
class GuardedChunkOutputTest(unittest.TestCase):
    def test_guarded_chunk_rule_is_deterministic_and_exact(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, TRITON_CACHE_DIR=cache, FLA_CACHE_RESULTS="0",
                       PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
            completed = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True,
                                       text=True, timeout=900, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr[-3000:])
        line = next(
            entry for entry in completed.stdout.splitlines() if entry.startswith("RESULT ")
        )
        results = json.loads(line.removeprefix("RESULT "))
        self.assertNotIn(8, results.pop("warps"))
        for label, result in results.items():
            self.assertTrue(result["bitwise"], f"{label}: repeated identical calls differ")
            self.assertLess(result["rel_err"], 0.02, f"{label}: {result['rel_err']:.4f} vs exact recurrence")


if __name__ == "__main__":
    unittest.main()
