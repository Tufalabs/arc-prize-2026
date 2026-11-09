import unittest

from sft.distillation.compute_metrics import (
    SCHEMA,
    aggregate_compute,
    qsa_edges,
    sum_floor,
    useful_model_flops,
)


def config():
    return {
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "hidden_size": 4,
        "moe_intermediate_size": 3,
        "shared_expert_intermediate_size": 2,
        "num_experts": 5,
        "num_experts_per_tok": 2,
        "hc_count": 2,
        "hc_lowrank": 1,
        "ple_layer_ids": [2],
        "ple_embed_dim": 4,
        "ple_conv_kernel_size": 4,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 2,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 4,
        "num_attention_heads": 2,
        "head_dim": 2,
        "num_key_value_heads": 1,
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 2,
        "vocab_size": 11,
    }


class ComputeMetricsTests(unittest.TestCase):
    def test_qsa_semantic_edges_against_explicit_selection(self):
        for n in range(1, 50):
            for ratio in (1, 2, 4):
                budget = 3 * ratio
                count = 0
                for j in range(1, n + 1):
                    complete, tail = divmod(j, ratio)
                    count += min(complete, budget // ratio) * ratio + tail
                self.assertEqual(qsa_edges(n, budget, ratio), count)
                self.assertEqual(
                    sum_floor(n, ratio), sum(j // ratio for j in range(1, n + 1))
                )

    def test_experts_frozen_backward_and_completion_head(self):
        c = config()
        report = useful_model_flops(c, 20, 3)
        # Layer0 has forward only; layer1 has forward+input gradient, no dW.
        per_forward = 6 * 20 * 4 * (2 * 3 + 2) + 2 * 20 * 4
        self.assertEqual(report["parts"]["experts"], 3 * per_forward)
        self.assertEqual(report["parts"]["routers"], (2 + 3) * 2 * 20 * 4 * 5)
        self.assertEqual(report["parts"]["output_head"], 4 * 3 * 4 * 11)
        other = useful_model_flops(c, 20, 7)
        self.assertEqual(
            other["useful_model_flops_estimate"]
            - report["useful_model_flops_estimate"],
            4 * (7 - 3) * 4 * 11,
        )
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(
            report["useful_model_flops_estimate"], sum(report["parts"].values())
        )

    def test_head_has_no_teacher_or_recomputation_factor(self):
        c = config()
        c["offload_every"] = 3
        a = useful_model_flops(c, 100, 5)
        c["offload_every"] = 1
        self.assertEqual(a, useful_model_flops(c, 100, 5))
        self.assertEqual(a["parts"]["output_head"], 4 * 5 * 4 * 11)
        with self.assertRaises(ValueError):
            useful_model_flops(c, 5, 5)
        c["layer_types"] = ["bad", "full_attention"]
        with self.assertRaises(ValueError):
            useful_model_flops(c, 10, 5)

    def test_aggregate_sum_flops_over_max_seconds(self):
        rows = [
            {
                "compute": {
                    "schema": SCHEMA,
                    "useful_model_flops_estimate": f,
                    "accounting_seconds": t,
                }
            }
            for f, t in [(100, 1), (200, 2), (300, 3), (400, 4)]
        ]
        result = aggregate_compute(rows)
        self.assertEqual(result["train/useful_model_flops_estimate"], 1000)
        self.assertEqual(result["train/useful_model_flops_per_second_estimate"], 250)
        self.assertEqual(len(result), 2)
        self.assertEqual(aggregate_compute([{}] * 4), {})
        rows[0]["compute"]["schema"] = "unknown"
        with self.assertRaises(ValueError):
            aggregate_compute(rows)
