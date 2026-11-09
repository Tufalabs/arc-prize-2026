"""Versioned useful student matmul/conv FLOP estimate, not hardware FLOP counters.

QSA counts semantically selected causal edges, not masked-out SDPA work.
DeltaNet uses recurrent-form matmul counts, not chunk algorithm/recompute work.
Pointwise activations, norms, softmax, top-k, optimizer and communication excluded.
"""

SCHEMA = "flash-router-useful-matmul-conv-v1"


def sum_floor(n, divisor):
    """sum(j // divisor for j in range(1, n + 1))."""
    q, r = divmod(n, divisor)
    return divisor * q * (q - 1) // 2 + q * (r + 1)


def qsa_edges(tokens, budget, ratio):
    if ratio < 1 or budget < 1 or budget % ratio:
        raise ValueError("invalid QSA block budget")
    prefix = min(tokens, budget + ratio - 1)
    tail_sum = (
        tokens * (tokens + 1) // 2
        - prefix * (prefix + 1) // 2
        - ratio * (sum_floor(tokens, ratio) - sum_floor(prefix, ratio))
    )
    return prefix * (prefix + 1) // 2 + budget * (tokens - prefix) + tail_sum


def useful_model_flops(config, tokens, completion_tokens):
    """One rank, one sequence; all body tokens, completion-only frozen head.

    All non-router parameters are frozen. Before layer0's router there is no
    differentiable input; later body operators propagate input gradients.
    Multiply+add counts as two FLOPs. The teacher and rematerialization are absent.
    """
    c = config
    if not 0 < completion_tokens < tokens:
        raise ValueError("need a nonempty prompt and completion")
    if (
        len(c["layer_types"]) != c["num_hidden_layers"]
        or c.get("model_type") != "qwen4_exp_text"
    ):
        raise ValueError("unsupported model layout")
    d, m, shared = (
        c["hidden_size"],
        c["moe_intermediate_size"],
        c["shared_expert_intermediate_size"],
    )
    e, k = c["num_experts"], c["num_experts_per_tok"]
    if not 0 < k <= e:
        raise ValueError("invalid MoE top-k")
    hc, low = c["hc_count"], c["hc_lowrank"]
    parts = {
        "experts": 0,
        "routers": 0,
        "attention_projections": 0,
        "attention_core": 0,
        "qsa_indexer": 0,
        "hyperconnections": 0,
        "ple": 0,
        "convolution": 0,
        "output_head": 0,
    }
    for layer, kind in enumerate(c["layer_types"]):
        # Frozen GEMMs need forward + input gradient only after the first router.
        factor = 1 if layer == 0 else 2
        parts["experts"] += factor * 6 * tokens * d * (k * m + shared)
        parts["experts"] += factor * 2 * tokens * d  # shared expert scalar gate
        # Router weight gradient always; input gradient only after layer0.
        parts["routers"] += (2 if layer == 0 else 3) * 2 * tokens * d * e
        parts["hyperconnections"] += (
            2 * factor * 2 * tokens * (2 * hc * d * low + hc * d * hc)
        )
        if layer + 1 in c["ple_layer_ids"]:
            # Embedding->key/value are frozen branches: forward only.
            parts["ple"] += 2 * tokens * c["ple_embed_dim"] * (hc * d + d)
            parts["convolution"] += (
                factor * 2 * tokens * hc * d * c["ple_conv_kernel_size"]
            )
        if kind == "linear_attention":
            keys = c["linear_num_key_heads"] * c["linear_key_head_dim"]
            values = c["linear_num_value_heads"] * c["linear_value_head_dim"]
            heads = c["linear_num_value_heads"]
            # qkv,z,a,b,out projections.
            parts["attention_projections"] += (
                factor * 2 * tokens * d * (2 * keys + 3 * values + 2 * heads)
            )
            parts["convolution"] += (
                factor * 2 * tokens * (2 * keys + values) * c["linear_conv_kernel_dim"]
            )
            # k^T S, rank-one update, q^T S: 6KV forward, 12KV backward.
            parts["attention_core"] += (
                (6 if layer == 0 else 18)
                * tokens
                * heads
                * c["linear_key_head_dim"]
                * c["linear_value_head_dim"]
            )
        elif kind == "full_attention":
            heads, hd, kv = (
                c["num_attention_heads"],
                c["head_dim"],
                c["num_key_value_heads"],
            )
            # q projection also generates the output gate.
            parts["attention_projections"] += (
                factor * 2 * tokens * d * hd * (3 * heads + 2 * kv)
            )
            budget, ratio = c["indexer_budget"], c["indexer_compress_ratio"]
            edges = qsa_edges(tokens, budget, ratio)
            parts["attention_core"] += (4 if layer == 0 else 12) * heads * hd * edges
            prefix = min(tokens, budget + ratio - 1)
            if tokens > prefix:
                ih, ikv, ihd = (
                    c["indexer_n_heads"],
                    c["indexer_kv_heads"],
                    c["indexer_head_dim"],
                )
                if ikv != 1:
                    raise ValueError("unsupported QSA indexer layout")
                # Index selection is no-grad, including after layer0.
                parts["qsa_indexer"] += 2 * tokens * d * (ih + ikv) * ihd
                parts["qsa_indexer"] += (
                    2 * ih * ihd * (sum_floor(tokens, ratio) - sum_floor(prefix, ratio))
                )
        else:
            raise ValueError("unsupported attention type")
    # Final hyperconnection mixer has two frozen projections, no injection gate.
    parts["hyperconnections"] += 4 * tokens * (2 * hc * d * low)
    # Student logits + input gradient; teacher head and repeated logits excluded.
    parts["output_head"] = 4 * completion_tokens * d * c["vocab_size"]
    return {
        "schema": SCHEMA,
        "parts": parts,
        "useful_model_flops_estimate": sum(parts.values()),
    }


def aggregate_compute(rows):
    """Sum rank work and divide by the slowest rank's elapsed time."""
    if not all("compute" in row for row in rows):
        return {}
    reports = [row["compute"] for row in rows]
    if any(report["schema"] != SCHEMA for report in reports):
        raise ValueError("FLOP estimator schema mismatch")
    flops = sum(report["useful_model_flops_estimate"] for report in reports)
    seconds = max(report["accounting_seconds"] for report in reports)
    if seconds <= 0:
        raise ValueError("invalid compute accounting duration")
    result = {
        "train/useful_model_flops_estimate": flops,
        "train/useful_model_flops_per_second_estimate": flops / seconds,
    }
    return result
