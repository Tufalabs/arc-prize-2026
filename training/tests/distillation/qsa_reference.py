"""FP32 QSA oracles, separate from group union/reverse-adjacency construction."""

import torch


@torch.no_grad()
def dense_reference(q, k, v, grad, chosen, lengths=None):
    """Materialize one document/head mask at a time (qualification sizes only)."""
    lengths = (q.shape[2],) if lengths is None else lengths
    out, dq = torch.zeros_like(q, dtype=torch.float32), torch.zeros_like(q, dtype=torch.float32)
    dk, dv = torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32)
    lse = torch.empty(q.shape[1:3], device=q.device, dtype=torch.float32)
    start, offset = 0, 0
    for n in lengths:
        pos = torch.arange(n, device=q.device)
        blocks = chosen[start:start + n].long() - offset
        selected = torch.zeros((n, (n + 3) // 4 + 1), device=q.device, dtype=torch.bool)
        selected.scatter_(1, torch.where(blocks >= 0, blocks, (n + 3) // 4), True)
        allow = (pos[None] <= pos[:, None]) & (
            (pos[:, None] < 2051) | selected[:, pos // 4]
            | (pos[None] >= ((pos[:, None] + 1) // 4) * 4)
        )
        for h in range(24):
            kh = h // 12
            # Autograd through unfused FP32 matmuls/softmax is independent of
            # every hand-written attention adjoint under test.
            with torch.enable_grad():
                a, b, c = (t[0, hh, start:start + n].detach().float().requires_grad_()
                           for t, hh in ((q, h), (k, kh), (v, kh)))
                do = grad[0, h, start:start + n].float()
                score = (a @ b.T / 16).masked_fill(~allow, -torch.inf)
                norm = score.logsumexp(-1)
                o = score.softmax(-1).nan_to_num(nan=0.0) @ c
                ga, gb, gc = torch.autograd.grad(o, (a, b, c), do)
            out[0, h, start:start + n] = o.detach()
            lse[h, start:start + n] = norm.detach() * 1.4426950408889634
            dq[0, h, start:start + n] = ga
            dk[0, kh, start:start + n] += gb
            dv[0, kh, start:start + n] += gc
        start += n
        offset += (n + 3) // 4
    return out, lse, dq, dk, dv


@torch.no_grad()
def errors(actual, reference):
    finite = torch.isfinite(reference)
    error = torch.where(finite, actual.float() - reference.float(),
                        torch.where(actual == reference, 0., torch.inf))
    norm = reference.float().masked_fill(~finite, 0).norm().clamp_min(1e-30)
    return {
        "relative_l2": float(error.norm() / norm),
        "max_abs": float(error.abs().max()),
    }


@torch.no_grad()
def sampled_reference(q, k, v, grad, chosen, queries, blocks, batch=16):
    """Exact FP32 sampled rows and key blocks, recomputing every needed row.

    Key gradients require all queries selecting a sampled key. For those queries
    we gather every allowed key, recompute FP32 softmax/output/delta, then sum the
    sampled-key contributions. No approximate/group-kernel LSE is reused.
    """
    device, n = q.device, q.shape[2]
    queries = torch.tensor(queries, device=device, dtype=torch.long)
    blocks = torch.tensor(blocks, device=device, dtype=torch.long)
    positions = torch.arange(n, device=device)
    needed = torch.zeros(n, device=device, dtype=torch.bool)
    needed[queries] = True
    for block in blocks:
        needed |= (chosen == block).any(-1)
        needed |= (positions < 2051) & (positions >= block * 4)
        needed |= (((positions + 1) // 4) == block) & ((positions + 1) % 4 != 0)
    rows = needed.nonzero().flatten()
    key_positions = (blocks[:, None] * 4 + torch.arange(4, device=device)).flatten()
    out = torch.empty((24, len(queries), 256), device=device)
    lse = torch.empty((24, len(queries)), device=device)
    dq = torch.empty_like(out)
    dk = torch.zeros((2, len(key_positions), 256), device=device)
    dv = torch.zeros_like(dk)
    for start in range(0, len(rows), batch):
        row = rows[start:start + batch]
        selected = chosen[row].long()
        keys = (selected[:, :, None] * 4 + torch.arange(4, device=device)).flatten(1)
        tail = ((row + 1) // 4)[:, None] * 4 + torch.arange(3, device=device)
        keys = torch.cat((keys, tail), 1)
        keys = torch.where(row[:, None] < 2051, torch.arange(2051, device=device)[None], keys)
        valid = (keys >= 0) & (keys <= row[:, None]) & (keys < n)
        safe = keys.clamp(0, n - 1)
        selected_queries = (row[:, None] == queries[None]).nonzero()
        for kv in range(2):
            heads = slice(kv * 12, (kv + 1) * 12)
            a = q[0, heads, row].transpose(0, 1).float()
            do = grad[0, heads, row].transpose(0, 1).float()
            b, c = k[0, kv, safe].float(), v[0, kv, safe].float()
            score = (torch.bmm(a, b.transpose(1, 2)) / 16).masked_fill(~valid[:, None], -torch.inf)
            p = score.softmax(-1).nan_to_num(nan=0.0)
            o = torch.bmm(p, c)
            ds = p * (torch.bmm(do, c.transpose(1, 2)) - (o * do).sum(-1, keepdim=True))
            for local, sample in selected_queries.tolist():
                out[heads, sample] = o[local]
                lse[heads, sample] = score[local].logsumexp(-1) * 1.4426950408889634
                dq[heads, sample] = ds[local] @ b[local] / 16
            for index, key in enumerate(key_positions):
                match = (keys == key) & valid
                probability = (p * match[:, None]).sum(-1)
                derivative = (ds * match[:, None]).sum(-1)
                dk[kv, index] += (derivative[:, :, None] * a).sum((0, 1)) / 16
                dv[kv, index] += (probability[:, :, None] * do).sum((0, 1))
    return (out, lse, dq, dk, dv), queries, key_positions, len(rows)
