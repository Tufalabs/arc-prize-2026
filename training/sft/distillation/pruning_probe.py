"""Fixed held-out router probe; temporary joint ablation never edits weights."""

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from sft.distillation.common import Dataset, dump, row_key, sha, validate_cache
from sft.distillation.loss import fp32_logits
from sft.distillation.pruning_metrics import bottom, layer_metrics, route_overlap
from sft.distillation.routing import RouteRecorder, attach


def validate_probe(config):
    c = config["probe"]
    if c["every"] < 1 or c["window"] < 2 or c["sample_tokens"] < 1:
        raise ValueError("invalid probe schedule/window/sample count")
    if not 0 < c["bottom_fraction"] < 1:
        raise ValueError("invalid diagnostic pruning fraction")
    for key in ("min_persistence", "max_window_kl_range"):
        if not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError("invalid probe threshold")
    if c["min_persistence"] > 1:
        raise ValueError("invalid persistence threshold")
    if c.get("max_ablation_delta_kl") is not None and (
        not math.isfinite(c["max_ablation_delta_kl"]) or c["max_ablation_delta_kl"] < 0
    ):
        raise ValueError("invalid ablation budget")
    data = Dataset(c["data"])
    if sha(c["data"]) != c["data_sha256"]:
        raise ValueError("fixed probe data changed")
    manifest = json.loads(Path(c["manifest"]).read_text())
    if (
        sha(c["manifest"]) != c["manifest_sha256"]
        or manifest["dataset_sha256"] != c["data_sha256"]
    ):
        raise ValueError("fixed probe manifest changed")
    if sha(manifest["split"]) != manifest["split_sha256"]:
        raise ValueError("probe split changed")
    split = json.loads(Path(manifest["split"]).read_text())["summary"]
    allowed = set(split["eval"]["games"]) - set(split["train"]["games"])
    games, keys = set(), set()
    for i in range(len(data)):
        row = data[i]
        if row["game_id"] not in allowed:
            raise ValueError("probe row outside held-out game split")
        games.add(row["game_id"])
        keys.add(row_key(row))
    train = Dataset(config["data"])
    for i in range(len(train)):
        row = train[i]
        if row["game_id"] in games or row_key(row) in keys:
            raise ValueError("training/probe leakage")
    return c


class ProbeRecorder(RouteRecorder):
    def __init__(self, layers, experts, device, sample_tokens):
        super().__init__(layers, experts, device)
        self.sample_tokens = sample_tokens
        self.routes = {}

    def begin(self, prompt_len):
        super().begin(prompt_len)
        self.routes = {}

    @torch.no_grad()
    def record(self, layer, indices, weights):
        if not self.enabled:
            return
        super().record(layer, indices, weights)
        idx = indices[self.prompt_len - 1 : -1]
        positions = torch.linspace(
            0, len(idx) - 1, min(self.sample_tokens, len(idx)), device=idx.device
        ).long()
        self.routes[layer] = idx[positions].to(device="cpu", dtype=torch.int16)


@contextmanager
def temporary_pruning(gates, candidates):
    old = [getattr(g, "probe_excluded_experts", None) for g in gates]
    try:
        if len(gates) != len(candidates):
            raise ValueError("candidate layer count mismatch")
        for gate, ids in zip(gates, candidates, strict=True):
            if (
                len(set(ids)) != len(ids)
                or gate.weight.shape[0] - len(ids) < gate.top_k
            ):
                raise ValueError("ablation must retain at least top-k experts")
            mask = torch.zeros(
                gate.weight.shape[0], dtype=torch.bool, device=gate.weight.device
            )
            mask[ids] = True
            gate.probe_excluded_experts = mask
        yield
    finally:
        for gate, value in zip(gates, old, strict=True):
            gate.probe_excluded_experts = value


@torch.no_grad()
def score(hidden, head, teacher_hidden, teacher_head, targets, temperature, chunk):
    if (
        len(hidden) != len(teacher_hidden)
        or len(hidden) != len(targets)
        or not len(targets)
    ):
        raise ValueError("probe completion alignment mismatch")
    if temperature <= 0 or not math.isfinite(temperature) or chunk < 1:
        raise ValueError("invalid probe objective")
    sums = torch.zeros(4, device=hidden.device, dtype=torch.float64)
    for start in range(0, len(targets), chunk):
        h = hidden[start : start + chunk]
        logits = fp32_logits(h, head)
        teacher = fp32_logits(
            teacher_hidden[start : start + chunk].to(h.device), teacher_head
        )
        logp = (logits / temperature).log_softmax(-1)
        logq = (teacher / temperature).log_softmax(-1)
        sums[0] += (
            ((logq.exp() * (logq - logp)).sum(-1) * temperature**2).sum().double()
        )
        sums[1] += F.cross_entropy(
            logits, targets[start : start + chunk], reduction="sum"
        ).double()
        sums[2] += (logits.argmax(-1) == teacher.argmax(-1)).sum()
        sums[3] += len(h)
    return sums


class PruningProbe:
    def __init__(
        self,
        config,
        student,
        model,
        head,
        teacher_head,
        temperature,
        chunk,
        out,
        recorder,
    ):
        self.config, self.model, self.head, self.teacher_head = (
            config,
            model,
            head,
            teacher_head,
        )
        self.temperature, self.chunk, self.out, self.train_recorder = (
            temperature,
            chunk,
            out,
            recorder,
        )
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.dataset = Dataset(config["data"])
        self.cache = Path(config["cache"])
        self.manifest = validate_cache(self.cache, config["data"], student)
        self.identity = {
            "config": config,
            "cache_sha256": sha(self.cache / "manifest.json"),
        }
        self.gates = [m for n, m in model.named_modules() if n.endswith("mlp.gate")]
        self.device = head.weight.device
        self.experts = self.gates[0].weight.shape[0]
        self.k = config.get("bottom_k") or max(
            1, math.ceil(config["bottom_fraction"] * self.experts)
        )
        if not 0 < self.k <= self.experts - self.gates[0].top_k:
            raise ValueError("probe ablation budget leaves fewer than top-k experts")
        self.recorder = ProbeRecorder(
            len(self.gates), self.experts, self.device, config["sample_tokens"]
        )
        self.state = {
            "identity": self.identity,
            "step": -1,
            "history": [],
            "routes": {},
        }

    def restore(self, state):
        if state["identity"] != self.identity:
            raise ValueError("resume probe identity changed")
        self.state = state

    def forward_pass(self, capture):
        total = torch.zeros(4, dtype=torch.float64, device=self.device)
        counts = torch.zeros_like(self.recorder.counts)
        mass = torch.zeros_like(self.recorder.mass)
        routes = {}
        for base in range(0, len(self.dataset), self.world):
            index = base + self.rank
            real = index < len(self.dataset)
            index = min(index, len(self.dataset) - 1)
            row = self.dataset[index]
            target = self.cache / f"{index:08d}.safetensors"
            entry = self.manifest["entries"][str(index)]
            if (
                row_key(row) != entry["row_sha256"]
                or sha(target) != entry["file_sha256"]
            ):
                raise ValueError("probe teacher cache corruption")
            teacher = load_file(target, device="cpu")["hidden"]
            prompt = len(row["input_token_ids"])
            ids = torch.tensor(
                [row["input_token_ids"] + row["output_token_ids"]], device=self.device
            )
            self.recorder.begin(prompt)
            self.recorder.enabled = capture and real
            try:
                hidden = self.model(input_ids=ids, use_cache=False).last_hidden_state
            finally:
                self.recorder.finish()
            if real:
                total += score(
                    hidden[0, prompt - 1 : -1],
                    self.head.weight,
                    teacher,
                    self.teacher_head,
                    ids[0, prompt:],
                    self.temperature,
                    self.chunk,
                )
                if capture:
                    counts += self.recorder.counts
                    mass += self.recorder.mass
                    routes[str(index)] = torch.stack(
                        [self.recorder.routes[i] for i in range(len(self.gates))]
                    )
            del hidden, ids, teacher
        for tensor in (total, counts, mass):
            dist.all_reduce(tensor)
        if not torch.isfinite(total).all() or total[3] <= 0:
            raise RuntimeError("nonfinite or empty probe")
        return total, counts[1].cpu().tolist(), mass[1].cpu().tolist(), routes

    @torch.no_grad()
    def evaluate(self, step, *, record_history=True):
        if step <= self.state["step"]:
            return
        began = time.monotonic()
        flags = [(m, m.training) for m in self.model.modules()]
        try:
            self.model.eval()
            attach(self.model, self.recorder)
            with (
                torch.random.fork_rng(
                    devices=[self.device.index] if self.device.type == "cuda" else []
                ),
                sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
            ):
                total, counts, masses, routes = self.forward_pass(True)
                baseline = (total[:3] / total[3]).tolist()
                history = self.state["history"][-(self.config["window"] - 1) :]
                report = {
                    "step": step,
                    "scheduled": record_history,
                    "data_sha256": self.config["data_sha256"],
                    "rows": len(self.dataset),
                    "completion_tokens": total[3].item(),
                    "bottom_k": self.k,
                    "baseline_kl": baseline[0],
                    "baseline_ce": baseline[1],
                    "teacher_top1_agreement": baseline[2],
                    "criteria": {},
                }
                snapshot = {"frequency": counts, "mass": masses, "kl": baseline[0]}
                for name, values in (("frequency", counts), ("mass", masses)):
                    layers = [
                        layer_metrics(
                            v,
                            history[-1][name][i] if history else None,
                            [h[name][i] for h in history],
                            self.k,
                        )
                        for i, v in enumerate(values)
                    ]
                    candidates = [layer["candidates"] for layer in layers]
                    with temporary_pruning(self.gates, candidates):
                        ablated, _, _, _ = self.forward_pass(False)
                    ablated = (ablated[:3] / ablated[3]).tolist()
                    report["criteria"][name] = {
                        "layers": layers,
                        "ablation_kl": ablated[0],
                        "ablation_ce": ablated[1],
                        "ablation_delta_kl": ablated[0] - baseline[0],
                        "ablation_delta_ce": ablated[1] - baseline[1],
                        "ablation_teacher_top1_agreement": ablated[2],
                        "min_overlap_previous": min(
                            (x["overlap_previous"] for x in layers), default=None
                        )
                        if history
                        else None,
                        "min_candidate_persistence": min(
                            x["candidate_persistence_min"] for x in layers
                        ),
                    }
                overlap = torch.zeros(
                    len(self.gates), 2, device=self.device, dtype=torch.float64
                )
                for key, value in routes.items():
                    if key in self.state["routes"]:
                        shared = route_overlap(value, self.state["routes"][key])
                        overlap[:, 0] += shared.sum(-1).to(self.device)
                        overlap[:, 1] += shared.shape[-1]
                dist.all_reduce(overlap)
                report["token_topk_overlap"] = [
                    (a / b).item() if b else None for a, b in overlap
                ]
                report["frequency_mass_candidate_overlap"] = [
                    len(set(bottom(c, self.k)) & set(bottom(m, self.k))) / self.k
                    for c, m in zip(counts, masses, strict=True)
                ]
                losses = [h["kl"] for h in history] + [baseline[0]]
                report["window_kl_range"] = max(losses) - min(losses)
                report["window_evaluations"] = len(losses)
                budget = self.config.get("max_ablation_delta_kl")
                report["advisory_ready"] = bool(
                    record_history
                    and budget is not None
                    and len(losses) >= self.config["window"]
                    and report["window_kl_range"] <= self.config["max_window_kl_range"]
                    and all(
                        v["min_candidate_persistence"] >= self.config["min_persistence"]
                        and v["ablation_delta_kl"] <= budget
                        for v in report["criteria"].values()
                    )
                )
                report["ablation_budget_configured"] = budget is not None
                report["seconds"] = time.monotonic() - began
                if record_history:
                    self.state = {
                        "identity": self.identity,
                        "step": step,
                        "history": (history + [snapshot])[-self.config["window"] :],
                        "routes": routes,
                    }
                if self.rank == 0:
                    dump(self.out / f"probe-step{step}.json", report)
        finally:
            self.recorder.finish()
            attach(self.model, self.train_recorder)
            for module, training in flags:
                module.training = training
