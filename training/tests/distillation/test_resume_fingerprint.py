"""Resuming a checkpoint saved under the retired NVFP4 expert storage.

Such a checkpoint's frozen fingerprint hashed the packed NVFP4 experts, which
the build now expands to BF16. The resume hashes the student checkpoint's
packed experts again as that storage did and compares them with the saved
hashes: the student's config, index and lineage files, which the manifest
pins, do not cover the expert shards' contents.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from sft.distillation.accumulation import (
    nvfp4_expert_digests,
    validate_resume_fingerprint,
    validate_resume_manifest,
)
from sft.distillation.common import sha
from tests.gloo_ranks import run_rank
from training.flash.reader import Reader

LAYERS, EXPERTS, HIDDEN, INTERMEDIATE, WORLD = 3, 5, 32, 16, 4
SOURCE = {"packed": ".weight", "scale": ".weight_scale", "outer": ".weight_scale_2"}
NVFP4 = [
    f"{projection}_{part}" for projection in ("gate", "up", "down") for part in SOURCE
]
BF16 = ["gate_up_proj", "down_proj"]
# What an NVFP4-era worker recorded in its checkpoint manifest.
LEGACY = {
    "expert_storage": "nvfp4",
    "nvfp4_distribution": "replicated",
    "model_distribution": "fsdp",
}
IDENTITY = ("config.json", "model.safetensors.index.json", "distillation-parent.json")


def expert_name(layer, expert, leaf):
    projection, part = leaf.split("_")
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}_proj{SOURCE[part]}"
    )


def write_student(root):
    """A student checkpoint in the source layout: per-expert NVFP4 tensors
    over two shards, an index, and the config and lineage files."""
    generator = torch.Generator().manual_seed(20261009)
    tensors = {"lm_head.weight": torch.randn(8, HIDDEN, generator=generator).bfloat16()}
    shapes = {"gate": (INTERMEDIATE, HIDDEN), "up": (INTERMEDIATE, HIDDEN),
              "down": (HIDDEN, INTERMEDIATE)}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            for projection, (rows, columns) in shapes.items():
                packed, scale, outer = (
                    expert_name(layer, expert, f"{projection}_{part}") for part in SOURCE
                )
                tensors[packed] = torch.randint(
                    0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator
                )
                tensors[scale] = (
                    torch.rand(rows, columns // 16, generator=generator) + 0.5
                ).to(torch.float8_e4m3fn)
                tensors[outer] = torch.rand((), generator=generator) + 0.5
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[::2],
              "model-00002-of-00002.safetensors": names[1::2]}
    for shard, members in shards.items():
        save_file({name: tensors[name] for name in members}, root / shard)
    weight_map = {name: shard for shard, members in shards.items() for name in members}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (root / "config.json").write_text(json.dumps({"text_config": {"num_experts": EXPERTS}}))
    (root / "distillation-parent.json").write_text(json.dumps({"teacher": "e307"}))


def rewrite(root, name):
    """Change one byte of ``name`` in its shard; names, shapes, dtypes, the
    index, config and lineage files stay as they were."""
    shard = root / Reader(root).mapping[name]
    tensors = load_file(shard)
    tensors[name].reshape(-1).view(torch.uint8)[0] ^= 1
    save_file(tensors, shard)


class RetiredExperts(nn.Module):
    """The retired FrozenNVFP4Experts' storage (training/flash/nvfp4_experts.py
    before the removal): each projection's tensors stacked over the experts,
    non-persistent buffers when replicated, frozen parameters under FSDP."""

    def __init__(self, reader, layer, distribution):
        super().__init__()
        for leaf in NVFP4:
            stacked = torch.stack(
                [reader.tensor(expert_name(layer, e, leaf)) for e in range(EXPERTS)]
            )
            if distribution == "fsdp":
                self.register_parameter(leaf, nn.Parameter(stacked, requires_grad=False))
            else:
                self.register_buffer(leaf, stacked, persistent=False)


class BF16Experts(nn.Module):
    """The build's experts: the same checkpoint tensors expanded to BF16."""

    def __init__(self, reader, layer):
        super().__init__()
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        gate_up = [
            torch.cat([reader.dequant(f"{prefix}{e}.{p}_proj", "cpu") for p in ("gate", "up")])
            for e in range(EXPERTS)
        ]
        down = [reader.dequant(f"{prefix}{e}.down_proj", "cpu") for e in range(EXPERTS)]
        self.gate_up_proj = nn.Parameter(torch.stack(gate_up), requires_grad=False)
        self.down_proj = nn.Parameter(torch.stack(down), requires_grad=False)


def student(reader, experts):
    """Checkpoint-wrapped layers with the routed experts and a trainable
    router, and a frozen head: the fingerprint's tensor names."""
    model = nn.Module()
    layers = []
    for layer in range(LAYERS):
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = experts(layer)
        block.mlp.gate = nn.Linear(HIDDEN, EXPERTS, bias=False)
        layers.append(checkpoint_wrapper(block))
    model.layers = nn.ModuleList(layers)
    model.lm_head = nn.Parameter(reader.tensor("lm_head.weight"), requires_grad=False)
    return model


def fingerprint(model):
    """The NVFP4-era worker's ``frozen_fingerprint``: the raw bytes of each
    frozen parameter (its local FSDP shard) and buffer."""
    hashes = {}
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if not tensor.requires_grad:
            value = tensor.to_local() if hasattr(tensor, "to_local") else tensor
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
            hashes[name] = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    return hashes


def bf16_fingerprint(root):
    """The current build's fingerprint of the student at ``root``."""
    reader = Reader(root)
    return fingerprint(student(reader, lambda layer: BF16Experts(reader, layer)))


def unexpected(parents):
    raise AssertionError("a BF16-era checkpoint never reads the packed experts")


class LegacyExpertResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        write_student(self.root)
        reader = Reader(self.root)
        self.saved = fingerprint(
            student(reader, lambda layer: RetiredExperts(reader, layer, "replicated"))
        )
        self.current = bf16_fingerprint(self.root)
        self.reads = []

    def source(self, parents):
        self.reads.append(parents)
        return nvfp4_expert_digests(Reader(self.root), parents, EXPERTS)

    def test_an_unchanged_student_resumes_and_the_next_checkpoint_is_bf16(self):
        validate_resume_fingerprint(LEGACY, self.saved, self.current, self.source)
        self.assertEqual(
            self.reads,
            [[f"layers.{i}._checkpoint_wrapped_module.mlp.experts" for i in range(LAYERS)]],
        )
        # The resumed run's checkpoints carry its own manifest, without the
        # retired keys, and the BF16 fingerprint: later resumes compare that
        # exactly and never read the packed experts again.
        self.assertFalse(validate_resume_manifest(LEGACY, {}))
        validate_resume_fingerprint({}, self.current, dict(self.current), unexpected)
        changed = dict(self.current)
        changed["layers.2._checkpoint_wrapped_module.mlp.experts.down_proj"] = "x"
        with self.assertRaisesRegex(ValueError, "frozen student differs"):
            validate_resume_fingerprint({}, self.current, changed, unexpected)

    def test_changed_expert_bytes_are_refused_although_the_pinned_files_match(self):
        identity = {name: sha(self.root / name) for name in IDENTITY}
        nonexpert = {
            name: digest for name, digest in self.current.items()
            if name.rpartition(".")[2] not in BF16
        }
        for layer, expert, leaf in [(1, 3, "up_packed"), (0, 0, "down_scale"),
                                    (2, 4, "gate_outer")]:
            tensor = expert_name(layer, expert, leaf)
            with self.subTest(leaf=leaf):
                rewrite(self.root, tensor)
                try:
                    self.assertEqual(
                        {name: sha(self.root / name) for name in IDENTITY}, identity
                    )
                    # The build expands the changed experts: at most that
                    # layer's BF16 projection hashes differently (a one-ulp
                    # outer scale rounds away), every other frozen tensor
                    # hashes as saved, and the expert layout is the saved one.
                    current = bf16_fingerprint(self.root)
                    projection = "down_proj" if leaf.startswith("down") else "gate_up_proj"
                    self.assertLessEqual(
                        {name for name in current if current[name] != self.current[name]},
                        {f"layers.{layer}._checkpoint_wrapped_module.mlp.experts.{projection}"},
                    )
                    self.assertLessEqual(nonexpert.items(), current.items())
                    with self.assertRaisesRegex(
                        ValueError,
                        "packed experts no longer hash as saved "
                        rf"\(layers\.{layer}\._checkpoint_wrapped_module\.mlp\.experts\.{leaf}\)",
                    ):
                        validate_resume_fingerprint(LEGACY, self.saved, current, self.source)
                finally:
                    rewrite(self.root, tensor)  # restored
        validate_resume_fingerprint(LEGACY, self.saved, self.current, self.source)

    def test_layout_and_nonexpert_hashes_are_still_checked(self):
        def keep(fingerprint, *leaves):
            return {
                name: digest for name, digest in fingerprint.items()
                if ".mlp.experts." not in name or name.rpartition(".")[2] in leaves
            }

        for manifest, old, new in [
            ({}, self.saved, self.current),
            ({"expert_storage": "bf16"}, self.saved, self.current),
            (LEGACY, self.saved, self.current | {"lm_head": "x"}),
            (LEGACY, self.saved, keep(self.current, "gate_up_proj")),
            (LEGACY, keep(self.saved, *NVFP4[:-1]), self.current),
            (LEGACY, keep(self.saved), keep(self.current)),
        ]:
            with self.assertRaisesRegex(ValueError, "frozen student differs"):
                validate_resume_fingerprint(manifest, old, new, self.source)
        self.assertEqual(self.reads, [])
        with self.assertRaisesRegex(ValueError, "unknown NVFP4 expert distribution"):
            nvfp4_expert_digests(Reader(self.root), [], EXPERTS, "sharded")

    def test_every_rank_recomputes_the_hashes_it_saved(self):
        # FSDP2 sharded the packed stacks' expert dimension (5 experts on 4
        # ranks: 2, 2, 1 and 0), replicated buffers stayed whole; the ranks
        # split the replicated layers (3 on 4 ranks) and gather.
        context = mp.get_context("spawn")
        output = str(self.root / "rank{}.json")
        workers = [
            context.Process(
                target=run_rank,
                args=(rank, _fsdp_rank, str(self.root), str(self.root / "rendezvous"),
                      output),
            )
            for rank in range(WORLD)
        ]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=120)
                self.assertEqual(worker.exitcode, 0)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
        results = [json.loads(Path(output.format(rank)).read_text()) for rank in range(WORLD)]
        for distribution in ("replicated", "fsdp"):
            for rank, result in enumerate(results):
                saved, recomputed = result[distribution]
                with self.subTest(distribution=distribution, rank=rank):
                    self.assertEqual(len(saved), LAYERS * len(NVFP4))
                    self.assertEqual(recomputed, saved)
        self.assertEqual(results[0]["replicated"], results[3]["replicated"])
        empty = hashlib.sha256().hexdigest()
        self.assertEqual(set(results[3]["fsdp"][0].values()), {empty})
        self.assertNotIn(empty, results[2]["fsdp"][0].values())


def _fsdp_rank(rank, root, rendezvous, output):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=WORLD,
        timeout=timedelta(seconds=60),
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cpu", (WORLD,))
        reader = Reader(root)
        result = {}
        for distribution in ("replicated", "fsdp"):
            model = student(
                reader, lambda layer: RetiredExperts(reader, layer, distribution)
            )
            # As the NVFP4-era build sharded it: each layer, then the root,
            # with the trainable routers left out.
            routers = {layer.mlp.gate.weight for layer in model.layers}
            for layer in model.layers:
                fully_shard(layer, mesh=mesh, ignored_params=routers)
            fully_shard(model, mesh=mesh, ignored_params=routers)
            saved = {
                name: digest for name, digest in fingerprint(model).items()
                if ".mlp.experts." in name
            }
            parents = sorted({name.rpartition(".")[0] for name in saved})
            recomputed = nvfp4_expert_digests(
                Reader(root), parents, EXPERTS, distribution, rank, WORLD
            )
            result[distribution] = [saved, recomputed]
        Path(output.format(rank)).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
