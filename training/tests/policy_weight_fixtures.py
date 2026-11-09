"""A two-layer served model in vLLM's packed layout, its checkpoint and merged publications.

A publication is its manifest and the tensors the learner broadcasts; ``wire``
plays the broadcast for a worker's receiver, ``deliver`` runs the three worker
steps (prepare, receive, verify) as /weights/start, update and finish do.

Layer 0 is an attention (QSA) layer and layer 1 a GDN layer, as Flash
alternates them; both carry a shared expert and a router. ``ServedModel``
loads HF names the way vLLM's stacked-parameter loaders do: q/k/v into
``qkv_proj``, in_proj_qkv/z into ``in_proj_qkvz``, in_proj_b/a into
``in_proj_ba``, gate/up into ``gate_up_proj``, by narrowing the packed
parameter and copying into it.
"""

from pathlib import Path

import torch

from training.flash_policy_weights import (
    hf_weight_name,
    manifest_sha256,
    parse_weight_name,
    policy_manifest,
    tensor_checksums,
)

HIDDEN = 4
SHAPES = {
    "self_attn.q_proj": (6, HIDDEN),
    "self_attn.k_proj": (2, HIDDEN),
    "self_attn.v_proj": (2, HIDDEN),
    "self_attn.o_proj": (HIDDEN, 6),
    "linear_attn.in_proj_qkv": (6, HIDDEN),
    "linear_attn.in_proj_z": (3, HIDDEN),
    "linear_attn.in_proj_b": (2, HIDDEN),
    "linear_attn.in_proj_a": (2, HIDDEN),
    "linear_attn.out_proj": (HIDDEN, 3),
    "mlp.shared_expert.gate_proj": (5, HIDDEN),
    "mlp.shared_expert.up_proj": (5, HIDDEN),
    "mlp.shared_expert.down_proj": (HIDDEN, 5),
    "mlp.gate": (7, HIDDEN),
}
LAYERS = {0: "self_attn", 1: "linear_attn"}
TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
    "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj",
]
PACKED = {
    "self_attn.qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "linear_attn.in_proj_qkvz": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
    "linear_attn.in_proj_ba": ["linear_attn.in_proj_b", "linear_attn.in_proj_a"],
    "mlp.shared_expert.gate_up_proj": ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"],
}


def layer_modules(layer):
    mixer = LAYERS[layer]
    return [module for module in SHAPES if not module.startswith(("self_attn.", "linear_attn.")) or
            module.startswith(mixer + ".")]


def checkpoint_tensors(seed=0):
    """Every publishable weight of the two layers, as the checkpoint stores them (BF16)."""
    generator = torch.Generator().manual_seed(seed)
    return {
        hf_weight_name(layer, module): torch.randn(*SHAPES[module], generator=generator).bfloat16()
        for layer in LAYERS for module in layer_modules(layer)
    }


def write_checkpoint(root: Path, seed=0):
    """``model.safetensors`` with the publishable weights and a norm the census must ignore."""
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    tensors = checkpoint_tensors(seed)
    save_file({**tensors, "model.language_model.layers.0.input_layernorm.weight": torch.ones(HIDDEN).bfloat16()},
              str(root / "model.safetensors"), metadata={"format": "pt"})
    return tensors


def publication_names(targets=TARGETS, routers=False):
    names = []
    for layer in LAYERS:
        for module in layer_modules(layer):
            leaf = module.rpartition(".")[2]
            if (module == "mlp.gate" and routers) or (module != "mlp.gate" and leaf in targets):
                names.append(hf_weight_name(layer, module))
    return names


def make_publication(checkpoint, *, iteration=3, targets=TARGETS, routers=False, seed=1, drop=(), extra=None,
                     reshape=None, checksum_offset=None, rounding="dither"):
    """A merged publication of ``checkpoint`` moved by a fixed delta: (manifest, tensors); faults on request.

    drop: names left out; extra: {name: tensor} added; reshape: a name whose
    tensor is transposed; checksum_offset: a name whose published checksum is
    off by one (bytes that changed after the learner took its checksums);
    rounding: the merge rounding the publication declares.
    """
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name in publication_names(targets, routers):
        if name in drop:
            continue
        value = (checkpoint[name].float() + 0.25 * torch.randn(checkpoint[name].shape, generator=generator)).bfloat16()
        tensors[name] = value.t().contiguous() if name == reshape else value
    tensors.update(extra or {})
    checksums = dict(zip(tensors, tensor_checksums(tensors.values()), strict=True))
    if checksum_offset is not None:
        checksums[checksum_offset] += 1
    manifest = policy_manifest(iteration=iteration, lora_targets=targets, routers=routers, rounding=rounding,
                               shapes={name: value.shape for name, value in tensors.items()}, checksums=checksums)
    return manifest, tensors


def wire(manifest, tensors, calls=None):
    """The learner's broadcast as a worker sees it: each tensor in manifest order, one per call."""
    feed = iter([tensors[name] for name in manifest["names"]])

    def receive(buffer):
        if calls is not None:
            calls.append(tuple(buffer.shape))
        buffer.copy_(next(feed))

    return receive


def deliver(receiver, model, manifest, tensors):
    """/weights/start, update and finish on one worker: prepare, receive off the wire, verify."""
    digest = manifest_sha256(manifest)
    assert receiver.prepare(model, manifest) == {"prepared": len(manifest["names"]), "sha256": digest}
    received = receiver.receive(model, digest, wire(manifest, tensors))
    assert received["received"] == len(manifest["names"]) and received["sha256"] == digest
    return receiver.verify(digest)


class _Linear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


class ServedModel(torch.nn.Module):
    """vLLM's view of the checkpoint: packed parameters, loaded by narrowing and copying.

    Parameters carry vLLM's names (``language_model.model.layers.<i>.<parent>.<packed or
    leaf>.weight``). Faults, each a set of HF names: ``skip`` load_weights silently ignores,
    ``misplace`` it writes one row too low (over the next shard), ``fill`` it zero-fills in
    place before copying, ``flip`` it copies with the columns reversed, ``rows`` with the
    rows reversed, ``transpose`` as the elements of the transpose in row-major order;
    ``swap`` maps a name to the name whose rows it is written into instead; ``pad`` names
    packed modules (``<parent>.<packed>``) built with one row more than their parts.
    """

    def __init__(self, checkpoint, *, skip=(), misplace=(), fill=(), flip=(), rows=(), transpose=(), swap=None,
                 pad=()):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.layers = torch.nn.ModuleDict({str(layer): torch.nn.Module() for layer in LAYERS})
        self.layout = {}
        for layer in LAYERS:
            modules = layer_modules(layer)
            for packed, parts in PACKED.items():
                if parts[0] not in modules:
                    continue
                values = [checkpoint[hf_weight_name(layer, part)] for part in parts]
                if packed in pad:
                    values.append(torch.zeros(1, HIDDEN).bfloat16())
                parameter = self._add(layer, packed, torch.cat(values).clone())
                offset = 0
                for part, value in zip(parts, values, strict=False):
                    self.layout[(layer, part)] = (parameter, offset)
                    offset += value.shape[0]
            for module in modules:
                if (layer, module) not in self.layout:
                    parameter = self._add(layer, module, checkpoint[hf_weight_name(layer, module)].clone())
                    self.layout[(layer, module)] = (parameter, 0)
        self.skip, self.misplace, self.fill = set(skip), set(misplace), set(fill)
        self.flip, self.rows, self.transpose, self.swap = set(flip), set(rows), set(transpose), dict(swap or {})

    def _add(self, layer, module, weight):
        holder = self.language_model.model.layers[str(layer)]
        *parents, leaf = module.split(".")
        for part in parents:
            if not hasattr(holder, part):
                holder.add_module(part, torch.nn.Module())
            holder = getattr(holder, part)
        holder.add_module(leaf, _Linear(weight))
        return getattr(holder, leaf).weight

    def served(self, name):
        layer, module = parse_weight_name(name)
        parameter, offset = self.layout[(layer, module)]
        rows = SHAPES[module][0]
        return parameter.detach()[offset : offset + rows]

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            if name in self.skip:
                continue
            layer, module = parse_weight_name(self.swap.get(name, name))
            parameter, offset = self.layout[(layer, module)]
            if name in self.misplace:
                offset += 1
            target = parameter.data.narrow(0, offset, tensor.shape[0])
            if name in self.fill:
                target.fill_(0)
            if name in self.flip:
                tensor = tensor.flip(1)
            if name in self.rows:
                tensor = tensor.flip(0)
            if name in self.transpose:
                tensor = tensor.t().reshape(tensor.shape)
            target.copy_(tensor)
            loaded.add(name)
        return loaded
