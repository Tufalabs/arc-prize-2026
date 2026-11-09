"""Merged Flash policy weights: what the learner publishes and the servers verify.

Every publication ships the policy as the dense BF16 weights it changes, never
as an adapter: each LoRA target ``W`` becomes ``W + alpha/r * B @ A``, computed
in FP32 from the learner's masters and rounded once to BF16, by default with a
fixed per-tensor dither (``merge_lora_weight``; ``training.flash.policy_weights_rounding``
selects round-to-nearest instead), and with ``training.flash.train_routers`` the
routers' BF16 casts ride along (training/FLASH_RL.md, "Serving"). The tensors
are named by the checkpoint's HF names
(``model.language_model.layers.{i}.<module>.weight``) and travel over NCCL
from the learner's rank zero straight into each server's GPU memory
(training/policy_sync/stateless_nccl.py); no publication bytes touch disk.

The publication's manifest (``policy_manifest``) lists them in canonical
(layer, then module) order, the broadcast's order, with each shape and one
checksum per tensor, a position-weighted sum of its BF16 bit patterns in HF
row-major order (``policy_checksum``) computed on the learner's device, plus
the iteration, LoRA targets, routers and rounding. Its identity is the sha256
of its canonical JSON (``manifest_sha256``): the checksums pin every element
in place, so the hash pins the served bytes.

The engine side (``PolicyWeightReceiver``, run in every vLLM worker) requires
exactly the tensors the checkpoint holds for the declared targets (no unknown,
no missing name; shapes and dtypes of the checkpoint), and maps each one to
the rows of the served matrix it belongs in, from the checkpoint's shapes and
vLLM's packing order rather than from the loader
(``PolicyWeightReceiver.homes``), before any byte moves. It loads each
received tensor through the model's own ``load_weights`` (the startup path:
HF to vLLM names, packed modules), requires every copy to land inside its
tensor's rows, and recomputes each checksum from those rows of the served
memory, read in HF order. Every weight keeps its storage, so CUDA graphs stay
valid.

CPU-only and dependency-free beyond torch, which only the tensor functions
import: the trainer's control plane reads manifests without it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import struct
import time
from typing import TYPE_CHECKING

from training.flash.lora_targets import DENSE_TARGETS, OPTIONAL_TARGETS

if TYPE_CHECKING:
    import torch

POLICY_MANIFEST = "policy_weights.json"
# /3: no file; the learner broadcasts the tensors over NCCL and the manifest
# travels in the publication. /2 files and /1 files (position-free checksums)
# are no longer read: a resume republishes from the adapter checkpoint.
FORMAT = "flash_policy_weights/3"
ROUTER_MODULE = "mlp.gate"
_PREFIX = "model.language_model.layers."
# parent module -> every leaf a publication may carry
_LEAVES = {
    parent: (*leaves, *OPTIONAL_TARGETS.get(parent, ())) for parent, leaves in DENSE_TARGETS.items()
}
_NAME = re.compile(
    re.escape(_PREFIX) + r"(\d+)\.("
    + "|".join(
        re.escape(parent) + r"\.(?:" + "|".join(map(re.escape, leaves)) + ")"
        for parent, leaves in _LEAVES.items()
    )
    + "|" + re.escape(ROUTER_MODULE) + r")\.weight"
)
_MANIFEST_KEYS = {"format", "iteration", "lora_targets", "routers", "rounding", "names", "shapes", "checksums"}
# How a merge rounds its FP32 sum to BF16 (merge_lora_weight).
ROUNDINGS = ("dither", "nearest")
# The checksum's position weights: element i of a matrix's row-major order
# counts (i mod CHECKSUM_PRIME) + 1 times (policy_checksum).
CHECKSUM_PRIME = 65521
_CHECKSUM_CHUNK = 64 * CHECKSUM_PRIME
# vLLM's packed parameters (the fork's packed_modules_mapping): with one
# tensor-parallel rank each holds its parts' rows, concatenated in this order.
PACKED_MODULES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}
_PACKED_INTO = {part: (packed, parts) for packed, parts in PACKED_MODULES.items() for part in parts}
# A served parameter's layer and module, whatever prefix vLLM's model gives it.
_SERVED_NAME = re.compile(r"(?:.*\.)?layers\.(\d+)\.(.+)")


def hf_weight_name(layer: int, module: str) -> str:
    """``module`` is ``<parent>.<leaf>`` of a LoRA target, or ``mlp.gate``."""
    name = f"{_PREFIX}{int(layer)}.{module}.weight"
    if _NAME.fullmatch(name) is None:
        raise ValueError(f"not a publishable Flash weight: {module!r}")
    return name


def parse_weight_name(name: str) -> tuple[int, str]:
    """(layer, module) of a publishable HF weight name; anything else raises."""
    match = _NAME.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        raise ValueError(f"not a publishable Flash weight name: {name!r}")
    return int(match[1]), match[2]


def _leaf(module: str) -> str | None:
    """The LoRA target leaf of a module, None for the router."""
    return None if module == ROUTER_MODULE else module.rpartition(".")[2]


def _order(name: str) -> tuple[int, str]:
    return parse_weight_name(name)


def _rounding(value) -> str:
    if value not in ROUNDINGS:
        raise ValueError(f"policy weights rounding must be one of {ROUNDINGS}, got {value!r}")
    return value


def _targets(value) -> tuple[str, ...]:
    from training.flash.lora_targets import validate_target_modules

    return validate_target_modules(list(value))


# --------------------------------------------------------------- tensors


def policy_checksum(tensor: torch.Tensor) -> torch.Tensor:
    """One BF16 matrix's checksum, a one-element int64 tensor on its device (no host sync).

    ``sum_i bits_i * ((i mod CHECKSUM_PRIME) + 1)`` over the elements in the
    matrix's logical row-major order (its HF layout), ``bits_i`` the element's
    BF16 bit pattern read as a signed int16. Position-sensitive: moving element
    i to position j changes the sum by ``bits_i * (w_j - w_i)``, and the weights
    of two positions differ whenever they are fewer than CHECKSUM_PRIME (a
    prime) elements apart, so any exchange of two unequal elements in one row
    or one column of a matrix with fewer than 65,521 rows and columns (every
    published one) changes it; a flipped, transposed or row-permuted layout
    moves most elements at once and passes only if the data happens to solve
    one linear equation. A plain sum (every weight 1) passes every permutation.
    Exact: ``|bits| <= 2**15`` and ``w <= 65521`` keep each product below
    ``2**31`` (an int32), so the int64 sum cannot overflow below ``2**32``
    elements, 8 GiB of BF16 in one matrix (a whole publication is 5.44 GiB).
    Computed in chunks of a multiple of CHECKSUM_PRIME elements, so one weight
    vector serves every chunk and the temporaries stay at 16 MiB each.
    """
    import torch

    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"policy weight checksums need BF16 tensors, got {tensor.dtype}")
    bits = tensor.detach().reshape(-1).view(torch.int16)
    total = torch.zeros(1, dtype=torch.int64, device=bits.device)
    if not bits.numel():
        return total
    weights = torch.arange(min(bits.numel(), _CHECKSUM_CHUNK), dtype=torch.int32, device=bits.device)
    weights = weights.remainder_(CHECKSUM_PRIME).add_(1)
    for start in range(0, bits.numel(), _CHECKSUM_CHUNK):
        chunk = bits[start : start + _CHECKSUM_CHUNK].to(torch.int32)
        total += chunk.mul_(weights[: chunk.numel()]).sum(dtype=torch.int64)
    return total


def tensor_checksums(tensors: Iterable[torch.Tensor]) -> list[int]:
    """``policy_checksum`` of each BF16 matrix, computed where it lives; one transfer returns them all."""
    import torch

    sums = [policy_checksum(tensor) for tensor in tensors]
    if not sums:
        return []
    return [int(value) for value in torch.cat([value.to("cpu") for value in sums]).tolist()]


@contextmanager
def _ieee_fp32_matmul():
    """FP32 matmuls without TF32, whatever the process has set."""
    import torch

    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_tf32
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        matmul.allow_tf32 = previous


def dither_seed(name: str) -> int:
    """The fixed rounding dither of one published tensor: the same in every publication."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") & (2**63 - 1)


def merge_lora_weight(
    weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, scale: float, *, seed: int,
    rounding: str = "dither",
) -> torch.Tensor:
    """``weight + scale * (lora_B @ lora_A)`` in FP32, rounded once to BF16, by default with a fixed dither.

    Round-to-nearest returns ``weight`` itself wherever the update is below half a BF16 ulp
    of it, so small updates would never reach the servers: at the recipe's iteration 50 the
    nearest-rounded policy carried 69% of the adapter's logprob effect (MW_REPORT.md). Here
    each element instead rounds away from zero with probability equal to its position between
    its two neighbouring BF16 values, decided by 16 random low bits drawn once per tensor from
    ``seed`` (``dither_seed`` of its name): unbiased, deterministic, and an element whose sum
    does not move does not change between publications. An exactly representable sum (the
    initial B = 0) comes back unchanged. ``weight`` is the frozen BF16 base (FP32 in CPU test
    models), exact in FP32.

    ``rounding="nearest"`` is the plain round-to-nearest cast of the same FP32 sum instead
    (``seed`` unused), kept to compare against the dither (``training.flash.policy_weights_rounding``).
    """
    import torch

    _rounding(rounding)
    if weight.dtype not in (torch.bfloat16, torch.float32) or weight.ndim != 2:
        raise ValueError(f"merged weights start from a BF16 matrix, got {weight.dtype} {tuple(weight.shape)}")
    if lora_A.dtype != torch.float32 or lora_B.dtype != torch.float32:
        raise ValueError("LoRA masters must be FP32")
    if lora_B.shape[0] != weight.shape[0] or lora_A.shape[1] != weight.shape[1] or lora_B.shape[1] != lora_A.shape[0]:
        raise ValueError(
            f"LoRA shapes {tuple(lora_B.shape)} @ {tuple(lora_A.shape)} do not fit weight {tuple(weight.shape)}"
        )
    with torch.no_grad(), _ieee_fp32_matmul():
        delta = lora_B.to(weight.device) @ lora_A.to(weight.device)
        exact = torch.add(weight.float(), delta, alpha=float(scale))
        if rounding == "nearest":
            return exact.to(torch.bfloat16)
        generator = torch.Generator(device=exact.device).manual_seed(int(seed))
        dither = torch.randint(0, 1 << 16, exact.shape, generator=generator, device=exact.device,
                               dtype=torch.int32)
        # The low 16 bits of an FP32 value are what BF16 drops: adding uniform bits there and
        # truncating rounds the magnitude up with exactly that probability.
        return ((exact.view(torch.int32) + dither) & -65536).view(torch.float32).to(torch.bfloat16)


# --------------------------------------------------------- the manifest


def _census_of(names, *, lora_targets, routers: bool) -> None:
    """A publication carries LoRA targets of the declared leaves and routers only if declared."""
    targets = set(_targets(lora_targets))
    for name in names:
        _, module = parse_weight_name(name)
        leaf = _leaf(module)
        if (leaf is None and not routers) or (leaf is not None and leaf not in targets):
            raise ValueError(f"{name} is not part of a publication of {sorted(targets)} (routers={routers})")


def policy_manifest(
    *, iteration: int, lora_targets, routers: bool, rounding: str, shapes: Mapping[str, Iterable[int]],
    checksums: Mapping[str, int],
) -> dict:
    """The publication's manifest: what the learner sends, in the order it sends it.

    ``shapes`` and ``checksums`` cover the same BF16 matrices under HF names;
    the names are listed in canonical (layer, then module) order, the order of
    the broadcast. ``rounding`` is how the LoRA targets were merged (routers
    are always nearest-rounded casts).
    """
    if not shapes:
        raise ValueError("no policy weights to publish")
    if set(checksums) != set(shapes):
        raise ValueError("policy weight checksums must cover exactly the published tensors")
    names = sorted(shapes, key=_order)
    return validate_policy_manifest({
        "format": FORMAT,
        "iteration": int(iteration),
        "lora_targets": list(_targets(lora_targets)),
        "routers": bool(routers),
        "rounding": _rounding(rounding),
        "names": names,
        "shapes": [[int(size) for size in shapes[name]] for name in names],
        "checksums": [int(checksums[name]) for name in names],
    })


def manifest_sha256(manifest: Mapping) -> str:
    """The publication's identity: sha256 of its canonical JSON, checksums included.

    The checksums pin every published element in place, so this hash pins the
    served bytes as the file hash of earlier formats did.
    """
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_policy_manifest(value) -> dict:
    """A manifest as policy_manifest builds it; anything else raises ValueError."""
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise ValueError("a Flash policy weights manifest has exactly " + ", ".join(sorted(_MANIFEST_KEYS)))
    if value["format"] != FORMAT:
        raise ValueError(f"policy weights manifest format {value['format']!r} is not {FORMAT}")
    names, shapes, checksums = value["names"], value["shapes"], value["checksums"]
    if (
        type(value["iteration"]) is not int or value["iteration"] < -1
        or type(value["routers"]) is not bool
        or not isinstance(value["lora_targets"], list)
        or not isinstance(names, list) or not names
        or not isinstance(shapes, list) or not isinstance(checksums, list)
        or not len(names) == len(shapes) == len(checksums)
        or len(set(names)) != len(names)
        or any(not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)
               for shape in shapes)
        or any(type(checksum) is not int for checksum in checksums)
    ):
        raise ValueError("invalid Flash policy weights manifest fields")
    _rounding(value["rounding"])
    _targets(value["lora_targets"])
    if names != sorted(names, key=_order):
        raise ValueError("policy weights manifest names must be in canonical (layer, module) order")
    _census_of(names, lora_targets=value["lora_targets"], routers=value["routers"])
    return value


def write_policy_manifest(directory: Path, manifest: Mapping) -> str:
    """Record a publication's manifest and identity beside its adapter checkpoint (atomic)."""
    validate_policy_manifest(dict(manifest))
    digest = manifest_sha256(manifest)
    path = Path(directory) / POLICY_MANIFEST
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps({"policy_sha256": digest, **manifest}, indent=1, sort_keys=True) + "\n")
    staging.replace(path)
    return digest


def read_policy_manifest(directory: Path) -> tuple[dict, str]:
    """(manifest, identity) recorded beside an adapter checkpoint; the identity must hash its manifest."""
    path = Path(directory) / POLICY_MANIFEST
    recorded = json.loads(path.read_text())
    if not isinstance(recorded, dict) or "policy_sha256" not in recorded:
        raise ValueError(f"{path} is not a Flash policy weights manifest")
    digest = recorded.pop("policy_sha256")
    manifest = validate_policy_manifest(recorded)
    if digest != manifest_sha256(manifest):
        raise ValueError(f"{path} records identity {digest!r}, not its manifest's hash")
    return manifest, digest


# ------------------------------------------------------------ the engine


def checkpoint_census(checkpoint_dir: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Every publishable weight the checkpoint holds: HF name -> (dtype, shape).

    Read from the safetensors headers the engine loaded at startup.
    """
    root = Path(checkpoint_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    elif (root / "model.safetensors").is_file():
        weight_map = None
    else:
        raise ValueError(f"{root} has no safetensors checkpoint")
    files: dict[str, list[str]] = {}
    if weight_map is None:
        files["model.safetensors"] = []
    else:
        for name, file in weight_map.items():
            if _NAME.fullmatch(name):
                files.setdefault(file, []).append(name)
    census = {}
    for file, names in files.items():
        with (root / file).open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        for name in names or [n for n in header if _NAME.fullmatch(n)]:
            entry = header[name]
            census[name] = (entry["dtype"], tuple(entry["shape"]))
    if not census:
        raise ValueError(f"{root} holds no publishable Flash weights")
    return census


class _ServedWrites:
    """Records every copy into a served weight's storage; any other write raises."""

    def __init__(self, owners: Mapping[int, str]):
        from torch.utils._python_dispatch import TorchDispatchMode

        recorder = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                recorder._observe(func, args)
                return func(*args, **(kwargs or {}))

        self.mode = Mode()
        self.owners = owners
        self.writes: list[tuple[str, object]] = []

    def _owner(self, value):
        import torch

        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            return None
        return self.owners.get(value.untyped_storage().data_ptr())

    def _observe(self, func, args):
        import torch

        if func is torch.ops.aten.copy_.default:
            owner = self._owner(args[0])
            if owner is not None:
                self.writes.append((owner, args[0]))
            return
        schema = func._schema
        if not schema.is_mutable:
            return
        for argument, value in zip(schema.arguments, args):
            if argument.alias_info is not None and argument.alias_info.is_write and self._owner(value) is not None:
                raise RuntimeError(f"{func} wrote a served weight outside a plain copy")


class _Home:
    """Rows ``[start, start + rows)`` of a served BF16 matrix: where one published tensor belongs."""

    def __init__(self, name: str, parameter, start: int, rows: int):
        self.name, self.parameter, self.start, self.rows = name, parameter, start, rows

    def matrix(self):
        """The served rows in the publication's (HF) layout."""
        return self.parameter.detach()[self.start : self.start + self.rows]

    def span(self) -> tuple[int, int, int]:
        """(storage, first element, end element) of the rows; the matrix is contiguous."""
        columns = self.parameter.shape[1]
        first = self.parameter.storage_offset() + self.start * columns
        return self.parameter.untyped_storage().data_ptr(), first, first + self.rows * columns

    def holds(self, view) -> bool:
        """Whether every element ``view`` can address lies in these rows."""
        storage, first, end = self.span()
        if view.numel() == 0:
            return True
        last = view.storage_offset() + sum((size - 1) * stride for size, stride in zip(view.shape, view.stride()))
        return view.untyped_storage().data_ptr() == storage and first <= view.storage_offset() and last < end

    def __str__(self):
        return f"{self.name} rows {self.start}:{self.start + self.rows}"


class PolicyWeightReceiver:
    """One vLLM worker's side of a publication, in three RPCs.

    ``prepare`` (at /weights/start) checks the manifest against the checkpoint
    and finds every tensor's home before any byte moves; ``receive`` (at
    /weights/update) takes each tensor off the NCCL broadcast in manifest order
    and loads it through the model's own ``load_weights``, auditing every
    write; ``verify`` (at /weights/finish) recomputes each checksum from the
    served rows and compares it with the learner's.
    """

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._census = None
        # (targets, routers, names) of the first publication; later ones must match.
        self.pinned = None
        self._pending: dict | None = None

    def census(self):
        if self._census is None:
            self._census = checkpoint_census(self.checkpoint_dir)
        return self._census

    def expected(self, lora_targets, routers: bool) -> set[str]:
        targets = set(_targets(lora_targets))
        names = set()
        for name in self.census():
            leaf = _leaf(parse_weight_name(name)[1])
            if (leaf is None and routers) or (leaf is not None and leaf in targets):
                names.add(name)
        return names

    def check(self, manifest: dict) -> None:
        """Exactly the checkpoint's tensors for the declared targets, as the checkpoint stores them."""
        names = manifest["names"]
        expected = self.expected(manifest["lora_targets"], manifest["routers"])
        unknown, missing = sorted(set(names) - expected), sorted(expected - set(names))
        if unknown or missing:
            raise RuntimeError(
                f"policy weights name {len(unknown)} tensor(s) the served model does not expect "
                f"{unknown[:3]} and lack {len(missing)} it expects {missing[:3]}"
            )
        census = self.census()
        for name, shape in zip(names, manifest["shapes"], strict=True):
            dtype, stored = census[name]
            if dtype != "BF16" or tuple(shape) != stored:
                raise RuntimeError(
                    f"policy weight {name} is BF16 {tuple(shape)}; the checkpoint holds {dtype} {stored}"
                )
        pinned = (tuple(manifest["lora_targets"]), manifest["routers"], frozenset(names))
        if self.pinned is not None and pinned != self.pinned:
            raise RuntimeError("policy weights publish other tensors than this engine's earlier publication")

    def homes(self, parameters: Mapping[str, torch.Tensor], names: Iterable[str]) -> dict[str, list[_Home]]:
        """Where each published tensor belongs in the served model, independent of its loader.

        A leaf vLLM packs (PACKED_MODULES) belongs at its running row offset among
        its parts in ``layers.<i>.<parent>.<packed>.weight``, a matrix whose rows
        the parts (the checkpoint's shapes) must tile exactly; any tensor may
        instead be the whole of ``layers.<i>.<parent>.<leaf>.weight``. Each home
        must be a contiguous BF16 matrix of that shape. Names aliasing one matrix
        count once; a tensor with no home is refused before any weight changes.
        """
        import torch

        served: dict[tuple[int, str], dict] = {}
        for served_name, parameter in parameters.items():
            match = _SERVED_NAME.fullmatch(served_name)
            if match is not None and parameter.device.type != "meta":
                identity = (parameter.untyped_storage().data_ptr(), parameter.storage_offset(),
                            tuple(parameter.shape), parameter.stride())
                served.setdefault((int(match[1]), match[2]), {}).setdefault(identity, (served_name, parameter))
        census = self.census()
        homes = {}
        for name in names:
            layer, module = parse_weight_name(name)
            parent, _, leaf = module.rpartition(".")
            rows, columns = census[name][1]
            layouts = [(f"{parent}.{leaf}.weight", 0, rows)]
            if leaf in _PACKED_INTO:
                packed, parts = _PACKED_INTO[leaf]
                stored = [census.get(hf_weight_name(layer, f"{parent}.{part}")) for part in parts]
                if all(entry is not None and entry[1][1] == columns for entry in stored):
                    heights = [entry[1][0] for entry in stored]
                    layouts.append((f"{parent}.{packed}.weight", sum(heights[: parts.index(leaf)]), sum(heights)))
            found = []
            for module_name, start, total in layouts:
                for served_name, parameter in served.get((layer, module_name), {}).values():
                    if (parameter.dtype != torch.bfloat16 or tuple(parameter.shape) != (total, columns)
                            or not parameter.is_contiguous()):
                        raise RuntimeError(
                            f"served weight {served_name} is {parameter.dtype} {tuple(parameter.shape)}; policy "
                            f"weight {name} belongs at rows {start}:{start + rows} of a contiguous BF16 "
                            f"({total}, {columns}) matrix"
                        )
                    found.append(_Home(served_name, parameter, start, rows))
            if not found:
                expected = " or ".join(f"layers.{layer}.{module_name}" for module_name, _, _ in layouts)
                raise RuntimeError(f"the served model has no {expected} to hold policy weight {name}")
            homes[name] = found
        return homes

    def prepare(self, model, manifest: dict) -> dict:
        """Check a publication's manifest and place its tensors; nothing is written."""
        self._pending = None
        manifest = validate_policy_manifest(manifest)
        if any(getattr(module, "is_fused_shared_expert_enabled", False) or getattr(module, "_fse_fuse_gate", False)
               for module in model.modules()):
            raise RuntimeError("fused shared experts keep private weight copies; policy weights are unsupported")
        self.check(manifest)
        homes = self.homes(dict(model.named_parameters(remove_duplicate=False)), manifest["names"])
        digest = manifest_sha256(manifest)
        self._pending = {"manifest": manifest, "sha256": digest, "homes": homes, "landed": None}
        return {"prepared": len(manifest["names"]), "sha256": digest}

    def _require_pending(self, sha256: str, *, received: bool) -> dict:
        pending, self._pending = self._pending, None
        if pending is None or pending["sha256"] != sha256 or (pending["landed"] is not None) != received:
            # Out of order or another publication: it starts over from prepare.
            raise RuntimeError(f"no {'received' if received else 'prepared'} policy weights publication {sha256}")
        return pending

    def receive(self, model, sha256: str, receive) -> dict:
        """Take every declared tensor off the broadcast, in manifest order, and load it in place.

        ``receive(buffer)`` fills an empty BF16 buffer with the next tensor
        (one NCCL broadcast from the learner). Each tensor goes through the
        model's own ``load_weights`` alone, so every copy into a served weight
        is attributed to it: all of them must land inside its rows, exactly
        once over, and no other write may touch a served weight. After a
        refusal the remaining tensors are still received (and dropped), so the
        learner's broadcast completes and the failure surfaces as this call's
        error instead of a hung collective.
        """
        import torch

        started = time.perf_counter()
        pending = self._require_pending(sha256, received=False)
        manifest, homes = pending["manifest"], pending["homes"]
        parameters = dict(model.named_parameters(remove_duplicate=False))
        owners = {}
        for name, parameter in parameters.items():
            if parameter.device.type != "meta":
                owners.setdefault(parameter.untyped_storage().data_ptr(), name)
        addresses = {name: parameter.data_ptr() for name, parameter in parameters.items()}
        recorder = _ServedWrites(owners)
        landed: dict[str, _Home] = {}
        error = None
        # vLLM may build its weights as inference tensors; in-place writes to
        # those (and to ordinary frozen tensors) are allowed in inference mode.
        with torch.inference_mode():
            for name, shape in zip(manifest["names"], manifest["shapes"], strict=True):
                buffer = torch.empty(shape, dtype=torch.bfloat16, device=homes[name][0].parameter.device)
                receive(buffer)
                if error is not None:
                    continue
                try:
                    landed[name] = self._load(model, recorder, name, buffer, homes[name])
                except Exception as exc:  # noqa: BLE001 - drain the broadcast, then raise
                    error = exc
            if any(parameter.is_cuda for parameter in parameters.values()):
                torch.cuda.synchronize()
        if error is not None:
            raise error
        if {name: parameter.data_ptr() for name, parameter in parameters.items()} != addresses:
            raise RuntimeError("a policy weight update reallocated a served weight")
        spans = sorted((home.span(), name) for name, home in landed.items())
        for ((storage, _, end), name), ((other, first, _), next_name) in zip(spans, spans[1:]):
            if storage == other and first < end:
                raise RuntimeError(f"policy weights {name} and {next_name} share served rows")
        self._pending = {**pending, "landed": landed}
        return {"received": len(landed), "sha256": sha256, "seconds": round(time.perf_counter() - started, 3)}

    @staticmethod
    def _load(model, recorder, name, tensor, homes):
        import torch

        recorder.writes = []
        with recorder.mode:
            loaded = model.load_weights([(name, tensor)])
        writes = recorder.writes
        if not loaded or not writes:
            raise RuntimeError(f"the served model did not load policy weight {name}")
        if any(view.dtype != torch.bfloat16 for _, view in writes):
            raise RuntimeError(f"policy weight {name} landed in a non-BF16 served weight")
        if sum(view.numel() for _, view in writes) != tensor.numel():
            raise RuntimeError(f"policy weight {name} did not land exactly once")
        # Every copy inside the rows the tensor belongs in, wherever the loader aimed.
        inside = [home for home in homes if all(home.holds(view) for _, view in writes)]
        if not inside:
            raise RuntimeError(f"policy weight {name} landed outside its served rows ({', '.join(map(str, homes))})")
        return inside[0]

    def verify(self, sha256: str) -> dict:
        """Each tensor's checksum from its own served rows, read in HF order, against the learner's.

        After every copy: a later tensor overwriting an earlier one's rows, or
        any permutation, shows here.
        """
        import torch

        started = time.perf_counter()
        pending = self._require_pending(sha256, received=True)
        manifest, landed = pending["manifest"], pending["landed"]
        names = manifest["names"]
        with torch.inference_mode():
            served = dict(zip(names, tensor_checksums(landed[name].matrix() for name in names), strict=True))
        published = dict(zip(names, manifest["checksums"], strict=True))
        wrong = [name for name in names if served[name] != published[name]]
        if wrong:
            raise RuntimeError(f"{len(wrong)} served policy weight(s) differ from the published checksums: {wrong[:3]}")
        self.pinned = (tuple(manifest["lora_targets"]), manifest["routers"], frozenset(names))
        return {
            "applied": len(names),
            "sha256": sha256,
            "iteration": manifest["iteration"],
            "checksums": served,
            "seconds": round(time.perf_counter() - started, 3),
        }
