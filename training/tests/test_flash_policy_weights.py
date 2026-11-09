"""Merged Flash policy weights: the merge, the manifest, and a worker's receive (CPU)."""

import hashlib
import json

import pytest
import torch

from policy_weight_fixtures import (
    TARGETS,
    ServedModel,
    deliver,
    make_publication,
    publication_names,
    wire,
    write_checkpoint,
)
from training.flash_policy_weights import (
    CHECKSUM_PRIME,
    POLICY_MANIFEST,
    PolicyWeightReceiver,
    dither_seed,
    manifest_sha256,
    merge_lora_weight,
    parse_weight_name,
    policy_manifest,
    read_policy_manifest,
    tensor_checksums,
    validate_policy_manifest,
    write_policy_manifest,
)


def _neighbours(exact):
    """The two BF16 values around each FP32 value: toward and away from zero."""
    bits = exact.view(torch.int32)
    toward = (bits & -65536).view(torch.float32)
    away = ((bits & -65536) + 65536).view(torch.float32)
    representable = (bits & 65535) == 0
    return toward, torch.where(representable, toward, away)


def _case(seed=0, scale=0.05):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(64, 48, generator=generator).bfloat16()
    lora_A = torch.randn(4, 48, generator=generator)
    lora_B = scale * torch.randn(64, 4, generator=generator)
    return weight, lora_A, lora_B


def test_merge_is_fp32_math_rounded_once_with_a_fixed_dither():
    weight, lora_A, lora_B = _case()
    exact = weight.float() + 2.0 * (lora_B @ lora_A)
    merged = merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=dither_seed("q"))
    assert merged.dtype == torch.bfloat16 and merged.shape == weight.shape
    # One rounding of the FP32 sum, to one of its two BF16 neighbours.
    toward, away = _neighbours(exact)
    assert bool(((merged.float() == toward) | (merged.float() == away)).all())
    # Deterministic per tensor, and another tensor's dither differs.
    assert torch.equal(merged, merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=dither_seed("q")))
    assert not torch.equal(merged, merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=dither_seed("k")))
    # A zero B (the initial policy) publishes the checkpoint's bytes.
    assert torch.equal(merge_lora_weight(weight, lora_A, torch.zeros_like(lora_B), 2.0, seed=1), weight)
    with pytest.raises(ValueError, match="FP32"):
        merge_lora_weight(weight, lora_A.bfloat16(), lora_B, 2.0, seed=1)
    with pytest.raises(ValueError, match="do not fit"):
        merge_lora_weight(weight, lora_A, lora_B[:-1], 2.0, seed=1)


def _dither_of_record(weight, lora_A, lora_B, scale, seed):
    """The dithered merge as it shipped before the rounding became a setting."""
    exact = torch.add(weight.float(), lora_B @ lora_A, alpha=float(scale))
    generator = torch.Generator().manual_seed(seed)
    dither = torch.randint(0, 1 << 16, exact.shape, generator=generator, dtype=torch.int32)
    return ((exact.view(torch.int32) + dither) & -65536).view(torch.float32).to(torch.bfloat16)


@pytest.mark.parametrize("scale", [0.05, 1e-4])
def test_rounding_dithers_by_default_and_nearest_is_the_plain_cast(scale):
    weight, lora_A, lora_B = _case(scale=scale)
    exact = weight.float() + 2.0 * (lora_B @ lora_A)
    seed = dither_seed("q")
    dithered = merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=seed)
    # The default is the dither as it shipped, bit for bit.
    assert torch.equal(dithered, _dither_of_record(weight, lora_A, lora_B, 2.0, seed))
    assert torch.equal(merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=seed, rounding="dither"), dithered)
    # Nearest is the plain BF16 cast of the same FP32 sum, whatever the seed.
    nearest = merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=seed, rounding="nearest")
    assert torch.equal(nearest, exact.to(torch.bfloat16))
    assert torch.equal(merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=seed + 1, rounding="nearest"), nearest)
    assert not torch.equal(nearest, dithered)
    # A zero B (the initial policy) publishes the checkpoint's bytes under either rounding.
    assert torch.equal(merge_lora_weight(weight, lora_A, torch.zeros_like(lora_B), 2.0, seed=1, rounding="nearest"),
                       weight)
    with pytest.raises(ValueError, match="rounding"):
        merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=seed, rounding="stochastic")


def test_dithered_merge_keeps_updates_nearest_rounding_drops():
    """Updates far below half a BF16 ulp vanish under round-to-nearest; the dither keeps them
    in expectation, and an element whose sum does not move keeps its value."""
    weight, lora_A, lora_B = _case(scale=1e-4)
    delta = 2.0 * (lora_B @ lora_A)
    exact = weight.float() + delta
    # Nearest rounding publishes almost nothing of it.
    nearest = merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=0, rounding="nearest")
    assert torch.equal(nearest, exact.bfloat16())
    assert float((nearest == weight).float().mean()) > 0.9
    draws = torch.stack([merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=seed).float()
                         for seed in range(400)])
    realized = (draws - weight.float()).mean(0)
    # The mean realized update tracks the true one (per-element dither noise averages out).
    assert float(torch.nn.functional.cosine_similarity(realized.flatten(), delta.flatten(), dim=0)) > 0.95
    assert abs(float((realized * delta).sum() / (delta * delta).sum()) - 1.0) < 0.05
    # A fixed dither: the same seed gives the same elements, whatever else moved.
    once = merge_lora_weight(weight, lora_A, lora_B, 2.0, seed=7)
    moved = lora_B.clone()
    moved[:32] *= 3
    again = merge_lora_weight(weight, lora_A, moved, 2.0, seed=7)
    assert torch.equal(once[32:], again[32:])


def test_merge_ignores_tf32_settings():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        weight = torch.zeros(2, 2).bfloat16()
        merge_lora_weight(weight, torch.ones(1, 2), torch.ones(2, 1), 1.0, seed=0)
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _weighted(value):
    """The checksum's definition, in Python integers: bits_i * ((i mod P) + 1) over the row-major order."""
    bits = value.contiguous().view(torch.int16).flatten().tolist()
    return sum(int(b) * (i % CHECKSUM_PRIME + 1) for i, b in enumerate(bits))


def test_checksum_weighs_each_bit_pattern_by_its_position_in_hf_order():
    value = torch.tensor([[1.0, -2.0, 3.0], [0.5, -0.0, 7.0]]).bfloat16()
    assert tensor_checksums([value]) == [_weighted(value)]
    # The logical row-major order counts, not the memory layout.
    assert tensor_checksums([value.t()]) == tensor_checksums([value.t().contiguous()]) == [_weighted(value.t())]
    # Every rearrangement of the same elements changes it; the position-free sum it replaced does not.
    for moved in (value.flip(1), value.flip(0), value.t().reshape(value.shape), value[:, [1, 0, 2]]):
        assert int(moved.view(torch.int16).sum()) == int(value.view(torch.int16).sum())
        assert tensor_checksums([moved]) != tensor_checksums([value])
    with pytest.raises(ValueError, match="BF16"):
        tensor_checksums([value.float()])


def test_checksum_is_exact_across_chunks_and_at_the_extremes():
    from training.flash_policy_weights import _CHECKSUM_CHUNK

    count = _CHECKSUM_CHUNK + 1000
    generator = torch.Generator().manual_seed(0)
    bits = torch.randint(-(2**15), 2**15, (2, count // 2), generator=generator, dtype=torch.int16)
    weights = torch.arange(count, dtype=torch.int64) % CHECKSUM_PRIME + 1
    assert tensor_checksums([bits.view(torch.bfloat16)]) == [int((bits.flatten().long() * weights).sum())]
    # The largest product in magnitude, everywhere: no int32 or int64 overflow.
    lowest = torch.full((2, count // 2), -(2**15), dtype=torch.int16).view(torch.bfloat16)
    cycles, rest = divmod(count, CHECKSUM_PRIME)
    total = cycles * CHECKSUM_PRIME * (CHECKSUM_PRIME + 1) // 2 + rest * (rest + 1) // 2
    assert tensor_checksums([lowest]) == [-(2**15) * total]


def _canonical(names):
    return sorted(names, key=lambda n: (int(n.split(".")[3]), n.split(".", 4)[4]))


@pytest.mark.parametrize("rounding", ["dither", "nearest"])
@pytest.mark.parametrize("routers", [False, True])
def test_manifest_lists_the_broadcast_and_its_hash_pins_every_checksum(tmp_path, routers, rounding):
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    manifest, tensors = make_publication(checkpoint, iteration=7, routers=routers, rounding=rounding)
    # Canonical (layer, module) order is the broadcast's order; routers ride only when trained.
    assert manifest["names"] == _canonical(publication_names(routers=routers)) == _canonical(tensors)
    assert any(name.endswith("mlp.gate.weight") for name in manifest["names"]) == routers
    assert manifest["shapes"] == [list(tensors[name].shape) for name in manifest["names"]]
    assert manifest["checksums"] == tensor_checksums(tensors[name] for name in manifest["names"])
    assert (manifest["format"], manifest["iteration"], manifest["rounding"], manifest["routers"]) == (
        "flash_policy_weights/3", 7, rounding, routers)
    assert manifest["lora_targets"] == TARGETS
    digest = manifest_sha256(manifest)
    assert digest == hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Any element moving changes a checksum, so the identity; so do the iteration and the rounding.
    for changed in (
        {**manifest, "checksums": [manifest["checksums"][0] + 1, *manifest["checksums"][1:]]},
        {**manifest, "iteration": 8},
        {**manifest, "rounding": "nearest" if rounding == "dither" else "dither"},
    ):
        assert manifest_sha256(validate_policy_manifest(changed)) != digest
    # Recorded beside the adapter checkpoint with its identity, and read back checked.
    assert write_policy_manifest(tmp_path, manifest) == digest
    assert read_policy_manifest(tmp_path) == (manifest, digest)
    recorded = json.loads((tmp_path / POLICY_MANIFEST).read_text())
    (tmp_path / POLICY_MANIFEST).write_text(json.dumps({**recorded, "iteration": 8}))
    with pytest.raises(ValueError, match="not its manifest's hash"):
        read_policy_manifest(tmp_path)


def test_manifest_refuses_anything_but_a_declared_canonical_publication():
    value = torch.zeros(2, 2).bfloat16()
    for name, targets, routers in (
        ("model.language_model.layers.0.self_attn.q_proj.weight", ["k_proj"], False),
        ("model.language_model.layers.0.mlp.gate.weight", ["q_proj"], False),
        ("model.language_model.layers.0.input_layernorm.weight", ["q_proj"], True),
        ("model.language_model.layers.0.mlp.experts.0.up_proj.weight", ["up_proj"], False),
    ):
        with pytest.raises(ValueError):
            policy_manifest(iteration=0, lora_targets=targets, routers=routers, rounding="dither",
                            shapes={name: value.shape}, checksums={name: 0})
    name = "model.language_model.layers.0.self_attn.q_proj.weight"
    with pytest.raises(ValueError, match="checksums"):
        policy_manifest(iteration=0, lora_targets=["q_proj"], routers=False, rounding="dither",
                        shapes={name: value.shape}, checksums={})
    with pytest.raises(ValueError, match="rounding"):
        policy_manifest(iteration=0, lora_targets=["q_proj"], routers=False, rounding="stochastic",
                        shapes={name: value.shape}, checksums={name: 0})
    manifest = policy_manifest(iteration=0, lora_targets=["q_proj", "k_proj"], routers=False, rounding="dither",
                               shapes={name: (2, 2), name.replace("q_proj", "k_proj"): (2, 2),
                                       name.replace("layers.0", "layers.10"): (2, 2)},
                               checksums={name: 1, name.replace("q_proj", "k_proj"): 2,
                                          name.replace("layers.0", "layers.10"): 3})
    # Layer 10 after layer 0's k_proj and q_proj: numeric layers, then module names.
    assert [n.split(".")[3] + n.split(".")[5] for n in manifest["names"]] == ["0k_proj", "0q_proj", "10q_proj"]
    for changed in (
        {**manifest, "names": manifest["names"][::-1]},
        {**manifest, "format": "flash_policy_weights/2"},
        {**manifest, "file": "policy_weights.safetensors"},
        {**manifest, "shapes": [[2, 2], [2, 0], [2, 2]]},
        {**manifest, "checksums": [1, 2]},
        {**manifest, "iteration": -2},
        {**manifest, "names": [manifest["names"][0]] * 3},
    ):
        with pytest.raises(ValueError):
            validate_policy_manifest(changed)


@pytest.mark.parametrize("routers", [False, True])
def test_receive_loads_through_load_weights_in_place_and_verifies_served_checksums(tmp_path, routers):
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    manifest, tensors = make_publication(checkpoint, iteration=5, routers=routers)
    model = ServedModel(checkpoint)
    addresses = {name: parameter.data_ptr() for name, parameter in model.named_parameters()}
    receiver = PolicyWeightReceiver(tmp_path / "checkpoint")
    result = deliver(receiver, model, manifest, tensors)
    assert result["applied"] == len(tensors) and result["iteration"] == 5
    assert result["sha256"] == manifest_sha256(manifest)
    assert result["checksums"] == dict(zip(manifest["names"], manifest["checksums"], strict=True))
    for name, value in tensors.items():
        assert torch.equal(model.served(name), value)
    if not routers:
        # The checkpoint's routers stay served when the routers were not trained.
        for layer in (0, 1):
            name = f"model.language_model.layers.{layer}.mlp.gate.weight"
            assert torch.equal(model.served(name), checkpoint[name])
    assert {name: parameter.data_ptr() for name, parameter in model.named_parameters()} == addresses
    # Each tensor's home: its parts' rows of vLLM's packed matrix, in the checkpoint's shapes.
    homes = receiver.homes(dict(model.named_parameters()), tensors)
    assert {name: [str(home) for home in found] for name, found in homes.items()
            if ".layers.1.linear_attn." in name} == {
        f"model.language_model.layers.1.linear_attn.{leaf}.weight": [
            f"language_model.model.layers.1.linear_attn.{served}.weight rows {rows}"]
        for leaf, served, rows in (("in_proj_qkv", "in_proj_qkvz", "0:6"), ("in_proj_z", "in_proj_qkvz", "6:9"),
                                   ("in_proj_b", "in_proj_ba", "0:2"), ("in_proj_a", "in_proj_ba", "2:4"),
                                   ("out_proj", "out_proj", "0:4"))
    }
    # A second publication of the same census goes through again.
    second, tensors = make_publication(checkpoint, iteration=6, routers=routers, seed=2)
    deliver(receiver, model, second, tensors)
    assert all(torch.equal(model.served(name), value) for name, value in tensors.items())


def test_receive_accepts_any_view_of_the_right_rows(tmp_path):
    """The check reads the served matrix, not the loader's view: rows written through a transposed view hold."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    manifest, tensors = make_publication(checkpoint)

    class Strided(ServedModel):
        def load_weights(self, weights):
            for name, tensor in weights:
                parameter, offset = self.layout[parse_weight_name(name)]
                parameter.data.narrow(0, offset, tensor.shape[0]).t().copy_(tensor.t())
            return {name for name, _ in weights}

    model = Strided(checkpoint)
    assert deliver(PolicyWeightReceiver(tmp_path / "checkpoint"), model, manifest, tensors)["applied"] == len(tensors)
    assert all(torch.equal(model.served(name), value) for name, value in tensors.items())


@pytest.mark.parametrize("fault", ["flip", "rows", "transpose"])
def test_verify_refuses_served_matrices_whose_elements_moved(tmp_path, fault):
    """Every tensor lands in its own rows with exactly the published values, in other positions:
    columns reversed (the review's reproduction), rows reversed, or the transpose read row-major.
    The position-free sum of each served matrix still equals the publication's; the checksum does not."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    manifest, tensors = make_publication(checkpoint)
    model = ServedModel(checkpoint, **{fault: set(tensors)})
    with pytest.raises(RuntimeError, match=rf"^{len(tensors)} served policy weight\(s\) differ from the published"):
        deliver(PolicyWeightReceiver(tmp_path / "checkpoint"), model, manifest, tensors)
    assert len(tensors) == 15
    for name, value in tensors.items():
        served = model.served(name)
        assert not torch.equal(served, value), name
        assert int(served.view(torch.int16).sum()) == int(value.view(torch.int16).sum()), name


def test_steps_run_in_order_for_one_publication(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    manifest, tensors = make_publication(checkpoint)
    digest = manifest_sha256(manifest)
    receiver = PolicyWeightReceiver(tmp_path / "checkpoint")
    model = ServedModel(checkpoint)
    with pytest.raises(RuntimeError, match="no prepared policy weights publication"):
        receiver.receive(model, digest, wire(manifest, tensors))
    receiver.prepare(model, manifest)
    with pytest.raises(RuntimeError, match="no received policy weights publication"):
        receiver.verify(digest)
    with pytest.raises(RuntimeError, match="no prepared policy weights publication"):
        receiver.receive(model, "0" * 64, wire(manifest, tensors))
    # A refused step leaves nothing pending: the publication must start over.
    with pytest.raises(RuntimeError, match="no prepared"):
        receiver.receive(model, digest, wire(manifest, tensors))


NAME = "model.language_model.layers.0.self_attn.q_proj.weight"
IN_PROJ_B = "model.language_model.layers.1.linear_attn.in_proj_b.weight"
IN_PROJ_A = "model.language_model.layers.1.linear_attn.in_proj_a.weight"
DOWN = "model.language_model.layers.{}.mlp.shared_expert.down_proj.weight"


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("extra", "does not expect"),
        ("extra_gdn_attention", "does not expect"),
        ("missing", "lack 1"),
        ("shape", "the checkpoint holds"),
        ("checksum", "differ from the published checksums"),
        # The learner sends other bytes than its checksums promise.
        ("wire", "differ from the published checksums"),
        ("skip", "did not load"),
        ("misplace", "landed outside its served rows"),
        ("fill", "outside a plain copy"),
        ("census", "other tensors than this engine's earlier publication"),
        # A packed matrix the checkpoint's parts do not tile is no home for them.
        ("pad", "\\(5, 4\\); policy weight .*in_proj_a.weight belongs at rows 2:4 of a contiguous BF16 \\(4, 4\\)"),
        # in_proj_b and in_proj_a written into each other's rows of in_proj_ba.
        ("swap_packed", "landed outside its served rows"),
        # Layer 0's shared-expert down_proj written into layer 1's and back.
        ("swap_layers", "landed outside its served rows"),
    ],
)
def test_receiver_refuses_anything_but_the_expected_publication(tmp_path, fault, match):
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    receiver = PolicyWeightReceiver(tmp_path / "checkpoint")
    swap = {"swap_packed": {IN_PROJ_B: IN_PROJ_A, IN_PROJ_A: IN_PROJ_B},
            "swap_layers": {DOWN.format(0): DOWN.format(1), DOWN.format(1): DOWN.format(0)}}.get(fault)
    model = ServedModel(checkpoint, skip={NAME} if fault == "skip" else (),
                        misplace={NAME} if fault == "misplace" else (), fill={NAME} if fault == "fill" else (),
                        swap=swap, pad={"linear_attn.in_proj_ba"} if fault == "pad" else ())
    kwargs = {
        "extra": {"extra": {"model.language_model.layers.2.self_attn.q_proj.weight": torch.zeros(6, 4).bfloat16()}},
        # Layer 1 is a GDN layer: it has no attention projections.
        "extra_gdn_attention": {"extra": {"model.language_model.layers.1.self_attn.q_proj.weight":
                                          torch.zeros(6, 4).bfloat16()}},
        "missing": {"drop": {NAME}},
        "shape": {"reshape": "model.language_model.layers.0.self_attn.o_proj.weight"},
        "checksum": {"checksum_offset": NAME},
    }.get(fault, {})
    served = checkpoint
    if fault == "census":
        first, served = make_publication(checkpoint)
        deliver(receiver, model, first, served)
        kwargs = {"targets": [target for target in TARGETS if target != "in_proj_b"]}
    manifest, tensors = make_publication(checkpoint, **kwargs)
    if fault == "wire":
        tensors = {**tensors, NAME: tensors[NAME].flip(1).contiguous()}
    with pytest.raises((RuntimeError, ValueError), match=match):
        deliver(receiver, model, manifest, tensors)
    if fault in ("extra", "extra_gdn_attention", "missing", "shape", "census", "pad"):
        # Refused before any byte moved.
        assert all(torch.equal(model.served(name), value) for name, value in served.items())


def test_a_refused_load_still_drains_the_broadcast(tmp_path):
    """The learner sends every declared tensor whatever a worker made of the first: the worker
    keeps receiving (and drops the rest), so the collective completes and the refusal is this
    call's error, never a hung broadcast."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    manifest, tensors = make_publication(checkpoint)
    model = ServedModel(checkpoint, misplace={NAME})
    receiver = PolicyWeightReceiver(tmp_path / "checkpoint")
    receiver.prepare(model, manifest)
    calls = []
    with pytest.raises(RuntimeError, match="landed outside its served rows"):
        receiver.receive(model, manifest_sha256(manifest), wire(manifest, tensors, calls))
    assert calls == [tuple(shape) for shape in manifest["shapes"]]
    # Tensors after the refused one were received and dropped, never loaded.
    later = manifest["names"][manifest["names"].index(NAME) + 1:]
    assert later and all(torch.equal(model.served(name), checkpoint[name]) for name in later
                         if ".layers.0.self_attn." not in name)


def test_census_comes_from_the_checkpoint_headers(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    receiver = PolicyWeightReceiver(tmp_path / "checkpoint")
    assert set(receiver.census()) == set(checkpoint)
    assert receiver.expected(["q_proj"], False) == {NAME}
    assert receiver.expected(["q_proj"], True) == {NAME} | {
        f"model.language_model.layers.{layer}.mlp.gate.weight" for layer in (0, 1)
    }
    # A sharded checkpoint's index is read the same way.
    index = {"weight_map": {name: "model.safetensors" for name in checkpoint}}
    (tmp_path / "checkpoint" / "model.safetensors.index.json").write_text(json.dumps(index))
    assert set(PolicyWeightReceiver(tmp_path / "checkpoint").census()) == set(checkpoint)
