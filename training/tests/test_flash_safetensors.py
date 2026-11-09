"""The canonical BF16 safetensors writer and reader the Flash router files use (CPU)."""

import hashlib
import json
import struct

import pytest
import torch

from training import flash_safetensors
from training.flash_routers import (
    ROUTER_FILE,
    hf_router_name,
    load_router_file,
    parse_routers,
    read_router_manifest,
    serialize_routers,
    write_router_publication,
)


def _routers(layers=3, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return {hf_router_name(layer): torch.randn(7, 4, generator=generator).bfloat16() for layer in range(layers)}


# The router writer as its module carried it before the writer was shared, kept to pin the bytes.


def _router_file_of_record(tensors):
    layers = {int(name.split(".")[3]): name for name in tensors}
    shape = None
    header = {"__metadata__": {"format": "pt"}}
    payload = []
    offset = 0
    for layer in range(len(layers)):
        name = layers[layer]
        value = tensors[name]
        shape = shape or tuple(value.shape)
        raw = value.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        payload.append(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return b"".join([struct.pack("<Q", len(encoded)), encoded, *payload])


def test_router_files_keep_their_bytes_and_round_trip(tmp_path):
    routers = _routers()
    shuffled = {name: routers[name] for name in reversed(list(routers))}
    data = serialize_routers(shuffled)
    assert data == _router_file_of_record(routers)
    manifest = write_router_publication(tmp_path, shuffled, iteration=4)
    assert (tmp_path / ROUTER_FILE).read_bytes() == data
    assert manifest == {"file": ROUTER_FILE, "sha256": hashlib.sha256(data).hexdigest(), "iteration": 4,
                        "tensors": 3}
    assert read_router_manifest(tmp_path) == manifest
    loaded = load_router_file(tmp_path / ROUTER_FILE, manifest["sha256"])
    assert list(loaded) == list(routers)
    assert all(torch.equal(loaded[name], value) for name, value in routers.items())
    assert all(torch.equal(parse_routers(data)[name], value) for name, value in routers.items())
    with pytest.raises(ValueError, match="has sha256"):
        load_router_file(tmp_path / ROUTER_FILE, "0" * 64)
    with pytest.raises(ValueError, match="router sha256 must be 64 lowercase hex digits"):
        load_router_file(tmp_path / ROUTER_FILE, manifest["sha256"].upper())


def test_routers_are_refused_unless_canonical():
    routers = _routers()
    with pytest.raises(ValueError, match="0..n-1"):
        serialize_routers({name: value for name, value in routers.items() if not name.endswith(".1.mlp.gate.weight")})
    with pytest.raises(ValueError, match="expected"):
        serialize_routers(routers | {hf_router_name(3): torch.zeros(5, 4).bfloat16()})
    with pytest.raises(ValueError, match="BF16 matrix"):
        serialize_routers(routers | {hf_router_name(0): routers[hf_router_name(0)].float()})
    # A well-formed safetensors file the module would not have written: other metadata, other order.
    other = flash_safetensors.serialize(list(routers.items()), {"format": "pt", "note": "x"}, kind="router")
    reordered = flash_safetensors.serialize(list(reversed(routers.items())), {"format": "pt"}, kind="router")
    for data in (other, reordered):
        with pytest.raises(ValueError, match="canonical form"):
            parse_routers(data)
    with pytest.raises(ValueError, match="not a Flash router tensor name"):
        parse_routers(flash_safetensors.serialize([("layers.0.gate", routers[hf_router_name(0)])], {"format": "pt"},
                                                  kind="router"))


def _file(entries, metadata=None, data=b""):
    header = {"__metadata__": {"format": "pt"} if metadata is None else metadata, **entries}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + data


ENTRY = {"dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]}


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"\x01\x00", "is truncated"),
        (struct.pack("<Q", 64) + b"{}", "header is truncated"),
        (struct.pack("<Q", 8) + b"not json", "header is not JSON"),
        (_file({"a": ENTRY}, metadata=[], data=b"\x00" * 4), "has no metadata"),
        (_file({"a": ENTRY}, metadata={"format": 1}, data=b"\x00" * 4), "map names to strings"),
        (_file({"a": ENTRY | {"dtype": "F32"}}, data=b"\x00" * 4), "a must be stored as BF16"),
        (_file({"a": ENTRY | {"extra": 1}}, data=b"\x00" * 4), "a must be stored as BF16"),
        (_file({"a": ENTRY | {"shape": [2]}}, data=b"\x00" * 4), "invalid shape or extent"),
        (_file({"a": ENTRY | {"data_offsets": [0, 6]}}, data=b"\x00" * 6), "invalid shape or extent"),
        (_file({"a": ENTRY | {"data_offsets": [2, 6]}}, data=b"\x00" * 6), "does not follow"),
        (_file({"a": ENTRY, "b": ENTRY}, data=b"\x00" * 8), "does not follow"),
        (_file({"a": ENTRY}, data=b"\x00" * 6), "does not cover the file exactly"),
    ],
)
def test_parse_refuses_anything_but_tiled_bf16_matrices(data, match):
    with pytest.raises(ValueError, match=match):
        flash_safetensors.parse(data, source="file")


def test_parse_returns_writable_views_in_header_order():
    first, second = torch.arange(6.0).reshape(2, 3).bfloat16(), torch.ones(1, 2).bfloat16()
    data = flash_safetensors.serialize([("z", first), ("a", second)], {"format": "pt"}, kind="tensor")
    metadata, tensors = flash_safetensors.parse(data, source="file")
    assert metadata == {"format": "pt"} and list(tensors) == ["z", "a"]
    assert torch.equal(tensors["z"], first) and torch.equal(tensors["a"], second)
    buffer = bytearray(data)
    _, views = flash_safetensors.parse(buffer, source="file")
    views["a"].zero_()
    assert buffer != bytearray(data)  # zero-copy over a bytearray
    with pytest.raises(ValueError, match="on the CPU"):
        flash_safetensors.serialize([("z", first.to("meta"))], {}, kind="tensor")
