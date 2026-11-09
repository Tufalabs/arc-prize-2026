"""Reject harness imports that drift from the verified capsule installation."""

import json
from types import SimpleNamespace

import pytest

from scripts.flash_mixed_cosine120 import provenance


@pytest.mark.parametrize("mutation", [None, "contents", "path"])
def test_imported_package_identity_detects_shadowing_and_tampering(tmp_path, monkeypatch, mutation):
    package = tmp_path / "seal"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("value = 1\n")
    receipt = tmp_path / "identity.json"
    receipt.write_text(json.dumps({"harness_commit": "pinned", "packages": {
        "seal": {"path": str(package), "sha256": provenance.tree_digest(package)},
    }}))
    monkeypatch.setenv("FLASH_MIXED_HARNESS_IDENTITY", str(receipt))
    if mutation == "contents":
        source.write_text("value = 2\n")
    imported = tmp_path if mutation == "path" else package
    monkeypatch.setattr(provenance.importlib.util, "find_spec", lambda _: SimpleNamespace(
        submodule_search_locations=[str(imported)],
    ))
    if mutation:
        with pytest.raises(ValueError, match="differs from its source capsule"):
            provenance.imported_harness_identity()
    else:
        assert provenance.imported_harness_identity()["harness_commit"] == "pinned"


def test_uncommitted_candidate_cannot_publish_data_readiness(tmp_path, monkeypatch):
    from scripts.flash_mixed_cosine120 import prepare

    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"passes": 2, "rows": {"data_sha256": "a" * 64}}))
    monkeypatch.setattr(prepare, "RECIPE", recipe)
    monkeypatch.setattr(prepare, "PUBLIC", tmp_path)
    with pytest.raises(ValueError, match="differs from the committed recipe"):
        prepare.publish({"passes": 2, "rows": {"data_sha256": "b" * 64}}, {})
    assert not (tmp_path / "DATA_READY.json").exists()
