"""Preparation must not reuse stale data after code, flags or converted inputs change."""

import json

import pytest

from sft.logprob.preparation_cache import cache_matches, cache_record


def conversion(code, *, thinking=True, archive="archive-v1"):
    return cache_record(
        "conversion",
        inputs={"active_traces": ["run-a", "run-b"], "archive_sha256": archive},
        options={"summary_thinking": thinking, "skip_unparseable_calls": True},
        code_files={"conversion/kimi_steps.py": code},
    )


@pytest.mark.parametrize("change", ["code", "summary_flag", "archive"])
def test_same_trace_names_cannot_hide_conversion_changes(tmp_path, change):
    code = tmp_path / "converter.py"
    code.write_text("version = 1\n")
    cache = tmp_path / "INPUT_TRACES.json"
    cache.write_text(json.dumps(conversion(code)))
    assert cache_matches(cache, conversion(code))
    if change == "code":
        code.write_text("version = 2\n")
    expected = conversion(code, thinking=change != "summary_flag",
                          archive="archive-v2" if change == "archive" else "archive-v1")
    assert not cache_matches(cache, expected)


@pytest.mark.parametrize("change", ["receipt", "code", "packing_option"])
def test_same_manifest_cannot_hide_packing_changes(tmp_path, change):
    code = tmp_path / "packer.py"
    code.write_text("version = 1\n")

    def packed(receipt="receipt-v1", limit=32):
        return cache_record(
            "packing", inputs={"manifest_sha256": "fixed", "converted_receipts": {"source": receipt}},
            options={"max_completions": limit}, code_files={"conversion/kimi_rows.py": code},
        )

    cache = tmp_path / "SOURCES_SHA256.json"
    cache.write_text(json.dumps(packed()))
    assert cache_matches(cache, packed())
    if change == "code":
        code.write_text("version = 2\n")
    expected = packed(receipt="receipt-v2" if change == "receipt" else "receipt-v1",
                      limit=64 if change == "packing_option" else 32)
    assert not cache_matches(cache, expected)


def test_cache_keys_survive_checkout_relocation(tmp_path):
    first, second = tmp_path / "job-a.py", tmp_path / "job-b.py"
    first.write_text("unchanged = True\n")
    second.write_bytes(first.read_bytes())
    assert conversion(first) == conversion(second)


@pytest.mark.parametrize("contents", [None, "{", '{"sha256":"legacy-trace-names-only"}'])
def test_missing_corrupt_and_legacy_cache_entries_are_rebuilt(tmp_path, contents):
    code = tmp_path / "converter.py"
    code.write_text("version = 1\n")
    cache = tmp_path / "INPUT_TRACES.json"
    if contents is not None:
        cache.write_text(contents)
    assert not cache_matches(cache, conversion(code))
