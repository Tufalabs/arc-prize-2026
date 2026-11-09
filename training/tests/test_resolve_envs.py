"""scripts/resolve_envs.py: a revision's environments are found by the identity
build_env.sbatch plans for it, in manifests and receipts, without Slurm or /data."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest
from env_fixtures import (
    ARC2_PIN,
    NAMES,
    PIN,
    ROOT,
    make_checkout,
    plan,
    source,
    write_artifact,
    write_receipt,
)

spec = importlib.util.spec_from_file_location(
    "resolve_envs", ROOT / "scripts/resolve_envs.py"
)
resolve_envs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolve_envs)


@pytest.fixture(scope="module")
def checkout(tmp_path_factory):
    path = tmp_path_factory.mktemp("checkout") / "arc-training"
    revision = make_checkout(path)
    plans = {recipe: plan(path, recipe, revision) for recipe in NAMES}
    return path, revision, plans


def resolve(checkout_path, revision, root, *recipes, quiet=False, run_dir=None):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/resolve_envs.py"),
            "--root",
            str(root),
            "--repo",
            str(checkout_path),
            "--revision",
            revision,
            *(["--quiet"] if quiet else []),
            *(["--run-dir", str(run_dir)] if run_dir else []),
            *recipes,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_builders_plan_matches_the_provenance_it_records(checkout, tmp_path):
    path, revision, plans = checkout
    python = plans["trainer"]["python"]
    assert (
        plans["trainer"]["harness_revision"]
        == plans["harness"]["harness_revision"]
        == PIN
    )
    assert plans["arc2-worker"]["arc2_harness_submodule_pin"] == ARC2_PIN
    tarballs = {
        recipe: write_artifact(
            tmp_path, recipe, str(index) * 16, source(recipe, python)
        )
        for index, recipe in enumerate(NAMES)
    }
    result = resolve(path, revision, tmp_path, *NAMES)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"{recipe} {tarballs[recipe]}" for recipe in NAMES
    ]
    # A ref resolves like its commit; the harness ignores this repository's own lock.
    write_artifact(
        tmp_path / "other",
        "harness",
        "a" * 16,
        source("harness", python, uv_lock_sha256="x"),
    )
    assert resolve(path, "HEAD", tmp_path / "other", "harness").returncode == 0


@pytest.mark.parametrize(
    ("recipe", "change"),
    [
        ("trainer", {"uv_lock_sha256": "0" * 64}),
        ("trainer", {"extras": ["flash-distillation", "fa4"]}),
        ("trainer", {"harness_packages": []}),
        ("trainer", {"harness_revision": "9" * 40}),
        ("trainer", {"python_source": "/other/python3.13"}),
        ("harness", {"harness_revision": "9" * 40}),
        ("harness", {"package": "platypus"}),
        ("arc2-worker", {"project_uv_lock_sha256": "0" * 64}),
        ("arc2-worker", {"arc2_harness_submodule_pin": "9" * 40}),
        ("arc2-worker", {"harness_revision": "9" * 40}),
    ],
)
def test_a_different_lock_pin_or_interpreter_does_not_match(
    checkout, tmp_path, recipe, change
):
    path, revision, plans = checkout
    write_artifact(
        tmp_path, recipe, "0" * 16, source(recipe, plans[recipe]["python"], **change)
    )
    result = resolve(path, revision, tmp_path, recipe)
    assert result.returncode == 3
    assert result.stdout == ""
    assert f"no {recipe} environment in {tmp_path} matches {revision}" in result.stderr
    # The refusal names the artifact it nearly matched and the field that differs.
    (field,) = change
    assert (
        f"the nearest is key {'0' * 16} ({NAMES[recipe]}-{'0' * 16}.json), "
        f"which differs in {field} (recorded "
    ) in result.stderr
    assert resolve(path, revision, tmp_path, recipe, quiet=True).stderr == ""


def test_a_refusal_names_the_nearest_record_and_both_values(checkout, tmp_path):
    path, revision, plans = checkout
    python = plans["arc2-worker"]["python"]
    empty = resolve(path, revision, tmp_path / "empty", "arc2-worker")
    assert empty.returncode == 3
    assert (
        f"there is no {NAMES['arc2-worker']}-<key> artifact in {tmp_path / 'empty'}"
        in empty.stderr
    )
    expected = plans["arc2-worker"]["project_uv_lock_sha256"]
    # Two fields off, newer; one field off, older: the nearer one is named.
    write_artifact(
        tmp_path,
        "arc2-worker",
        "a" * 16,
        source(
            "arc2-worker",
            python,
            project_uv_lock_sha256="0" * 64,
            arc2_harness_submodule_pin="9" * 40,
        ),
        created_at="2026-10-05T00:00:00+00:00",
    )
    write_artifact(
        tmp_path,
        "arc2-worker",
        "b" * 16,
        source("arc2-worker", python, harness_revision="9" * 40),
        created_at="2026-10-04T00:00:00+00:00",
    )
    result = resolve(path, revision, tmp_path, "arc2-worker")
    assert result.returncode == 3
    assert (
        f"resolve_envs:   the nearest is key {'b' * 16} ({NAMES['arc2-worker']}-{'b' * 16}.json), "
        f"which differs in harness_revision (recorded {'9' * 40}, this revision {PIN})\n"
    ) in result.stderr
    # A receipt is a record of its own: a later build of key a with only the lock
    # off (as near as key b, and newer).
    write_receipt(
        tmp_path,
        "arc2-worker",
        "a" * 16,
        source("arc2-worker", python, project_uv_lock_sha256="0" * 64),
        job="2919046",
        created_at="2026-10-06T00:00:00+00:00",
    )
    result = resolve(path, revision, tmp_path, "arc2-worker")
    assert (
        f"the nearest is key {'a' * 16} (receipts/{NAMES['arc2-worker']}-{'a' * 16}-2919046.json), "
        f"which differs in project_uv_lock_sha256 (recorded {'0' * 64}, this revision {expected})\n"
    ) in result.stderr


def test_a_receipt_matches_an_artifact_first_published_for_another_revision(
    checkout, tmp_path
):
    path, revision, plans = checkout
    python = plans["trainer"]["python"]
    # Published under an older lock; this revision's build found the same key.
    tarball = write_artifact(
        tmp_path,
        "trainer",
        "c" * 16,
        source("trainer", python, uv_lock_sha256="0" * 64),
    )
    assert resolve(path, revision, tmp_path, "trainer").returncode == 3
    write_receipt(tmp_path, "trainer", "c" * 16, source("trainer", python), job="77")
    result = resolve(path, revision, tmp_path, "trainer")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["trainer", str(tarball)]


def test_the_newest_complete_match_wins(checkout, tmp_path):
    _, _, plans = checkout
    trainer = plans["trainer"]
    matching = source("trainer", trainer["python"])
    older = write_artifact(
        tmp_path, "trainer", "a" * 16, matching, created_at="2026-09-01T00:00:00+00:00"
    )
    newer = write_artifact(
        tmp_path, "trainer", "b" * 16, matching, created_at="2026-09-02T00:00:00+00:00"
    )
    assert resolve_envs.select(trainer, tmp_path) == newer
    # A receipt dates a match by the build that recorded it.
    write_receipt(
        tmp_path,
        "trainer",
        "a" * 16,
        matching,
        job="5",
        created_at="2026-09-03T00:00:00+00:00",
    )
    assert resolve_envs.select(trainer, tmp_path) == older
    # Incomplete artifacts never match: a manifest without its tarball, a
    # receipt for an artifact that is not in the root, a tarball without a manifest.
    unpublished = write_artifact(tmp_path / "partial", "trainer", "d" * 16, matching)
    unpublished.unlink()
    write_receipt(tmp_path / "partial", "trainer", "e" * 16, matching, job="6")
    (tmp_path / "partial" / f"{NAMES['trainer']}-{'f' * 16}.tar.zst").write_bytes(b"")
    assert resolve_envs.select(trainer, tmp_path / "partial") is None


def test_names_are_matched_exactly(checkout, tmp_path):
    _, _, plans = checkout
    harness = plans["harness"]
    # harness-platypus-<key> is another environment, not harness-<key>.
    tarball = tmp_path / f"harness-platypus-{'a' * 16}.tar.zst"
    tarball.write_bytes(b"")
    manifest = {
        "name": "harness-platypus",
        "source": source("harness", harness["python"]),
    }
    (tmp_path / f"harness-platypus-{'a' * 16}.json").write_text(json.dumps(manifest))
    assert resolve_envs.select(harness, tmp_path) is None
    assert resolve_envs.select(harness, tmp_path / "missing") is None


def test_a_revision_the_builder_cannot_plan_exits_2(checkout, tmp_path):
    path, _, _ = checkout
    result = resolve(path, "f" * 40, tmp_path, "trainer")
    assert result.returncode == 2
    assert "is not a commit in" in result.stderr


def test_the_builders_receipt_is_what_the_resolver_reads(checkout, tmp_path):
    path, revision, plans = checkout
    python = plans["trainer"]["python"]
    builder = (ROOT / "scripts/build_env.sbatch").read_text()
    program = builder.split('"$build_python" -I -B -c \'', 1)[1].split(
        '\' "$work/provenance.json"', 1
    )[0]
    # Published for an older lock; a build of this revision found the same key.
    tarball = write_artifact(
        tmp_path,
        "trainer",
        "c" * 16,
        source("trainer", python, uv_lock_sha256="0" * 64),
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(source("trainer", python)))
    (tmp_path / "receipts").mkdir()
    receipt = tmp_path / "receipts" / f"{NAMES['trainer']}-{'c' * 16}-99.json"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            program,
            str(provenance),
            NAMES["trainer"],
            "c" * 16,
            str(receipt),
        ],
        env={"SLURM_JOB_ID": "99"},
        check=True,
    )
    record = json.loads(receipt.read_text())
    assert record["slurm_job"] == "99"
    assert record["tarball"] == tarball.name
    assert record["source"] == json.loads(provenance.read_text())
    result = resolve(path, revision, tmp_path, "trainer")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["trainer", str(tarball)]


def test_a_resumed_run_keeps_the_environments_it_started_with(checkout, tmp_path):
    path, revision, plans = checkout
    python = plans["trainer"]["python"]
    current = write_artifact(tmp_path, "trainer", "a" * 16, source("trainer", python))
    # The run started before a lock change that gave the trainer a new key.
    started = write_artifact(
        tmp_path,
        "trainer",
        "b" * 16,
        source("trainer", python, uv_lock_sha256="0" * 64),
    )
    harness = write_artifact(tmp_path, "harness", "c" * 16, source("harness", python))
    run = tmp_path / "run"
    run.mkdir()
    staged = "/cache/users/1000/venvs/{}-{}/bin/python"
    fingerprint = {
        "schema_version": 1,
        "training": {"python": staged.format(NAMES["trainer"], "b" * 16)},
        # Staged from another root: not here, so the revision's match is used.
        "server": {"python": staged.format("harness", "d" * 16)},
    }
    (run / "runtime_fingerprint.json").write_text(json.dumps(fingerprint))
    fresh = resolve(path, revision, tmp_path, "trainer", "harness")
    assert fresh.stdout.splitlines() == [f"trainer {current}", f"harness {harness}"]
    resumed = resolve(path, revision, tmp_path, "trainer", "harness", run_dir=run)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.stdout.splitlines() == [f"trainer {started}", f"harness {harness}"]
    # A run directory without a fingerprint changes nothing.
    assert resolve(
        path, revision, tmp_path, "trainer", run_dir=tmp_path / "new"
    ).stdout.split() == [
        "trainer",
        str(current),
    ]
