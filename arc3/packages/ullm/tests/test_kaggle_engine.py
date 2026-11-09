"""The Kaggle engine descriptor: one object says everything a deployment
needs about the engine, and it agrees with the workspace's pin."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest
import ullm.backends.victim_offload as victim_offload
import ullm.kaggle as ullm_kaggle
import ullm.vllm_overlay as vllm_overlay


def test_the_fork_engine_says_the_whole_deployment() -> None:
    """Wheelhouse as the kernel source; the install (the record beside its
    wheels replayed, pip where it does not match), then the CuteDSL path the
    install leaves unreachable, then the overlay -- in that order, since each
    reads what the one before it laid down; the fork's repo kept out of the
    bundle; the overlay as the one payload file."""
    engine = ullm_kaggle.VLLM_FORK
    assert engine.kernel_sources() == (ullm_kaggle.KAGGLE_VLLM_WHEELS,)
    assert engine.setup_commands() == (
        ullm_kaggle.kaggle_site_install_command(*ullm_kaggle.KAGGLE_VLLM_REQUIREMENTS),
        ullm_kaggle.kaggle_cutlass_dsl_path_command(),
        ullm_kaggle.kaggle_vllm_overlay_command(),
    )
    assert engine.exclude_source_repos() == (ullm_kaggle.VLLM_FORK_REPO,)


def test_the_stock_engine_ships_no_fork() -> None:
    engine = ullm_kaggle.VLLM_STOCK
    assert engine.wheels == ullm_kaggle.VLLM_FORK.wheels
    # The path export is the box's problem, not the fork's, so a stock
    # engine on the same image needs it too.
    assert engine.setup_commands() == (
        ullm_kaggle.kaggle_site_install_command(*ullm_kaggle.KAGGLE_VLLM_REQUIREMENTS),
        ullm_kaggle.kaggle_cutlass_dsl_path_command(),
    )
    assert engine.exclude_source_repos() == ()
    assert engine.bundle_files(pathlib.Path("/anywhere")) == []


def test_the_overlay_is_built_against_the_named_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    seen: list[pathlib.Path | None] = []

    def fake_build(dest_dir: pathlib.Path | None = None, *, workspace: pathlib.Path | None = None) -> pathlib.Path:
        seen.append(workspace)
        return tmp_path / vllm_overlay.OVERLAY_PAYLOAD_NAME

    monkeypatch.setattr(vllm_overlay, "build_for_deploy", fake_build)
    assert ullm_kaggle.VLLM_FORK.bundle_files(tmp_path) == [str(tmp_path / vllm_overlay.OVERLAY_PAYLOAD_NAME)]
    assert seen == [tmp_path]


def test_the_install_command_resolves_the_engine_s_own_wheelhouse() -> None:
    """An engine on another wheelhouse attaches it AND installs from it: the
    ref the kernel source names is the ref the resolver on the kernel is
    asked for, not the shipped default."""
    custom = ullm_kaggle.KaggleEngine(wheels="owner/custom-wheels", requirements=("vllm==0.27.1",), fork_overlay=False)
    install, _dsl_path = custom.setup_commands()
    assert install.split()[3] == "owner/custom-wheels"
    assert ullm_kaggle.KAGGLE_VLLM_WHEELS not in install
    shipped, _dsl_path, _overlay = ullm_kaggle.VLLM_FORK.setup_commands()
    assert shipped.split()[3] == ullm_kaggle.KAGGLE_VLLM_WHEELS
    with pytest.raises(ValueError, match="owner/slug"):
        ullm_kaggle.kaggle_site_install_command("vllm==0.27.1", wheels="no-owner")


def _line(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def test_the_engine_line_is_said_once_and_everything_agrees() -> None:
    """The wheelhouse's slug, the requirement set's engine pin, the connector's
    line guard and the venv's installed engine all name the same line, and
    the venv's fork revision is the workspace's pin. Any of these drifting
    apart is what a version bump forgets; here it fails before a deploy."""
    engine = ullm_kaggle.VLLM_FORK
    (vllm_req,) = [r for r in engine.requirements if r.startswith("vllm==")]
    stock_version = vllm_req.removeprefix("vllm==")
    # The slug names the line, in whatever separator style it was created
    # with: "0-29-0" and "0290" both say 0.29.0.
    assert stock_version.replace(".", "") in engine.wheels.replace("-", "").replace(".", "")
    assert _line(stock_version) == victim_offload.SUPPORTED_VLLM_LINE

    pytest.importorskip("vllm", reason="the installed engine is the other half of the check")
    # Kaggle and this workspace must render and tokenize on one release, so
    # the requirement names the version the lock resolved rather than a floor.
    transformers = pytest.importorskip("transformers")
    assert ullm_kaggle.KAGGLE_TRANSFORMERS_REQUIREMENT == f"transformers=={transformers.__version__}", (
        "the Kaggle pin and the workspace lock disagree on transformers"
    )
    installed = vllm_overlay.read_installed_vllm()
    assert _line(installed.version) == victim_offload.SUPPORTED_VLLM_LINE
    workspace = pathlib.Path(__file__).resolve().parents[3]
    pinned = vllm_overlay.pinned_fork_commit(workspace)
    assert pinned is not None, "the workspace root pyproject pins the fork"
    assert installed.fork_commit == pinned, "the venv is behind the pin: uv sync"


def test_the_prebuilt_kernel_pin_is_the_installed_flashinfer() -> None:
    """The cubins and the jit cache are keyed to the FlashInfer build by
    content hash: a pin that disagrees with the installed one does not fail
    on the kernel, it misses and compiles, which is the whole cost these two
    wheels exist to remove. Its own test, so a missing flashinfer skips this
    check alone and not the engine-line checks beside it."""
    flashinfer = pytest.importorskip("flashinfer")
    assert ullm_kaggle.KAGGLE_FLASHINFER_VERSION == flashinfer.__version__, (
        "the Kaggle prebuilt-kernel pin and the workspace lock disagree on flashinfer"
    )


def test_the_kernels_run_on_the_image_the_record_was_built_on() -> None:
    """The record replays only onto the installed state it started from, so
    the kernels are pinned to the image its notebook is pinned to."""
    builder = pathlib.Path(__file__).resolve().parents[3] / "tools/kaggle/arc3-site-0290/kernel-metadata.json"
    assert json.loads(builder.read_text())["docker_image"] == ullm_kaggle.KAGGLE_DOCKER_IMAGE
    assert ullm_kaggle.VLLM_FORK.docker_image == ullm_kaggle.KAGGLE_DOCKER_IMAGE
    assert ullm_kaggle.VLLM_STOCK.docker_image == ullm_kaggle.KAGGLE_DOCKER_IMAGE


def test_a_checkpoint_in_one_dataset_is_read_where_it_is_mounted() -> None:
    checkpoint = ullm_kaggle.KaggleCheckpoint("owner/drafter")
    assert checkpoint.sources == ("owner/drafter",)
    assert checkpoint.path == f"{ullm_kaggle.KAGGLE_DATASET_MOUNT}/owner/drafter"
    assert checkpoint.assemble_command() is None


def test_a_split_checkpoint_is_read_where_its_assembly_puts_it() -> None:
    """Every dataset mounted, the engine pointed at the assembled directory,
    and the command that makes it naming both, the checkpoint's own first."""
    checkpoint = ullm_kaggle.KaggleCheckpoint("owner/prune", shards_from=("owner/base",))
    assert checkpoint.sources == ("owner/prune", "owner/base")
    assert checkpoint.path == f"{ullm_kaggle.KAGGLE_ASSEMBLED_ROOT}/owner/prune"
    assert checkpoint.assemble_command() == (
        f'"$PYTHON" -m ullm.kaggle_assemble {checkpoint.path} owner/prune owner/base'
    )
    with pytest.raises(ValueError, match="owner/slug"):
        ullm_kaggle.KaggleCheckpoint("owner/prune", shards_from=("no-owner",))


def test_a_split_checkpoint_can_take_its_shards_from_a_model_version() -> None:
    """A base whose files moved to a Kaggle model: mounted as a model, not a
    dataset, and named to the assembly as the version it is."""
    base = "owner/base/Transformers/default/1"
    checkpoint = ullm_kaggle.KaggleCheckpoint("owner/prune", shards_from=(base,))
    assert checkpoint.sources == ("owner/prune",)
    assert checkpoint.model_sources == (base,)
    assert checkpoint.path == f"{ullm_kaggle.KAGGLE_ASSEMBLED_ROOT}/owner/prune"
    assert checkpoint.assemble_command() == (f'"$PYTHON" -m ullm.kaggle_assemble {checkpoint.path} owner/prune {base}')
    assert ullm_kaggle.model_mount(base) == f"{ullm_kaggle.KAGGLE_MODEL_MOUNT}/owner/base/transformers/default/1"
    with pytest.raises(ValueError, match="<owner>/<slug>/<framework>/<variation>/<version>"):
        ullm_kaggle.KaggleCheckpoint("owner/prune", shards_from=("owner/base/Transformers/default/latest",))


def test_a_checkpoint_s_own_part_can_be_a_model_version() -> None:
    """A prune whose own shards are a Kaggle model too: no dataset mounted,
    read where its assembly puts it, or, whole, where the model mounts."""
    prune, base = "owner/prune/Transformers/default/1", "owner/base/Transformers/default/1"
    split = ullm_kaggle.KaggleCheckpoint(prune, shards_from=(base,))
    assert split.sources == ()
    assert split.model_sources == (prune, base)
    assert split.path == f"{ullm_kaggle.KAGGLE_ASSEMBLED_ROOT}/{prune}"
    assert split.assemble_command() == f'"$PYTHON" -m ullm.kaggle_assemble {split.path} {prune} {base}'
    whole = ullm_kaggle.KaggleCheckpoint(prune)
    assert whole.model_sources == (prune,) and whole.sources == ()
    assert whole.path == ullm_kaggle.model_mount(prune)
    assert whole.assemble_command() is None


def test_a_split_checkpoint_is_assembled_before_anything_reads_it() -> None:
    """Before the read-ahead, which reads the assembled directory."""
    checkpoint = ullm_kaggle.KaggleCheckpoint("owner/prune", shards_from=("owner/base",))
    engine = dataclasses.replace(ullm_kaggle.VLLM_FORK, prefetch=(checkpoint.path,), assemble=(checkpoint,))
    commands = engine.setup_commands()
    assert commands[:2] == (checkpoint.assemble_command(), ullm_kaggle.kaggle_prefetch_command(checkpoint.path))
    assert commands[2:] == ullm_kaggle.VLLM_FORK.setup_commands()


def test_a_model_is_read_where_its_version_is_mounted() -> None:
    """The ref's own path under the models mount, its framework in lower case
    as the kernel lays it out; a ref naming no single version is refused."""
    assert ullm_kaggle.model_mount("owner/slug/Transformers/default/1") == (
        f"{ullm_kaggle.KAGGLE_MODEL_MOUNT}/owner/slug/transformers/default/1"
    )
    for ref in ("owner/slug", "owner/slug/Transformers/default", "owner/slug/Transformers/default/latest"):
        with pytest.raises(ValueError, match="<version>"):
            ullm_kaggle.model_mount(ref)
